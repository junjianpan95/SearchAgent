"""SearchAgent 多轮 agent loop（verl 0.10 自定义 rollout）。

## 为什么需要它

verl 的 `actor_rollout_ref.rollout.agent.default_agent_loop` 默认是
`single_turn_agent`：模型拿到一次 prompt 后连续生成到 EOS/长度上限，
**全程不执行任何工具**。对本项目而言这有两个致命后果：

1. prompt 与环境不匹配。SFT 训练样本是「逐步决策」格式，
   instruction 形如
       Question: <q>
       Evidence:
       - <evidence>
       Search results:
       - <page>: <text>
       Next action:
   而单轮 rollout 只会喂 system + 原始问题，模型处于严重的 out-of-distribution；
2. 轨迹里永远不会出现 `Answer:` 行，奖励函数解析出的 answer_f1 ≈ 0
   （实测 0.0005，而同一 adapter 在真实评测harness 下是 0.0319）。

本模块实现与 `scripts/run_compare.py` 的评测循环**完全一致**的多轮交互：
  生成一个动作 -> 真实检索/读页/加证据 -> 把更新后的状态重新渲染成
  instruction 追加为下一条 user 消息 -> 继续生成，直到 Answer 或预算耗尽。

这样训练与评测同分布，奖励（含 answer_f1 / citation_validity /
supporting-fact 命中 / 四类惩罚）才有意义。

## 观测渲染

渲染逻辑与 `src/data_gen/build_data.py::trajectory_to_steps` 严格保持一致，
两者必须同步修改，否则又会引入训练/评测分布偏移。
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SYSTEM_PROMPT = "You are a helpful assistant."

# ---------------------------------------------------------------- 动作解析

_ACTION_PATTERNS = [
    ("Search", re.compile(r"^\s*Search\s*:\s*(.*)$")),
    ("Read", re.compile(r"^\s*Read\s*:\s*(.*)$")),
    ("AddEvidence", re.compile(r"^\s*AddEvidence\s*:\s*(.*)$")),
    ("Answer", re.compile(r"^\s*Answer\s*:\s*(.*)$")),
]


def parse_first_action(text: str) -> Optional[dict]:
    """从模型单轮输出中解析第一个动作（逐行匹配，与评测 harness 一致）。"""
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        for name, pat in _ACTION_PATTERNS:
            m = pat.match(line)
            if m:
                return {"type": name, "arg": m.group(1).strip()}
    return None


def _strip_citations(rest: str) -> tuple[str, list[int]]:
    m = re.search(r"\[citations:\s*([^\]]*)\]", rest)
    if not m:
        return rest.strip(), []
    cites = [int(x) for x in m.group(1).split(",") if x.strip().isdigit()]
    return rest[: m.start()].strip(), cites


# ---------------------------------------------------------------- 观测渲染


def render_instruction(question: str, evidence_lines: list[str],
                       last_action_type: Optional[str],
                       last_hits: Optional[list[dict]]) -> str:
    """渲染下一轮的 instruction。

    必须与 `src/data_gen/build_data.py::trajectory_to_steps` 完全一致：
        Question / Evidence / Search results / Next action
    其中 Search results 只在「上一步动作是 Search」时出现，且取前 3 条。
    """
    ctx = [f"Question: {question}"]
    if evidence_lines:
        ctx.append("Evidence:")
        ctx.extend(f"- {e}" for e in evidence_lines)
    if last_action_type == "Search" and last_hits:
        ctx.append("Search results:")
        for h in last_hits[:3]:
            ctx.append(f"- {h.get('page', '')}: {h.get('text', '')[:100]}")
    ctx.append("Next action:")
    return "\n".join(ctx)


def extract_question(instruction: str) -> str:
    m = re.search(r"^Question:\s*(.*)$", instruction or "", re.MULTILINE)
    return m.group(1).strip() if m else (instruction or "").strip()


# ---------------------------------------------------------------- 检索环境

_RETRIEVER_ACTOR = None
_RETRIEVER_NAME = "searchagent_retriever_actor"


def _get_retriever():
    """进程内惰性获取共享检索器 Ray actor。

    每个 AgentLoopWorker 都是独立进程，若各自加载 BM25 + BGE-M3 + 104K 向量
    会重复占用数 GB。这里用命名 actor 做全局限定单例，所有 worker 共享一份索引。
    """
    global _RETRIEVER_ACTOR
    if _RETRIEVER_ACTOR is not None:
        return _RETRIEVER_ACTOR

    import ray

    try:
        _RETRIEVER_ACTOR = ray.get_actor(_RETRIEVER_NAME)
        return _RETRIEVER_ACTOR
    except Exception:
        pass

    from src.train.retriever_actor import RetrieverActor

    try:
        # 非 detached：actor 归 driver 所有，训练结束时自动回收，
        # 不会在 Ray 集群里残留一个占 5GB 的僵尸 actor。
        actor = RetrieverActor.options(
            name=_RETRIEVER_NAME, num_cpus=1
        ).remote(device=os.getenv("SA_RETRIEVER_DEVICE", "cpu"))
        # 触发一次同步，确保索引真的加载成功
        ray.get(actor.ping.remote())
        _RETRIEVER_ACTOR = actor
    except Exception:
        # 并发创建竞争：别的 worker 已经建好了，直接复用
        _RETRIEVER_ACTOR = ray.get_actor(_RETRIEVER_NAME)
    return _RETRIEVER_ACTOR


# ---------------------------------------------------------------- Agent loop


@register("searchagent_multihop")
class SearchAgentLoop(AgentLoopBase):
    """多轮 search agent：文本动作格式，真实执行工具。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_turns = int(os.getenv("SA_MAX_TURNS", "6"))
        self.top_k = int(os.getenv("SA_TOP_K", "5"))
        self.max_evidence = int(os.getenv("SA_MAX_EVIDENCE", "5"))
        self.use_retriever = os.getenv("SA_USE_RETRIEVER", "1") == "1"

    # ---------------------------------------------------------- helpers
    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    async def _execute(self, action: dict) -> tuple[Optional[list[dict]], Optional[str]]:
        """执行动作，返回 (hits, evidence_line)。

        - Search  -> 真实混合检索，返回 top_k 命中
        - Read    -> 读取页面正文（本项目把命中文本直接作为证据，故无额外观测）
        - AddEvidence -> 生成一条证据行
        """
        t = action["type"]
        arg = action["arg"]

        if t == "Search":
            if not self.use_retriever:
                return [], None
            actor = _get_retriever()
            hits = await actor.search.remote(arg, self.top_k)
            return list(hits or []), None

        if t == "AddEvidence":
            if "|" in arg:
                page, text = arg.split("|", 1)
            else:
                page, text = "", arg
            return None, f"{page.strip()}: {text.strip()[:120]}"

        return None, None

    # ---------------------------------------------------------- main loop
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex

        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        question = extract_question(last_user)

        metrics: dict[str, Any] = {}
        num_turns = 0
        evidence_lines: list[str] = []
        last_action_type: Optional[str] = None
        last_hits: Optional[list[dict]] = None

        # ---------- 初始 prompt ----------
        prompt_ids = await self.ct_build_initial_tokens(messages)
        response_mask: list[int] = []
        # 注意：必须是 []，不能是 None。
        # ct_merge_assistant_token 的校验是
        #   "response_logprobs is required when assistant_logprobs is provided"
        # 首轮 response_logprobs 尚不存在，但 vLLM 已经返回了 assistant_logprobs，
        # 传 None 会直接抛 ValueError；传 [] 则合法（ToolAgentLoop 的 AgentData
        # 也是这样初始化的）。
        response_logprobs: list[float] = []
        final_action: Optional[dict] = None

        for turn in range(self.max_turns):
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=f"{request_id}-{turn}",
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    priority=priority,
                )
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

            merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                prompt_ids,
                output.token_ids,
                response_mask,
                response_logprobs if (response_logprobs or output.log_probs) else None,
                assistant_logprobs=output.log_probs if output.log_probs else None,
            )
            if response_logprobs is None:
                response_logprobs = []
            prompt_ids = merge_result.token_ids
            num_turns += 1

            assistant_text = self._decode(output.token_ids)
            messages.append({"role": "assistant", "content": assistant_text})

            if len(response_mask) >= self.response_length:
                break

            action = parse_first_action(assistant_text)
            if action is None:
                # 无法解析 -> 结束（奖励函数会对非法/空动作扣分）
                break

            final_action = action
            if action["type"] == "Answer":
                break

            hits, ev_line = await self._execute(action)
            last_action_type = action["type"]
            last_hits = hits
            if ev_line:
                if len(evidence_lines) < self.max_evidence:
                    evidence_lines.append(ev_line)

            # ---------- 追加下一轮 user 观测 ----------
            instruction = render_instruction(question, evidence_lines, last_action_type, last_hits)
            previous_messages = list(messages)
            messages.append({"role": "user", "content": instruction})
            merge_result, response_mask, response_logprobs = await self.ct_merge_context_msg(
                previous_messages,
                messages,
                prompt_ids,
                response_mask,
                response_logprobs if response_logprobs else None,
            )
            if response_logprobs is None:
                response_logprobs = []
            prompt_ids = merge_result.token_ids

            if len(response_mask) >= self.response_length:
                break

        # ---------- 收尾 ----------
        resp_ids = prompt_ids[-len(response_mask):] if response_mask else []
        p_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        result = AgentLoopOutput(
            prompt_ids=p_ids,
            response_ids=resp_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=max(num_turns + 1, 2),
            metrics=metrics,
        )
        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result
