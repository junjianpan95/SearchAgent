"""SearchAgent：多跳检索主循环。

把动作空间、Evidence Memory、预算约束、检索后端串起来，
实现"逐轮生成查询→选择支持句→带可核验引用输出答案"的完整闭环。

LLM 生成动作是可插拔的（llm_fn），训练/评测时可替换为 SFT/RL 后的模型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .action_space import Action, ActionType
from .memory import EvidenceMemory
from .runtime import Budget


@dataclass
class AgentState:
    """一次 episode 的完整状态（含最终答案与轨迹，供 SFT/RL 采集）。"""
    question: str
    memory: EvidenceMemory
    budget: Budget
    trajectory: list[dict] = field(default_factory=list)   # 完整轨迹（动作+观察）
    answer: Optional[str] = None
    citations: list[int] = field(default_factory=list)
    done: bool = False


@dataclass
class SearchAgent:
    """多跳 Search Agent。

    Args:
        retriever: 检索后端（本地混合 / Tavily），需实现 search(query)->list[Hit]
        llm_fn: 动作生成函数，输入状态上下文，输出 dict（type+args），可替换
        max_search / max_read / max_evidence / token_budget: 预算（来自 config）
        dedup_queries / bridge_entity_direct_read / recovery_feedback: 开关
    """
    retriever: Any
    llm_fn: Callable[..., dict]
    max_search: int = 4
    max_read: int = 6
    max_evidence: int = 5
    token_budget: int = 8192
    dedup_queries: bool = True
    bridge_entity_direct_read: bool = True
    recovery_feedback: bool = True

    def run(self, question: str, gold_supporting: Optional[list[str]] = None) -> AgentState:
        mem = EvidenceMemory(max_evidence=self.max_evidence)
        budget = Budget(
            max_search=self.max_search,
            max_read=self.max_read,
            token_budget=self.token_budget,
            max_evidence=self.max_evidence,
        )
        state = AgentState(question=question, memory=mem, budget=budget)

        context: list[str] = [f"Question: {question}"]

        # 硬性步数上限（安全阀）：防止重复查询/非法动作不消耗预算导致死循环
        max_steps = self.max_search + self.max_read + 1
        step = 0
        while not state.done and not budget.exhausted and step < max_steps:
            step += 1
            action = self._next_action(context, state)
            state.trajectory.append({"action": action.to_dict()})

            if action.type == ActionType.SEARCH:
                self._do_search(action, context, state, budget)
            elif action.type == ActionType.READ:
                self._do_read(action, context, state, budget)
            elif action.type == ActionType.ADD_EVIDENCE:
                self._do_add_evidence(action, context, state)
            elif action.type == ActionType.ANSWER:
                self._do_answer(action, state)
            else:
                # 非法动作：记录但不推进，触发惩罚信号
                state.trajectory[-1]["illegal"] = True
                context.append("[Illegal action, ignored]")

        # 预算耗尽仍无答案 → 强制基于已有证据回答
        if not state.done:
            state.answer = self._forced_answer(state)
            state.done = True
        return state

    # ------------------------------------------------------------ 动作执行
    def _next_action(self, context: list[str], state: AgentState) -> Action:
        raw = self.llm_fn(context=context, state=state)
        return self._parse(raw)

    def _do_search(self, action: Action, context: list[str], state: AgentState, budget: Budget) -> None:
        query = action.args.get("query", "")
        # 历史查询去重
        if self.dedup_queries and not state.memory.record_query(query):
            state.trajectory[-1]["dup_query"] = True
            context.append(f"[Duplicate query: {query}]")
            return
        budget.spend_search()
        hits = self.retriever.search(query)
        budget.spend_tokens(self._estimate_tokens(hits))
        # 桥接实体 direct-read：命中页面直接可读
        result = {"query": query, "hits": [h.to_dict() for h in hits]}
        state.trajectory[-1]["result"] = result
        context.append(f"Search('{query}') -> {len(hits)} hits")

    def _do_read(self, action: Action, context: list[str], state: AgentState, budget: Budget) -> None:
        page = action.args.get("page", "")
        budget.spend_read()
        content = self.retriever.read(page)
        budget.spend_tokens(self._estimate_tokens(content))
        state.trajectory[-1]["result"] = {"page": page, "content": content[:500]}
        context.append(f"Read('{page}') -> {len(content)} chars")

    def _do_add_evidence(self, action: Action, context: list[str], state: AgentState) -> None:
        from .memory import Evidence
        text = action.args.get("text", "")
        page = action.args.get("page", "")
        ev = Evidence(text=text, page=page, index=len(state.memory))
        ok = state.memory.add(ev)
        if not ok:
            state.trajectory[-1]["redundant_evidence"] = True
            context.append("[Evidence memory full or duplicate, rejected]")
        else:
            context.append(f"Evidence[{ev.index}] added from {page}")

    def _do_answer(self, action: Action, state: AgentState) -> None:
        state.answer = action.args.get("answer", "")
        state.citations = action.args.get("citations", [])
        state.done = True

    def _forced_answer(self, state: AgentState) -> str:
        if len(state.memory) == 0:
            return "[No evidence found]"
        joined = " ".join(e.text for e in state.memory.items)
        return f"{joined[:200]}"

    # ------------------------------------------------------------ 辅助
    @staticmethod
    def _parse(raw: dict) -> Action:
        t = raw.get("type", "")
        if t not in {a.value for a in ActionType}:
            # 非法/无法解析的动作 → 直接 Answer，避免空 Search 死循环
            return Action(ActionType.ANSWER, {"answer": "", "citations": []})
        return Action(ActionType(t), raw.get("args", {}))

    @staticmethod
    def _estimate_tokens(obj: Any) -> int:
        import json
        # Hit 对象列表转 dict 再序列化
        if isinstance(obj, list):
            obj = [getattr(o, "to_dict", lambda: o)() for o in obj]
        return len(json.dumps(obj, ensure_ascii=False)) // 4
