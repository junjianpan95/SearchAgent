"""四方案横向对比评测：{LoRA, DoRA} × {GRPO, GSPO} + Base ReAct 基线。

在页面隔离的 sealed test（AgenticRagTrace 划分）上评测五指标：
Answer F1 / Joint F1 / 证据链 All-Hit / 冗余率 / 平均工具调用。

用法：
  python scripts/run_compare.py --n 30
  python scripts/run_compare.py --n 30 --methods lora dora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ROOT
from src.utils.metrics import f1_score, evidence_all_hit, redundancy_rate, joint_f1


# ------------------------------------------------------------ sealed test
def load_sealed_test(n: int) -> list[dict]:
    """从 AgenticRagTrace 加载 sealed test（用后 200 条作为页面隔离评测集）。"""
    from src.data_gen import SearchTrace
    st = SearchTrace({})
    samples = st.load_agenticragtracer()
    # 用尾部作为 held-out（训练用前 1105，评测用后 200）
    test_samples = samples[-200:]
    if n < len(test_samples):
        test_samples = test_samples[:n]
    out = []
    for s in test_samples:
        out.append({
            "question": s.question,
            "answer": s.answer,
            "supporting_facts": s.supporting_facts,
            "supporting_pages": s.supporting_pages,
        })
    return out


# ------------------------------------------------------------ LLM 工厂
def make_adapter_llm(adapter_dir: Path):
    """加载 base + PEFT adapter 的 LLM。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = ROOT / "data" / "models" / "Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    # merge_and_unload：把 LoRA/DoRA 适配器合并进基座权重，推理速度与全量模型一致。
    # 关键：DoRA 在推理时每个 forward 都要做 magnitude/direction 重参数化，
    # 在 transformers 5.x + peft 0.18 下慢约 4-5 倍（实测 lora 5 条 44s vs dora 217s），
    # merge 后 dora 与 lora 速度对齐且数值等价。
    model = model.merge_and_unload()
    model.eval()

    def llm_fn(context=None, state=None, **kw):
        question = context[0].replace("Question: ", "") if context else ""
        # 构造 instruction，精确匹配 SFT 训练格式（Question/Evidence/Search results/Next action）
        lines = [f"Question: {question}"]
        if state and state.memory and len(state.memory) > 0:
            lines.append("Evidence:")
            for e in state.memory.items:
                lines.append(f"- {e.text[:100]}")
        # 上一步的检索结果
        if state and state.trajectory:
            prev = state.trajectory[-1]
            if prev.get("action", {}).get("type") == "Search":
                hits = (prev.get("result") or {}).get("hits", [])
                if hits:
                    lines.append("Search results:")
                    for h in hits[:3]:
                        lines.append(f"- {h.get('page', '')}: {h.get('text', '')[:100]}")
        lines.append("Next action:")
        instruction = "\n".join(lines)
        # 用 chat template（与 LLaMA-Factory 训练格式一致）
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return parse_text_action(text)

    return llm_fn


def parse_text_action(text: str) -> dict:
    """解析文本动作（与 SFT 训练格式 + RL thinking 输出对齐）。

    SFT 模型（lora/dora）直接输出「Search: ...」等动作行；
    RL 模型（grpo/gspo）经 verl 用原始 Qwen3 tokenizer 做 rollout，
    默认开启 thinking，会先输出一段 <think>...</think> 再跟动作行。
    因此这里不能只看第一行，而要跳过 <think> 块逐行找第一个动作行，
    与 src/train/reward_adapter.py::_parse_actions 的解析逻辑保持一致。
    """
    import re
    in_think = False
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # 跳过 <think>...</think> 块
        if line.startswith("<think") or line.startswith("</think"):
            in_think = line.startswith("<think") and not line.startswith("</think")
            continue
        if in_think:
            continue
        # 跳过 markdown 强调符（RL 模型可能输出 **answer** 而非 Answer: 动作）
        if line.startswith("Search:"):
            return {"type": "Search", "args": {"query": line[len("Search:"):].strip()}}
        if line.startswith("Read:"):
            return {"type": "Read", "args": {"page": line[len("Read:"):].strip()}}
        if line.startswith("AddEvidence:"):
            rest = line[len("AddEvidence:"):].strip()
            if "|" in rest:
                page, txt = rest.split("|", 1)
                return {"type": "AddEvidence", "args": {"page": page.strip(), "text": txt.strip()}}
            return {"type": "AddEvidence", "args": {"page": "", "text": rest}}
        if line.startswith("Answer:"):
            rest = line[len("Answer:"):].strip()
            cites = []
            m = re.search(r"\[citations:\s*([^\]]*)\]", rest)
            if m:
                try:
                    cites = [int(x) for x in m.group(1).split(",") if x.strip().isdigit()]
                except ValueError:
                    cites = []
                rest = rest[:m.start()].strip()
            return {"type": "Answer", "args": {"answer": rest, "citations": cites}}
    # 兜底：没有动作行时，取整段文本当 Answer（含 thinking 内容则去掉）
    stripped = text.replace("<think>", "").replace("</think>", "").strip()
    return {"type": "Answer", "args": {"answer": stripped, "citations": []}}


def make_rule_llm(retriever):
    """Base ReAct 规则基线。"""
    class RuleAgent:
        def __init__(self):
            self._step = 0
            self._ev = []
            self._hits = []

        def __call__(self, context=None, state=None, **kw):
            q = context[0].replace("Question: ", "") if context else ""
            self._step += 1
            if self._step == 1:
                return {"type": "Search", "args": {"query": q}}
            if self._step == 2:
                self._hits = retriever.search(q, top_k=5)
                return {"type": "Read", "args": {"page": self._hits[0].page}} if self._hits else \
                    {"type": "Answer", "args": {"answer": "", "citations": []}}
            idx = self._step - 3
            if idx < len(self._hits):
                h = self._hits[idx]
                if h.text not in self._ev:
                    self._ev.append(h.text)
                return {"type": "AddEvidence", "args": {"text": h.text, "page": h.page}}
            ans = self._ev[0].rstrip(".") if self._ev else ""
            return {"type": "Answer", "args": {"answer": ans, "citations": list(range(len(self._ev)))}}

    return RuleAgent()


# ------------------------------------------------------------ 评测主流程
def eval_one(cfg, data, retriever, llm_fn, name):
    from src.agent import SearchAgent
    import time
    preds, golds = [], []
    t0 = time.time()
    for i, item in enumerate(data):
        # 规则基线是有状态的，每个问题需重置
        if hasattr(llm_fn, "_step"):
            llm_fn._step = 0
            llm_fn._ev = []
            llm_fn._hits = []
        agent = SearchAgent(
            retriever=retriever, llm_fn=llm_fn,
            max_search=cfg.agent["max_search"], max_read=cfg.agent["max_read"],
            max_evidence=cfg.agent["max_evidence"], token_budget=cfg.agent["token_budget"],
            dedup_queries=cfg.agent["dedup_queries"],
            bridge_entity_direct_read=cfg.agent["bridge_entity_direct_read"],
            recovery_feedback=cfg.agent["recovery_feedback"],
        )
        state = agent.run(item["question"])
        preds.append({
            "answer": state.answer,
            "citations": state.citations,
            "evidence_texts": [e.text for e in state.memory.items],
            "selected_ids": list(range(len(state.memory))),
            "n_tool_calls": state.budget.snapshot()["tool_calls"],
        })
        golds.append({"answer": item["answer"], "supporting_facts": item["supporting_facts"]})
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{len(data)}  elapsed={time.time()-t0:.0f}s", flush=True)

    from evals.run_eval import evaluate_trajectories
    r = evaluate_trajectories(preds, golds)
    r["method"] = name
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--methods", nargs="*", default=["base_react", "lora", "dora", "grpo", "gspo"],
                    help="评测方案列表：base_react/lora/dora/grpo/gspo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = load_sealed_test(args.n)
    print(f"[EVAL] sealed test {len(data)} 条")

    # 混合索引（缓存）。BGE-M3 稠密检索器放 CPU，把 GPU 显存独享给 Qwen3 生成模型，
    # 避免 16GB 单卡上「检索器 + 生成模型」同时驻留导致的 OOM（100 条多轮 query 编码量小，CPU 足够）。
    from src.retrieval.index_loader import CachedHybridRetriever
    retriever = CachedHybridRetriever(device="cpu")

    results = {}
    for m in args.methods:
        print(f"\n[EVAL] 评测方案: {m}")
        if m == "base_react":
            llm = make_rule_llm(retriever)
        else:
            # SFT 方案（lora/dora）的 adapter 在 data/models/sft_<m>；
            # RL 方案（grpo/gspo）的 adapter 由 verl checkpoint 合并而来，
            # 落盘到 data/models/rl_<m>（见 docs/项目介绍.md 附录 A）。
            if m in ("grpo", "gspo"):
                # verl model_merger 把 PEFT adapter 导出到 rl_<m>/lora_adapter/ 子目录，
                # 外层目录是完整合并模型（model.safetensors），需指向子目录。
                adapter = ROOT / "data" / "models" / f"rl_{m}" / "lora_adapter"
            else:
                adapter = ROOT / "data" / "models" / f"sft_{m}"
            if not adapter.exists():
                print(f"  [SKIP] adapter 不存在: {adapter}")
                continue
            llm = make_adapter_llm(adapter)
        r = eval_one(cfg, data, retriever, llm, m)
        results[m] = r
        for k, v in r.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif k == "method":
                print(f"  {k}: {v}")

    # 保存 + 对比表
    out = ROOT / "data" / "compare_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[EVAL] 对比报告 -> {out}")

    # 打印对比表
    print("\n===== 四方案横向对比 =====")
    metrics = ["answer_f1", "joint_f1", "evidence_all_hit", "redundancy_rate", "avg_tool_calls"]
    header = "method".ljust(12) + "".join(m.ljust(18) for m in metrics)
    print(header)
    for m, r in results.items():
        row = m.ljust(12)
        for metric in metrics:
            v = r.get(metric, float("nan"))
            row += f"{v:.4f}".ljust(18) if isinstance(v, (int, float)) else str(v).ljust(18)
        print(row)


if __name__ == "__main__":
    main()
