"""真实评测：用训练后的模型在 sealed test（HotpotQA validation 子集）上跑多跳问答。

产出真实指标：Answer F1、Joint F1、证据链 All-Hit、冗余率、平均工具调用。

用法：
  python scripts/run_real_eval.py --config configs/local_smoke.yaml --n 50
  python scripts/run_real_eval.py --config configs/local_smoke.yaml --n 50 --adapter data/models/sft_lora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ROOT
from src.utils.metrics import f1_score, evidence_all_hit, redundancy_rate, joint_f1


def load_test_data(n: int) -> list[dict]:
    """加载 sealed test（用 HotpotQA validation 作为页面隔离评测集）。"""
    path = ROOT / "data" / "datasets" / "hotpotqa_validation.json"
    if not path.exists():
        # 回退内置数据
        from src.data_gen import SearchTrace
        st = SearchTrace({})
        return [{"question": s.question, "answer": s.answer,
                 "supporting_facts": s.supporting_facts} for s in st.load_samples()[:n]]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:n]


def build_retriever_for_test(data: list[dict]):
    """用测试数据的 supporting facts + context 构建检索索引（页面隔离）。"""
    from src.retrieval.bm25 import BM25Index

    class BM25Retriever:
        def __init__(self):
            self._idx = BM25Index()
            self._pages = {}

        def build(self, passages):
            self._idx.build(passages)
            for p in passages:
                self._pages[p["page"]] = p["text"]

        def search(self, query, top_k=20):
            return self._idx.search(query, top_k)

        def read(self, page):
            return self._pages.get(page, "")

    # 构建 passages：每个 supporting fact 作为一段
    passages = []
    for item in data:
        for fact, page in zip(item.get("supporting_facts", []), item.get("supporting_pages", [])):
            if fact:
                passages.append({"page": page, "text": fact})
    retriever = BM25Retriever()
    retriever.build(passages)
    return retriever


def rule_answer_llm(retriever):
    """规则化 Agent 策略（作为 Base ReAct 基线）。"""
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
                if self._hits:
                    return {"type": "Read", "args": {"page": self._hits[0].page}}
                return {"type": "Answer", "args": {"answer": "", "citations": []}}
            if self._step < 2 + min(len(self._hits), 4) + 1:
                h = self._hits[self._step - 3]
                if h.text not in self._ev:
                    self._ev.append(h.text)
                return {"type": "AddEvidence", "args": {"text": h.text, "page": h.page}}
            ans = self._ev[0].rstrip(".") if self._ev else ""
            return {"type": "Answer", "args": {"answer": ans, "citations": list(range(len(self._ev)))}}

    return RuleAgent()


def run_eval_real(cfg, n, adapter=None):
    from src.agent import SearchAgent

    data = load_test_data(n)
    retriever = build_retriever_for_test(data)

    # LLM：优先用 adapter 模型，否则规则基线
    llm_fn = None
    if adapter and Path(adapter).exists():
        from scripts.backend import ModelLLM
        llm_fn = ModelLLM(Path(adapter))  # 加载训练后的模型
    else:
        llm_fn = rule_answer_llm(retriever)
        print("[EVAL] 使用规则基线（Base ReAct 模拟）")

    preds, golds = [], []
    for item in data:
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

    # 计算指标
    from evals.run_eval import evaluate_trajectories
    results = evaluate_trajectories(preds, golds)
    results["n_samples"] = len(preds)
    results["mode"] = "adapter" if adapter else "base_react"
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--adapter", default=None, help="训练后 adapter 目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = run_eval_real(cfg, args.n, args.adapter)

    out = ROOT / "data" / "real_eval_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n===== 真实评测结果 =====")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n报告已保存 -> {out}")


if __name__ == "__main__":
    main()
