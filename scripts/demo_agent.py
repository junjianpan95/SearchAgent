"""端到端 Demo：无需下载大模型，用内置样例 + 规则化策略模型跑通多跳检索 Agent。

展示：SearchAgent 如何逐轮 Search → Read → AddEvidence → Answer，
完成跨文档桥接推理，输出带可核验引用的最终答案。

用途：
1. 本地立刻验证 Agent 逻辑（零依赖大模型）
2. 产出轨迹 JSON 供前端 Demo 可视化
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 让 scripts/ 下的脚本能导入项目根目录的 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ROOT
from src.agent import SearchAgent
from src.retrieval import build_retriever


def _builtin_passages() -> list[dict]:
    return [
        {"page": "The Shining (film)",
         "text": "The Shining is a 1980 horror film based on Stephen King's 1977 novel of the same name."},
        {"page": "Stephen King",
         "text": "Stephen King is an American author who wrote the novel The Shining."},
        {"page": "Inception",
         "text": "Inception is a 2010 science fiction action film directed by Christopher Nolan."},
        {"page": "Christopher Nolan",
         "text": "Christopher Nolan is a British-American filmmaker born in London, England."},
        {"page": "2010 FIFA World Cup",
         "text": "The 2010 FIFA World Cup Golden Boot was won by Thomas Müller of Germany."},
        {"page": "Thomas Müller",
         "text": "Thomas Müller is a German footballer who played for Bayern Munich in 2010."},
    ]


class BM25OnlyRetriever:
    """纯 BM25 检索（demo 用，无需 BGE-M3 大模型）。"""
    def __init__(self):
        self._pages = {}

    def build(self, passages):
        from src.retrieval.bm25 import BM25Index
        self._bm25 = BM25Index()
        self._bm25.build(passages)
        for d in passages:
            self._pages[d["page"]] = d["text"]

    def search(self, query, top_k=20):
        return self._bm25.search(query, top_k)

    def read(self, page):
        return self._pages.get(page, "")


def make_rule_llm(gold_answer: str, supporting_pages: list[str], supporting_facts: list[str]):
    """规则化策略模型：模拟一个"已经学会"的 Agent，按桥接推理逐步检索。

    真实训练后，这里替换为 SFT/RL 后的 Qwen 模型（llm_fn 接口不变）。
    """
    step = {"i": 0}
    pages = list(supporting_pages)
    facts = list(supporting_facts)

    def llm_fn(context=None, state=None):
        i = step["i"]
        step["i"] += 1
        # 依次：Search 每个桥接页面 → AddEvidence → 最后 Answer
        if i < len(pages):
            return {"type": "Search", "args": {"query": pages[i]}}
        if i < len(pages) * 2:
            idx = i - len(pages)
            return {"type": "AddEvidence",
                    "args": {"text": facts[idx], "page": pages[idx]}}
        return {"type": "Answer",
                "args": {"answer": gold_answer,
                         "citations": list(range(len(pages)))}}

    return llm_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    ap.add_argument("--question", default=None, help="指定问题（默认跑全部内置样例）")
    ap.add_argument("--out", default="data/demo_trajectories.json")
    ap.add_argument("--no-dense", action="store_true", help="纯 BM25 检索（无需 BGE-M3 大模型）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_dense:
        retriever = BM25OnlyRetriever()
    else:
        retriever = build_retriever(cfg.retrieval)
    retriever.build(_builtin_passages())

    samples = [
        ("Who is the author of the novel that the movie 'The Shining' was based on?",
         "Stephen King", ["The Shining (film)", "Stephen King"],
         ["The Shining is a 1980 horror film based on Stephen King's 1977 novel.",
          "Stephen King wrote the novel The Shining."]),
        ("What country does the director of 'Inception' come from?",
         "United Kingdom", ["Inception", "Christopher Nolan"],
         ["Inception was directed by Christopher Nolan.",
          "Christopher Nolan is a British-American filmmaker born in London."]),
        ("Which club did the 2010 World Cup Golden Boot winner play for that year?",
         "Bayern Munich", ["2010 FIFA World Cup", "Thomas Müller"],
         ["The 2010 FIFA World Cup Golden Boot was won by Thomas Müller.",
          "Thomas Müller played for Bayern Munich in 2010."]),
    ]

    all_demos = []
    for question, answer, pages, facts in samples:
        llm = make_rule_llm(answer, pages, facts)
        agent = SearchAgent(
            retriever=retriever,
            llm_fn=llm,
            max_search=cfg.agent["max_search"],
            max_read=cfg.agent["max_read"],
            max_evidence=cfg.agent["max_evidence"],
            token_budget=cfg.agent["token_budget"],
            dedup_queries=cfg.agent["dedup_queries"],
            bridge_entity_direct_read=cfg.agent["bridge_entity_direct_read"],
            recovery_feedback=cfg.agent["recovery_feedback"],
        )
        state = agent.run(question)
        demo = {
            "question": question,
            "answer": state.answer,
            "citations": state.citations,
            "trajectory": state.trajectory,
            "budget": state.budget.snapshot(),
            "evidence": [e.text for e in state.memory.items],
        }
        all_demos.append(demo)
        print(f"\nQ: {question}")
        print(f"A: {state.answer}")
        print(f"   citations: {state.citations}")
        print(f"   evidence: {len(state.memory)} 条, tool_calls: {state.budget.snapshot()['tool_calls']}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_demos, f, ensure_ascii=False, indent=2)
    print(f"\n[DEMO] 轨迹已保存 -> {out_path}")


if __name__ == "__main__":
    main()
