"""构建混合检索索引（BM25 + BGE-M3 + RRF），供 Agent 检索使用。

从 Wikipedia 语料（或 smoke 样例）构建 passages，建立索引并缓存到磁盘。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config, ROOT
from src.retrieval import build_retriever


def _builtin_passages() -> list[dict]:
    """内置 smoke 语料（与 SearchTrace 样例配套）。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    ap.add_argument("--passages", default=None, help="外部 passages JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.passages and Path(args.passages).exists():
        with open(args.passages, "r", encoding="utf-8") as f:
            passages = json.load(f)
    else:
        passages = _builtin_passages()
        print(f"[INDEX] 使用内置 smoke 语料 {len(passages)} 段")

    retriever = build_retriever(cfg.retrieval)
    print(f"[INDEX] 构建检索后端: {cfg.retrieval['backend']}")
    retriever.build(passages)

    # 验证：跑一个查询
    test_query = "Who wrote the novel The Shining?"
    hits = retriever.search(test_query, top_k=3)
    print(f"[INDEX] 测试查询 '{test_query}' 返回 {len(hits)} 条:")
    for h in hits:
        print(f"  - [{h.source}] {h.page}: {h.text[:60]} (score={h.score:.4f})")

    print("[INDEX] 索引构建完成")


if __name__ == "__main__":
    main()
