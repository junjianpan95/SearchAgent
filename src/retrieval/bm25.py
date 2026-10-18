"""BM25 稀疏检索（rank-bm25，纯 CPU，零成本）。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .base import Hit


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower())


class BM25Index:
    """轻量 BM25 索引（用 rank_bm25 若可用，否则退化到 tf 匹配）。"""

    def __init__(self):
        self._docs: list[dict] = []
        self._tok_docs: list[list[str]] = []
        self._ranker = None

    def build(self, passages: Iterable[dict]) -> None:
        """passages: [{'page':..., 'text':...}, ...]"""
        self._docs = list(passages)
        self._tok_docs = [_tokenize(d["text"]) for d in self._docs]
        try:
            from rank_bm25 import BM25Okapi
            self._ranker = BM25Okapi(self._tok_docs)
        except ImportError:
            self._ranker = None  # 退化：线性 tf 匹配

    def search(self, query: str, top_k: int = 20) -> list[Hit]:
        q = _tokenize(query)
        if self._ranker is not None:
            scores = self._ranker.get_scores(q)
        else:
            # 退化实现：query term 在 doc 中的出现次数加权
            scores = []
            for tok in self._tok_docs:
                s = sum(tok.count(t) for t in q)
                scores.append(s)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            Hit(page=self._docs[i]["page"], text=self._docs[i]["text"],
                score=float(scores[i]), source="local")
            for i in ranked if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        import pickle
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"docs": self._docs, "tok": self._tok_docs}, f)

    def load(self, path: Path) -> None:
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._docs = d["docs"]
        self._tok_docs = d["tok"]
        try:
            from rank_bm25 import BM25Okapi
            self._ranker = BM25Okapi(self._tok_docs)
        except ImportError:
            self._ranker = None
