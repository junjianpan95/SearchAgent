"""混合检索：BM25 + BGE-M3 + RRF 融合 + Reranker 重排。

RRF（Reciprocal Rank Fusion）：
    score(d) = Σ 1 / (k + rank_i(d))，k 默认 60。
"""
from __future__ import annotations

from pathlib import Path

from .base import Hit, Retriever
from .bm25 import BM25Index
from .dense import DenseIndex


def _resolve_path(name: str) -> str:
    """本地路径优先：若 name 是相对路径且文件存在，转绝对路径；否则当作 HF 模型名。"""
    p = Path(name)
    if p.exists():
        return str(p.resolve())
    root = Path(__file__).resolve().parent.parent.parent  # 项目根
    joined = root / name
    if joined.exists():
        return str(joined)
    return name


def rrf_fuse(hits_lists: list[list[Hit]], k: int = 60, top_k: int = 20) -> list[Hit]:
    """RRF 融合多路召回结果。"""
    score_map: dict[tuple, float] = {}
    doc_map: dict[tuple, Hit] = {}
    for hits in hits_lists:
        for rank, h in enumerate(hits):
            key = (h.page, h.text[:60])
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = h
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for key, score in ranked:
        h = doc_map[key]
        out.append(Hit(page=h.page, text=h.text, score=score, source=h.source))
    return out


class HybridRetriever:
    """BM25 + 稠密 + RRF (+ 可选 reranker) 的混合检索后端。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bm25 = BM25Index() if cfg.get("bm25", {}).get("enabled", True) else None
        self.dense = DenseIndex(
            _resolve_path(cfg.get("dense_model", "BAAI/bge-m3")),
            device=cfg.get("device", "cpu"),
        )
        self.rrf_k = cfg.get("rrf_k", 60)
        self.top_k = cfg.get("top_k", 20)
        self.use_reranker = cfg.get("use_reranker", False)
        self._reranker = None
        self._pages: dict[str, str] = {}   # page title -> full text

    def build(self, passages: list[dict]) -> None:
        for d in passages:
            self._pages[d["page"]] = d["text"]
        if self.bm25 is not None:
            self.bm25.build(passages)
        self.dense.build(passages)

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.top_k
        lists = []
        if self.bm25 is not None:
            lists.append(self.bm25.search(query, top_k))
        lists.append(self.dense.search(query, top_k))
        fused = rrf_fuse(lists, k=self.rrf_k, top_k=top_k)
        if self.use_reranker:
            fused = self._rerank(query, fused, top_k)
        return fused

    def read(self, page: str) -> str:
        return self._pages.get(page, "")

    def _rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                _resolve_path(self.cfg.get("reranker_model", "BAAI/bge-reranker-base")),
                device=self.cfg.get("device", "cpu"),
            )
        pairs = [(query, h.text) for h in hits]
        scores = self._reranker.predict(pairs)
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for i in order:
            h = hits[i]
            out.append(Hit(page=h.page, text=h.text, score=float(scores[i]), source=h.source))
        return out
