"""混合索引加载器：从 data/index/ 缓存加载 BM25 + BGE-M3，提供统一检索接口。

缓存由 scripts/build_index.py 生成：
- data/index/corpus.jsonl   语料（与向量对齐）
- data/index/bm25.pkl       BM25 索引
- data/index/embeddings.npy BGE-M3 向量
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def tokenize(t: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", t.lower())


class CachedHybridRetriever:
    """从缓存加载的混合检索器：BM25 + BGE-M3 + RRF。"""

    def __init__(self, index_dir: str | None = None, device: str = "cuda"):
        self.index_dir = Path(index_dir) if index_dir else ROOT / "data" / "index"
        self.device = device
        self._passages: list[dict] = []
        self._bm25 = None
        self._emb: np.ndarray | None = None
        self._model = None
        self.rrf_k = 60
        self._load()

    def _load(self):
        # 语料
        with open(self.index_dir / "corpus.jsonl", encoding="utf-8") as f:
            self._passages = [json.loads(l) for l in f if l.strip()]
        # BM25
        with open(self.index_dir / "bm25.pkl", "rb") as f:
            self._bm25 = pickle.load(f)["bm25"]
        # 向量
        self._emb = np.load(self.index_dir / "embeddings.npy")
        print(f"[INDEX] 加载 {len(self._passages)} 段，向量 {self._emb.shape}")

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_path = str(ROOT / "data" / "models" / "bge-m3")
            self._model = SentenceTransformer(model_path, device=self.device)
        return self._model

    def search(self, query: str, top_k: int = 20):
        """BM25 + BGE-M3 + RRF 混合检索。"""
        # BM25
        bm25_scores = self._bm25.get_scores(tokenize(query))
        # dense
        model = self._ensure_model()
        qe = model.encode([query], normalize_embeddings=True).astype(np.float32)
        dense_scores = (self._emb @ qe[0]).flatten()

        def rank_map(scores):
            order = np.argsort(-scores)
            return {i: r for r, i in enumerate(order)}

        rb = rank_map(bm25_scores)
        rd = rank_map(dense_scores)
        fused = {
            i: 1.0 / (self.rrf_k + rb[i] + 1) + 1.0 / (self.rrf_k + rd[i] + 1)
            for i in range(len(self._passages))
        }
        top = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

        from src.retrieval.base import Hit
        return [Hit(page=self._passages[i]["page"], text=self._passages[i]["text"],
                    score=fused[i], source="hybrid") for i, _ in top]

    def read(self, page: str) -> str:
        for p in self._passages:
            if p["page"] == page:
                return p["text"]
        return ""

    def build(self, passages=None):
        """兼容接口（已从缓存加载，无需重建）。"""
        pass
