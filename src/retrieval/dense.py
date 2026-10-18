"""BGE-M3 稠密检索（向量召回，本地 GPU/CPU 均可）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Hit


class DenseIndex:
    """BGE-M3 稠密检索。惰性加载模型，支持 CPU/GPU。"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._docs: list[dict] = []
        self._emb: np.ndarray | None = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def build(self, passages: list[dict]) -> None:
        self._docs = passages
        model = self._ensure_model()
        texts = [d["text"] for d in passages]
        self._emb = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True,
            batch_size=64,
        ).astype(np.float32)

    def search(self, query: str, top_k: int = 20) -> list[Hit]:
        model = self._ensure_model()
        q = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores = (self._emb @ q[0]).flatten()
        idx = np.argsort(-scores)[:top_k]
        return [
            Hit(page=self._docs[i]["page"], text=self._docs[i]["text"],
                score=float(scores[i]), source="local")
            for i in idx
        ]
