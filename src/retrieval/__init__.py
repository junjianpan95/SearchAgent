"""检索后端工厂：根据 config 创建对应 retriever。"""
from __future__ import annotations

from .base import Hit, Retriever
from .hybrid import HybridRetriever
from .tavily import TavilyRetriever


def build_retriever(cfg: dict) -> Retriever:
    backend = cfg.get("backend", "hybrid")
    if backend == "tavily":
        return TavilyRetriever(cfg.get("tavily", {}))
    if backend == "hybrid":
        return HybridRetriever(cfg)
    raise ValueError(f"Unknown backend: {backend}")


__all__ = ["Hit", "Retriever", "build_retriever", "HybridRetriever", "TavilyRetriever"]
