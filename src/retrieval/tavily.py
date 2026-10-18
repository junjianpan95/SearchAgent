"""Tavily 在线搜索后端（可插拔）。

按次计费，rollout 阶段会放大成本。默认 disabled，需设置 TAVILY_API_KEY。
"""
from __future__ import annotations

import os

from .base import Hit


class TavilyRetriever:
    """Tavily 在线搜索。在线搜索的取舍：覆盖更广但不可复现、按次计费、引入网页噪声。"""

    def __init__(self, cfg: dict):
        self.api_key = os.environ.get(cfg.get("api_key_env", "TAVILY_API_KEY"), "")
        self.max_results = cfg.get("max_results", 5)
        self._client = None
        self._cache: dict[str, str] = {}

    def _ensure_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("TAVILY_API_KEY not set")
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("tavily-python not installed: pip install tavily-python")
        return self._client

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        client = self._ensure_client()
        top_k = top_k or self.max_results
        resp = client.search(query=query, max_results=top_k, search_depth="advanced")
        hits = []
        for r in resp.get("results", []):
            hits.append(Hit(
                page=r.get("title", ""),
                text=r.get("content", ""),
                score=float(r.get("score", 0.0)),
                source="tavily",
            ))
        return hits

    def read(self, page: str) -> str:
        # 从缓存或重新提取（简化：返回缓存）
        return self._cache.get(page, "")
