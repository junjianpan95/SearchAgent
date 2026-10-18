"""检索后端抽象：本地混合检索 + Tavily 在线搜索（可插拔）。

统一接口：
- search(query) -> list[Hit]
- read(page) -> str

后端切换：hybrid / bm25 / dense / tavily
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class Hit:
    """一条检索命中。"""
    page: str
    text: str
    score: float
    source: str = "local"   # local | tavily

    def to_dict(self) -> dict:
        return {"page": self.page, "text": self.text, "score": self.score, "source": self.source}


class Retriever(Protocol):
    """检索后端接口。"""

    def search(self, query: str, top_k: int = 20) -> list[Hit]:
        ...

    def read(self, page: str) -> str:
        ...
