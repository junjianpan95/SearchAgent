"""固定容量 Evidence Memory + 历史查询去重。

设计要点：
- Evidence Memory：固定容量（max_evidence），存 supporting fact（原文 + 来源页）。
- 历史查询去重：记录所有已发起的 Search query，重复查询直接拦截。
- 桥接实体 direct-read：从当前证据中识别桥接实体，直接读对应页面。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evidence:
    """一条证据（supporting fact）。"""
    text: str
    page: str            # 来源页标题
    index: int           # 在检索结果中的全局 id
    score: float = 0.0


@dataclass
class EvidenceMemory:
    max_evidence: int = 5
    _items: list[Evidence] = field(default_factory=list)
    _history_queries: set[str] = field(default_factory=set)

    def add(self, ev: Evidence) -> bool:
        """加入证据，超出容量时返回 False（触发冗余惩罚信号）。"""
        if len(self._items) >= self.max_evidence:
            return False
        # 去重：相同页面+相似文本不重复加入
        if any(e.page == ev.page and e.text.strip()[:50] == ev.text.strip()[:50]
               for e in self._items):
            return False
        ev.index = len(self._items)
        self._items.append(ev)
        return True

    def record_query(self, query: str) -> bool:
        """记录查询，返回是否为新查询（重复则 False）。"""
        q = query.strip().lower()
        if q in self._history_queries:
            return False
        self._history_queries.add(q)
        return True

    @property
    def items(self) -> list[Evidence]:
        return list(self._items)

    @property
    def queries(self) -> set[str]:
        return set(self._history_queries)

    def is_full(self) -> bool:
        return len(self._items) >= self.max_evidence

    def clear(self) -> None:
        self._items.clear()
        self._history_queries.clear()

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class BridgeEntity:
    """桥接实体：跨文档推理时从证据中抽取的连接点。"""
    name: str
    source_page: str
    target_pages: list[str] = field(default_factory=list)
