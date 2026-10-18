"""Runtime：预算约束与停止条件（deterministic）。

控制多跳检索的停止：Search/Read 次数上限、token 预算、Evidence Memory 容量。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Budget:
    max_search: int = 4
    max_read: int = 6
    token_budget: int = 8192
    max_evidence: int = 5

    _search_used: int = 0
    _read_used: int = 0
    _tokens_used: int = 0

    def can_search(self) -> bool:
        return self._search_used < self.max_search

    def can_read(self) -> bool:
        return self._read_used < self.max_read

    def can_spend_tokens(self, n: int) -> bool:
        return self._tokens_used + n <= self.token_budget

    def spend_search(self) -> None:
        self._search_used += 1

    def spend_read(self) -> None:
        self._read_used += 1

    def spend_tokens(self, n: int) -> None:
        self._tokens_used += n

    @property
    def exhausted(self) -> bool:
        """是否必须停止（预算耗尽）。"""
        return (not self.can_search() and not self.can_read()) \
            or self._tokens_used >= self.token_budget

    def snapshot(self) -> dict:
        return {
            "search_used": self._search_used,
            "read_used": self._read_used,
            "tokens_used": self._tokens_used,
            "tool_calls": self._search_used + self._read_used,
        }
