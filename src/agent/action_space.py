"""结构化动作空间定义。

动作类型：Search / Read / AddEvidence / Answer
每个动作带一个 JSON-schema 化的参数签名，供 Agent 生成与合法性校验。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    SEARCH = "Search"
    READ = "Read"
    ADD_EVIDENCE = "AddEvidence"
    ANSWER = "Answer"


# 合法动作集合（用于非法动作惩罚）
LEGAL_ACTIONS = {a.value for a in ActionType}


@dataclass
class Action:
    """一次工具动作。"""
    type: ActionType
    args: dict[str, Any] = field(default_factory=dict)
    # 结果回填（执行后）
    result: Any = None

    @property
    def is_legal(self) -> bool:
        return self.type.value in LEGAL_ACTIONS

    def to_dict(self) -> dict:
        return {"type": self.type.value, "args": self.args}


def parse_action(raw: dict) -> Action:
    """从 LLM 生成的原始输出解析动作，非法时返回 None 语义（由调用方处理）。"""
    t = raw.get("type", "")
    if t not in LEGAL_ACTIONS:
        return Action(ActionType.SEARCH, {})  # 占位，合法性由 is_legal 判定
    return Action(ActionType(t), raw.get("args", {}))
