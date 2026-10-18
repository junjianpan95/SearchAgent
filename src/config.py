"""配置加载与访问。

统一加载 local_smoke / server_full 两套配置，通过 --config 切换。
所有训练/评测/数据模块都从这里读取参数，保证"一套代码、双模式"。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """扁平化的运行时配置，从 YAML 解析而来。"""
    mode: str
    model: dict
    retrieval: dict
    agent: dict
    data: dict
    sft: dict
    rl: dict
    eval: dict
    _raw: dict = field(repr=False)

    @property
    def is_smoke(self) -> bool:
        return self.mode == "local_smoke"

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        # 允许相对项目根目录的 configs/xxx.yaml
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        mode=raw["mode"],
        model=raw["model"],
        retrieval=raw["retrieval"],
        agent=raw["agent"],
        data=raw["data"],
        sft=raw["sft"],
        rl=raw["rl"],
        eval=raw.get("eval", {}),
        _raw=raw,
    )


def data_dir(cfg: Config) -> Path:
    d = ROOT / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d
