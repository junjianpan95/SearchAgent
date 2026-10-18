"""src 包初始化，暴露 ROOT 与配置入口。"""
from .config import Config, load_config, data_dir, ROOT

__all__ = ["Config", "load_config", "data_dir", "ROOT"]
