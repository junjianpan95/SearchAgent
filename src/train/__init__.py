"""train 包。"""
from .sft import prepare_sft_data, run_sft
from .grpo import run_grpo, run_grpo_verl, run_grpo_trl

__all__ = ["prepare_sft_data", "run_sft", "run_grpo", "run_grpo_verl", "run_grpo_trl"]
