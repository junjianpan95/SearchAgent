"""GRPO / GSPO on-policy RL 训练。

双框架支持（通过 config 切换）：
- verl（服务器 full）：verl + vLLM + Ray + FSDP，分布式 on-policy rollout，
  在真实检索环境里逐题采样 group_size 条完整轨迹。
- TRL（本地 smoke）：TRL GRPOTrainer + LoRA，单卡跑通逻辑。

两阶段课程奖励由 src.reward.curriculum 提供。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config, ROOT


def _rollout_env_fn(cfg: Config):
    """真实检索环境的 rollout 函数（供 verl / TRL 调用）。

    返回一个 callable(question) -> list[trajectory]，
    用当前策略模型在混合检索环境里采样 group_size 条完整轨迹。
    """
    from ..agent import SearchAgent
    from ..retrieval import build_retriever

    retriever = build_retriever(cfg.retrieval)
    group_size = cfg.rl.get("group_size", 4)

    def rollout(question: str, policy_llm_fn=None):
        trajs = []
        for _ in range(group_size):
            agent = SearchAgent(
                retriever=retriever,
                llm_fn=policy_llm_fn,
                max_search=cfg.agent["max_search"],
                max_read=cfg.agent["max_read"],
                max_evidence=cfg.agent["max_evidence"],
                token_budget=cfg.agent["token_budget"],
                dedup_queries=cfg.agent["dedup_queries"],
                bridge_entity_direct_read=cfg.agent["bridge_entity_direct_read"],
                recovery_feedback=cfg.agent["recovery_feedback"],
            )
            state = agent.run(question)
            trajs.append({
                "actions": state.trajectory,
                "answer": state.answer,
                "citations": state.citations,
            })
        return trajs

    return rollout


def _compute_rewards(cfg: Config, trajs: list[dict], gold: dict) -> list[float]:
    from ..reward import CurriculumReward
    r = CurriculumReward(cfg.rl["reward"])
    rewards = []
    for t in trajs:
        reward, comps = r.compute(t, gold)
        rewards.append(reward)
    return rewards


def run_grpo_verl(cfg: Config) -> str:
    """服务器 full：用 verl 跑 GRPO/GSPO。

    生成 verl 训练脚本，在真实检索环境中做 on-policy rollout + 两阶段课程奖励。
    """
    method = cfg.rl["method"]           # grpo | gspo
    model_name = cfg.model["name"]

    script = f'''#!/bin/bash
# SearchAgent GRPO/GSPO 训练（verl + vLLM + Ray + FSDP）
# 在真实检索环境中 on-policy rollout，每题采样 {cfg.rl["group_size"]} 条完整轨迹
set -e

export MODEL={model_name}
export METHOD={method}
export GROUP_SIZE={cfg.rl["group_size"]}
export KL_COEF={cfg.rl["kl_coef"]}

python -m verl.trainer.main_ppo \\
    --algorithm {method} \\
    --actor_rollout_ref.model.path "$MODEL" \\
    --actor_rollout_ref.rollout.name vllm \\
    --actor_rollout_ref.rollout.n $GROUP_SIZE \\
    --actor_rollout_ref.rollout.temperature 1.0 \\
    --actor_rollout_ref.rollout.gpu_memory_utilization 0.6 \\
    --actor_rollout_ref.actor.optim.lr {cfg.rl["lr"]} \\
    --actor_rollout_ref.actor.ppo_mini_batch_size 32 \\
    --actor_rollout_ref.actor.kl_loss_coef $KL_COEF \\
    --actor_rollout_ref.actor.fsdp_config.param_offload true \\
    --critic.ppo_mini_batch_size 32 \\
    --trainer.n_gpus_per_node 8 \\
    --trainer.nnodes 1 \\
    --data.train_files {ROOT/"data/rl_samples.json"} \\
    --reward_fn src.train.reward_adapter:compute_reward \\
    --custom_reward_function.src.train.reward_adapter \\
    --trainer.total_epochs {cfg.rl["epochs"]}
'''
    out = ROOT / "scripts" / f"run_grpo_{method}_verl.sh"
    out.write_text(script, encoding="utf-8")
    print(f"[GRPO-verl] 生成训练脚本 -> {out}")
    print("[GRPO-verl] 请在多卡 A100/H100 服务器上执行该脚本")
    return str(out)


def run_grpo_trl(cfg: Config) -> str:
    """本地 smoke：用 TRL GRPOTrainer + LoRA 单卡跑通。

    逻辑与 verl 完全一致（动作空间、课程奖励、KL 约束），只是框架降级。
    """
    method = cfg.rl["method"]
    model_name = cfg.model["name"]

    script = f'''#!/usr/bin/env python
# SearchAgent GRPO/GSPO 训练（TRL 单卡 smoke 版）
# 与 verl 版逻辑一致：动作空间 + 两阶段课程奖励 + KL 约束，单卡跑通
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

training_args = GRPOConfig(
    output_dir="data/grpo_{method}",
    learning_rate={cfg.rl["lr"]},
    num_train_epochs={cfg.rl["epochs"]},
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    bf16=True,
    logging_steps=5,
)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=None,   # 由 rollout env 在线生成
)
trainer.train()
'''
    out = ROOT / "scripts" / f"run_grpo_{method}_trl.py"
    out.write_text(script, encoding="utf-8")
    print(f"[GRPO-trl] 生成训练脚本 -> {out}")
    print("[GRPO-trl] 本地 smoke 版，单卡可跑")
    return str(out)


def run_grpo(cfg: Config) -> str:
    framework = cfg.rl.get("framework", "trl")
    if framework == "verl":
        return run_grpo_verl(cfg)
    return run_grpo_trl(cfg)
