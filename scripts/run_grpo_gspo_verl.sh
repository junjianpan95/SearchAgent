#!/bin/bash
# SearchAgent GRPO/GSPO 训练（verl + vLLM + Ray + FSDP）
# 在真实检索环境中 on-policy rollout，每题采样 4 条完整轨迹
set -e

export MODEL=Qwen/Qwen3-8B
export METHOD=gspo
export GROUP_SIZE=4
export KL_COEF=0.04

python -m verl.trainer.main_ppo \
    --algorithm gspo \
    --actor_rollout_ref.model.path "$MODEL" \
    --actor_rollout_ref.rollout.name vllm \
    --actor_rollout_ref.rollout.n $GROUP_SIZE \
    --actor_rollout_ref.rollout.temperature 1.0 \
    --actor_rollout_ref.rollout.gpu_memory_utilization 0.6 \
    --actor_rollout_ref.actor.optim.lr 1e-06 \
    --actor_rollout_ref.actor.ppo_mini_batch_size 32 \
    --actor_rollout_ref.actor.kl_loss_coef $KL_COEF \
    --actor_rollout_ref.actor.fsdp_config.param_offload true \
    --critic.ppo_mini_batch_size 32 \
    --trainer.n_gpus_per_node 8 \
    --trainer.nnodes 1 \
    --data.train_files E:\Life\SearchAgent\data\rl_samples.json \
    --reward_fn src.train.reward_adapter:compute_reward \
    --custom_reward_function.src.train.reward_adapter \
    --trainer.total_epochs 1
