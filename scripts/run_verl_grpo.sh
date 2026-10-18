#!/bin/bash
# =============================================================================
# SearchAgent — verl 0.10 GRPO / GSPO 单卡训练脚本（WSL2 Ubuntu 内执行）
#
# 用法：
#   conda activate verl
#   VERL_USE_UV=0 bash scripts/run_verl_grpo.sh grpo     # 跑 GRPO
#   VERL_USE_UV=0 bash scripts/run_verl_grpo.sh gspo     # 跑 GSPO
#
# 关键设计（单卡 16G 跑 Qwen3-1.7B）：
#   - LoRA 训练（rank 16），仅更新 adapter，显存可控
#   - FSDP param_offload + optimizer_offload，参数/优化器卸载到 CPU
#   - vLLM rollout gpu_memory_utilization 0.5，给 actor 留显存
#   - 每题采样 group_size=4 条轨迹，on-policy rollout
#   - 两阶段课程奖励 + 四惩罚（reward 在 src/train/reward_adapter.py）
#
# 前置：先跑 bash scripts/build_rl_parquet.py 生成 parquet 数据
# =============================================================================
set -euo pipefail

METHOD=${1:-grpo}          # grpo | gspo
GROUP_SIZE=${GROUP_SIZE:-4}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# RL 起点策略：SFT-LoRA 合并进基座后的完整模型（data/models/qwen3_1.7b_sft_merged）。
# 若仍用原始 Qwen3-1.7B 作起点，则 actor 是一个"随机初始化的 LoRA 挂在裸基座"上，
# rollout 几乎不会产生合法动作（实测 answer_f1≈0.0005），组内奖励全同 -> 优势恒 0 -> 无梯度。
# 起点改为 SFT 合并模型后，RL 只在其上做"最后一公里"的策略修正。
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/data/models/qwen3_1.7b_sft_merged}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/traces/rl_train.parquet}"
KL_COEF=${KL_COEF:-0.001}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-1}
NGPUS=${NGPUS:-1}
SAVE_FREQ=${SAVE_FREQ:-25}
# vLLM 显存占比：单卡 16GB 需同时容纳 actor(3.4G)+ref(3.4G)+桌面占用(1.7G)，
# 留给 vLLM 的预算需下调到 0.35 左右，否则报 "No available memory for the cache blocks"
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
# vLLM 上下文上限：若不显式收窄，vLLM 会按 Qwen3 的默认 max_position_embeddings
# (40960) 预留 KV cache 预算，在本机上会直接算出"无可用 cache 空间"而启动失败。
# 多轮 agent loop 的 prompt+response = 512 + 1536 = 2048，故设为 2048。
ROLLOUT_MAX_LEN=${ROLLOUT_MAX_LEN:-2048}
# 验证集：用从【训练段】切分出的 rl_val.parquet（100 条）。
# 注意不要把 rl_val 指向密封测试集——那会让 test_freq 的验证指标变成"窥探测试集"，
# 与 SFT/评测阶段的隔离约定冲突。切分逻辑见 scripts/build_rl_parquet.py --skip。
VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/traces/rl_val.parquet}"
# 多轮轨迹的总响应长度（含工具观测 token）：prompt 512 + response 1536 <= max_model_len 2048
RESPONSE_LEN=${RESPONSE_LEN:-1536}

# ---- 多轮 agent loop（SearchAgent 自定义 rollout）----
# 必须用多轮：verl 默认 default_agent_loop=single_turn_agent，只生成一次、
# 不执行任何工具，模型拿到的 prompt 是自造格式（OOD），轨迹里不会出现 Answer 行，
# 奖励 answer_f1 实测仅 0.0005（同 adapter 真实评测为 0.0319）。
# 详见 src/train/searchagent_loop.py 顶部说明。
AGENT_LOOP_CONFIG="$PROJECT_ROOT/configs/agent_loop_searchagent.yaml"
AGENT_LOOP_NAME=${AGENT_LOOP_NAME:-searchagent_multihop}
# 多轮预算：与 SFT 轨迹长度对齐（Search/AddEvidence.../Answer，实测 2~5 步）
SA_MAX_TURNS=${SA_MAX_TURNS:-6}
SA_TOP_K=${SA_TOP_K:-5}
SA_MAX_EVIDENCE=${SA_MAX_EVIDENCE:-5}
# 检索器共享 Ray actor 跑在 CPU（GPU 已被 vLLM/FSDP 占满）
SA_RETRIEVER_DEVICE=${SA_RETRIEVER_DEVICE:-cpu}
SA_USE_RETRIEVER=${SA_USE_RETRIEVER:-1}

# 算法 adv_estimator
# 坑：verl 0.10 里 `gspo` 【不是】合法的 adv_estimator（register_adv_est 只注册了
#   grpo/gae/rloo/opo...），它是 register_policy_loss 里的一个 loss_mode。
#   写 algorithm.adv_estimator=gspo 会直接报 "not supported"。
#   官方 examples/gspo_trainer/run_qwen3_8b_fsdp.sh 的正确配法是：
#     algorithm.adv_estimator=grpo                       # advantage 仍用 GRPO 组内归一化
#     actor_rollout_ref.actor.policy_loss.loss_mode=gspo
#     actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
ADV_ESTIMATOR="grpo"
if [ "$METHOD" = "gspo" ]; then
    GSPO_ARGS=(
        actor_rollout_ref.actor.policy_loss.loss_mode=gspo
        actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
        # 官方 GSPO 默认的极窄 clip（序列级重要性比几乎恒为 1）
        actor_rollout_ref.actor.clip_ratio_low=3e-4
        actor_rollout_ref.actor.clip_ratio_high=4e-4
        actor_rollout_ref.actor.clip_ratio_c=10.0
    )
else
    GSPO_ARGS=()
fi

echo "============================================================"
echo " SearchAgent verl 训练"
echo "   方法          : $METHOD (adv_estimator=$ADV_ESTIMATOR)"
echo "   Agent loop    : $AGENT_LOOP_NAME (多轮, max_turns=$SA_MAX_TURNS, top_k=$SA_TOP_K)"
echo "   响应长度      : $RESPONSE_LEN (max_model_len=$ROLLOUT_MAX_LEN)"
echo "   模型          : $MODEL_PATH"
echo "   数据          : $TRAIN_FILE"
echo "   验证集        : $VAL_FILE"
echo "   每题采样      : $GROUP_SIZE 条轨迹"
echo "   KL 系数       : $KL_COEF"
echo "   学习率        : $LR"
echo "============================================================"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误：数据文件不存在 $TRAIN_FILE"
    echo "请先运行：python scripts/build_rl_parquet.py --in data/traces/rl_samples.json --out data/traces/rl_train.parquet"
    exit 1
fi

cd "$PROJECT_ROOT"

# ---- 进程环境变量（必须在 ray start 之前导出！）----
# 坑：Ray worker 继承的是 **raylet 进程** 的环境，而不是 driver 进程的。
# 因此所有需要 worker 可见的变量都必须在 `ray start --head` 之前 export，
# 否则 driver 侧生效、worker 侧拿不到（曾因此让 VERL_DISABLE_CUDA_IPC 静默失效）。
export VERL_USE_UV=0

# ---- WSL2 CUDA IPC 回退 ----
# 见 scripts/patch_verl_wsl_ipc.py。WSL2 的 GPU 经 /dev/dxg 半虚拟化，
# CUDA IPC 跨进程句柄不可用，actor(FSDP) -> vLLM worker 传权重时报
#   torch.AcceleratorError: CUDA error: invalid resource handle
# 位置：bucketed_weight_transfer.py:51 rebuild_ipc -> torch/storage.py:1464
# 置 1 后 is_ipc_supported() 返回 False（platform_cuda.py 本地补丁），
# verl 自动回退到 multiprocessing.shared_memory + ZeroMQ 的权重传输路径
# （功能等价，多一次 CPU 往返拷贝）。
export VERL_DISABLE_CUDA_IPC=${VERL_DISABLE_CUDA_IPC:-1}

# ---- 多轮 agent loop（SearchAgent 自定义 rollout）----
# 变量本体在文件顶部定义（含完整背景说明）；这里负责在 ray start 之前
# export 给 worker —— 顺序很关键，见上方"进程环境变量"注释。
export AGENT_LOOP_CONFIG AGENT_LOOP_NAME
export SA_MAX_TURNS SA_TOP_K SA_MAX_EVIDENCE SA_RETRIEVER_DEVICE SA_USE_RETRIEVER

# ---- Ray 内存治理 ----
# 症状：verl 跑到 vLLM engine 初始化 + FSDP 参数驻留的峰值时刻，raylet 报
#   node_manager.cc:3544: N Workers killed due to memory pressure (OOM)
# 根因：Ray 的 memory monitor 默认以"节点已用内存 / 节点总内存"为判据，且把
#   page cache 也算进已用内存。vLLM 以 load_format=dummy 载入权重、FSDP 把
#   1.7B 主干映射进地址空间时会产生瞬时 cache 峰值，被 Ray 误判为内存压力，
#   于是在一个采样周期内批量击杀 worker -> 训练卡死（driver 仍在，worker 全丢）。
#
# 解法（来自 raylet 自身的报错提示）：
#   "To disable worker killing, set RAY_memory_monitor_refresh_ms to zero."
# 单机 / 单容器场景下，Ray 的粗粒度 OOM Killer 弊大于利：它不认识 page cache
# 的可回收性，而 Linux 内核的 OOM Killer 有完善的 reclaim 逻辑，且有 8GB swap
# 兜底。因此这里直接【禁用】Ray 的 worker killing，把内存裁决权交还内核。
#
# 注意：该环境变量必须在 raylet 启动前导出才生效（见下方 ray start 之前）。
# 踩坑记录：曾把它设为 1000，误以为"拉长采样间隔"可缓解，实际 0 才是禁用；
# 非 0 值只是把采样周期改成 1s（默认约 100ms），反而更容易命中瞬时峰值。
RAY_OBJ_STORE_MB=${RAY_OBJ_STORE_MB:-1500}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.98}
export RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-0}
# 若将来需要保留 Ray OOM Killer，可改用按 owner group 的旧策略 + 降低空闲阈值：
#   export RAY_worker_killing_policy_by_group=true
#   export RAY_idle_worker_killing_memory_threshold_bytes=100000000
if ! ray status > /dev/null 2>&1; then
    echo "[RAY] 预启动受限 Ray head（object store = ${RAY_OBJ_STORE_MB}MB, OOM killer = OFF）"
    ray start --head --object-store-memory=$((RAY_OBJ_STORE_MB * 1000 * 1000)) \
        --disable-usage-stats > /dev/null 2>&1 || true
    sleep 3
fi
echo "[ENV] VERL_DISABLE_CUDA_IPC=$VERL_DISABLE_CUDA_IPC | " \
     "RAY_memory_monitor_refresh_ms=$RAY_memory_monitor_refresh_ms | " \
     "RAY_memory_usage_threshold=$RAY_memory_usage_threshold"
ray status 2>&1 | head -8 || true

# 让 verl 能 import 项目里的 reward_adapter（绝对路径直接用，无需 PYTHONPATH）
REWARD_PATH="$PROJECT_ROOT/src/train/reward_adapter.py"

# 数据参数
DATA=(
    algorithm.adv_estimator=$ADV_ESTIMATOR
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size=16
    data.max_prompt_length=512
    data.max_response_length=$RESPONSE_LEN
    data.filter_overlong_prompts=True
    data.truncation=error
    algorithm.use_kl_in_reward=False
)

# 模型参数
MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=16
    actor_rollout_ref.model.lora_alpha=32
    actor_rollout_ref.model.target_modules=all-linear
    # 本机未编译 flash-attn（RTX 5060 Ti sm_120），回退 PyTorch 原生 SDPA
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    # ---- LoRA 权重同步走 merge 模式（而非 adapter 模式）----
    # 背景：adapter 模式下 verl 会调 vLLM 的 add_lora()，把 adapter 张量交给
    # vLLM 的 LoRA manager（TensorLoRARequest / fully-sharded LoRA 路径）。
    # 在本机这套前沿组合（vLLM 0.31.0 + torch 2.13 + sm_120）上，该路径在
    # vllm/lora/model_manager.py:920
    #     first_lora = next(iter(lora_model.loras.values()))
    # 抛出 StopIteration —— 说明 load_weights 没能把任何一层匹配成 LoRA 层，
    # lora_model.loras 为空字典，next(iter({}.values())) 直接炸。
    #
    # merge 模式（见 fsdp/transformer_impl.py:get_per_tensor_param）：
    #     merge_lora = model_config.lora.get("merge", False)
    #     ... else: return self._merged_lora_per_tensor_param(), None
    # 返回的 peft_config 为 None，于是 _update_weights 走普通
    # model.load_weights() 分支，彻底绕开 vLLM 的 LoRA manager。
    #
    # 代价：每步要传「base+adapter 合并后」的全量权重（1.7B bf16 ≈ 3.4GB），
    # 而不是 70MB 的 adapter；且每步多一次 LoRA 合并计算。冒烟与正确性优先。
    # 注意：这里不能加 "+" 前缀（hydra 报 "Either remove + prefix"），
    # 因为 actor_rollout_ref.model.lora 是已存在的 dict，直接覆盖其 merge 键即可。
    actor_rollout_ref.model.lora.merge=True
)

# Actor（LoRA 微调：adapter 仅 ~70MB，无需 CPU offload）
# 注意：对 LoRA 而言 param/optimizer offload 是负优化——它把 1.7B 主干参数
# 与优化器状态整体搬到 CPU，导致 WSL 侧内存被吃满（实测 22.6/23.5GB）后
# 被 Ray OOM Killer 杀掉。关闭后 CPU 峰值降到 ~8GB。
ACTOR=(
    actor_rollout_ref.actor.optim.lr=$LR
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

# Rollout（vLLM，单卡 TP=1）
ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL
    actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_LEN
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=$GROUP_SIZE
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=0.95
    # ---- 多轮 agent loop：真实执行 Search/Read/AddEvidence ----
    actor_rollout_ref.rollout.agent.default_agent_loop=$AGENT_LOOP_NAME
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG"
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS:-4}
)

# Ref policy（参数常驻 CPU：ref 只在算 KL 时前向，卸载后 GPU 腾出 3.4GB）
REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# Trainer
TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=searchagent
    trainer.experiment_name="${METHOD}_qwen3_1.7b"
    trainer.n_gpus_per_node=$NGPUS
    trainer.nnodes=1
    trainer.default_local_dir="$PROJECT_ROOT/checkpoints/$METHOD"
    trainer.save_freq=$SAVE_FREQ
    trainer.max_actor_ckpt_to_keep=1
    trainer.test_freq=$SAVE_FREQ
    trainer.total_epochs=$EPOCHS
    trainer.use_v1=False
    transfer_queue.enable=False
)

# Custom reward —— 注意：verl 0.10 的 reward_loop 架构使用 reward.* 命名空间
# 坑：本版本里 migrate_legacy_reward_impl()（职责就是把旧版顶层的
#   custom_reward_function.* / reward_model.* 搬迁到 reward.*）在整个代码库中
#   【没有任何调用点】（grep 无结果），因此写旧 key 会被【静默忽略】，
#   最终退化到内置 default_compute_score，报：
#     NotImplementedError: Reward function is not implemented for data_source='...'
#   必须直接写新 key。字段定义见 verl/trainer/config/reward/reward.yaml：
#     reward.custom_reward_function.{path,name}
#     reward.reward_manager.name
REWARD=(
    reward.reward_manager.name=naive
    # 默认 8 个 RewardLoop worker，每个常驻 ~800MB；本机降到 2 个以省内存
    reward.num_workers=2
    reward.custom_reward_function.path="$REWARD_PATH"
    reward.custom_reward_function.name=compute_score
)

python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${GSPO_ARGS[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}"

echo ""
echo "训练完成。adapter 保存在 ./outputs 下。"
