#!/bin/bash
# =============================================================================
# 把 verl 的 FSDP checkpoint 合并/导出为可评测的 PEFT adapter。
#
# 背景：verl 训练时保存的是 FSDP 分片 checkpoint（model_world_size_1_rank_0.pt
# 一类），而评测脚本 scripts/run_compare.py 通过 PeftModel.from_pretrained
# 加载适配器。verl 自带的 model_merger 能从 state_dict 里抽出 lora_* 参数，
# 结合同目录的 lora_train_meta.json（含 r / lora_alpha / task_type）
# 生成标准 PEFT 适配器目录（adapter_config.json + adapter_model.safetensors）。
#
# 用法（在 WSL2 内执行）：
#   bash scripts/merge_verl_ckpt.sh grpo
#   bash scripts/merge_verl_ckpt.sh gspo
#
# 产物：data/models/rl_<method>/
# =============================================================================
set -eu

METHOD=${1:?用法: bash scripts/merge_verl_ckpt.sh <grpo|gspo>}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CKPT_ROOT="$PROJECT_ROOT/checkpoints/$METHOD"
TARGET="$PROJECT_ROOT/data/models/rl_$METHOD"

if [ ! -d "$CKPT_ROOT" ]; then
    echo "错误：checkpoint 目录不存在 $CKPT_ROOT"
    exit 1
fi

# 取编号最大的 global_step_N
STEP_DIR=$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | \
           sed 's/.*global_step_//' | sort -n | tail -1)

if [ -z "${STEP_DIR:-}" ]; then
    echo "错误：未找到 global_step_* 目录"
    exit 1
fi

ACTOR_DIR="$CKPT_ROOT/global_step_${STEP_DIR}/actor"
echo "============================================================"
echo " 合并 checkpoint"
echo "   method   : $METHOD"
echo "   step     : $STEP_DIR"
echo "   local_dir: $ACTOR_DIR"
echo "   target   : $TARGET"
echo "============================================================"

ls -la "$ACTOR_DIR" | head -20

# --backend fsdp：读取 FSDP 分片并合并；model_merger 会自动识别 lora_train_meta.json
# 并额外导出 PEFT adapter。
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET"

echo ""
echo "=== 产物 ==="
ls -la "$TARGET" 2>/dev/null || true
