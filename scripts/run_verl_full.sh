#!/bin/bash
# =============================================================================
# SearchAgent —— 正式 RL 训练串联脚本（GRPO -> GSPO）
#
# 用法（在 WSL2 Ubuntu 内执行）：
#   bash scripts/run_verl_full.sh
#
# 说明：
#   - 数据使用【无泄漏版】：rl_train.parquet (1105) / rl_val.parquet (200)
#   - 两个方法各自独立输出到 logs/rl_<method>_full.log
#   - 单个方法失败不阻断下一个（便于一次性拿到两套结果）
#   - checkpoint 分别落在 checkpoints/<method>/，互不干扰
# =============================================================================
set -u

source /home/pan/miniconda3/etc/profile.d/conda.sh
conda activate verl
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

TRAIN_FILE="$PROJECT_ROOT/data/traces/rl_train.parquet"
VAL_FILE="$PROJECT_ROOT/data/traces/rl_val.parquet"

echo "============================================================"
echo " SearchAgent verl 正式训练（串联）"
echo "   训练集 : $TRAIN_FILE"
echo "   验证集 : $VAL_FILE"
echo "   开始   : $(date '+%F %T')"
echo "============================================================"

for method in grpo gspo; do
    # 关键：清掉该方法的旧 checkpoint，否则 trainer 读 latest_checkpointed_iteration.txt
    # 后会「续跑」而不是从零开始（冒烟测试留下的 global_step_1 会污染正式结果）。
    rm -rf "$PROJECT_ROOT/checkpoints/$method" 2>/dev/null || true

    echo ""
    echo ">>> [$method] 开始 $(date '+%F %T')"
    VERL_USE_UV=0 \
    TRAIN_FILE="$TRAIN_FILE" \
    VAL_FILE="$VAL_FILE" \
    bash scripts/run_verl_grpo.sh "$method" > "logs/rl_${method}_full.log" 2>&1
    rc=$?
    echo ">>> [$method] 结束 $(date '+%F %T')，退出码 $rc"
done

echo ""
echo "============================================================"
echo " 全部结束 : $(date '+%F %T')"
echo "============================================================"
