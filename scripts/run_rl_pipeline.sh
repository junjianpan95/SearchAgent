#!/bin/bash
# =============================================================================
# SearchAgent — GRPO → GSPO 串行编排（不关机版）
#
# 由 systemd-run 托管运行，彻底脱离 WorkBuddy 会话，避免会话结束时被
# SIGTERM 回收（上次 GRPO 正因会话结束被 kill，停在 step 3）。
#
# 流程：GRPO(1005 条) → ray 清理 → GSPO(1005 条) → 结束
# 日志：logs/pipeline.log（编排层） + logs/grpo_full.log + logs/gspo_full.log
# =============================================================================
set -u
ROOT=/home/pan/SearchAgent
PIPELINE_LOG="$ROOT/logs/pipeline.log"
CONDA_SH=/home/pan/miniconda3/etc/profile.d/conda.sh

log() { echo "[$(date '+%F %T')] $*" >> "$PIPELINE_LOG"; }

source "$CONDA_SH"
conda activate verl
cd "$ROOT" || exit 1
mkdir -p logs

log "===== Pipeline 启动：GRPO -> GSPO（不关机）====="
log "PID=$$  PPID=$PPID"

# ---- 阶段 1/2：GRPO ----
log "----- 阶段 1/2: GRPO -----"
bash scripts/run_verl_grpo.sh grpo >> logs/grpo_full.log 2>&1
log "GRPO 退出码 = $?"
ray stop --force >> "$PIPELINE_LOG" 2>&1
sleep 5

# ---- 阶段 2/2：GSPO ----
log "----- 阶段 2/2: GSPO -----"
bash scripts/run_verl_grpo.sh gspo >> logs/gspo_full.log 2>&1
log "GSPO 退出码 = $?"
ray stop --force >> "$PIPELINE_LOG" 2>&1

log "===== Pipeline 全部结束 ====="
