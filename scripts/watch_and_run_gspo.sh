#!/bin/bash
# =============================================================================
# 守护脚本：等 GRPO 段结束 → 接管 → 用【修正版】配置跑 GSPO
#
# 背景：
#   正在运行的 run_verl_full.sh 串联脚本里，GSPO 段走的是 run_verl_grpo.sh，
#   而后者把方法名直接当成 adv_estimator（ADV_ESTIMATOR="$METHOD"）。
#   但 verl 0.10 中 `gspo` 不是合法的 adv_estimator（它在 policy_loss 注册表里），
#   所以那段【必然失败】。
#
#   由于 run_verl_grpo.sh 此刻正在被 GRPO 进程执行，
#   直接改写运行中的 bash 脚本会导致其文件偏移错乱、行为未定义，
#   因此这里采用「不动运行中文件 + 另起守护进程接管」的方式：
#     1) 轮询等待 GRPO 段结束（driver 打出 "gspo] 开始" 或串联进程退出）
#     2) 终止串联脚本及其启动的那次注定失败的 GSPO
#     3) 清理 Ray / vLLM / stale ZMQ socket
#     4) 用 scripts/run_verl_gspo.sh（adv_estimator=grpo + policy_loss=gspo）重跑
#
# 用法：bash scripts/watch_and_run_gspo.sh
# =============================================================================
set -u

cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
DRIVER_LOG="$PROJECT_ROOT/logs/rl_full_driver.log"

log() { echo "[WATCH] $(date '+%F %T') $*"; }

log "启动守护，等待 GRPO 段结束 ..."

# ---- 1. 等待 GRPO 段结束 ----
while true; do
    if grep -q "gspo\] 开始" "$DRIVER_LOG" 2>/dev/null; then
        log "检测到串联脚本已进入 GSPO 段（GRPO 已结束）"
        break
    fi
    if ! pgrep -f "[r]un_verl_full.sh" > /dev/null 2>&1; then
        log "串联脚本已退出"
        break
    fi
    sleep 30
done

# 给串联脚本一点时间真正起停
sleep 10

# ---- 2. 终止串联脚本与那次失败的 GSPO ----
log "终止串联脚本及其 GSPO 子进程"
pkill -9 -f "[r]un_verl_full.sh" 2>/dev/null || true
pkill -9 -f "[m]ain_ppo" 2>/dev/null || true
pkill -9 -f "[V]LLMHttpServer" 2>/dev/null || true
pkill -9 -f "[V]LLM::" 2>/dev/null || true
pkill -9 -f "[A]gentLoopWorker" 2>/dev/null || true
pkill -9 -f "[R]ewardLoop" 2>/dev/null || true
pkill -9 -f "[r]un_verl_grpo.sh" 2>/dev/null || true
sleep 5

# ---- 3. 清理 Ray 与残留 ----
source /home/pan/miniconda3/etc/profile.d/conda.sh
conda activate verl
ray stop --force > /dev/null 2>&1 || true
pkill -9 -f "[r]ay::" 2>/dev/null || true
sleep 5
rm -f /tmp/rl-colocate-zmq-*.sock 2>/dev/null || true

log "清理后残留进程数：$(ps -eo pid,comm 2>/dev/null | grep -icE 'vllm|main_ppo|ray')"

# ---- 4. 用修正版配置跑 GSPO ----
log "启动修正版 GSPO（adv_estimator=grpo + policy_loss.loss_mode=gspo）"
VERL_USE_UV=0 \
TRAIN_FILE="$PROJECT_ROOT/data/traces/rl_train.parquet" \
VAL_FILE="$PROJECT_ROOT/data/traces/rl_val.parquet" \
bash scripts/run_verl_gspo.sh > "$PROJECT_ROOT/logs/rl_gspo_full.log" 2>&1
rc=$?
log "GSPO 结束，退出码 $rc"
