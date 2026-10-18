#!/bin/bash
# =============================================================================
# GSPO 专用入口（薄封装）
#
# GSPO 在 verl 0.10 里不是独立的 adv_estimator，而是 actor 的 policy_loss 模式，
# 因此现在统一由 run_verl_grpo.sh 按 METHOD 分支处理，本脚本只做转发以保留旧命令：
#     bash scripts/run_verl_gspo.sh          <=>  bash scripts/run_verl_grpo.sh gspo
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/run_verl_grpo.sh" gspo "$@"
