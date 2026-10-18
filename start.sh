#!/usr/bin/env bash
# SearchAgent 一键启动（Windows 用 Git Bash 或 WSL 运行）
# 等价于：bash scripts/start_web.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/start_web.sh" "$@"
