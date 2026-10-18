#!/usr/bin/env bash
# SearchAgent 一键停止
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/start_web.sh" --stop
