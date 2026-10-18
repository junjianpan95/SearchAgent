#!/usr/bin/env bash
# SearchAgent 前端 Demo 启动/停止脚本
# 用法：
#   bash scripts/start_web.sh          # 启动前端（默认 8000 端口）
#   bash scripts/start_web.sh --stop   # 停止
#   bash scripts/start_web.sh --port 9000
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT=8000
MODE=start

for arg in "$@"; do
  case $arg in
    --stop) MODE=stop ;;
    --port) shift; PORT=$1 ;;
  esac
done

PIDFILE="$ROOT/data/web.pid"

stop_web() {
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null && echo "==> 已停止前端服务 (PID $PID)"
    fi
    rm -f "$PIDFILE"
  else
    echo "==> 前端服务未在运行"
  fi
}

if [ "$MODE" = "stop" ]; then
  stop_web
  exit 0
fi

# 启动（先用 .venv python，回退系统 python）
PY="$ROOT/.venv/Scripts/python.exe"
[ -f "$PY" ] || PY=$(which python || which python3)

echo "==> 启动前端 Demo: http://localhost:$PORT"
nohup "$PY" -m http.server "$PORT" --directory "$ROOT/web" > "$ROOT/data/web.log" 2>&1 &
echo $! > "$PIDFILE"
sleep 1
echo "==> 前端已启动 (PID $(cat "$PIDFILE"))，浏览器打开 http://localhost:$PORT"
echo "==> 停止: bash scripts/start_web.sh --stop"
