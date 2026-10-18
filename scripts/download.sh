#!/usr/bin/env bash
# SearchAgent 数据与模型下载
# --mode local_smoke: 下载 Qwen3-1.7B + BGE-M3（约 6GB）
# --mode server_full: 下载 Qwen3-8B + BGE-M3 + BGE-reranker（约 20GB，服务器执行）
set -e

MODE=${1:-local_smoke}
HUB=${HF_ENDPOINT:-https://huggingface.co}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$ROOT/data/models"

echo "==> 下载模式: $MODE"
mkdir -p "$MODEL_DIR"

if [ "$MODE" = "local_smoke" ]; then
    echo "==> 下载 Qwen3-1.7B（约 3.5GB）"
    huggingface-cli download Qwen/Qwen3-1.7B --local-dir "$MODEL_DIR/Qwen3-1.7B" || \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B', local_dir='$MODEL_DIR/Qwen3-1.7B')"
    echo "==> 下载 BGE-M3（约 2.3GB）"
    huggingface-cli download BAAI/bge-m3 --local-dir "$MODEL_DIR/bge-m3" || \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='$MODEL_DIR/bge-m3')"
elif [ "$MODE" = "server_full" ]; then
    echo "==> 下载 Qwen3-8B（约 16GB）"
    huggingface-cli download Qwen/Qwen3-8B --local-dir "$MODEL_DIR/Qwen3-8B"
    echo "==> 下载 BGE-M3"
    huggingface-cli download BAAI/bge-m3 --local-dir "$MODEL_DIR/bge-m3"
    echo "==> 下载 BGE-reranker-base"
    huggingface-cli download BAAI/bge-reranker-base --local-dir "$MODEL_DIR/bge-reranker-base"
else
    echo "未知模式: $MODE（可选 local_smoke / server_full）"
    exit 1
fi

echo "==> 下载完成，模型位于 $MODEL_DIR"
