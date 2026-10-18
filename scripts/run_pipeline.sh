#!/usr/bin/env bash
# SearchAgent 全链路 pipeline：数据 → 检索 → Agent → SFT → 评测
# 本地 smoke 默认跑通逻辑；服务器 full 出真实数字
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG=${1:-configs/local_smoke.yaml}
echo "==> 使用配置: $CONFIG"

echo "==> [1/4] 数据构造（SearchTrace 三类轨迹）"
python -m src.data_gen.build_data --config "$CONFIG"

echo "==> [2/4] 构建混合检索索引（BM25 + BGE-M3 + RRF）"
python -m src.retrieval.build_index --config "$CONFIG"

echo "==> [3/4] SFT 训练（LoRA/DoRA）"
python -m src.train.sft --config "$CONFIG"

echo "==> [4/4] 评测（Answer F1 / Joint F1 / 证据链 / 冗余率 / 工具调用）"
python -m evals.run_eval --config "$CONFIG"

echo "==> 全链路完成"
