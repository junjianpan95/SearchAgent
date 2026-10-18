#!/usr/bin/env bash
# SearchAgent 依赖安装
set -e

echo "==> 检查 Python 环境"
PY=$(which python || which python3)
$PY --version

echo "==> 安装核心依赖（本地 smoke 最小集）"
# 核心库：配置、数据处理、BM25、指标
$PY -m pip install --quiet \
    pyyaml \
    rank-bm25 \
    numpy \
    sentence-transformers \
    transformers \
    datasets \
    torch

echo "==> 可选：LLaMA-Factory（SFT，需 CUDA）"
echo "    有 GPU 时执行：$PY -m pip install llmfactory 或按官方文档安装"
echo "==> 可选：TRL（本地 GRPO smoke）"
echo "    $PY -m pip install trl"

echo "==> 可选：verl + vLLM + Ray（服务器 full，仅多卡环境）"
echo "    pip install verl vllm ray"

echo "==> 可选：Tavily（在线搜索后端）"
echo "    $PY -m pip install tavily-python"

echo "==> 依赖安装完成"
