#!/bin/bash
# =============================================================================
# SearchAgent — WSL2 verl 环境搭建脚本（在 WSL2 Ubuntu 内执行）
#
# 目标：在 WSL2 + 单卡 RTX 5060 Ti 16GB 上搭建 verl/vLLM/Ray/FSDP 环境，
#       跑 Qwen3-1.7B 的 on-policy GRPO / GSPO。
#
# 前置条件（Windows 侧）：
#   1. WSL2 已启用，`wsl -l -v` 有 Ubuntu 发行版
#   2. NVIDIA WSL 驱动已装（WSL 内 `nvidia-smi` 能出 GPU）
#
# 用法（在 WSL2 Ubuntu 终端）：
#   cd /mnt/e/Life/1/SearchAgent
#   bash scripts/setup_wsl.sh
# =============================================================================
set -e

echo "============================================================"
echo "[1/6] 检查 GPU 直通"
echo "============================================================"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误：WSL 内无 nvidia-smi。请先在 Windows 装 NVIDIA 驱动并启用 WSL GPU 直通。"
    exit 1
fi
nvidia-smi

echo ""
echo "============================================================"
echo "[2/6] 安装 miniconda（若无）"
echo "============================================================"
if ! command -v conda >/dev/null 2>&1; then
    echo "下载并安装 miniconda（清华镜像加速）..."
    wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
else
    eval "$(conda shell.bash hook)"
fi
# pip 清华镜像加速
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true

echo ""
echo "============================================================"
echo "[3/6] 创建 verl 环境（Python 3.10）"
echo "============================================================"
conda create -n verl python=3.10 -y || true
conda activate verl

echo ""
echo "============================================================"
echo "[4/6] 安装 CUDA torch（cu130，Linux 版，适配驱动 617/CUDA 13.4）"
echo "============================================================"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo ""
echo "============================================================"
echo "[5/6] 安装 vLLM + Ray + flash-attn"
echo "============================================================"
# vLLM（含 CUDA kernel）
pip install vllm
# Ray（verl 依赖）
pip install ray[default]
# flash-attn（可选，装不上可跳过，verl 会退回 sdpa）
pip install flash-attn --no-build-isolation || echo "flash-attn 安装失败，将使用 sdpa 后端（可接受）"

echo ""
echo "============================================================"
echo "[6/6] 安装 verl"
echo "============================================================"
# 推荐从源码装最新版（支持 GSPO）
if [ ! -d "$HOME/verl" ]; then
    git clone https://github.com/volcengine/verl.git "$HOME/verl"
fi
cd "$HOME/verl"
pip install -e .

echo ""
echo "============================================================"
echo "验证安装"
echo "============================================================"
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import ray; print('ray', ray.__version__)"
python -c "import verl; print('verl OK')"

echo ""
echo "============================================================"
echo "环境搭建完成！"
echo "下一步："
echo "  conda activate verl"
echo "  bash scripts/run_verl_grpo.sh grpo   # 或 gspo"
echo "============================================================"
