#!/bin/bash
# 在 WSL2 Ubuntu 内执行：bash scripts/wsl_install_verl.sh
# 分步安装 verl 环境（miniconda + torch cu130 + vLLM + Ray + verl）
set -e

export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "============================================================"
echo "[1/5] 创建 verl conda 环境 (Python 3.10)"
echo "============================================================"
conda config --set auto_activate_base false 2>/dev/null || true
# 接受 ToS + 换清华镜像源（避免 anaconda 官方源 ToS 卡住，且更快）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main 2>/dev/null || true
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free 2>/dev/null || true
conda config --set show_channel_urls yes 2>/dev/null || true
if conda env list | grep -q "^verl "; then
    echo "verl 环境已存在，跳过创建"
else
    conda create -n verl python=3.10 -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
fi
conda activate verl
python --version
which python

echo ""
echo "============================================================"
echo "[2/5] 安装 torch cu130 (适配驱动 617 / CUDA 13.4)"
echo "============================================================"
pip install --upgrade pip 2>/dev/null || true
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo ""
echo "============================================================"
echo "[3/5] 验证 torch CUDA"
echo "============================================================"
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"

echo ""
echo "============================================================"
echo "[4/5] 安装 vLLM + Ray"
echo "============================================================"
pip install vllm
pip install "ray[default]"

echo ""
echo "============================================================"
echo "[5/5] 安装 verl (源码)"
echo "============================================================"
if [ ! -d "$HOME/verl" ]; then
    git clone https://github.com/volcengine/verl.git "$HOME/verl"
fi
cd "$HOME/verl"
pip install -e .

echo ""
echo "============================================================"
echo "最终验证"
echo "============================================================"
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1
python -c "import ray; print('ray', ray.__version__)" 2>&1 | tail -1
python -c "import verl; print('verl OK')" 2>&1 | tail -1

echo ""
echo "环境搭建完成！"
