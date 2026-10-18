"""下载模型权重（优先 ModelScope 国内源，回退 HuggingFace）。

用法：
  python scripts/download_models.py --all            # 下载全部（1.7B + BGE-M3 + 数据集）
  python scripts/download_models.py --model qwen3-1.7b
  python scripts/download_models.py --model bge-m3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models"


# ModelScope 模型 ID 映射
MODELS = {
    "qwen3-1.7b": {
        "ms": "Qwen/Qwen3-1.7B",
        "hf": "Qwen/Qwen3-1.7B",
        "dir": "Qwen3-1.7B",
    },
    "qwen3-4b": {
        "ms": "Qwen/Qwen3-4B",
        "hf": "Qwen/Qwen3-4B",
        "dir": "Qwen3-4B",
    },
    "qwen3-8b": {
        "ms": "Qwen/Qwen3-8B",
        "hf": "Qwen/Qwen3-8B",
        "dir": "Qwen3-8B",
    },
    "bge-m3": {
        "ms": "BAAI/bge-m3",
        "hf": "BAAI/bge-m3",
        "dir": "bge-m3",
    },
}


def download_from_modelscope(model_id: str, local_dir: Path) -> bool:
    """用 ModelScope 下载（国内快）。"""
    try:
        from modelscope import snapshot_download
        snapshot_download(model_id, local_dir=str(local_dir))
        return True
    except Exception as e:
        print(f"  [ModelScope] 失败: {e}")
        return False


def download_from_hf(model_id: str, local_dir: Path) -> bool:
    """回退 HuggingFace。"""
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, local_dir=str(local_dir))
        return True
    except Exception as e:
        print(f"  [HuggingFace] 失败: {e}")
        return False


def download_model(name: str) -> bool:
    spec = MODELS[name]
    target = MODEL_DIR / spec["dir"]
    if target.exists() and any(target.iterdir()):
        print(f"[SKIP] {name} 已存在 -> {target}")
        return True

    print(f"[DOWNLOAD] {name} -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    ok = download_from_modelscope(spec["ms"], target)
    if not ok:
        ok = download_from_hf(spec["hf"], target)
    if ok:
        print(f"[OK] {name} 下载完成 -> {target}")
    else:
        print(f"[FAIL] {name} 下载失败")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        names = list(MODELS.keys())
    elif args.model:
        names = [args.model]
    else:
        names = list(MODELS.keys())

    for n in names:
        download_model(n)
        print()


if __name__ == "__main__":
    main()
