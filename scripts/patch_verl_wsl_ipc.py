#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[LOCAL PATCH] 让 verl 在 WSL2 下回退到 shared-memory 权重传输。

背景
----
verl 0.10 的 colocated 模式下，actor(FSDP) 会把更新后的权重通过 CUDA IPC
句柄传给 vLLM 的 rollout worker：

    vllm_rollout.py:143        self.use_shm = not is_support_ipc()
    vllm_rollout.py:239        update_weights_from_ipc(..., use_shm=self.use_shm)
    utils.py:339               receiver.receive_weights(...)
    bucketed_weight_transfer.py:322  _init_buffer()
    bucketed_weight_transfer.py:51   rebuild_ipc()
    torch/storage.py:1464            _new_shared_cuda()

而 platform_cuda.py 里 is_ipc_supported() 硬编码返回 True，
于是 use_shm=False，强制走 IPC 路径。

WSL2 的 GPU 经 /dev/dxg 半虚拟化，CUDA IPC 跨进程句柄不可用，接收端报：

    torch.AcceleratorError: CUDA error: invalid resource handle

旁证：同一次运行 vLLM 打印
    "Pinned memory is not available on this platform; falling back to
     device memory for UVA buffers."

修复
----
给 is_ipc_supported() 加一个环境变量开关 VERL_DISABLE_CUDA_IPC=1，
返回 False 即可让 use_shm=True，走 verl 内置的 multiprocessing.shared_memory
回退路径（ZeroMQ + CPU 共享内存），代价是多一次 CPU 往返拷贝，功能等价。

本脚本幂等：已打过补丁则跳过。
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

TARGET = Path("/home/pan/verl/verl/plugin/platform/platform_cuda.py")

OLD = """    def is_ipc_supported(self) -> bool:
        return True
"""

NEW = '''    def is_ipc_supported(self) -> bool:
        # [LOCAL PATCH] WSL2 适配 —— 见 scripts/patch_verl_wsl_ipc.py
        # ------------------------------------------------------------------
        # 上游无条件返回 True，导致 vllm_rollout.py 里
        #     self.use_shm = not is_support_ipc()   ->   False
        # 强制走 CUDA IPC：actor(FSDP) -> vLLM worker 之间用
        # cudaIpcGetMemHandle / cudaIpcOpenMemHandle 共享权重缓冲区。
        #
        # 但 WSL2 的 GPU 经 /dev/dxg 半虚拟化，CUDA IPC 跨进程句柄不可用，
        # 接收端 _new_shared_cuda 抛：
        #     torch.AcceleratorError: CUDA error: invalid resource handle
        # 调用链：utils.py:339 -> bucketed_weight_transfer.py:322 _init_buffer
        #         -> bucketed_weight_transfer.py:51 rebuild_ipc
        #         -> torch/storage.py:1464 _new_shared_cuda
        #
        # 旁证：同一运行中 vLLM 打印
        #   "Pinned memory is not available on this platform; falling back to
        #    device memory for UVA buffers."
        #
        # verl 已内置 shared memory 回退（multiprocessing.shared_memory +
        # ZeroMQ），只是被此处的硬编码 True 挡住。本补丁用环境变量放开它。
        #
        #   export VERL_DISABLE_CUDA_IPC=1   -> 回退到 shm（WSL2 需要）
        #   不设 / 设 0                       -> 保持上游原生 IPC 行为
        if os.environ.get("VERL_DISABLE_CUDA_IPC", "0") == "1":
            return False
        return True
'''


def main() -> int:
    if not TARGET.exists():
        print(f"[FAIL] 目标文件不存在：{TARGET}")
        return 1

    src = TARGET.read_text(encoding="utf-8")

    if "VERL_DISABLE_CUDA_IPC" in src:
        print(f"[SKIP] 已打过补丁：{TARGET}")
        return 0

    if OLD not in src:
        print("[FAIL] 未匹配到待替换片段，verl 版本可能已变化。")
        print("       请手动检查 is_ipc_supported() 的实现。")
        return 2

    if not re.search(r"^import os$", src, flags=re.M):
        print("[FAIL] 目标文件未 import os，需先补 import。")
        return 3

    backup = TARGET.with_suffix(".py.bak")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"[BACKUP] {backup}")

    TARGET.write_text(src.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"[OK] 已打补丁：{TARGET}")
    print("     启用方式：export VERL_DISABLE_CUDA_IPC=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
