#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 SFT LoRA adapter 合并进基座模型，产出 RL 的起点策略。

## 为什么必须做这一步

verl 的 `actor_rollout_ref.model.path` 如果不指向已经会"说动作语言"的策略，
RL 就是在教一个只会说自然语言的基座模型输出 `Search: ...` / `Answer: ...`。
实测（冒烟 smoke_mt4）：起点是原始 Qwen3-1.7B + 随机初始化 LoRA 时，
奖励函数解析到的动作数为 0（`tool_cost=0.0`）、`answer_f1=0.0`、
`num_turns` 恒为 2（模型第一轮输出无法解析成动作，agent loop 立刻终止），
组内奖励方差为 0 -> GRPO/GSPO 拿不到任何梯度。

正确流程是 SFT -> RL：RL 从 SFT 策略出发，只在"决策质量"上继续优化。

## 做法

`PeftModel.merge_and_unload()` 把 adapter 权重按 `lora_alpha/r` 缩放后加到
基座权重上，导出一个普通的 HF 模型目录，verl 直接当 `model.path` 用。
（比走 verl 的 `lora_adapter_path` 初始化路径更稳，避免再碰 LoRA manager。）

用法：
    python scripts/merge_sft_adapter.py \
        --base  data/models/Qwen3-1.7B \
        --adapter data/models/sft_lora \
        --out   data/models/qwen3_1.7b_sft_merged
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"[SKIP] 目标已存在且非空: {out}")
        return

    print(f"[1/4] 加载基座 {args.base} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )

    print(f"[2/4] 加载并合并 adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.eval()

    print(f"[3/4] 保存到 {out}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    # 把 SFT 侧可能存在的 chat_template 一并带上（保持推理格式一致）
    tpl = Path(args.adapter) / "chat_template.jinja"
    if tpl.exists():
        shutil.copy2(tpl, out / "chat_template.jinja")

    print("[4/4] 完成。目录内容：")
    total = 0
    for p in sorted(out.iterdir()):
        sz = p.stat().st_size
        total += sz
        if sz > 1024 * 1024:
            print(f"    {p.name:40s} {sz / 1024 / 1024:8.1f} MB")
        else:
            print(f"    {p.name:40s} {sz:8d} B")
    print(f"    合计 {total / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()
