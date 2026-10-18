#!/usr/bin/env python3
"""诊断 RL parquet 的 prompt token 长度分布，定位 filter_overlong_prompts 全过滤原因。"""
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

TOK_PATH = "/home/pan/SearchAgent/data/models/qwen3_1.7b_sft_merged"
TRACE_DIR = "/home/pan/SearchAgent/data/traces"


def to_text(p):
    if isinstance(p, str):
        try:
            obj = json.loads(p)
        except Exception:
            return p
        return to_text(obj)
    if isinstance(p, list):
        return "".join(
            m.get("content", "") if isinstance(m, dict) else str(m) for m in p
        )
    return str(p)


def main():
    tok = AutoTokenizer.from_pretrained(TOK_PATH)
    for name in ["rl_train_smoke", "rl_train", "rl_val"]:
        path = f"{TRACE_DIR}/{name}.parquet"
        try:
            d = pd.read_parquet(path)
        except Exception as e:
            print(f"{name}: READ ERR {e}")
            continue
        lens = [len(tok(to_text(p))["input_ids"]) for p in d["prompt"]]
        a = np.array(lens)
        print(
            f"{name:16s} n={len(a):5d}  min={a.min():5d}  p50={int(np.median(a)):5d}  "
            f"p90={int(np.percentile(a, 90)):5d}  max={a.max():5d}  "
            f"over512={int((a > 512).sum())}"
        )
    # 打印一条样本，确认 prompt 结构与长度来源
    d = pd.read_parquet(f"{TRACE_DIR}/rl_train_smoke.parquet")
    raw = d["prompt"].iloc[0]
    print("\n--- prompt 字段类型:", type(raw))
    print("--- 样本(前 1500 字符) ---")
    print(to_text(raw)[:1500])


if __name__ == "__main__":
    main()
