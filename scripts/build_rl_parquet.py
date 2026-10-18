#!/usr/bin/env python3
"""把 rl_samples.json 转成 verl 需要的 parquet 格式。

verl 0.10 数据要求（naive reward manager）：
- data_source: 字符串，用于选择 reward 函数
- prompt: 列表，每条一个 chat message（[{"role":"user","content":...}]）
- reward_model: dict，含 ground_truth（供 reward 函数取标准答案）
- 额外字段 extra_info: dict（可选）

用法：
    python scripts/build_rl_parquet.py --in data/traces/rl_samples.json --out data/traces/rl_train.parquet
"""
import argparse
import json

import pandas as pd

# 必须与 SFT / 评测 harness 保持完全一致：
#   - SFT 用 LLaMA-Factory `template: qwen`，其 default_system 即 "You are a helpful assistant."
#   - 评测 scripts/run_compare.py::make_adapter_llm 也用同一句
#   - 初始 instruction 是「逐步决策」格式的第 1 步：只有 Question + Next action:
# 早期版本这里写的是自造的 "You are a multi-hop search agent..."，并要求模型
# 一次输出整条轨迹，导致 rollout 严重 OOD：模型几乎产不出可解析的 Answer 行，
# 奖励函数算出的 answer_f1 只有 0.0005（同 adapter 在真实评测下是 0.0319）。
SYSTEM_PROMPT = "You are a helpful assistant."


def build_instruction(question: str) -> str:
    """第 1 步的 instruction，与 src/data_gen/build_data.py 的渲染格式一致。"""
    return f"Question: {question}\nNext action:"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_file", required=True)
    p.add_argument("--out", dest="out_file", required=True)
    p.add_argument("--skip", type=int, default=0, help="跳过前 N 条")
    p.add_argument("--n", type=int, default=0, help="只取前 N 条（0=全部）")
    args = p.parse_args()

    with open(args.in_file, encoding="utf-8") as f:
        samples = json.load(f)

    if args.skip:
        samples = samples[args.skip:]
    if args.n > 0:
        samples = samples[: args.n]

    rows = []
    for s in samples:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_instruction(s["question"])},
        ]
        rows.append({
            "data_source": "searchagent_multihop",
            "prompt": prompt,
            "reward_model": {
                "ground_truth": s["answer"],
                "supporting_facts": s.get("supporting_facts", []),
            },
            "extra_info": {
                "question": s["question"],
                "supporting_facts": s.get("supporting_facts", []),
            },
        })

    df = pd.DataFrame(rows)
    df.to_parquet(args.out_file, index=False)
    print(f"[PARQUET] 生成 {len(rows)} 条 -> {args.out_file}")


if __name__ == "__main__":
    main()
