#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 verl RL 训练日志中抽取真实指标轨迹。

背景：verl 的进度条把多个 step 用 \r 挤在同一"行"里，且指标值被包成
np.float64(...) / np.int32(...)，直接 grep 会漏掉绝大部分 step。
本脚本先把 \r 归一化为 \n，再逐 step 提取。

用法：
    python scripts/analyze_rl_log.py logs/rl_grpo_full.log
    python scripts/analyze_rl_log.py logs/rl_grpo_full.log --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

# reward loop 输出的指标键（后缀 /mean@1 由 verl 的 val-aux 前缀机制生成）
METRIC_KEYS = {
    "reward":         "core/searchagent_multihop/reward/mean@1",
    "answer_f1":      "aux/searchagent_multihop/answer_f1/mean@1",
    "answer_em":      "aux/searchagent_multihop/answer_em/mean@1",
    "citation":       "aux/searchagent_multihop/citation_validity/mean@1",
    "new_fact":       "aux/searchagent_multihop/new_supporting_fact/mean@1",
    "dup_query":      "aux/searchagent_multihop/dup_query/mean@1",
    "redundant_ev":   "aux/searchagent_multihop/redundant_evidence/mean@1",
    "illegal_action": "aux/searchagent_multihop/illegal_action/mean@1",
    "tool_cost":      "aux/searchagent_multihop/tool_cost/mean@1",
    "num_turns":      "aux/num_turns/mean",
}

# 训练侧指标（不含 val 前缀）
TRAIN_KEYS = {
    "critic_score": r"critic/score/mean:([-0-9.eE]+)",
    "pg_loss":      r"actor/pg_loss:np\.float64\(([-0-9.eE]+)\)",
    "kl_loss":      r"actor/kl_loss:np\.float64\(([-0-9.eE]+)\)",
    "grad_norm":    r"actor/grad_norm:np\.float64\(([-0-9.eE]+)\)",
    "entropy":      r"actor/entropy:([-0-9.eE]+)",
}


def parse(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # 关键：verl 用 \r 刷新进度条，一个物理"行"里塞了几十个 step
    lines = raw.replace("\r", "\n").splitlines()

    rows: dict[int, dict[str, float]] = {}
    for line in lines:
        m = re.search(r"step:(\d+) - ", line)
        if not m:
            continue
        step = int(m.group(1))
        d = rows.setdefault(step, {})
        for short, key in METRIC_KEYS.items():
            mm = re.search(re.escape(key) + r":np\.float64\(([-0-9.eE]+)\)", line)
            if mm:
                d[short] = float(mm.group(1))
        for short, pat in TRAIN_KEYS.items():
            mm = re.search(pat, line)
            if mm:
                d[short] = float(mm.group(1))

    # 只保留含核心 reward 的 step
    steps = sorted(s for s in rows if "reward" in rows[s])
    return {"steps": steps, "rows": rows}


def summarize(data: dict) -> None:
    steps, rows = data["steps"], data["rows"]
    if not steps:
        print("未解析到任何 step（检查日志路径/格式）")
        return

    print(f"解析到 {len(steps)} 个含 reward 的 step（step {steps[0]}..{steps[-1]}）\n")

    cols = ["reward", "answer_f1", "citation", "dup_query", "tool_cost", "num_turns"]
    print(f"{'step':>5} " + " ".join(f"{c:>10}" for c in cols))
    for s in steps:
        if s % 8 == 0 or s >= steps[-2]:
            vals = [rows[s].get(c) for c in cols]
            print(f"{s:>5} " + " ".join(
                f"{v:>+10.4f}" if v is not None else f"{'--':>10}" for v in vals))

    n = len(steps)
    h = max(1, n // 3)
    print("\n=== 前 1/3 vs 后 1/3 均值 ===")
    for c in METRIC_KEYS:
        a = [rows[s][c] for s in steps[:h] if c in rows[s]]
        b = [rows[s][c] for s in steps[-h:] if c in rows[s]]
        if not a or not b:
            continue
        delta = statistics.mean(b) - statistics.mean(a)
        arrow = "↑" if delta > 1e-6 else ("↓" if delta < -1e-6 else "=")
        print(f"  {c:16s} 前={statistics.mean(a):+.4f}  后={statistics.mean(b):+.4f}  "
              f"{arrow} {delta:+.4f}")

    print("\n=== 训练侧 ===")
    for c in TRAIN_KEYS:
        vals = [rows[s][c] for s in steps if c in rows[s]]
        if vals:
            print(f"  {c:16s} 首={vals[0]:+.4f}  末={vals[-1]:+.4f}  "
                  f"min={min(vals):+.4f}  max={max(vals):+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="verl 训练日志路径")
    ap.add_argument("--json", type=Path, default=None, help="把解析结果导出为 JSON")
    args = ap.parse_args()

    if not args.log.exists():
        raise SystemExit(f"日志不存在: {args.log}")

    data = parse(args.log)
    summarize(data)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已导出: {args.json}")


if __name__ == "__main__":
    main()
