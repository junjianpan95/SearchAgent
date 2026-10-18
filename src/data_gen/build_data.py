"""数据构造入口：加载 AgenticRagTrace 问题集，构建三类轨迹，产出 SFT 与 RL 数据。

输出：
- data/traces/sft_trajectories.json   原始轨迹（含动作序列，供评测/可视化）
- data/traces/sft_data.json           llamafactory alpaca 格式（instruction=问题, output=动作序列文本）
- data/traces/rl_samples.json         verl RL 样本（question/answer/supporting_facts）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config, ROOT
from src.data_gen import SearchTrace


def action_to_text(a: dict) -> str:
    t = a["type"]
    args = a.get("args", {})
    if t == "Search":
        return f"Search: {args.get('query', '')}"
    if t == "Read":
        return f"Read: {args.get('page', '')}"
    if t == "AddEvidence":
        return f"AddEvidence: {args.get('page', '')} | {args.get('text', '')[:120]}"
    if t == "Answer":
        cites = ",".join(str(c) for c in args.get("citations", []))
        return f"Answer: {args.get('answer', '')} [citations: {cites}]"
    return f"{t}"


def trajectory_to_steps(traj) -> list[dict]:
    """把一条轨迹展开成"逐步决策"样本（每个动作一个样本）。

    每个样本的 input 包含：问题 + 已有证据 + 上一步检索结果；
    output 是下一个动作。这样模型学会"根据环境反馈逐步决策"，
    与推理时的 Agent 循环格式一致。
    """
    steps = []
    evidence_lines: list[str] = []
    for i, a in enumerate(traj.actions):
        t = a["type"]
        args = a.get("args", {})
        # 构造当前状态（历史证据 + 上一步的观察）
        ctx = []
        ctx.append(f"Question: {traj.question}")
        if evidence_lines:
            ctx.append("Evidence:")
            ctx.extend(f"- {e}" for e in evidence_lines)
        # 上一步动作的观察（检索结果/读到的内容）
        if i > 0:
            prev = traj.actions[i - 1]
            if prev["type"] == "Search":
                hits = (prev.get("result") or {}).get("hits", [])
                if hits:
                    ctx.append("Search results:")
                    for h in hits[:3]:
                        ctx.append(f"- {h.get('page', '')}: {h.get('text', '')[:100]}")
        ctx.append("Next action:")
        instruction = "\n".join(ctx)
        output = action_to_text(a)
        steps.append({"instruction": instruction, "input": "", "output": output})
        # 更新证据状态
        if t == "AddEvidence" and args.get("text"):
            evidence_lines.append(f"{args.get('page', '')}: {args.get('text', '')[:120]}")
    return steps


def trajectory_to_text(traj) -> str:
    """把一条轨迹转成可训练的文本（动作序列，保留用于展示）。"""
    return "\n".join(action_to_text(a) for a in traj.actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    ap.add_argument("--samples", default=None, help="外部问题集 JSON（覆盖默认 AgenticRagTrace）")
    ap.add_argument("--n", type=int, default=None, help="抽样条数（默认按 config）")
    ap.add_argument("--train-upto", type=int, default=1105,
                    help="训练/评测隔离：只用前 N 条构造训练数据，其余留作 sealed test（默认 1105）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    st = SearchTrace(cfg.data)

    # 加载问题集：默认 AgenticRagTrace，可覆盖
    if args.samples and Path(args.samples).exists():
        samples = st.load_samples(Path(args.samples))
    else:
        samples = st.load_samples()
    if args.n:
        samples = samples[: args.n]
    elif cfg.is_smoke:
        samples = samples[: cfg.data.get("n_samples_smoke", 300)]

    # ---- 训练 / 评测隔离（修复数据泄漏）----
    # sealed test 定义为样本尾部 200 条（见 scripts/run_compare.py），
    # 因此训练集必须严格截断到前 N 条，否则评测集会被训练数据覆盖。
    n_all = len(samples)
    if args.train_upto and 0 < args.train_upto < n_all:
        samples = samples[: args.train_upto]
        print(f"[SPLIT] 训练/评测隔离：总计 {n_all} 条 -> 训练 {len(samples)} 条，"
              f"预留 sealed test {n_all - len(samples)} 条")

    print(f"[DATA] 加载 {len(samples)} 条多跳问题（来源: {cfg.data.get('dataset')}）")

    # 构建三类轨迹
    trajs = st.build_trajectories(samples, cfg.data["mix_ratio"])
    stats = st.stats(trajs)
    print(f"[DATA] 轨迹统计: {stats}")

    # 保存
    out_dir = ROOT / "data" / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 原始轨迹
    st.save(trajs, out_dir / "sft_trajectories.json")

    # 2. llamafactory alpaca 格式（逐步决策样本：每个动作一个样本）
    sft_data = []
    for t in trajs:
        sft_data.extend(trajectory_to_steps(t))
    with open(out_dir / "sft_data.json", "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    # 3. RL 样本（verl 格式）
    id2sample = {s.id: s for s in samples}
    rl_samples = []
    seen = set()
    for t in trajs:
        if t.sample_id in seen:
            continue
        seen.add(t.sample_id)
        s = id2sample.get(t.sample_id)
        if s is None:
            continue
        rl_samples.append({
            "id": t.sample_id,
            "question": t.question,
            "answer": s.answer,
            "supporting_facts": s.supporting_facts,
            "supporting_pages": s.supporting_pages,
        })
    with open(out_dir / "rl_samples.json", "w", encoding="utf-8") as f:
        json.dump(rl_samples, f, ensure_ascii=False, indent=2)

    print(f"[DATA] SFT 轨迹 -> {out_dir / 'sft_trajectories.json'}")
    print(f"[DATA] SFT 数据(alpaca) -> {out_dir / 'sft_data.json'} ({len(sft_data)} 条)")
    print(f"[DATA] RL 样本 -> {out_dir / 'rl_samples.json'} ({len(rl_samples)} 条)")


if __name__ == "__main__":
    main()
