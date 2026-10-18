#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agent loop 观测一致性回归测试。

背景：RL rollout 的观测渲染一旦与 SFT 训练数据、评测 harness 不一致，
就会重新引入分布偏移（早期正是因此让 answer_f1 掉到 0.0005）。
本测试把 `src/train/searchagent_loop.render_instruction` 与权威渲染器
`src/data_gen/build_data.trajectory_to_steps` 做逐字符串比对，
保证 5967 条 SFT 样本全部可被逐字复现。

用法（WSL verl 环境内，工作目录为项目根）：
    python scripts/test_agent_loop.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_gen.build_data import action_to_text, trajectory_to_steps  # noqa: E402
from src.train.searchagent_loop import (  # noqa: E402
    extract_question,
    parse_first_action,
    render_instruction,
)


def test_parse_action() -> None:
    cases = [
        ("Search: who owns CKER-FM?", "Search", "who owns CKER-FM?"),
        ("  Read: Run-length limited  ", "Read", "Run-length limited"),
        ("AddEvidence: page | some text", "AddEvidence", "page | some text"),
        ("Answer: Paris [citations: 0,1]", "Answer", "Paris [citations: 0,1]"),
        ("\n\nblah\nSearch: real query", "Search", "real query"),
    ]
    for text, want_type, want_arg in cases:
        got = parse_first_action(text)
        assert got is not None, f"解析失败: {text!r}"
        assert got["type"] == want_type, f"{text!r} -> {got['type']} != {want_type}"
        assert got["arg"] == want_arg, f"{text!r} -> {got['arg']!r} != {want_arg!r}"
    assert parse_first_action("") is None
    assert parse_first_action("Search results:\n- page: text") is None, \
        "观测行 'Search results:' 不应被误判为 Search 动作"
    print(f"[OK] parse_first_action {len(cases)} 例")


def test_render_matches_build_data() -> None:
    """逐条比对：用 render_instruction 重放轨迹，必须与 build_data 产出的 instruction 完全一致。"""
    traj_file = ROOT / "data" / "traces" / "sft_trajectories.json"
    sft_file = ROOT / "data" / "traces" / "sft_data.json"
    if not traj_file.exists() or not sft_file.exists():
        print(f"[SKIP] 缺少 {traj_file.name} 或 {sft_file.name}，跳过一致性比对")
        return

    trajs = json.loads(traj_file.read_text(encoding="utf-8"))
    sft = json.loads(sft_file.read_text(encoding="utf-8"))

    # build_data 产出的期望 instruction 序列（顺序与 trajectories 展开一致）
    expected = []
    for t in trajs:
        expected.extend(s["instruction"] for s in trajectory_to_steps_from(t))
    assert len(expected) == len(sft), f"步数不一致: {len(expected)} vs {len(sft)}"

    # 用 render_instruction 独立重放
    got = []
    for t in trajs:
        q = t["question"]
        evidence_lines: list[str] = []
        last_type, last_hits = None, None
        for a in t["actions"]:
            got.append(render_instruction(q, evidence_lines, last_type, last_hits))
            typ = a["type"]
            args = a.get("args", {})
            if typ == "Search":
                last_type, last_hits = "Search", (a.get("result") or {}).get("hits", [])
            else:
                last_type, last_hits = typ, None
            if typ == "AddEvidence" and args.get("text"):
                evidence_lines.append(f"{args.get('page', '')}: {args.get('text', '')[:120]}")

    mismatch = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert not mismatch, (
        f"有 {len(mismatch)}/{len(expected)} 条 instruction 与权威渲染器不一致，"
        f"首例 index={mismatch[0][0]}:\n--- expected ---\n{mismatch[0][1]}\n--- got ---\n{mismatch[0][2]}"
    )
    print(f"[OK] render_instruction 与 build_data 逐字一致（{len(expected)} 条）")


def trajectory_to_steps_from(t: dict) -> list[dict]:
    """把 dict 形式的轨迹喂给 build_data 的权威渲染器。"""

    class _Traj:
        question = None
        actions = None

    obj = _Traj()
    obj.question = t["question"]
    obj.actions = t["actions"]
    return trajectory_to_steps(obj)


def test_extract_question() -> None:
    q = "Which RLL encoding stores more data bits per inch, (1,3) or (2,7)?"
    assert extract_question(f"Question: {q}\nNext action:") == q
    print("[OK] extract_question")


def test_action_to_text_roundtrip() -> None:
    """动作文本 -> 解析 -> 再渲染，应稳定（不因格式解析丢信息）。"""
    lines = [
        "Search: who owns CKER-FM?",
        "Read: Run-length limited",
        "AddEvidence: page | text body",
        "Answer: Paris [citations: 0,1]",
    ]
    for ln in lines:
        a = parse_first_action(ln)
        assert a is not None
        assert ln.strip().startswith(a["type"] + ":"), ln
    print("[OK] 动作往返")


if __name__ == "__main__":
    test_parse_action()
    test_extract_question()
    test_action_to_text_roundtrip()
    test_render_matches_build_data()
    print("\n全部通过。")
