"""评测：Answer F1 / Joint F1 / 证据链 All-Hit / 冗余率 / 工具调用次数 + 四方案横向对比。

评测集：页面隔离 sealed test + MuSiQue zero-shot。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import Config, load_config, ROOT
from src.utils.metrics import (
    f1_score,
    validate_citations,
    evidence_all_hit,
    redundancy_rate,
    joint_f1,
)


def evaluate_trajectories(preds: list[dict], golds: list[dict]) -> dict:
    """对一批预测轨迹计算所有指标。

    preds: [{'answer','citations','evidence_texts','n_tool_calls','selected_ids'}, ...]
    golds: [{'answer','supporting_facts'}, ...]
    """
    n = len(preds)
    if n == 0:
        return {}

    answer_f1s, joint_f1s, all_hits, redunds, tool_calls = [], [], [], [], []
    for p, g in zip(preds, golds):
        af1 = f1_score(p["answer"], g["answer"])
        hit = evidence_all_hit(set(p.get("selected_ids", [])),
                               set(range(len(g.get("supporting_facts", [])))))
        jf1 = joint_f1(af1, hit)
        red = redundancy_rate(p.get("evidence_texts", []))
        answer_f1s.append(af1)
        joint_f1s.append(jf1)
        all_hits.append(hit)
        redunds.append(red)
        tool_calls.append(p.get("n_tool_calls", 0))

    return {
        "answer_f1": sum(answer_f1s) / n,
        "joint_f1": sum(joint_f1s) / n,
        "evidence_all_hit": sum(all_hits) / n,
        "redundancy_rate": sum(redunds) / n,
        "avg_tool_calls": sum(tool_calls) / n,
        "n_samples": n,
    }


def run_eval(cfg: Config, eval_data: dict, model_ckpt: str) -> dict:
    """对单个 checkpoint 跑评测。

    eval_data: {'preds':[...], 'golds':[...]}
    model_ckpt: checkpoint 名称（用于记录）
    """
    results = evaluate_trajectories(eval_data["preds"], eval_data["golds"])
    results["checkpoint"] = model_ckpt
    return results


def compare_combos(cfg: Config, all_results: dict[str, dict]) -> dict:
    """四方案横向对比：{LoRA, DoRA} × {GRPO, GSPO}。"""
    return {
        combo: all_results.get(combo, {}) for combo in cfg.eval.get("compare_combos", [])
    }


def save_report(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[EVAL] 评测结果已保存 -> {path}")


def _run_smoke_eval(cfg: Config) -> dict:
    """本地 smoke：用内置样例构造 gold 与一个模拟 Base ReAct 预测，跑通评测逻辑。"""
    from src.data_gen import SearchTrace
    st = SearchTrace(cfg.data)
    samples = st.load_samples()

    golds = [{"answer": s.answer, "supporting_facts": s.supporting_facts}
             for s in samples]

    # 模拟 Base ReAct 预测：答案空/部分，证据不全（体现可提升空间）
    preds = []
    for i, s in enumerate(samples):
        # 模拟：只命中了第一个 supporting fact，答案有偏差
        preds.append({
            "answer": s.answer if i % 2 == 0 else "",
            "citations": [0],
            "evidence_texts": [s.supporting_facts[0]] if s.supporting_facts else [],
            "selected_ids": [0],
            "n_tool_calls": 5,   # Base ReAct 调用较多
        })

    results = evaluate_trajectories(preds, golds)
    results["checkpoint"] = "base_react_smoke"
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    ap.add_argument("--eval-data", default=None, help="外部评测 JSON（full 模式用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.eval_data and Path(args.eval_data).exists():
        with open(args.eval_data, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = run_eval(cfg, data, "checkpoint")
    else:
        results = _run_smoke_eval(cfg)

    save_report(results, ROOT / "data" / "eval_report.json")
    print("\n===== 评测结果 =====")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
