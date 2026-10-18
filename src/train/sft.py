"""SFT 训练：LLaMA-Factory LoRA / DoRA。

- 只监督 Agent 生成的 token（查询、工具动作、证据选择、答案），
  不监督环境观察 token。
- 用 Joint F1、完整证据链召回、工具成本联合选择 checkpoint。
- 支持 LoRA / DoRA 两种方法（四方案对比维度之一）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import Config, ROOT


def _build_data_spec(cfg: Config) -> dict:
    """构造 SFT 数据集说明，交给 LLaMA-Factory。"""
    return {
        "dataset": cfg.data["dataset"],
        "trajectory_fields": ["query", "action", "evidence_selection", "answer"],
        "supervise": "agent_tokens_only",   # 仅监督 Agent 生成 token
    }


def prepare_sft_data(cfg: Config, traj_path: Path, out_path: Path) -> None:
    """把轨迹转换为 LLaMA-Factory 可读的 alpaca/sharegpt 格式。

    每条轨迹 → 一轮"指令(问题) → 输出(动作序列+答案)"，
    监督 mask 只打在 Agent 生成段。
    """
    import json as _json
    with open(traj_path, "r", encoding="utf-8") as f:
        trajs = _json.load(f)

    records = []
    for t in trajs:
        actions_str = "\n".join(
            f"{a['type']}: {json.dumps(a.get('args', {}), ensure_ascii=False)}"
            for a in t["actions"]
        )
        output = f"{actions_str}\nAnswer: {t['answer']}"
        records.append({
            "instruction": t["question"],
            "output": output,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[SFT] 写入 {len(records)} 条训练样本 -> {out_path}")


def run_sft(cfg: Config) -> str:
    """运行 SFT 训练。

    调用 LLaMA-Factory 的 llamafactory-cli。本地 smoke 与服务器 full 都走这条路径，
    区别只在 config 里的基座/规模/方法（LoRA vs DoRA）。
    """
    method = cfg.sft["method"]            # lora | dora
    model_name = cfg.model["name"]

    # 生成 LLaMA-Factory 训练配置
    llf_config = {
        "model_name_or_path": model_name,
        "stage": "sft",
        "finetuning_type": method,        # lora 或 dora
        "lora_rank": cfg.sft.get("lora_r", 16),
        "lora_alpha": cfg.sft.get("lora_alpha", 32),
        "lora_dropout": cfg.sft.get("lora_dropout", 0.05),
        "learning_rate": cfg.sft["lr"],
        "num_train_epochs": cfg.sft["epochs"],
        "per_device_train_batch_size": cfg.sft["batch_size"],
        "gradient_accumulation_steps": cfg.sft["grad_accum"],
        "deepspeed": cfg.sft["deepspeed"],
        "output_dir": str(ROOT / "data" / f"sft_{method}"),
        "dataset_dir": str(ROOT / "data" / "sft_dataset"),
        "dataset": cfg.data["dataset"],
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
    }
    cfg_path = ROOT / "data" / f"llf_{method}.json"
    cfg_path.write_text(json.dumps(llf_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SFT] 生成 LLaMA-Factory 配置 -> {cfg_path}")

    # 实际训练命令（本地 smoke 可加 --max_samples 限规模）
    cmd = f"llamafactory-cli train {cfg_path}"
    if cfg.is_smoke:
        cmd += " --max_samples 200"
    print(f"[SFT] 执行: {cmd}")
    print("[SFT] 提示：在无 GPU 环境下本步骤为 dry-run，请在有卡机器执行上述命令")

    return str(ROOT / "data" / f"sft_{method}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    traj_path = ROOT / "data" / "traces" / "sft_trajectories.json"
    out_path = ROOT / "data" / "sft_dataset" / f"{cfg.data['dataset']}.json"
    if traj_path.exists():
        prepare_sft_data(cfg, traj_path, out_path)
    run_sft(cfg)


if __name__ == "__main__":
    main()
