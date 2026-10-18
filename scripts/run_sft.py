"""真实可执行的 SFT 训练脚本（transformers + PEFT LoRA/DoRA）。

直接加载 Qwen3-1.7B + HotpotQA 轨迹数据，跑 LoRA SFT，
保存 adapter 到 data/models/sft_lora（原始权重不变，另存 adapter）。

用法：
  python scripts/run_sft.py --config configs/local_smoke.yaml --epochs 1 --max-samples 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ROOT


def build_training_records(max_samples: int = None) -> list[dict]:
    """从 SearchTrace 轨迹构造 SFT 训练样本（instruction -> output）。

    只监督 Agent 生成段（query/action/evidence/answer），
    环境观察（检索结果）作为 input 的一部分但不进 loss。
    这里简化为 instruction=问题，output=动作序列+答案。
    """
    traj_path = ROOT / "data" / "traces" / "sft_trajectories.json"
    if not traj_path.exists():
        # 回退：用 HotpotQA 数据直接构造简单 SFT 样本
        return _from_hotpotqa(max_samples)

    with open(traj_path, "r", encoding="utf-8") as f:
        trajs = json.load(f)

    records = []
    for t in trajs[:max_samples]:
        actions_str = "\n".join(
            f"{a['type']}: {json.dumps(a.get('args', {}), ensure_ascii=False)}"
            for a in t["actions"]
        )
        records.append({
            "instruction": t["question"],
            "output": f"{actions_str}\nAnswer: {t['answer']}",
        })
    return records


def _from_hotpotqa(max_samples=None) -> list[dict]:
    ds = ROOT / "data" / "datasets" / "hotpotqa_train.json"
    if not ds.exists():
        # 内置样例回退
        return [
            {"instruction": "Who is the author of the novel The Shining?", "output": "Answer: Stephen King"},
            {"instruction": "Where was Christopher Nolan born?", "output": "Answer: London"},
            {"instruction": "Which club did Thomas Müller play for in 2010?", "output": "Answer: Bayern Munich"},
        ]
    with open(ds, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for item in data[:max_samples]:
        records.append({
            "instruction": item["question"],
            "output": f"Answer: {item['answer']}",
        })
    return records


def run_sft_real(cfg, epochs, max_samples, method):
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    model_dir = ROOT / "data" / "models" / "Qwen3-1.7B"
    if not model_dir.exists():
        print(f"[SFT] 模型不存在: {model_dir}，请先下载")
        return

    print(f"[SFT] 加载 {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False

    # LoRA / DoRA
    use_dora = (method == "dora")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.sft.get("lora_r", 16),
        lora_alpha=cfg.sft.get("lora_alpha", 32),
        lora_dropout=cfg.sft.get("lora_dropout", 0.05),
        use_dora=use_dora,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    records = build_training_records(max_samples)
    print(f"[SFT] 训练样本 {len(records)} 条")

    # 组装 prompt
    def format_example(r):
        return f"### Instruction\n{r['instruction']}\n\n### Response\n{r['output']}"

    texts = [format_example(r) for r in records]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc
        def __len__(self):
            return self.enc["input_ids"].shape[0]
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}

    args = TrainingArguments(
        output_dir=str(ROOT / "data" / "models" / f"sft_{method}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg.sft.get("batch_size", 4),
        gradient_accumulation_steps=cfg.sft.get("grad_accum", 8),
        learning_rate=cfg.sft["lr"],
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=DS(enc),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()

    # 保存 adapter（原始权重不变，只存 LoRA adapter）
    save_dir = ROOT / "data" / "models" / f"sft_{method}"
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"[SFT] adapter 已保存 -> {save_dir}")
    print(f"[SFT] 原始模型权重保持不变 -> {model_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--method", choices=["lora", "dora"], default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    method = args.method or cfg.sft.get("method", "lora")
    run_sft_real(cfg, args.epochs, args.max_samples, method)


if __name__ == "__main__":
    main()
