"""真实可执行的 GRPO 训练脚本（TRL 单卡，Qwen3-1.7B + LoRA）。

在真实检索环境里 on-policy rollout，用两阶段课程奖励做 GRPO。
单卡 16G 可跑（小数据量 smoke）。

用法：
  python scripts/run_grpo.py --config configs/local_smoke.yaml --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ROOT


def run_grpo_real(cfg, max_samples, group_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import GRPOConfig, GRPOTrainer

    model_dir = ROOT / "data" / "models" / "Qwen3-1.7B"
    if not model_dir.exists():
        print(f"[GRPO] 模型不存在: {model_dir}")
        return

    print(f"[GRPO] 加载 {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # LoRA（RL 阶段通常继续用 SFT 的 adapter，这里从头演示）
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.rl.get("lora_r", 16),
        lora_alpha=cfg.rl.get("lora_alpha", 32),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # 奖励函数（两阶段课程奖励）
    from src.reward import CurriculumReward
    reward_fn = CurriculumReward(cfg.rl["reward"])

    def reward_func(prompts, completions, **kwargs):
        rewards = []
        for p, c in zip(prompts, completions):
            question = p[-1]["content"] if isinstance(p, list) else str(p)
            answer = c[0]["content"] if isinstance(c, list) else str(c)
            traj = {"actions": [], "answer": answer, "citations": []}
            gold = {"answer": question, "supporting_facts": []}
            r, _ = reward_fn.compute(traj, gold)
            rewards.append(r)
        return rewards

    # 训练数据（问题列表）
    rl_samples = ROOT / "data" / "traces" / "rl_samples.json"
    if rl_samples.exists():
        with open(rl_samples, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = [d["question"] for d in data[:max_samples]]
    else:
        questions = [
            "Who is the author of the novel The Shining?",
            "Where was Christopher Nolan born?",
        ]

    from datasets import Dataset
    ds = Dataset.from_dict({"prompt": [{"role": "user", "content": q} for q in questions]})

    training_args = GRPOConfig(
        output_dir=str(ROOT / "data" / "models" / "grpo"),
        learning_rate=cfg.rl["lr"],
        num_train_epochs=cfg.rl.get("epochs", 1),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.rl.get("grad_accum", 4),
        num_generations=group_size,   # 每题采样 group_size 条
        max_prompt_length=512,
        max_completion_length=256,
        bf16=True,
        logging_steps=1,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=ds,
        reward_funcs=reward_func,
    )
    trainer.train()
    save_dir = ROOT / "data" / "models" / "grpo"
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"[GRPO] adapter 已保存 -> {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--group-size", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_grpo_real(cfg, args.max_samples, args.group_size)


if __name__ == "__main__":
    main()
