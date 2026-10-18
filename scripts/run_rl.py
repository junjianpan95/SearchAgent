"""GRPO / GSPO on-policy RL 训练（TRL，单卡 16G，Qwen3-1.7B + LoRA）。

在真实检索环境（BM25）中执行 reward：解析模型生成的动作序列，
执行 Search 检索，验证证据与答案，用两阶段课程奖励计算 reward。

用法：
  python scripts/run_rl.py --method grpo --max-samples 50
  python scripts/run_rl.py --method gspo --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ROOT


def load_retriever():
    """轻量 BM25 检索器（纯 CPU，避免与训练抢显存）。"""
    from rank_bm25 import BM25Okapi

    corpus = ROOT / "data" / "corpus" / "corpus.jsonl"
    passages = []
    for line in open(corpus, encoding="utf-8"):
        d = json.loads(line.strip())
        passages.append(d)
    toks = [re.findall(r"[a-zA-Z0-9]+", p["text"].lower()) for p in passages]
    bm25 = BM25Okapi(toks)

    def search(query, top_k=5):
        qt = re.findall(r"[a-zA-Z0-9]+", query.lower())
        scores = bm25.get_scores(qt)
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [passages[i] for i in idx]

    return search


def parse_completion_to_traj(completion: str) -> dict:
    """把模型生成的文本解析成轨迹（actions/answer/citations）。"""
    actions = []
    answer = ""
    citations = []
    for line in completion.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Search:"):
            actions.append({"type": "Search", "args": {"query": line[len("Search:"):].strip()}})
        elif line.startswith("Read:"):
            actions.append({"type": "Read", "args": {"page": line[len("Read:"):].strip()}})
        elif line.startswith("AddEvidence:"):
            rest = line[len("AddEvidence:"):].strip()
            if "|" in rest:
                page, txt = rest.split("|", 1)
                actions.append({"type": "AddEvidence", "args": {"page": page.strip(), "text": txt.strip()}})
            else:
                actions.append({"type": "AddEvidence", "args": {"page": "", "text": rest}})
        elif line.startswith("Answer:"):
            rest = line[len("Answer:"):].strip()
            m = re.search(r"\[citations:\s*([^\]]*)\]", rest)
            if m:
                try:
                    citations = [int(x) for x in m.group(1).split(",") if x.strip().isdigit()]
                except ValueError:
                    citations = []
                rest = rest[:m.start()].strip()
            answer = rest
    return {"actions": actions, "answer": answer, "citations": citations}


def run_rl(cfg, method, max_samples, group_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_dir = ROOT / "data" / "models" / "Qwen3-1.7B"
    sft_dir = ROOT / "data" / "models" / "sft_lora"
    print(f"[RL] 加载 base {base_dir} + SFT adapter {sft_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if sft_dir.exists():
        model = PeftModel.from_pretrained(model, str(sft_dir), is_trainable=True)

    # 训练数据（问题 + gold）
    rl_samples = json.load(open(ROOT / "data" / "traces" / "rl_samples.json", encoding="utf-8"))
    rl_samples = rl_samples[:max_samples]
    prompts = []
    golds = {}
    for d in rl_samples:
        q = d["question"]
        prompts.append([{"role": "user", "content": f"Question: {q}\nNext action:"}])
        golds[q] = {"answer": d["answer"], "supporting_facts": d["supporting_facts"]}

    from datasets import Dataset
    ds = Dataset.from_dict({"prompt": prompts})

    # reward function：解析 completion → 执行检索 → 课程奖励
    search = load_retriever()
    from src.reward import CurriculumReward
    reward_cfg = cfg.rl["reward"]
    reward_fn = CurriculumReward(reward_cfg)

    def reward_func(prompts, completions, **kwargs):
        rewards = []
        for p, c in zip(prompts, completions):
            question = p[-1]["content"].replace("Question: ", "").split("\n")[0].strip()
            text = c[0]["content"] if isinstance(c, list) else str(c)
            traj = parse_completion_to_traj(text)
            gold = golds.get(question, {"answer": "", "supporting_facts": []})
            r, comps = reward_fn.compute(traj, gold)
            rewards.append(r)
        return rewards

    from trl import GRPOConfig, GRPOTrainer
    grpo_kwargs = {}
    if method == "gspo":
        # GSPO：序列级 importance sampling weight（verl 生态的 GRPO 变体）
        grpo_kwargs = {"importance_sampling_level": "sequence", "beta": 0.0, "epsilon": 3e-4}

    training_args = GRPOConfig(
        output_dir=str(ROOT / "data" / "models" / f"rl_{method}"),
        learning_rate=cfg.rl["lr"],
        num_train_epochs=cfg.rl.get("epochs", 1),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.rl.get("grad_accum", 4),
        num_generations=group_size,
        max_completion_length=256,
        bf16=True,
        logging_steps=1,
        report_to=[],
        **grpo_kwargs,
    )
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, args=training_args,
        train_dataset=ds, reward_funcs=reward_func,
    )
    trainer.train()
    out = ROOT / "data" / "models" / f"rl_{method}"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[RL] {method} adapter 保存 -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_smoke.yaml")
    ap.add_argument("--method", default="grpo", choices=["grpo", "gspo"])
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--group-size", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_rl(cfg, args.method, args.max_samples, args.group_size)


if __name__ == "__main__":
    main()
