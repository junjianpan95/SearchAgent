"""下载 Wikipedia 多跳问答数据集（HotpotQA 等）。

HotpotQA 是标准多跳 QA 基准：每个问题带 supporting facts + 上下文段落。
用 datasets 库从 HuggingFace（hf-mirror 镜像）拉取，转成 SearchTrace 需要的 JSON 格式。

数据格式输出（data/datasets/）：
  hotpotqa_train.json / hotpotqa_dev.json
  [{"id","question","answer","supporting_facts":[...],"supporting_pages":[...],"context":[...]}]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "datasets"


def download_hotpotqa(split: str, max_n: int = None) -> list[dict]:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    ds = load_dataset("hotpotqa", "distractor", split=split, trust_remote_code=True)
    out = []
    for i, item in enumerate(ds):
        if max_n and i >= max_n:
            break
        # supporting_facts 是 [["title", "sent_idx"], ...] 格式
        titles = []
        facts = []
        for title, sent_idx in item.get("supporting_facts", []):
            if title not in titles:
                titles.append(title)
            facts.append(f"{title}::{sent_idx}")

        # 收集 supporting 段落文本
        supporting_texts = []
        context = item.get("context", {})
        # context 是 {title: [sent1, sent2, ...]}
        for title, sents in context.items():
            if title in titles:
                supporting_texts.append(" ".join(sents))

        out.append({
            "id": item.get("id", str(i)),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "supporting_facts": supporting_texts,
            "supporting_pages": titles,
            "supporting_fact_ids": facts,
        })
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, max_n in [("train", None), ("validation", None)]:
        print(f"[DATA] 下载 HotpotQA {split} ...")
        data = download_hotpotqa(split)
        path = DATA_DIR / f"hotpotqa_{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DATA] {split}: {len(data)} 条 -> {path}")


if __name__ == "__main__":
    main()
