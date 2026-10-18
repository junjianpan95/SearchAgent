"""把 HotpotQA parquet 转成 SearchTrace JSON 格式。

读 data/datasets/raw/*.parquet，输出 data/datasets/hotpotqa_{split}.json
字段对齐 SearchTrace 的 MultiHopSample。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def convert(parquet_path: Path, out_path: Path, max_n: int = None):
    import pandas as pd
    df = pd.read_parquet(parquet_path)

    out = []
    for i, row in df.iterrows():
        if max_n and i >= max_n:
            break
        # supporting_facts: {'title': array, 'sent_id': array}
        sf = row.get("supporting_facts", {})
        titles = [str(t) for t in sf.get("title", [])]
        sent_ids = [int(s) for s in sf.get("sent_id", [])]
        fact_texts = [f"{t} (sent {s})" for t, s in zip(titles, sent_ids)]

        # context: {'title': array, 'sentences': array}
        context = row.get("context", {})
        ctx_titles = [str(t) for t in context.get("title", [])]
        ctx_sentences = list(context.get("sentences", []))

        # 抽取 supporting 句子的实际文本
        supporting_texts = []
        for t, s in zip(titles, sent_ids):
            t_str = str(t)
            if t_str in ctx_titles:
                idx = ctx_titles.index(t_str)
                sents = ctx_sentences[idx]
                # sents 可能是 list/tuple 或 numpy array
                try:
                    if s < len(sents):
                        supporting_texts.append(str(sents[s]))
                except TypeError:
                    supporting_texts.append(str(sents))

        out.append({
            "id": row.get("id", str(i)),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "supporting_facts": supporting_texts,
            "supporting_pages": titles,
            "supporting_fact_ids": fact_texts,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[CONVERT] {len(out)} 条 -> {out_path}")
    return out


def main():
    raw_dir = ROOT / "data" / "datasets" / "raw"
    if not raw_dir.exists():
        print("无 parquet 文件，请先下载")
        return

    for p in sorted(raw_dir.glob("*.parquet")):
        split = "validation" if "val" in p.name else "train"
        convert(p, ROOT / "data" / "datasets" / f"hotpotqa_{split}.json", max_n=None)


if __name__ == "__main__":
    main()
