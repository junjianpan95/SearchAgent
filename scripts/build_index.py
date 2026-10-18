"""构建混合检索索引（BM25 + BGE-M3），缓存到磁盘供后端/Agent 加载。

流程：
1. 加载 data/corpus/corpus.jsonl（全量语料）
2. BM25 索引（纯 CPU，秒级）
3. BGE-M3 稠密向量（GPU，几分钟）
4. 缓存：data/index/{corpus.jsonl, embeddings.npy, bm25.pkl}

后续通过 src.retrieval.hybrid 的缓存加载路径复用。
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def tokenize(t: str) -> list[str]:
    import re
    return re.findall(r"[a-zA-Z0-9]+", t.lower())


def load_corpus() -> list[dict]:
    corpus = ROOT / "data" / "corpus" / "corpus.jsonl"
    passages = []
    for line in open(corpus, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            passages.append({"page": d["page"], "text": d["text"]})
    return passages


def build_bm25(passages):
    from rank_bm25 import BM25Okapi
    corpus_tok = [tokenize(p["text"]) for p in passages]
    bm25 = BM25Okapi(corpus_tok)
    return bm25


def build_dense(passages, model_path, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, device=device)
    texts = [p["text"] for p in passages]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                       batch_size=64).astype(np.float32)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/bge-m3", help="BGE-M3 模型路径")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-docs", type=int, default=None, help="限制文档数（调试用）")
    args = ap.parse_args()

    passages = load_corpus()
    if args.max_docs:
        passages = passages[: args.max_docs]
    print(f"[INDEX] 语料 {len(passages)} 段")

    out_dir = ROOT / "data" / "index"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. BM25
    t0 = time.time()
    bm25 = build_bm25(passages)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25}, f)
    print(f"[INDEX] BM25 完成 {time.time()-t0:.1f}s")

    # 2. BGE-M3 稠密
    t0 = time.time()
    emb = build_dense(passages, args.model, args.device)
    np.save(out_dir / "embeddings.npy", emb)
    print(f"[INDEX] BGE-M3 稠密完成 {time.time()-t0:.1f}s，向量 {emb.shape}")

    # 3. 语料副本（含顺序，与向量对齐）
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[INDEX] 索引已缓存到 {out_dir}/")
    print(f"[INDEX] 完成：BM25 + BGE-M3({emb.shape[1]}维) 混合索引")


if __name__ == "__main__":
    main()
