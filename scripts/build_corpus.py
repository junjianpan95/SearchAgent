"""构建全量检索语料。

数据源：
1. AgenticRAGTracer（1299 条多跳问题，每个 hop 带 doc 段落，格式 '"标题"\\n正文'）
2. HotpotQA（context 段落：title + sentences）
3. MuSiQue（paragraphs 段落，可选）

输出：data/corpus/corpus.jsonl，每行 {id, page, text}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_doc(doc: str) -> tuple[str, str]:
    """解析 '"标题"\\n正文' 格式的 doc，返回 (title, text)。"""
    doc = doc.strip()
    if not doc:
        return "", ""
    # 匹配开头的 "标题"
    m = re.match(r'^"([^"]+)"\s*\n(.*)$', doc, re.DOTALL)
    if m:
        title = m.group(1).strip()
        text = m.group(2).strip()
        return title, text
    # 没有引号标题：整段作为 text，title 取前 40 字符
    return doc[:40], doc


def from_agenticragtracer(out: list[dict], seen: set) -> int:
    """从 AgenticRAGTracer 6 个 jsonl 提取 hop 级 doc。"""
    d = ROOT / "data" / "datasets" / "agenticragtracer"
    if not d.exists():
        return 0
    cnt = 0
    for f in sorted(d.glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for k in item:
                if not k.startswith("hop_"):
                    continue
                doc = item[k].get("doc", "")
                title, text = parse_doc(doc)
                if not text:
                    continue
                key = title.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append({"id": f"art_{len(out)}", "page": title, "text": text})
                cnt += 1
    return cnt


def from_hotpotqa(out: list[dict], seen: set, max_items: int = None) -> int:
    """从 HotpotQA parquet 提取 context 段落（title + sentences）。"""
    import pandas as pd

    raw = ROOT / "data" / "datasets" / "raw"
    parquets = [raw / "hotpotqa_val.parquet"]
    for p in sorted(raw.glob("hotpotqa_train_*.parquet")):
        parquets.append(p)
    cnt = 0
    for p in parquets:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if max_items:
            df = df.head(max_items)
        for _, row in df.iterrows():
            ctx = row.get("context", {})
            titles = [str(t) for t in ctx.get("title", [])]
            sentences = list(ctx.get("sentences", []))
            for title, sents in zip(titles, sentences):
                key = title.lower()
                if key in seen:
                    continue
                text = " ".join(str(s) for s in sents)
                if not text:
                    continue
                seen.add(key)
                out.append({"id": f"hp_{len(out)}", "page": title, "text": text})
                cnt += 1
    return cnt


def from_musique(out: list[dict], seen: set, max_items: int = None) -> int:
    """从 MuSiQue dev 提取 paragraphs 段落（用于 zero-shot 检索语料）。"""
    import json

    d = ROOT / "data" / "datasets" / "musique"
    if not d.exists():
        return 0
    cnt = 0
    for f in sorted(d.glob("*.jsonl")):
        for i, line in enumerate(open(f, encoding="utf-8")):
            if max_items and i >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            paragraphs = item.get("paragraphs", [])
            for para in paragraphs:
                title = para.get("title", "")
                text = para.get("paragraph_text", "")
                key = (title + text[:50]).lower()
                if key in seen or not text:
                    continue
                seen.add(key)
                out.append({"id": f"mq_{len(out)}", "page": title, "text": text})
                cnt += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpotqa-max", type=int, default=5000,
                    help="HotpotQA 取多少条构造 context（默认 5000，全量会很大）")
    ap.add_argument("--musique-max", type=int, default=2000,
                    help="MuSiQue 取多少条 paragraphs")
    args = ap.parse_args()

    out: list[dict] = []
    seen: set = set()

    n1 = from_agenticragtracer(out, seen)
    print(f"[CORPUS] AgenticRAGTracer: {n1} 段")

    n2 = from_hotpotqa(out, seen, max_items=args.hotpotqa_max)
    print(f"[CORPUS] HotpotQA: {n2} 段")

    n3 = from_musique(out, seen, max_items=args.musique_max)
    print(f"[CORPUS] MuSiQue: {n3} 段")

    out_dir = ROOT / "data" / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "corpus.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[CORPUS] 总计 {len(out)} 段 -> {out_path}")

    # 顺便输出统计
    pages = {}
    for item in out:
        pages[item["page"]] = pages.get(item["page"], 0) + 1
    print(f"[CORPUS] 去重后页面数: {len(pages)}")


if __name__ == "__main__":
    main()
