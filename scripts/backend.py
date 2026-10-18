"""在线问答后端（Flask）。

提供 POST /api/ask，用户输入问题 → 加载真实检索索引 + 模型 → 跑多跳检索 Agent → 返回轨迹+答案。

支持两种模式：
1. 真实模型（默认）：加载 Qwen3-1.7B，用提示词引导生成 Search/Read/AddEvidence/Answer 动作
2. 规则回退（--rule）：无模型时用检索+答案拼接，仍能演示完整多跳流程
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.config import load_config, ROOT
from src.agent import SearchAgent

app = Flask(__name__)
CORS(app)

# 全局：检索器 + LLM
RETRIEVER = None
LLM = None
CORPUS = []  # [{page, text}]


# ============================================================ 语料加载
def load_corpus() -> list[dict]:
    """加载全量 Wikipedia 语料作为检索 passages。

    优先用 build_corpus.py 生成的全量语料 corpus.jsonl（10 万段）；
    否则回退到 HotpotQA supporting facts；最后用内置样例。
    """
    corpus = ROOT / "data" / "corpus" / "corpus.jsonl"
    if corpus.exists():
        passages = []
        for line in open(corpus, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            passages.append({"page": d.get("page", ""), "text": d.get("text", "")})
        print(f"[CORPUS] 从全量语料加载 {len(passages)} 段")
        return passages
    # 内置样例回退
    return [
        {"page": "The Shining (film)", "text": "The Shining is a 1980 horror film based on Stephen King's 1977 novel."},
        {"page": "Stephen King", "text": "Stephen King is an American author who wrote the novel The Shining."},
        {"page": "Inception", "text": "Inception is a 2010 science fiction action film directed by Christopher Nolan."},
        {"page": "Christopher Nolan", "text": "Christopher Nolan is a British-American filmmaker born in London."},
        {"page": "2010 FIFA World Cup", "text": "The 2010 FIFA World Cup Golden Boot was won by Thomas Müller."},
        {"page": "Thomas Müller", "text": "Thomas Müller is a German footballer who played for Bayern Munich in 2010."},
    ]


# ============================================================ LLM 动作生成
class ModelLLM:
    """用 Qwen3-1.7B 生成动作（真实模型推理）。"""

    ACTION_PROMPT = """You are a multi-hop retrieval agent. Given a question, decide the next action.
Available actions:
- {"type":"Search","args":{"query":"..."}}  search for a query
- {"type":"Read","args":{"page":"..."}}    read a page
- {"type":"AddEvidence","args":{"text":"...","page":"..."}}  add supporting evidence
- {"type":"Answer","args":{"answer":"...","citations":[0,1]}}  final answer

Current evidence: {evidence}

Question: {question}

Output ONLY a JSON object (one action), no explanation:
"""

    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        print(f"[LLM] 加载模型 {model_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        print("[LLM] 模型加载完成")

    def __call__(self, context=None, state=None, **kw):
        question = context[0].replace("Question: ", "") if context else ""
        evidence = "\n".join(e.text for e in state.memory.items) if state else ""
        # 用 replace 而非 format，避免 JSON 花括号被当成占位符
        prompt = (self.ACTION_PROMPT
                  .replace("{question}", question)
                  .replace("{evidence}", evidence or "(none)"))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_action(text)

    def _parse_action(self, text: str) -> dict:
        """从模型输出提取 JSON 动作。"""
        import re
        # 提取第一个 {...}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"type": "Answer", "args": {"answer": text.strip(), "citations": []}}
        try:
            obj = json.loads(m.group(0))
            return obj
        except json.JSONDecodeError:
            return {"type": "Answer", "args": {"answer": text.strip(), "citations": []}}


class RuleLLM:
    """规则回退：无模型时，用检索结果模拟多跳动作序列。

    策略：Search(问题关键词) → Read(top 页面) → AddEvidence(逐条) → Answer(拼接命中文本)。
    完整走一遍动作空间，让前端能展示 Search→Read→AddEvidence→Answer 全过程。
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self._step = 0
        self._evidence = []
        self._hits = []

    def __call__(self, context=None, state=None, **kw):
        question = context[0].replace("Question: ", "") if context else ""
        self._step += 1

        # 第1步：Search
        if self._step == 1:
            return {"type": "Search", "args": {"query": question}}

        # 第2步：Read 第一个命中页面
        if self._step == 2:
            self._hits = self.retriever.search(question, top_k=3)
            if self._hits:
                return {"type": "Read", "args": {"page": self._hits[0].page}}
            return {"type": "Answer", "args": {"answer": "No answer found", "citations": []}}

        # 第3步起：AddEvidence（逐条加入命中文本）
        if self._step < 2 + len(self._hits) + 1:
            h = self._hits[self._step - 3]
            if h.text not in self._evidence:
                self._evidence.append(h.text)
                return {"type": "AddEvidence", "args": {"text": h.text, "page": h.page}}
            return {"type": "AddEvidence", "args": {"text": h.text, "page": h.page}}

        # 最后：Answer（用证据中最相关的片段作答）
        answer = self._extract_answer(question, self._evidence)
        return {"type": "Answer", "args": {"answer": answer,
                                            "citations": list(range(len(self._evidence)))}}

    def _extract_answer(self, question, evidence):
        """从证据中抽取答案：取与问题关键词最相关的句子。"""
        if not evidence:
            return "No answer found"
        # 简化：返回证据中第一句（真实场景由模型生成）
        best = max(evidence, key=len)
        return best.rstrip(".")


# ============================================================ API
@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is empty"}), 400

    from src.config import load_config
    cfg = load_config(ROOT / "configs" / "local_smoke.yaml")

    # RuleLLM 有状态，每次请求新建一个（模型 LLM 无状态可复用）
    llm = LLM
    if isinstance(LLM, RuleLLM):
        llm = RuleLLM(RETRIEVER)

    agent = SearchAgent(
        retriever=RETRIEVER,
        llm_fn=llm,
        max_search=cfg.agent["max_search"],
        max_read=cfg.agent["max_read"],
        max_evidence=cfg.agent["max_evidence"],
        token_budget=cfg.agent["token_budget"],
        dedup_queries=cfg.agent["dedup_queries"],
        bridge_entity_direct_read=cfg.agent["bridge_entity_direct_read"],
        recovery_feedback=cfg.agent["recovery_feedback"],
    )
    state = agent.run(question)

    return jsonify({
        "question": question,
        "answer": state.answer,
        "citations": state.citations,
        "trajectory": state.trajectory,
        "budget": state.budget.snapshot(),
        "evidence": [e.text for e in state.memory.items],
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "corpus_size": len(CORPUS),
                    "model_loaded": LLM is not None, "backend": "model" if isinstance(LLM, ModelLLM) else "rule"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--rule", action="store_true", help="无模型规则回退（无需 GPU/模型）")
    ap.add_argument("--model-dir", default=None, help="模型目录（默认 data/models/Qwen3-1.7B）")
    args = ap.parse_args()

    global RETRIEVER, LLM, CORPUS

    # 检索器：用 BM25（避免 BGE-M3 需要额外下载时也能跑）
    from src.retrieval.bm25 import BM25Index
    from src.retrieval.base import Hit

    class BM25Retriever:
        def __init__(self):
            self._idx = BM25Index()
            self._pages = {}

        def build(self, passages):
            self._idx.build(passages)
            for d in passages:
                self._pages[d["page"]] = d["text"]

        def search(self, query, top_k=20):
            return self._idx.search(query, top_k)

        def read(self, page):
            return self._pages.get(page, "")

    CORPUS = load_corpus()
    RETRIEVER = BM25Retriever()
    RETRIEVER.build(CORPUS)
    print(f"[RETRIEVER] BM25 索引已构建，共 {len(CORPUS)} 段")

    if args.rule:
        LLM = RuleLLM(RETRIEVER)
        print("[LLM] 规则回退模式（无需模型）")
    else:
        model_dir = args.model_dir or str(ROOT / "data" / "models" / "Qwen3-1.7B")
        if not Path(model_dir).exists():
            print(f"[WARN] 模型目录不存在: {model_dir}，回退到规则模式")
            LLM = RuleLLM(RETRIEVER)
        else:
            LLM = ModelLLM(Path(model_dir))

    print(f"[SERVER] 启动在线问答后端 http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
