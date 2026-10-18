"""SearchTrace：多跳问题集与工具轨迹构造。

对标 AgenticRagTrace 的替代自造词。职责：
1. 构造 Wikipedia 多跳问题集（问题 + 答案 + supporting facts + 页面）
2. 生成三类轨迹（供 SFT）：
   - Oracle 轨迹（成功路径，60%）
   - Teacher 多样化轨迹（25%）
   - 查询漂移恢复轨迹（15%）
3. 生成 RL 样本

数据来源：优先本地已有 Wikipedia 语料；否则用内置小型样例集做 smoke。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 项目根目录（避免循环导入 src.config）
ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class MultiHopSample:
    """一条多跳问答样本。"""
    question: str
    answer: str
    supporting_facts: list[str]          # gold supporting facts
    supporting_pages: list[str]          # 支撑证据所在页面
    id: str = ""
    metadata: dict = field(default_factory=dict)
    hops: list[dict] = field(default_factory=list)  # hop 级中间问题/答案/文档


@dataclass
class Trajectory:
    """一条工具轨迹（SFT/RL 用）。"""
    sample_id: str
    question: str
    actions: list[dict]                  # [{'type','args','result'}, ...]
    answer: str
    citations: list[int]
    traj_type: str                       # oracle | teacher | drift_recovery


class SearchTrace:
    """多跳问题集与轨迹构造器。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.samples: list[MultiHopSample] = []

    # ------------------------------------------------------------ 问题集
    def load_samples(self, path: Optional[Path] = None) -> list[MultiHopSample]:
        """加载多跳问题集。smoke 模式用内置样例，full 模式从外部 JSON 加载。"""
        if path is not None and Path(path).exists():
            self.samples = self._load_json(path)
            return self.samples
        # 优先从 AgenticRagTrace 数据加载
        art = self.load_agenticragtracer()
        if art:
            self.samples = art
            return self.samples
        # smoke：内置样例（真实多跳结构，无版权问题）
        self.samples = self._builtin_samples()
        return self.samples

    def load_agenticragtracer(self, dir_path: Optional[Path] = None) -> list[MultiHopSample]:
        """从 AgenticRAGTracer 数据集（hop 级结构）加载多跳样本。

        每个 jsonl 一行：{'hop_1': {question, answer, doc}, 'hop_2': {...},
                          'final_question', 'final_answer'}
        doc 格式为 '"标题"\\n正文段落'。
        """
        import json
        import re

        if dir_path is None:
            dir_path = ROOT / "data" / "datasets" / "agenticragtracer"
        if not Path(dir_path).exists():
            return []

        def parse_doc(doc: str):
            doc = doc.strip()
            m = re.match(r'^"([^"]+)"\s*\n(.*)$', doc, re.DOTALL)
            if m:
                return m.group(1).strip(), m.group(2).strip()
            return doc[:40], doc

        out: list[MultiHopSample] = []
        for f in sorted(Path(dir_path).glob("*.jsonl")):
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                hops = []
                facts, pages = [], []
                for k in sorted(d.keys()):
                    if not k.startswith("hop_"):
                        continue
                    h = d[k]
                    title, text = parse_doc(h.get("doc", ""))
                    hops.append({
                        "question": h.get("question", ""),
                        "answer": h.get("answer", ""),
                        "page": title,
                        "text": text,
                    })
                    if text:
                        facts.append(text)
                        pages.append(title)
                if not hops:
                    continue
                out.append(MultiHopSample(
                    question=d.get("final_question", hops[-1]["question"]),
                    answer=str(d.get("final_answer", hops[-1]["answer"])),
                    supporting_facts=facts,
                    supporting_pages=pages,
                    id=str(d.get("id", f"art_{len(out)}")),
                    metadata={"source": f.name, "num_hops": len(hops)},
                    hops=hops,
                ))
        return out

    def _load_json(self, path: Path) -> list[MultiHopSample]:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for d in data:
            # answer 可能是 list（HotpotQA 多别名），取第一个
            ans = d.get("answer", "")
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            # supporting_facts 可能是 [{"page","text"}] 或纯文本 list
            facts = d.get("supporting_facts", [])
            pages = d.get("supporting_pages", [])
            if facts and isinstance(facts[0], dict):
                pages = [f.get("page", "") for f in facts]
                facts = [f.get("text", "") for f in facts]
            out.append(MultiHopSample(
                question=d["question"],
                answer=str(ans),
                supporting_facts=list(facts),
                supporting_pages=list(pages),
                id=str(d.get("id", len(out))),
                metadata=d.get("metadata", {}),
            ))
        return out

    def _builtin_samples(self) -> list[MultiHopSample]:
        """内置 smoke 样例：真实的两跳/三跳桥接结构。"""
        samples = [
            MultiHopSample(
                question="Who is the author of the novel that the movie 'The Shining' was based on?",
                answer="Stephen King",
                supporting_facts=[
                    "The Shining is a 1980 horror film based on Stephen King's 1977 novel of the same name.",
                    "Stephen King wrote the novel The Shining.",
                ],
                supporting_pages=["The Shining (film)", "Stephen King"],
                id="s0",
            ),
            MultiHopSample(
                question="What country does the director of 'Inception' come from?",
                answer="United Kingdom",
                supporting_facts=[
                    "Inception is a 2010 film directed by Christopher Nolan.",
                    "Christopher Nolan is a British-American filmmaker born in London.",
                ],
                supporting_pages=["Inception", "Christopher Nolan"],
                id="s1",
            ),
            MultiHopSample(
                question="Which team did the player who won the 2010 FIFA World Cup Golden Boot play for at club level that year?",
                answer="Bayern Munich",
                supporting_facts=[
                    "The 2010 FIFA World Cup Golden Boot was won by Thomas Müller.",
                    "Thomas Müller played for Bayern Munich in 2010.",
                ],
                supporting_pages=["2010 FIFA World Cup", "Thomas Müller"],
                id="s2",
            ),
        ]
        return samples

    # ------------------------------------------------------------ 轨迹构造
    def build_trajectories(self, samples: list[MultiHopSample],
                           mix: dict) -> list[Trajectory]:
        """按配比生成三类轨迹。

        mix = {'oracle':0.6, 'teacher':0.25, 'drift_recovery':0.15}
        """
        n = len(samples)
        n_oracle = int(n * mix["oracle"])
        n_teacher = int(n * mix["teacher"])
        n_drift = n - n_oracle - n_teacher

        trajs: list[Trajectory] = []
        for i, s in enumerate(samples[:n_oracle]):
            trajs.append(self._oracle_traj(s))
        for i, s in enumerate(samples[n_oracle:n_oracle + n_teacher]):
            trajs.append(self._teacher_traj(s))
        for i, s in enumerate(samples[n_oracle + n_teacher:]):
            trajs.append(self._drift_recovery_traj(s))
        return trajs

    def _oracle_traj(self, s: MultiHopSample) -> Trajectory:
        """Oracle 成功轨迹：按 hop 顺序逐跳查询桥接实体，命中支持句。"""
        actions = []
        # 有 hop 级信息时，用 hop 的中间问题作为查询（最真实）
        if s.hops:
            for i, hop in enumerate(s.hops):
                q = hop["question"] or hop["page"]
                actions.append({"type": "Search", "args": {"query": q},
                                "result": {"hits": [{"page": hop["page"], "text": hop["text"]}]}})
                actions.append({"type": "AddEvidence",
                                "args": {"text": hop["text"], "page": hop["page"]},
                                "result": {"ok": True}})
        else:
            for page in s.supporting_pages:
                actions.append({"type": "Search", "args": {"query": page},
                                "result": {"hits": [{"page": page, "text": s.supporting_facts[0]}]}})
                actions.append({"type": "AddEvidence",
                                "args": {"text": s.supporting_facts[0], "page": page},
                                "result": {"ok": True}})
        actions.append({"type": "Answer",
                        "args": {"answer": s.answer, "citations": list(range(len(s.supporting_pages)))},
                        "result": {}})
        return Trajectory(sample_id=s.id, question=s.question, actions=actions,
                          answer=s.answer, citations=list(range(len(s.supporting_pages))),
                          traj_type="oracle")

    def _teacher_traj(self, s: MultiHopSample) -> Trajectory:
        """Teacher 多样化轨迹：不同的查询措辞/顺序，仍能到达答案。"""
        actions = [
            {"type": "Search", "args": {"query": s.question.split()[0] + " information"},
             "result": {"hits": [{"page": s.supporting_pages[0], "text": s.supporting_facts[0]}]}},
            {"type": "Read", "args": {"page": s.supporting_pages[0]},
             "result": {"content": s.supporting_facts[0]}},
            {"type": "AddEvidence",
             "args": {"text": s.supporting_facts[0], "page": s.supporting_pages[0]},
             "result": {"ok": True}},
            {"type": "Search", "args": {"query": s.supporting_pages[-1]},
             "result": {"hits": [{"page": s.supporting_pages[-1], "text": s.supporting_facts[-1]}]}},
            {"type": "AddEvidence",
             "args": {"text": s.supporting_facts[-1], "page": s.supporting_pages[-1]},
             "result": {"ok": True}},
            {"type": "Answer",
             "args": {"answer": s.answer, "citations": list(range(len(s.supporting_pages)))},
             "result": {}},
        ]
        return Trajectory(sample_id=s.id, question=s.question, actions=actions,
                          answer=s.answer, citations=list(range(len(s.supporting_pages))),
                          traj_type="teacher")

    def _drift_recovery_traj(self, s: MultiHopSample) -> Trajectory:
        """查询漂移恢复轨迹：先走偏查询，通过 recovery feedback 回到正轨。"""
        actions = [
            {"type": "Search", "args": {"query": "unrelated " + s.question.split()[0]},
             "result": {"hits": []}},   # 漂移：空结果
            {"type": "Search", "args": {"query": s.supporting_pages[0]},
             "result": {"hits": [{"page": s.supporting_pages[0], "text": s.supporting_facts[0]}]}},
            {"type": "AddEvidence",
             "args": {"text": s.supporting_facts[0], "page": s.supporting_pages[0]},
             "result": {"ok": True}},
            {"type": "Search", "args": {"query": s.supporting_pages[-1]},
             "result": {"hits": [{"page": s.supporting_pages[-1], "text": s.supporting_facts[-1]}]}},
            {"type": "AddEvidence",
             "args": {"text": s.supporting_facts[-1], "page": s.supporting_pages[-1]},
             "result": {"ok": True}},
            {"type": "Answer",
             "args": {"answer": s.answer, "citations": list(range(len(s.supporting_pages)))},
             "result": {}},
        ]
        return Trajectory(sample_id=s.id, question=s.question, actions=actions,
                          answer=s.answer, citations=list(range(len(s.supporting_pages))),
                          traj_type="drift_recovery")

    # ------------------------------------------------------------ 序列化
    def save(self, trajs: list[Trajectory], path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([t.__dict__ for t in trajs], f, ensure_ascii=False, indent=2)

    def stats(self, trajs: list[Trajectory]) -> dict:
        from collections import Counter
        c = Counter(t.traj_type for t in trajs)
        total_actions = sum(len(t.actions) for t in trajs)
        return {
            "total_trajectories": len(trajs),
            "by_type": dict(c),
            "total_actions": total_actions,
            "avg_actions": total_actions / len(trajs) if trajs else 0,
        }
