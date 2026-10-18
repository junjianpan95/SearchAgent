"""两阶段课程奖励（用于 GRPO/GSPO on-policy RL）。

Stage 1（检索决策）：优化新增 supporting-fact、引用有效性、grounding。
Stage 2（答案质量）：强化 Answer EM/F1，同时保留证据约束。

另有四类惩罚：重复查询、冗余证据、非法动作、工具成本。
所有 reward component 可监控（抑制策略退化）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.metrics import (
    f1_score,
    validate_citations,
    evidence_all_hit,
)


@dataclass
class RewardComponents:
    """各 reward component 的数值（供分布监控）。"""
    new_supporting_fact: float = 0.0
    citation_validity: float = 0.0
    grounding: float = 0.0
    answer_f1: float = 0.0
    answer_em: float = 0.0
    dup_query_penalty: float = 0.0
    redundant_evidence_penalty: float = 0.0
    illegal_action_penalty: float = 0.0
    tool_cost_penalty: float = 0.0

    def as_dict(self) -> dict:
        return {
            "new_supporting_fact": self.new_supporting_fact,
            "citation_validity": self.citation_validity,
            "grounding": self.grounding,
            "answer_f1": self.answer_f1,
            "answer_em": self.answer_em,
            "dup_query_penalty": self.dup_query_penalty,
            "redundant_evidence_penalty": self.redundant_evidence_penalty,
            "illegal_action_penalty": self.illegal_action_penalty,
            "tool_cost_penalty": self.tool_cost_penalty,
        }


class CurriculumReward:
    """两阶段课程奖励计算器。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.stage = cfg.get("stage", 1)   # 1 或 2
        self.w_stage1 = cfg.get("stage1_weights", {
            "new_supporting_fact": 1.0,
            "citation_validity": 1.0,
            "grounding": 1.0,
        })
        self.w_stage2 = cfg.get("stage2_weights", {
            "answer_em_f1": 1.0,
            "evidence_consistency": 0.5,
        })
        self.penalty_w = cfg.get("penalty_weights", {
            "dup_query": 0.1,
            "redundant_evidence": 0.1,
            "illegal_action": 0.2,
            "tool_cost": 0.05,
        })

    def compute(self, trajectory: dict, gold: dict) -> tuple[float, RewardComponents]:
        """计算一条轨迹的总奖励与各分量。

        trajectory: {'actions':[...], 'answer':..., 'citations':[...]}
        gold: {'answer':..., 'supporting_facts':[...]}
        """
        comps = RewardComponents()
        answer = trajectory.get("answer", "")
        citations = trajectory.get("citations", [])
        actions = trajectory.get("actions", [])

        # 证据：从 AddEvidence 动作提取
        evidence_texts = [a["args"].get("text", "") for a in actions
                          if a.get("type") == "AddEvidence"]
        selected_ids = set(range(len(evidence_texts)))
        gold_ids = set(range(len(gold.get("supporting_facts", []))))

        # Stage 1 分量
        comps.new_supporting_fact = self._new_supporting_fact(evidence_texts, gold)
        comps.citation_validity = validate_citations(answer, evidence_texts)
        comps.grounding = self._grounding(answer, evidence_texts)
        comps.answer_f1 = f1_score(answer, gold.get("answer", ""))
        comps.answer_em = float(answer.strip() == gold.get("answer", "").strip())

        # 惩罚
        comps.dup_query_penalty = self._count(actions, "dup_query") * self.penalty_w["dup_query"]
        comps.redundant_evidence_penalty = self._count(actions, "redundant_evidence") * self.penalty_w["redundant_evidence"]
        comps.illegal_action_penalty = self._count(actions, "illegal") * self.penalty_w["illegal_action"]
        comps.tool_cost_penalty = len(actions) * self.penalty_w["tool_cost"]

        # 总奖励
        if self.stage == 1:
            reward = (
                comps.new_supporting_fact * self.w_stage1["new_supporting_fact"]
                + comps.citation_validity * self.w_stage1["citation_validity"]
                + comps.grounding * self.w_stage1["grounding"]
            )
        else:
            reward = (
                comps.answer_f1 * self.w_stage2["answer_em_f1"]
                + comps.citation_validity * self.w_stage2["evidence_consistency"]
            )
        reward -= (
            comps.dup_query_penalty
            + comps.redundant_evidence_penalty
            + comps.illegal_action_penalty
            + comps.tool_cost_penalty
        )
        return reward, comps

    # ------------------------------------------------------------ 分量实现
    def _new_supporting_fact(self, evidence_texts: list[str], gold: dict) -> float:
        """新增 supporting-fact：选中的证据中命中 gold supporting fact 的比例。"""
        gold_facts = gold.get("supporting_facts", [])
        if not gold_facts:
            return 1.0
        hit = 0
        for gt in gold_facts:
            if any(gt[:30].lower() in e[:60].lower() for e in evidence_texts):
                hit += 1
        return hit / len(gold_facts)

    def _grounding(self, answer: str, evidence_texts: list[str]) -> float:
        """grounding：答案是否被证据支撑（答案关键词出现在证据中）。"""
        if not answer or not evidence_texts:
            return 0.0
        import re
        kw = set(re.findall(r"[A-Za-z]{4,}", answer.lower()))
        if not kw:
            return 0.0
        hit = 0
        for k in kw:
            if any(k in e.lower() for e in evidence_texts):
                hit += 1
        return hit / len(kw)

    @staticmethod
    def _count(actions: list[dict], flag: str) -> int:
        return sum(1 for a in actions if a.get(flag))
