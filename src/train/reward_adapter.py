"""reward_adapter：供 verl 0.10 调用的 custom reward 函数。

verl 0.10 的 custom_reward_function 签名：
    compute_score(data_source, solution_str, ground_truth, extra_info) -> float | dict

- data_source: 数据源标识字符串
- solution_str: 模型 rollout 生成的完整文本（动作序列）
- ground_truth: 标准答案（parquet 里 reward_model.ground_truth）
- extra_info: 额外信息（含 supporting_facts 等）

返回 float 或 {"score": float}。这里返回 dict 以便 verl 记录各 reward 分量。
"""
from __future__ import annotations

import re


def _parse_actions(response: str) -> dict:
    """把模型 rollout 文本解析成 {actions, answer, citations}。

    文本动作格式（与 SFT 逐步决策格式一致）：
        Search: <query>
        Read: <page>
        AddEvidence: <page> | <text>
        Answer: <answer> [citations: i,j]
    """
    actions = []
    answer = ""
    citations = []
    for line in response.split("\n"):
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
                page, text = rest.split("|", 1)
                actions.append({"type": "AddEvidence",
                                "args": {"page": page.strip(), "text": text.strip()}})
            else:
                actions.append({"type": "AddEvidence", "args": {"page": "", "text": rest}})
        elif line.startswith("Answer:"):
            rest = line[len("Answer:"):].strip()
            m = re.search(r"\[citations:\s*([^\]]*)\]", rest)
            if m:
                answer = rest[:m.start()].strip()
                citations = [int(x) for x in m.group(1).split(",") if x.strip().isdigit()]
            else:
                answer = rest
    return {"actions": actions, "answer": answer, "citations": citations}


def _f1(pred: str, gold: str) -> float:
    """token 级 F1（用于答案质量）。"""
    def toks(s: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9]+", s.lower()))
    p, g = toks(pred), toks(gold)
    if not p or not g:
        return 0.0
    inter = p & g
    if not inter:
        return 0.0
    precision = len(inter) / len(p)
    recall = len(inter) / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> dict:
    """两阶段课程奖励（Stage 2：强化 Answer F1 + 证据约束 + 四惩罚）。

    verl 0.10 custom reward 入口。返回 dict，score 字段为总奖励。
    """
    traj = _parse_actions(solution_str or "")
    answer = traj["answer"]
    actions = traj["actions"]
    citations = traj["citations"]

    gold_answer = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
    supporting_facts = (extra_info or {}).get("supporting_facts", [])

    # --- Stage 2 答案质量分量 ---
    answer_f1 = _f1(answer, gold_answer)
    answer_em = float(answer.strip().lower() == gold_answer.strip().lower())

    # --- 证据约束：引用是否落在已选证据内 ---
    evidence_texts = [a["args"].get("text", "") for a in actions if a.get("type") == "AddEvidence"]
    citation_validity = 0.0
    if citations and evidence_texts:
        valid = sum(1 for c in citations if 0 <= c < len(evidence_texts))
        citation_validity = valid / len(citations)
    elif not citations and evidence_texts:
        citation_validity = 0.0
    elif not citations and not evidence_texts:
        citation_validity = 0.0

    # --- 新增 supporting-fact 命中率 ---
    new_supporting_fact = 1.0
    if supporting_facts:
        hit = 0
        for gt in supporting_facts:
            gt_short = gt[:30].lower()
            if any(gt_short in e[:60].lower() for e in evidence_texts):
                hit += 1
        new_supporting_fact = hit / len(supporting_facts)

    # --- 四类惩罚 ---
    queries = [a["args"].get("query", "") for a in actions if a.get("type") == "Search"]
    dup_query = max(0, len(queries) - len(set(queries)))  # 重复查询数
    n_add = len(evidence_texts)
    redundant_evidence = max(0, n_add - len(set(evidence_texts)))  # 冗余证据
    illegal_action = sum(1 for a in actions if a.get("type") not in
                         {"Search", "Read", "AddEvidence", "Answer"})
    tool_cost = len(actions)  # 工具调用成本

    # --- 加权总奖励（Stage 2：answer_em_f1 权重 1.0 + evidence 0.5）---
    reward = (
        max(answer_f1, answer_em) * 1.0
        + citation_validity * 0.5
    )
    reward -= (
        dup_query * 0.1
        + redundant_evidence * 0.1
        + illegal_action * 0.2
        + tool_cost * 0.05
    )

    return {
        "score": reward,
        "answer_f1": answer_f1,
        "answer_em": answer_em,
        "citation_validity": citation_validity,
        "new_supporting_fact": new_supporting_fact,
        "dup_query": dup_query,
        "redundant_evidence": redundant_evidence,
        "illegal_action": illegal_action,
        "tool_cost": tool_cost,
    }
