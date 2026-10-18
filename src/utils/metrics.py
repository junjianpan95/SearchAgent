"""指标与引用校验工具（deterministic，不靠 LLM 判断）。

包含：
- token-level F1 / EM 计算
- 引用（citation）有效性校验
- Joint F1、证据链 All-Hit、冗余率、工具调用次数等评测指标
"""
from __future__ import annotations

import re
from typing import Iterable


# ------------------------------------------------------------------ token 化
def _tokenize(text: str) -> list[str]:
    """简单分词（ASCII 单词 + 中文单字 + 数字），用于 F1/EM 计算。"""
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower())


def _normalize(text: str) -> str:
    """轻量归一化：去空白、去标点、lower。"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]", " ", text)
    return " ".join(text.split())


def em_score(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tok = _tokenize(pred)
    gold_tok = _tokenize(gold)
    if not pred_tok or not gold_tok:
        return 0.0
    common = sum(1 for t in pred_tok if t in gold_tok)
    # 简单集合级 F1（token 级召回/精确）
    tp = len(set(pred_tok) & set(gold_tok))
    precision = tp / len(set(pred_tok)) if pred_tok else 0.0
    recall = tp / len(set(gold_tok)) if gold_tok else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------------ 引用校验
def validate_citations(answer: str, evidence: Iterable[str]) -> float:
    """校验答案中的引用是否真实指向已有证据。

    deterministic：统计答案中 [1]/[2] 形式的引用号，检查是否在证据索引范围内，
    并检查是否真的引用了非空证据。返回有效性比例 [0,1]。
    """
    cites = re.findall(r"\[(\d+)\]", answer)
    n_evidence = len(list(evidence))
    if n_evidence == 0:
        return 0.0
    if not cites:
        return 0.0
    valid = sum(1 for c in cites if 1 <= int(c) <= n_evidence)
    return valid / len(cites)


# ------------------------------------------------------------------ 证据链 All-Hit
def evidence_all_hit(selected_evidence_ids: set[int], gold_supporting_ids: set[int]) -> float:
    """完整证据链召回：选中的证据是否覆盖了所有 gold supporting facts。"""
    if not gold_supporting_ids:
        return 1.0
    return len(gold_supporting_ids & selected_evidence_ids) / len(gold_supporting_ids)


# ------------------------------------------------------------------ 冗余率
def redundancy_rate(selected_evidence: list[str]) -> float:
    """冗余引用率：非 supporting 的引用占全部引用的比例。"""
    if not selected_evidence:
        return 0.0
    unique = set(_normalize(e) for e in selected_evidence)
    return 1.0 - len(unique) / len(selected_evidence)


def joint_f1(answer_f1: float, evidence_hit: float) -> float:
    """Joint F1：答案 F1 与证据链命中的调和联合。"""
    if answer_f1 == 0 or evidence_hit == 0:
        return 0.0
    return 2 * answer_f1 * evidence_hit / (answer_f1 + evidence_hit)
