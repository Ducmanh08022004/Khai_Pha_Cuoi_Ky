from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np


@dataclass
class AlertResult:
    score: float
    severity: str
    reasons: List[str]


def severity_label(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def combine_risk_scores(
    fraud_probability: float,
    cluster_risk: float = 0.0,
    rule_hits: int = 0,
    probability_weight: float = 0.6,
    cluster_weight: float = 0.2,
    rule_weight: float = 0.2,
) -> float:
    rule_component = min(rule_hits, 5) / 5.0
    score = (
        probability_weight * float(fraud_probability)
        + cluster_weight * float(cluster_risk)
        + rule_weight * rule_component
    )
    return float(np.clip(score, 0.0, 1.0))


def build_alert(
    fraud_probability: float,
    cluster_risk: float = 0.0,
    rule_hits: int = 0,
    reasons: Iterable[str] | None = None,
) -> AlertResult:
    score = combine_risk_scores(fraud_probability, cluster_risk=cluster_risk, rule_hits=rule_hits)
    severity = severity_label(score)
    reason_list = list(reasons or [])
    if rule_hits:
        reason_list.append(f"Matched {rule_hits} suspicious association rules")
    return AlertResult(score=score, severity=severity, reasons=reason_list)
