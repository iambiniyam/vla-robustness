"""Evaluation metrics for binary success and selective prediction."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        error += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(error)


def risk_coverage(scores: np.ndarray, correct: np.ndarray) -> list[dict[str, float]]:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-scores)
    rows = []
    for k in range(1, len(order) + 1):
        selected = order[:k]
        rows.append({"coverage": k / len(order), "risk": float(1.0 - correct[selected].mean())})
    return rows


def summarize(successes: int, total: int, confidences: np.ndarray | None = None, correct: np.ndarray | None = None) -> dict[str, object]:
    low, high = wilson_interval(successes, total)
    result: dict[str, object] = {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "wilson_low": low,
        "wilson_high": high,
    }
    if confidences is not None and correct is not None and len(confidences):
        result["ece"] = expected_calibration_error(confidences, correct)
    return result
