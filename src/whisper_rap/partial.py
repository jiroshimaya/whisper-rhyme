"""Construct partial clear-word inputs from frozen, matched PCM recordings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def replace_phrase_regions(
    cv: np.ndarray, all_clear: np.ndarray, *, sample_rate: int, bpm: float,
    phrases: Sequence[int] = (2, 4), phrase_count: int = 4,
) -> np.ndarray:
    """Select one-based clear-word phrases, preserving all other PCM samples exactly."""
    if cv.ndim != 1 or cv.shape != all_clear.shape or cv.dtype != all_clear.dtype:
        raise ValueError("need matching mono PCM arrays with the same dtype")
    if sample_rate < 1 or bpm <= 0 or phrase_count < 1:
        raise ValueError("positive sample rate, bpm and phrase count required")
    if len(set(phrases)) != len(phrases) or any(p not in range(1, phrase_count + 1) for p in phrases):
        raise ValueError("phrase numbers must be unique and within the recording")
    if len(cv) < round(phrase_count * 4 * 60 / bpm * sample_rate):
        raise ValueError("recording is shorter than the phrase grid")
    result = cv.copy()
    for phrase in phrases:
        start = round((phrase - 1) * 4 * 60 / bpm * sample_rate)
        end = round(phrase * 4 * 60 / bpm * sample_rate)
        result[start:end] = all_clear[start:end]
    return result


def mora_count_summary(rows: Iterable[dict[str, Any]], target: int = 48) -> dict[str, Any]:
    """Compare output pronunciation length to the input CV count, without timing claims."""
    values = [row["mora_count"] for row in rows if row.get("mora_count") is not None]
    if not values:
        return {"sample_count": 0, "target_moras": target}
    return {
        "sample_count": len(values), "target_moras": target,
        "mean_moras": float(np.mean(values)), "median_moras": float(np.median(values)),
        "minimum_moras": min(values), "maximum_moras": max(values),
        "exact_count": values.count(target),
        "mean_absolute_count_error": float(np.mean([abs(value - target) for value in values])),
    }
