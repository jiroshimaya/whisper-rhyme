"""Clear-word overlays and lexical metrics for hybrid nonsense-speech experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ANCHOR_VARIANTS: dict[str, tuple[str, ...]] = {
    "夜": ("夜", "よる", "ヨル"),
    "街": ("街", "まち", "マチ"),
    "影": ("影", "かげ", "カゲ"),
    "鼓動": ("鼓動", "こどう", "コドウ"),
}

THEME_TERM_VARIANTS: dict[str, tuple[str, ...]] = {
    "夜": ("夜",),
    "街": ("街",),
    "影": ("影",),
    "鼓動": ("鼓動",),
}

ANCHOR_READINGS = {
    "夜": "ヨル",
    "街": "マチ",
    "影": "カゲ",
    "鼓動": "コドウ",
}

NEUTRAL_ANCHOR_READINGS = {
    "これ": "コレ",
    "ただ": "タダ",
    "いま": "イマ",
    "ことば": "コトバ",
}

RELATED_VARIANTS: dict[str, tuple[str, ...]] = {
    "月": ("月",),
    "星": ("星",),
    "闇": ("闇",),
    "暗": ("暗",),
    "灯": ("灯", "明かり"),
    "ネオン": ("ネオン",),
    "都会": ("都会", "都市"),
    "路地": ("路地",),
    "交差点": ("交差点",),
    "ビル": ("ビル",),
    "駅": ("駅",),
    "心臓": ("心臓",),
    "脈": ("脈",),
    "リズム": ("リズム",),
    "胸": ("胸",),
    "光": ("光",),
    "眠": ("眠",),
}


def _mono_resampled(path: str | Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if source_rate != sample_rate:
        mono = resample_poly(mono, sample_rate, source_rate).astype(np.float32)
    return mono


def trim_silence(audio: np.ndarray, sample_rate: int, padding: float = 0.025) -> np.ndarray:
    """Trim leading/trailing TTS silence while preserving a small natural margin."""
    mono = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak < 1e-7:
        return mono
    active = np.flatnonzero(np.abs(mono) >= peak * 0.025)
    if not len(active):
        return mono
    pad = round(padding * sample_rate)
    start = max(0, int(active[0]) - pad)
    end = min(len(mono), int(active[-1]) + pad + 1)
    return mono[start:end]


def overlay_word_anchors(
    base_audio: np.ndarray,
    *,
    sample_rate: int,
    anchor_files: Sequence[str | Path],
    starts: Sequence[float],
    anchor_peak: float = 0.78,
    duck_gain: float = 0.18,
) -> np.ndarray:
    """Replace short phrase-initial regions with clearly synthesized word anchors."""
    if len(anchor_files) != len(starts):
        raise ValueError("anchor_files and starts must have the same length")
    output = np.asarray(base_audio, dtype=np.float32).copy()
    for path, start_seconds in zip(anchor_files, starts, strict=True):
        anchor = trim_silence(_mono_resampled(path, sample_rate), sample_rate)
        if not len(anchor):
            continue
        anchor = anchor * (anchor_peak / max(float(np.max(np.abs(anchor))), 1e-7))
        start = round(start_seconds * sample_rate)
        if start >= len(output):
            continue
        count = min(len(anchor), len(output) - start)
        fade = min(round(0.015 * sample_rate), count // 2)
        duck = np.full(count, duck_gain, dtype=np.float32)
        if fade:
            duck[:fade] = np.linspace(1.0, duck_gain, fade, endpoint=False)
            duck[-fade:] = np.linspace(duck_gain, 1.0, fade, endpoint=False)
            anchor[:fade] *= np.linspace(0.0, 1.0, fade, endpoint=False)
            anchor[count - fade : count] *= np.linspace(1.0, 0.0, fade, endpoint=False)
        output[start : start + count] *= duck
        output[start : start + count] += anchor[:count]
    peak = float(np.max(np.abs(output))) if len(output) else 0.0
    return output if peak < 1e-7 else (output * (0.92 / peak)).astype(np.float32)


def _variant_hits(text: str, variants: dict[str, tuple[str, ...]]) -> dict[str, int]:
    return {
        concept: max((text.count(surface) for surface in surfaces), default=0)
        for concept, surfaces in variants.items()
    }


def _reading_hits(reading: str, targets: dict[str, str]) -> dict[str, int]:
    return {concept: reading.count(target) for concept, target in targets.items()}


def lexical_measurements(text: str, reading: str | None = None) -> dict[str, Any]:
    """Separate direct anchor copying from related vocabulary expansion."""
    anchor_hits = (
        _reading_hits(reading, ANCHOR_READINGS)
        if reading is not None
        else _variant_hits(text, ANCHOR_VARIANTS)
    )
    neutral_anchor_hits = _reading_hits(reading or "", NEUTRAL_ANCHOR_READINGS)
    theme_term_hits = _variant_hits(text, THEME_TERM_VARIANTS)
    related_hits = _variant_hits(text, RELATED_VARIANTS)
    anchor_count = sum(anchor_hits.values())
    related_count = sum(related_hits.values())
    visible_count = len("".join(text.split()))
    return {
        "anchor_hits": anchor_hits,
        "anchor_copy_count": anchor_count,
        "anchor_copy_distinct": sum(count > 0 for count in anchor_hits.values()),
        "anchor_copy_sample": anchor_count > 0,
        "neutral_anchor_hits": neutral_anchor_hits,
        "neutral_anchor_copy_count": sum(neutral_anchor_hits.values()),
        "neutral_anchor_copy_distinct": sum(count > 0 for count in neutral_anchor_hits.values()),
        "neutral_anchor_copy_sample": any(count > 0 for count in neutral_anchor_hits.values()),
        "theme_term_hits": theme_term_hits,
        "theme_term_count": sum(theme_term_hits.values()),
        "theme_term_distinct": sum(count > 0 for count in theme_term_hits.values()),
        "theme_term_sample": any(count > 0 for count in theme_term_hits.values()),
        "related_hits": related_hits,
        "related_term_count": related_count,
        "related_distinct_count": sum(count > 0 for count in related_hits.values()),
        "related_term_sample": related_count > 0,
        "related_terms_per_100_chars": (
            100.0 * related_count / visible_count if visible_count else None
        ),
    }


def summarize_lexical_cells(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['anchor_mode']}|{row['rhyme_mode']}"
        grouped.setdefault(key, []).append(row)

    result: dict[str, Any] = {}
    for key, items in grouped.items():
        phrase_count = sum(int(row["recognized_phrase_count"]) for row in items)
        per_100 = [
            float(row["related_terms_per_100_chars"])
            for row in items
            if row.get("related_terms_per_100_chars") is not None
        ]
        result[key] = {
            "sample_count": len(items),
            "nonempty_count": sum(bool(str(row.get("text", "")).strip()) for row in items),
            "anchor_copy_sample_rate": sum(bool(row["anchor_copy_sample"]) for row in items)
            / len(items),
            "present_anchor_copy_sample_rate": sum(
                bool(row["present_anchor_copy_sample"]) for row in items
            )
            / len(items),
            "theme_term_sample_rate": sum(bool(row["theme_term_sample"]) for row in items)
            / len(items),
            "related_term_sample_rate": sum(bool(row["related_term_sample"]) for row in items)
            / len(items),
            "mean_anchor_copy_count": float(np.mean([row["anchor_copy_count"] for row in items])),
            "mean_present_anchor_copy_count": float(
                np.mean([row["present_anchor_copy_count"] for row in items])
            ),
            "mean_theme_term_count": float(np.mean([row["theme_term_count"] for row in items])),
            "mean_related_distinct_count": float(
                np.mean([row["related_distinct_count"] for row in items])
            ),
            "mean_related_terms_per_100_chars": float(np.mean(per_100)) if per_100 else None,
            "mean_character_count": float(np.mean([row["character_count"] for row in items])),
            "mean_japanese_character_ratio": float(
                np.mean([row["japanese_character_ratio"] for row in items])
            ),
            "recognized_phrase_count": phrase_count,
            "target_last_1_rate": (
                sum(int(row["target_last_1_hits"]) for row in items) / phrase_count
                if phrase_count
                else None
            ),
            "target_last_2_rate": (
                sum(int(row["target_last_2_hits"]) for row in items) / phrase_count
                if phrase_count
                else None
            ),
            "texts": [str(row.get("text", "")) for row in items],
        }
    return result


def paired_cell_comparison(
    rows: Iterable[dict[str, Any]],
    *,
    left: tuple[str, str],
    right: tuple[str, str],
    metrics: tuple[str, ...],
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap comparison for two anchor/rhyme cells."""
    indexed = {
        (str(row["anchor_mode"]), str(row["rhyme_mode"]), int(row["temperature_sample_id"])): row
        for row in rows
    }
    sample_ids = sorted(
        sample_id
        for anchor, rhyme, sample_id in indexed
        if (anchor, rhyme) == left and (*right, sample_id) in indexed
    )
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "left": "|".join(left),
        "right": "|".join(right),
        "matched_sample_count": len(sample_ids),
    }
    for metric in metrics:
        differences = np.asarray(
            [
                float(indexed[(*left, sample_id)][metric])
                - float(indexed[(*right, sample_id)][metric])
                for sample_id in sample_ids
                if indexed[(*left, sample_id)].get(metric) is not None
                and indexed[(*right, sample_id)].get(metric) is not None
            ],
            dtype=float,
        )
        if not len(differences):
            result[metric] = {"pair_count": 0, "mean_difference": None, "bootstrap_95_ci": None}
            continue
        draws = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
        low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
        result[metric] = {
            "pair_count": len(differences),
            "mean_difference": float(differences.mean()),
            "bootstrap_95_ci": [float(low), float(high)],
        }
    return result
