"""Audio conditions and measurements for acoustic vowel-rhyme experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Literal

import numpy as np

from .audio import CV_CHOICES, AudioConfig, MoraEvent

VowelCondition = Literal["balanced", "a_heavy", "a_shuffled", "a_final"]
VOWEL_CONDITIONS: tuple[VowelCondition, ...] = (
    "balanced",
    "a_heavy",
    "a_shuffled",
    "a_final",
)


def build_vowel_condition_events(
    *,
    condition: VowelCondition,
    seed: int,
    phrase_count: int = 8,
    moras_per_phrase: int = 8,
    phrase_seconds: float = 1.75,
) -> list[MoraEvent]:
    """Build matched CV events where only vowel identity differs between conditions."""
    if condition not in VOWEL_CONDITIONS:
        raise ValueError(f"unsupported vowel condition: {condition}")
    if phrase_count < 1 or moras_per_phrase < 2:
        raise ValueError("need at least one phrase and two moras per phrase")

    count = phrase_count * moras_per_phrase
    rng = np.random.default_rng(seed)
    consonants = [str(rng.choice(CV_CHOICES))[0] for _ in range(count)]
    balanced = list(("aiueo" * ((count + 4) // 5))[:count])
    rng.shuffle(balanced)

    final = balanced.copy()
    for phrase in range(phrase_count):
        end = (phrase + 1) * moras_per_phrase
        final[end - 2 : end] = ["a", "a"]

    if condition == "balanced":
        vowels = balanced
    elif condition == "a_final":
        vowels = final
    elif condition == "a_shuffled":
        vowels = final.copy()
        rng.shuffle(vowels)
    else:
        vowels = balanced.copy()
        non_a = [index for index, vowel in enumerate(vowels) if vowel != "a"]
        rng.shuffle(non_a)
        target_a_count = round(0.70 * count)
        for index in non_a[: max(0, target_a_count - vowels.count("a"))]:
            vowels[index] = "a"

    events: list[MoraEvent] = []
    spacing = 0.155
    for index, (consonant, vowel) in enumerate(zip(consonants, vowels, strict=True)):
        phrase = index // moras_per_phrase
        within = index % moras_per_phrase
        start = 0.22 + phrase * phrase_seconds + within * spacing
        duration = 0.22 if within == moras_per_phrase - 1 else 0.125
        events.append(MoraEvent(start=start, duration=duration, symbol=consonant + vowel))
    return events


def vowel_condition_config(base: AudioConfig) -> AudioConfig:
    """Force the comparable CV synthesis mode used by the vowel experiment."""
    return replace(base, voice_mode="cv")


def rhyme_measurements(row: dict[str, Any], target: str = "aa") -> dict[str, Any]:
    """Measure target vowels at recognized phrase endings and across all output."""
    endings = [
        "".join(item.get(f"last_{len(target)}", []))
        for item in row.get("phrase_end_vowels", [])
        if item.get("last_1")
    ]
    final_vowels = [ending[-1] for ending in endings if ending]
    vowels = [str(vowel) for vowel in row.get("vowels", [])]
    pair_count = len(final_vowels) * (len(final_vowels) - 1) // 2
    agreeing_pairs = sum(count * (count - 1) // 2 for count in Counter(final_vowels).values())
    return {
        "rhyme_target": target,
        "recognized_phrase_count": len(endings),
        "target_last_1_hits": sum(ending.endswith(target[-1]) for ending in endings),
        "target_last_2_hits": sum(ending.endswith(target) for ending in endings),
        "target_last_1_rate": (
            sum(ending.endswith(target[-1]) for ending in endings) / len(endings) if endings else None
        ),
        "target_last_2_rate": (
            sum(ending.endswith(target) for ending in endings) / len(endings) if endings else None
        ),
        "target_vowel_share": vowels.count(target[-1]) / len(vowels) if vowels else None,
        "phrase_end_pair_agreement": agreeing_pairs / pair_count if pair_count else None,
    }


def summarize_vowel_conditions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate phrase-weighted rhyme measurements by acoustic condition."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["vowel_condition"]), []).append(row)

    summary: dict[str, Any] = {}
    for condition, condition_rows in grouped.items():
        phrase_count = sum(int(row["recognized_phrase_count"]) for row in condition_rows)
        vowel_values = [row["target_vowel_share"] for row in condition_rows if row["target_vowel_share"] is not None]
        rhyme_values = [
            row["phrase_end_pair_agreement"]
            for row in condition_rows
            if row["phrase_end_pair_agreement"] is not None
        ]
        summary[condition] = {
            "sample_count": len(condition_rows),
            "nonempty_count": sum(bool(str(row.get("text", "")).strip()) for row in condition_rows),
            "recognized_phrase_count": phrase_count,
            "target_last_1_hits": sum(int(row["target_last_1_hits"]) for row in condition_rows),
            "target_last_1_rate": (
                sum(int(row["target_last_1_hits"]) for row in condition_rows) / phrase_count
                if phrase_count
                else None
            ),
            "target_last_2_hits": sum(int(row["target_last_2_hits"]) for row in condition_rows),
            "target_last_2_rate": (
                sum(int(row["target_last_2_hits"]) for row in condition_rows) / phrase_count
                if phrase_count
                else None
            ),
            "mean_target_vowel_share": (
                float(np.mean(vowel_values)) if vowel_values else None
            ),
            "mean_phrase_end_pair_agreement": (
                float(np.mean(rhyme_values)) if rhyme_values else None
            ),
            "mean_character_count": float(
                np.mean([int(row.get("character_count", 0)) for row in condition_rows])
            ),
            "texts": [str(row.get("text", "")) for row in condition_rows],
        }
    return summary


def paired_condition_comparison(
    rows: Iterable[dict[str, Any]],
    *,
    left: str,
    right: str,
    metrics: tuple[str, ...] = (
        "target_last_1_rate",
        "target_last_2_rate",
        "target_vowel_share",
        "phrase_end_pair_agreement",
    ),
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare conditions using matched decoder sample IDs and a bootstrap interval."""
    indexed = {
        (str(row["vowel_condition"]), int(row["temperature_sample_id"])): row for row in rows
    }
    sample_ids = sorted(
        sample_id
        for condition, sample_id in indexed
        if condition == left and (right, sample_id) in indexed
    )
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {"left": left, "right": right, "matched_sample_count": len(sample_ids)}
    for metric in metrics:
        differences = np.asarray(
            [
                float(indexed[(left, sample_id)][metric])
                - float(indexed[(right, sample_id)][metric])
                for sample_id in sample_ids
                if indexed[(left, sample_id)].get(metric) is not None
                and indexed[(right, sample_id)].get(metric) is not None
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
