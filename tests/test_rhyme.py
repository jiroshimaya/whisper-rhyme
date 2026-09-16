from __future__ import annotations

from collections import Counter

from whisper_rap.rhyme import (
    VOWEL_CONDITIONS,
    build_vowel_condition_events,
    paired_condition_comparison,
    rhyme_measurements,
    summarize_vowel_conditions,
)


def _vowels(events):
    return [event.symbol[-1] for event in events]


def test_vowel_conditions_match_timing_and_isolate_position() -> None:
    conditions = {
        name: build_vowel_condition_events(condition=name, seed=12) for name in VOWEL_CONDITIONS
    }
    baseline = conditions["balanced"]
    for events in conditions.values():
        assert [(event.start, event.duration) for event in events] == [
            (event.start, event.duration) for event in baseline
        ]
        assert [event.symbol[0] for event in events] == [event.symbol[0] for event in baseline]

    final_vowels = _vowels(conditions["a_final"])
    assert all(final_vowels[index - 2 : index] == ["a", "a"] for index in range(8, 65, 8))
    assert Counter(final_vowels) == Counter(_vowels(conditions["a_shuffled"]))
    assert _vowels(conditions["a_heavy"]).count("a") == round(0.70 * len(baseline))


def test_rhyme_measurements_and_summary() -> None:
    row = {
        "vowel_condition": "a_final",
        "text": "花　まだ　夜",
        "character_count": 5,
        "vowels": list("aaaao"),
        "phrase_end_vowels": [
            {"last_1": ["a"], "last_2": ["a", "a"]},
            {"last_1": ["a"], "last_2": ["a", "a"]},
            {"last_1": ["o"], "last_2": ["o"]},
        ],
    }
    row.update(rhyme_measurements(row))
    assert row["target_last_1_hits"] == 2
    assert row["target_last_2_hits"] == 2
    assert row["phrase_end_pair_agreement"] == 1 / 3
    summary = summarize_vowel_conditions([row])["a_final"]
    assert summary["target_last_1_rate"] == 2 / 3


def test_paired_condition_comparison_matches_sample_ids() -> None:
    rows = []
    for condition, values in {"a_final": [0.5, 1.0], "a_shuffled": [0.0, 0.5]}.items():
        for sample_id, value in enumerate(values):
            rows.append(
                {
                    "vowel_condition": condition,
                    "temperature_sample_id": sample_id,
                    "target_last_1_rate": value,
                    "target_last_2_rate": value,
                    "target_vowel_share": value,
                    "phrase_end_pair_agreement": value,
                }
            )
    result = paired_condition_comparison(rows, left="a_final", right="a_shuffled", seed=3)
    assert result["matched_sample_count"] == 2
    assert result["target_last_1_rate"]["mean_difference"] == 0.5
    assert result["target_last_1_rate"]["bootstrap_95_ci"] == [0.5, 0.5]
