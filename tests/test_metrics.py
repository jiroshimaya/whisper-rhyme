from __future__ import annotations

from whisper_rap.metrics import (
    aggregate_records,
    analyze_text,
    japanese_character_ratio,
    moras_to_vowels,
)


def test_japanese_character_ratio() -> None:
    assert japanese_character_ratio("夜abc") == 0.25
    assert japanese_character_ratio("") == 0.0


def test_analyze_text_uses_jamorasep() -> None:
    result = analyze_text("シャンプーハット。", audio_duration=2.0)
    assert result["moras"] == ["シャ", "ン", "プ", "ー", "ハ", "ッ", "ト"]
    assert result["mora_count"] == 7
    assert result["mora_per_second"] == 3.5
    assert result["vowels"] == ["a", "u", "u", "a", "o"]


def test_phrase_vowels_split_whitespace_and_punctuation() -> None:
    result = analyze_text("未来　期待、花")
    assert [item["last_2"] for item in result["phrase_end_vowels"]] == [
        ["a", "i"],
        ["a", "i"],
        ["a", "a"],
    ]
    assert moras_to_vowels(["コ", "ー", "ド", "ン"]) == ["o", "o", "o"]


def test_aggregate_records() -> None:
    result = aggregate_records(
        [
            {"text": "夜", "character_count": 1, "japanese_character_ratio": 1.0},
            {"text": "夜", "character_count": 1, "japanese_character_ratio": 1.0},
            {"text": "朝", "character_count": 1, "japanese_character_ratio": 1.0},
        ]
    )
    assert result["unique_output_ratio"] == 2 / 3
    assert result["usable_candidate_ratio"] == 0.0
    assert result["top_outputs"][0] == {"text": "夜", "count": 2, "ratio": 2 / 3}
