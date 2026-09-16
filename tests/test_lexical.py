from __future__ import annotations

import numpy as np
import soundfile as sf

from whisper_rap.lexical import (
    lexical_measurements,
    overlay_word_anchors,
    paired_cell_comparison,
    summarize_lexical_cells,
    trim_silence,
)


def test_trim_and_overlay_word_anchor(tmp_path) -> None:
    sample_rate = 1_000
    anchor = np.concatenate((np.zeros(100), np.ones(200) * 0.5, np.zeros(100))).astype(np.float32)
    path = tmp_path / "anchor.wav"
    sf.write(path, anchor, sample_rate)
    trimmed = trim_silence(anchor, sample_rate, padding=0.0)
    assert len(trimmed) == 200

    base = np.ones(1_000, dtype=np.float32) * 0.1
    mixed = overlay_word_anchors(
        base, sample_rate=sample_rate, anchor_files=[path], starts=[0.4], anchor_peak=0.5
    )
    assert len(mixed) == len(base)
    assert np.max(np.abs(mixed)) <= 0.921
    assert not np.array_equal(mixed, base)


def test_lexical_measurements_separate_copy_and_expansion() -> None:
    result = lexical_measurements("待ちの陰で月とネオンを見た", "マチノカゲデツキトネオンヲミタ")
    assert result["anchor_copy_count"] == 2
    assert result["anchor_copy_distinct"] == 2
    assert result["related_term_count"] == 2
    assert result["related_distinct_count"] == 2
    assert result["anchor_copy_sample"] is True
    assert result["theme_term_count"] == 0
    assert result["related_term_sample"] is True
    assert result["neutral_anchor_copy_count"] == 0

    compound = lexical_measurements("今夜の待ちで夢中", "コンヤノマチデムチュウ")
    assert compound["anchor_copy_count"] == 1
    assert compound["theme_term_count"] == 1
    assert compound["related_term_count"] == 0


def test_summary_and_paired_comparison() -> None:
    rows = []
    for anchor_mode, related in (("theme", 1), ("neutral", 0)):
        row = {
            "anchor_mode": anchor_mode,
            "rhyme_mode": "off",
            "temperature_sample_id": 0,
            "text": "夜",
            "character_count": 1,
            "japanese_character_ratio": 1.0,
            "recognized_phrase_count": 1,
            "target_last_1_hits": 0,
            "target_last_2_hits": 0,
            "anchor_copy_sample": anchor_mode == "theme",
            "anchor_copy_count": int(anchor_mode == "theme"),
            "present_anchor_copy_sample": anchor_mode == "theme",
            "present_anchor_copy_count": int(anchor_mode == "theme"),
            "theme_term_sample": anchor_mode == "theme",
            "theme_term_count": int(anchor_mode == "theme"),
            "related_term_sample": bool(related),
            "related_distinct_count": related,
            "related_terms_per_100_chars": float(related),
        }
        rows.append(row)
    summary = summarize_lexical_cells(rows)
    assert summary["theme|off"]["anchor_copy_sample_rate"] == 1.0
    comparison = paired_cell_comparison(
        rows,
        left=("theme", "off"),
        right=("neutral", "off"),
        metrics=("related_distinct_count",),
        seed=2,
    )
    assert comparison["related_distinct_count"]["mean_difference"] == 1.0
