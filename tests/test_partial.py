from __future__ import annotations

import numpy as np
import pytest

from whisper_rap.partial import mora_count_summary, replace_phrase_regions


def test_partial_selects_only_phrases_two_and_four_without_rescaling() -> None:
    cv = np.arange(20, dtype=np.int16)
    clear = -cv
    partial = replace_phrase_regions(cv, clear, sample_rate=1, bpm=60)
    assert np.array_equal(partial[:4], cv[:4])
    assert np.array_equal(partial[4:8], clear[4:8])
    assert np.array_equal(partial[8:12], cv[8:12])
    assert np.array_equal(partial[12:16], clear[12:16])
    assert np.array_equal(partial[16:], cv[16:])
    assert np.array_equal(cv, np.arange(20, dtype=np.int16))


@pytest.mark.parametrize("phrases", [(0,), (5,), (2, 2)])
def test_invalid_phrase_selections_rejected(phrases) -> None:
    audio = np.zeros(16, dtype=np.int16)
    with pytest.raises(ValueError, match="phrase numbers"):
        replace_phrase_regions(audio, audio, sample_rate=1, bpm=60, phrases=phrases)


def test_mora_summary_includes_empty_outputs_and_excludes_missing_measurements() -> None:
    result = mora_count_summary([{"mora_count": n} for n in (48, 0, 50, None)])
    assert result["sample_count"] == 3
    assert result["exact_count"] == 1
    assert result["median_moras"] == 48
    assert result["mean_absolute_count_error"] == 50 / 3
