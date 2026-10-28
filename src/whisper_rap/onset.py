"""Move frozen clear words to the first two CV slots without changing the beat."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .article import FINAL_WORDS, MORAS_PER_PHRASE, PHRASES, article_events
from .audio import AudioConfig
from .lexical import _mono_resampled, trim_silence


def onset_words(
    cv: np.ndarray, beat: np.ndarray, config: AudioConfig, anchor_files: Sequence[Path],
    *, shared_scale: float, phrases: Sequence[int] = (2, 4),
) -> tuple[np.ndarray, list[dict]]:
    """Replace two CVs per selected phrase; keep any following CV underneath the word tail."""
    if cv.ndim != 1 or cv.shape != beat.shape or len(anchor_files) != PHRASES:
        raise ValueError("need matching mono audio and four anchor files")
    if len(set(phrases)) != len(phrases) or any(p not in range(1, PHRASES + 1) for p in phrases):
        raise ValueError("phrase numbers must be unique and within the recording")
    if shared_scale <= 0:
        raise ValueError("shared scale must be positive")
    events = article_events(config, True)
    result = cv.copy()
    placements = []
    for phrase in phrases:
        offset = (phrase - 1) * MORAS_PER_PHRASE
        first, second, third = events[offset:offset + 3]
        anchor = trim_silence(
            _mono_resampled(anchor_files[phrase - 1], config.sample_rate), config.sample_rate,
        )
        if not len(anchor) or np.max(np.abs(anchor)) < 1e-7:
            raise ValueError("empty clear word")
        start = round(first.start * config.sample_rate)
        erase_end = round((second.start + second.duration) * config.sample_rate)
        word_end = start + len(anchor)
        # A short tail may overlap CV 3, which is retained, never silently deleted.
        if word_end > round((third.start + third.duration) * config.sample_rate):
            raise ValueError("word extends beyond the third CV slot")
        result[start:erase_end] = beat[start:erase_end]
        result[start:word_end] += anchor * (0.72 / np.max(np.abs(anchor))) * shared_scale
        placements.append({
            "phrase": phrase, "word": FINAL_WORDS[phrase - 1], "start": first.start,
            "duration": len(anchor) / config.sample_rate,
            "removed_cv_indices": [offset, offset + 1],
            "overlap_with_third_cv_seconds": max(
                0, (word_end - round(third.start * config.sample_rate)) / config.sample_rate,
            ),
        })
    if np.max(np.abs(result)) >= 1:
        raise ValueError("new placement clips; do not renormalize matched audio")
    return result, placements
