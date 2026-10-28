from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from whisper_rap.article import article_audio, article_events, clear_word_measurements
from whisper_rap.audio import AudioConfig


def test_rhyme_change_preserves_consonants_and_timing() -> None:
    config = AudioConfig(duration=12, seed=3579)
    balanced = article_events(config, False)
    final = article_events(config, True)
    assert len(balanced) == len(final) == 48
    for index, (left, right) in enumerate(zip(balanced, final, strict=True)):
        assert (left.start, left.duration, left.symbol[0]) == (
            right.start, right.duration, right.symbol[0]
        )
        if index % 12 >= 10:
            assert right.symbol.endswith("a")
        else:
            assert left == right


def test_clear_words_replace_only_phrase_endings_and_preserve_shared_beat(tmp_path) -> None:
    config = AudioConfig(duration=12, seed=3579, voice_gain=0.72, beat_gain=0.16)
    word = tmp_path / "word.wav"
    t = np.arange(6400) / 16000
    sf.write(word, np.sin(2 * np.pi * 220 * t) * 0.5, 16000)
    cells, metadata = article_audio(config, [word] * 4)
    assert set(cells) == {"beat", "cv_balanced", "cv_final", "clear_final"}
    modified = np.zeros(len(cells["beat"]), dtype=bool)
    for event, placement in zip(
        article_events(config, True)[10::12],
        metadata["conditions"]["final"]["clear_words"], strict=True,
    ):
        assert placement["start"] == event.start
        start = round(event.start * 16000)
        end = round((event.start + max(placement["duration"], 0.125 + 1 / 6 + 0.22)) * 16000)
        modified[start:end] = True
    assert np.array_equal(cells["cv_final"][~modified], cells["clear_final"][~modified])
    assert not np.array_equal(cells["cv_final"], cells["clear_final"])
    assert max(np.max(np.abs(audio)) for audio in cells.values()) <= 0.92001
    # Both CV conditions also preserve every sample outside their rhyme slots.
    assert np.array_equal(cells["cv_balanced"][~modified], cells["cv_final"][~modified])


def test_overlong_clear_word_rejected(tmp_path) -> None:
    path = tmp_path / "too-long.wav"
    sf.write(path, np.ones(32000) * 0.5, 16000)
    with pytest.raises(ValueError, match="does not fit"):
        article_audio(AudioConfig(duration=12), [path] * 4)


def test_clear_word_surface_and_reading_are_distinct() -> None:
    assert clear_word_measurements("鼻", "ハナ")["clear_word_reading_sample"]
    assert not clear_word_measurements("鼻", "ハナ")["clear_word_sample"]
