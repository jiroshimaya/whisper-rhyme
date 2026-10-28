from __future__ import annotations

import numpy as np
import soundfile as sf

from whisper_rap.article import article_audio, article_events
from whisper_rap.audio import AudioConfig
from whisper_rap.onset import onset_words


def test_onset_preserves_rhyme_slots_and_cv_three_even_with_word_tail_overlap(tmp_path):
    config = AudioConfig(duration=12, seed=3579, voice_gain=0.72, beat_gain=0.16)
    word = tmp_path / "word.wav"
    # Slightly longer than two 1/4-beat slots, so its tail overlaps CV 3.
    sf.write(word, np.full(5500, 0.5), config.sample_rate)
    cells, meta = article_audio(config, [word] * 4)
    onset, placements = onset_words(
        cells["cv_final"], cells["beat"], config, [word] * 4,
        shared_scale=meta["shared_scale"],
    )
    changed = np.zeros(len(onset), dtype=bool)
    events = article_events(config, True)
    for placement in placements:
        first = events[(placement["phrase"] - 1) * 12]
        third = events[(placement["phrase"] - 1) * 12 + 2]
        start = round(first.start * config.sample_rate)
        end = start + 5500
        changed[start:end] = True
        assert placement["overlap_with_third_cv_seconds"] > 0
        third_start = round(third.start * config.sample_rate)
        # CV 3 is retained underneath the extra word tail.
        np.testing.assert_allclose(
            onset[third_start:end] - cells["cv_final"][third_start:end],
            0.72 * meta["shared_scale"], atol=1e-6,
        )
    assert [p["phrase"] for p in placements] == [2, 4]
    assert np.array_equal(onset[~changed], cells["cv_final"][~changed])
