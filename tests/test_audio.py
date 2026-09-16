from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from whisper_rap.audio import (
    AudioConfig,
    degrade_tts_voice,
    generate_condition,
    load_mora_events,
    save_condition,
)


def test_all_conditions_are_deterministic_and_have_expected_content() -> None:
    config = AudioConfig(sample_rate=16_000, duration=1.5, bpm=90, seed=42)
    generated = {}
    for condition in "ABCD":
        first, events = generate_condition(condition, config)
        second, _ = generate_condition(condition, config)
        assert len(first) == 24_000
        assert first.dtype == np.float32
        assert np.max(np.abs(first)) <= 0.921
        assert np.array_equal(first, second)
        generated[condition] = first
        assert bool(events) is (condition != "A")
    assert np.array_equal(generated["C"], generated["D"])
    assert not np.array_equal(generated["A"], generated["B"])


def test_save_condition_writes_audio_and_metadata(tmp_path) -> None:
    output = tmp_path / "audio.wav"
    metadata = save_condition(output, "C", AudioConfig(duration=0.5, seed=7))
    assert output.exists()
    assert sf.info(output).samplerate == 16_000
    saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["condition"] == metadata["condition"] == "C"


def test_load_explicit_mora_events(tmp_path) -> None:
    path = tmp_path / "rhythm.json"
    path.write_text('[{"start": 0.0, "duration": 0.2, "symbol": "ta"}]', encoding="utf-8")
    events = load_mora_events(path)
    assert events[0].symbol == "ta"


def test_degraded_tts_is_deterministic_and_requires_a_source() -> None:
    config = AudioConfig(duration=0.5, seed=9, voice_mode="degraded_tts")
    source = np.sin(2 * np.pi * 220 * np.arange(8_000) / 16_000).astype(np.float32)
    first = degrade_tts_voice(source, 16_000, config, np.random.default_rng(3))
    second = degrade_tts_voice(source, 16_000, config, np.random.default_rng(3))
    assert np.array_equal(first, second)
    assert len(first) == 8_000
    with np.testing.assert_raises_regex(ValueError, "external voice"):
        generate_condition("C", config)
