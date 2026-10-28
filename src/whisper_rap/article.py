"""Matched phrase-final acoustic inputs for the article experiment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .audio import AudioConfig, MoraEvent, generate_beat, generate_pseudo_voice
from .lexical import _mono_resampled, trim_silence

FINAL_WORDS = ("朝", "花", "山", "坂")
FINAL_READINGS = ("アサ", "ハナ", "ヤマ", "サカ")
PHRASES = 4
MORAS_PER_PHRASE = 12


def article_events(config: AudioConfig, final: bool) -> list[MoraEvent]:
    rng = np.random.default_rng(config.seed)
    count = PHRASES * MORAS_PER_PHRASE
    consonants = rng.choice(list("tkrnmsdbp"), count)
    vowels = list(("aiueo" * ((count + 4) // 5))[:count])
    rng.shuffle(vowels)
    events = []
    beat = 60 / config.bpm
    for i, (consonant, vowel) in enumerate(zip(consonants, vowels, strict=True)):
        within = i % MORAS_PER_PHRASE
        if final and within >= MORAS_PER_PHRASE - 2:
            vowel = "a"
        events.append(MoraEvent(
            start=(i // MORAS_PER_PHRASE) * 4 * beat + 0.12 + within * beat / 4,
            duration=0.22 if within == MORAS_PER_PHRASE - 1 else 0.125,
            symbol=str(consonant) + vowel,
        ))
    return events


def article_audio(config: AudioConfig, anchor_files: Sequence[Path]) -> tuple[dict, dict]:
    if len(anchor_files) != PHRASES:
        raise ValueError("need one clear word per phrase")
    if config.voice_mode != "cv" or config.duration < PHRASES * 4 * 60 / config.bpm:
        raise ValueError("need CV mode and enough duration for four bars")
    beat_seed, voice_seed = np.random.SeedSequence(config.seed).spawn(2)
    beat = generate_beat(config, np.random.default_rng(beat_seed)) * config.beat_gain
    cells = {"beat": beat.copy()}
    metadata = {}
    for name, final in (("balanced", False), ("final", True)):
        events = article_events(config, final)
        voice, _ = generate_pseudo_voice(config, np.random.default_rng(voice_seed), events)
        voice *= config.voice_gain
        anchored = voice.copy()
        placements = []
        for phrase, path in enumerate(anchor_files):
            first = events[phrase * MORAS_PER_PHRASE + MORAS_PER_PHRASE - 2]
            last = events[(phrase + 1) * MORAS_PER_PHRASE - 1]
            anchor = trim_silence(_mono_resampled(path, config.sample_rate), config.sample_rate)
            if not len(anchor) or np.max(np.abs(anchor)) < 1e-7:
                raise ValueError(f"empty clear word: {path}")
            start = round(first.start * config.sample_rate)
            boundary = (phrase + 1) * 4 * 60 / config.bpm - 0.04
            if first.start + len(anchor) / config.sample_rate > boundary:
                raise ValueError(f"clear word does not fit phrase: {path}")
            end = max(round((last.start + last.duration) * config.sample_rate), start + len(anchor))
            anchored[start:end] = 0
            anchored[start:start + len(anchor)] = anchor * (0.72 / np.max(np.abs(anchor)))
            placements.append({"word": FINAL_WORDS[phrase], "start": first.start,
                               "duration": len(anchor) / config.sample_rate})
        cells[f"cv_{name}"] = beat + voice
        if final:
            cells["clear_final"] = beat + anchored
        metadata[name] = {"events": [vars(event) for event in events], "clear_words": placements}
    # One shared scale keeps the beat and all unmodified samples identical across cells.
    peak = max(float(np.max(np.abs(audio))) for audio in cells.values())
    scale = 0.92 / max(peak, 1e-7)
    return {name: (audio * scale).astype(np.float32) for name, audio in cells.items()}, {
        "conditions": metadata, "shared_scale": scale,
    }


def clear_word_measurements(text: str, reading: str) -> dict:
    return {
        "clear_word_sample": any(word in text for word in FINAL_WORDS),
        "clear_word_reading_sample": any(word in reading for word in FINAL_READINGS),
        "clear_word_hits": {word: text.count(word) for word in FINAL_WORDS},
    }
