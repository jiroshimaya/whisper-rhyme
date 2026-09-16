"""Deterministic, copyright-free beat and pseudo-voice synthesis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

Condition = Literal["A", "B", "C", "D"]
VoiceMode = Literal["vowel", "cv", "hum"]

VOWEL_FORMANTS: dict[str, tuple[float, float, float]] = {
    "a": (800, 1200, 2500),
    "i": (300, 2300, 3000),
    "u": (350, 1300, 2500),
    "e": (500, 1900, 2600),
    "o": (500, 900, 2400),
}
CV_CHOICES = ("ta", "ka", "ra", "na", "ma", "sa", "da", "ba", "pa")


@dataclass(frozen=True)
class MoraEvent:
    start: float
    duration: float
    symbol: str | None = None


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    duration: float = 30.0
    bpm: float = 90.0
    seed: int = 1234
    voice_mode: VoiceMode = "cv"
    vowels: str = "aiueo"
    voice_gain: float = 0.72
    beat_gain: float = 0.52

    def validate(self) -> None:
        if self.sample_rate < 8_000:
            raise ValueError("sample_rate must be at least 8000")
        if self.duration <= 0 or self.duration > 30:
            raise ValueError("duration must be in (0, 30]")
        if self.bpm <= 0:
            raise ValueError("bpm must be positive")
        if self.voice_mode not in {"vowel", "cv", "hum"}:
            raise ValueError(f"unsupported voice mode: {self.voice_mode}")
        if not self.vowels or any(v not in VOWEL_FORMANTS for v in self.vowels):
            raise ValueError("vowels must contain only a, i, u, e, o")


def _add_at(target: np.ndarray, sound: np.ndarray, start: int) -> None:
    if start >= len(target) or start + len(sound) <= 0:
        return
    src_start = max(0, -start)
    dst_start = max(0, start)
    count = min(len(sound) - src_start, len(target) - dst_start)
    target[dst_start : dst_start + count] += sound[src_start : src_start + count]


def _fade(sound: np.ndarray, sample_rate: int, seconds: float = 0.008) -> np.ndarray:
    n = min(int(seconds * sample_rate), len(sound) // 2)
    if n:
        sound[:n] *= np.linspace(0, 1, n, endpoint=False)
        sound[-n:] *= np.linspace(1, 0, n, endpoint=False)
    return sound


def _kick(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    length = int(0.34 * sample_rate)
    t = np.arange(length) / sample_rate
    frequency = 43 + 95 * np.exp(-t * 22)
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    body = np.sin(phase) * np.exp(-t * 12)
    click = rng.normal(0, 1, length) * np.exp(-t * 90)
    return _fade((0.95 * body + 0.10 * click).astype(np.float32), sample_rate)


def _snare(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    length = int(0.25 * sample_rate)
    t = np.arange(length) / sample_rate
    noise = rng.normal(0, 1, length)
    noise = sosfilt(butter(3, (700, 6500), btype="bandpass", fs=sample_rate, output="sos"), noise)
    tone = np.sin(2 * np.pi * 185 * t)
    sound = 0.65 * noise * np.exp(-t * 18) + 0.3 * tone * np.exp(-t * 24)
    sound /= max(np.max(np.abs(sound)), 1e-6)
    return _fade(sound.astype(np.float32), sample_rate)


def _hat(sample_rate: int, rng: np.random.Generator, open_hat: bool = False) -> np.ndarray:
    seconds = 0.16 if open_hat else 0.055
    length = int(seconds * sample_rate)
    t = np.arange(length) / sample_rate
    noise = rng.normal(0, 1, length)
    noise = sosfilt(butter(3, 5_000, btype="highpass", fs=sample_rate, output="sos"), noise)
    sound = noise * np.exp(-t * (25 if open_hat else 70))
    sound /= max(np.max(np.abs(sound)), 1e-6)
    return _fade(sound.astype(np.float32), sample_rate, 0.002)


def generate_beat(config: AudioConfig, rng: np.random.Generator) -> np.ndarray:
    """Create a simple 4/4 kick/snare/hi-hat beat."""
    samples = round(config.duration * config.sample_rate)
    audio = np.zeros(samples, dtype=np.float32)
    beat_seconds = 60.0 / config.bpm
    total_beats = int(np.ceil(config.duration / beat_seconds))
    for beat in range(total_beats):
        when = beat * beat_seconds
        beat_in_bar = beat % 4
        if beat_in_bar in (0, 2):
            _add_at(audio, _kick(config.sample_rate, rng), round(when * config.sample_rate))
        if beat_in_bar in (1, 3):
            _add_at(audio, _snare(config.sample_rate, rng), round(when * config.sample_rate))
        for subdivision in range(2):
            hat_time = when + subdivision * beat_seconds / 2
            is_open = beat_in_bar == 3 and subdivision == 1
            _add_at(
                audio,
                0.30 * _hat(config.sample_rate, rng, is_open),
                round(hat_time * config.sample_rate),
            )
    return audio


def default_mora_events(config: AudioConfig, rng: np.random.Generator) -> list[MoraEvent]:
    """Generate a speech-like, syncopated pattern aligned loosely to sixteenth notes."""
    step = (60.0 / config.bpm) / 4
    events: list[MoraEvent] = []
    cursor = 0.25
    index = 0
    while cursor < config.duration - 0.1:
        bar_step = index % 16
        keep_probability = 0.86 if bar_step in {0, 2, 4, 6, 8, 11, 12, 14} else 0.55
        if rng.random() < keep_probability:
            jitter = rng.uniform(-0.025, 0.025)
            duration = step * rng.choice([0.72, 0.88, 1.55], p=[0.45, 0.4, 0.15])
            events.append(MoraEvent(max(0.0, cursor + jitter), float(duration)))
        cursor += step
        index += 1
    return events


def load_mora_events(path: str | Path) -> list[MoraEvent]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("rhythm JSON must be a list")
    events = [
        MoraEvent(float(item["start"]), float(item["duration"]), item.get("symbol"))
        for item in data
    ]
    if any(event.start < 0 or event.duration <= 0 for event in events):
        raise ValueError("event starts must be non-negative and durations must be positive")
    return events


def _voiced_vowel(
    vowel: str,
    duration: float,
    sample_rate: int,
    f0: float,
    rng: np.random.Generator,
    formant_shift: float,
    hum: bool = False,
) -> np.ndarray:
    length = max(1, round(duration * sample_rate))
    t = np.arange(length) / sample_rate
    vibrato = 1 + 0.018 * np.sin(2 * np.pi * rng.uniform(4.2, 6.1) * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(f0 * vibrato) / sample_rate
    formants = np.asarray(VOWEL_FORMANTS[vowel]) * formant_shift
    bandwidths = np.asarray((100, 150, 240))
    signal = np.zeros(length, dtype=np.float64)
    max_harmonic = min(40, int((sample_rate / 2 - 100) / f0))
    for harmonic in range(1, max_harmonic + 1):
        hz = harmonic * f0
        formant_energy = np.sum(np.exp(-0.5 * ((hz - formants) / bandwidths) ** 2))
        amplitude = (0.08 + formant_energy) / harmonic
        signal += amplitude * np.sin(harmonic * phase + rng.uniform(-0.04, 0.04))
    if hum:
        signal += 0.35 * np.sin(phase / 2)
        signal = sosfilt(
            butter(2, 1_800, btype="lowpass", fs=sample_rate, output="sos"), signal
        )
    breath = rng.normal(0, 0.018, length)
    envelope = np.sin(np.linspace(0, np.pi, length)) ** 0.45
    signal = (signal + breath) * envelope
    signal /= max(np.max(np.abs(signal)), 1e-6)
    return _fade(signal.astype(np.float32), sample_rate)


def _consonant(kind: str, duration: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    length = max(1, round(duration * sample_rate))
    t = np.arange(length) / sample_rate
    noise = rng.normal(0, 1, length)
    if kind in {"s"}:
        noise = sosfilt(butter(3, 3_000, btype="highpass", fs=sample_rate, output="sos"), noise)
        envelope = np.sin(np.linspace(0, np.pi, length)) ** 0.5
    elif kind in {"m", "n"}:
        noise = np.sin(2 * np.pi * rng.uniform(110, 155) * t) + 0.2 * noise
        noise = sosfilt(butter(2, 1_100, btype="lowpass", fs=sample_rate, output="sos"), noise)
        envelope = np.linspace(0.2, 1, length)
    else:
        cutoff = rng.uniform(900, 1800)
        noise = sosfilt(butter(2, cutoff, btype="highpass", fs=sample_rate, output="sos"), noise)
        envelope = np.exp(-t * rng.uniform(22, 45))
    sound = noise * envelope
    sound /= max(np.max(np.abs(sound)), 1e-6)
    return _fade(sound.astype(np.float32), sample_rate, 0.003)


def generate_pseudo_voice(
    config: AudioConfig,
    rng: np.random.Generator,
    events: Sequence[MoraEvent] | None = None,
) -> tuple[np.ndarray, list[MoraEvent]]:
    """Synthesize deliberately ambiguous vowel, CV-like, or humming vocalizations."""
    if events is None:
        events = default_mora_events(config, rng)
    resolved: list[MoraEvent] = []
    audio = np.zeros(round(config.duration * config.sample_rate), dtype=np.float32)
    vowel_index = 0
    for event in events:
        if event.start >= config.duration:
            continue
        if config.voice_mode == "cv":
            symbol = event.symbol or str(rng.choice(CV_CHOICES))
            vowel = symbol[-1] if symbol[-1] in VOWEL_FORMANTS else str(rng.choice(list(config.vowels)))
            consonant_duration = min(event.duration * 0.28, rng.uniform(0.035, 0.075))
            onset = _consonant(symbol[0], consonant_duration, config.sample_rate, rng)
            vowel_sound = _voiced_vowel(
                vowel,
                max(0.04, event.duration - consonant_duration),
                config.sample_rate,
                rng.uniform(105, 185),
                rng,
                rng.uniform(0.88, 1.12),
            )
            sound = np.concatenate((0.35 * onset, vowel_sound))
        else:
            symbol = event.symbol or config.vowels[vowel_index % len(config.vowels)]
            vowel_index += 1
            vowel = symbol[-1] if symbol[-1] in VOWEL_FORMANTS else str(rng.choice(list(config.vowels)))
            sound = _voiced_vowel(
                vowel,
                event.duration,
                config.sample_rate,
                rng.uniform(95, 175),
                rng,
                rng.uniform(0.90, 1.10),
                hum=config.voice_mode == "hum",
            )
        _add_at(audio, sound, round(event.start * config.sample_rate))
        resolved.append(MoraEvent(event.start, event.duration, symbol))
    return audio, resolved


def _normalize(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    current = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio if current < 1e-8 else (audio * (peak / current)).astype(np.float32)


def generate_condition(
    condition: Condition,
    config: AudioConfig,
    rhythm_events: Sequence[MoraEvent] | None = None,
) -> tuple[np.ndarray, list[MoraEvent]]:
    """Generate one of conditions A-D. D has the same audio as C; its prompt is decoder-side."""
    condition = condition.upper()  # type: ignore[assignment]
    if condition not in {"A", "B", "C", "D"}:
        raise ValueError("condition must be A, B, C, or D")
    config.validate()
    seed_sequence = np.random.SeedSequence(config.seed)
    beat_rng, voice_rng = [np.random.default_rng(seed) for seed in seed_sequence.spawn(2)]
    beat = generate_beat(config, beat_rng) if condition in {"A", "C", "D"} else 0.0
    if condition in {"B", "C", "D"}:
        voice, events = generate_pseudo_voice(config, voice_rng, rhythm_events)
    else:
        voice, events = 0.0, []
    mixed = config.beat_gain * beat + config.voice_gain * voice
    return _normalize(np.asarray(mixed, dtype=np.float32)), events


def save_condition(
    output_path: str | Path,
    condition: Condition,
    config: AudioConfig,
    rhythm_events: Sequence[MoraEvent] | None = None,
) -> dict[str, object]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audio, events = generate_condition(condition, config, rhythm_events)
    sf.write(output, audio, config.sample_rate, subtype="PCM_16")
    metadata: dict[str, object] = {
        "condition": condition.upper(),
        "audio_file": str(output),
        "config": asdict(config),
        "mora_events": [asdict(event) for event in events],
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
