#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from whisper_rap.audio import AudioConfig, load_mora_events, save_condition


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate beat/pseudo-voice experiment audio")
    result.add_argument("--condition", choices=list("ABCD"), required=True)
    result.add_argument("--duration", type=float, default=30.0)
    result.add_argument("--bpm", type=float, default=90.0)
    result.add_argument("--sample-rate", type=int, default=16_000)
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument(
        "--voice-mode", choices=["vowel", "cv", "hum", "degraded_tts"], default="cv"
    )
    result.add_argument("--tts-source", type=Path, help="Original local TTS WAV for degraded_tts")
    result.add_argument("--vowels", default="aiueo", help="Vowel cycle using only a/i/u/e/o")
    result.add_argument("--rhythm-file", type=Path, help="Optional mora event JSON")
    result.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    config = AudioConfig(
        sample_rate=args.sample_rate,
        duration=args.duration,
        bpm=args.bpm,
        seed=args.seed,
        voice_mode=args.voice_mode,
        vowels=args.vowels,
    )
    output = args.output or Path(
        f"generated_audio/condition_{args.condition}_seed_{args.seed}_{args.voice_mode}.wav"
    )
    events = load_mora_events(args.rhythm_file) if args.rhythm_file else None
    if args.voice_mode == "degraded_tts" and not args.tts_source:
        raise SystemExit("--voice-mode degraded_tts requires --tts-source")
    metadata = save_condition(output, args.condition, config, events, args.tts_source)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
