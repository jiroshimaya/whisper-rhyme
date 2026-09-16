#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from whisper_rap.audio import AudioConfig, save_condition
from whisper_rap.decode import decode_samples, load_model, prepare_mel, write_jsonl

SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run direct Whisper large-v3 decoding samples")
    result.add_argument("--condition", choices=list("ABCD"), required=True)
    result.add_argument("--model", default="large-v3")
    result.add_argument("--temperature", type=float, action="append")
    result.add_argument("--temperature-sweep", action="store_true")
    result.add_argument("--n-samples", type=int, default=50)
    result.add_argument("--best-of", type=int, default=1)
    result.add_argument("--sample-len", type=int)
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument("--prompt-file", type=Path)
    result.add_argument("--audio-file", type=Path)
    result.add_argument("--duration", type=float, default=30.0)
    result.add_argument("--bpm", type=float, default=90.0)
    result.add_argument("--voice-mode", choices=["vowel", "cv", "hum"], default="cv")
    result.add_argument("--device", choices=["cuda", "cpu"])
    result.add_argument("--download-root")
    result.add_argument("--suppress-blank", action="store_true")
    result.add_argument(
        "--suppress-tokens",
        help="Whisper token IDs, comma-separated. Default is no token suppression.",
    )
    result.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.condition == "D" and not args.prompt_file:
        raise SystemExit("condition D requires --prompt-file")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else None
    temperatures = SWEEP if args.temperature_sweep else (args.temperature or [0.8])
    audio_file = args.audio_file or Path(
        f"generated_audio/condition_{args.condition}_seed_{args.seed}_{args.voice_mode}.wav"
    )
    if not audio_file.exists():
        config = AudioConfig(
            duration=args.duration,
            bpm=args.bpm,
            seed=args.seed,
            voice_mode=args.voice_mode,
        )
        save_condition(audio_file, args.condition, config)
    audio_info = sf.info(audio_file)
    audio_duration = audio_info.frames / audio_info.samplerate

    print(f"Loading {args.model} on {args.device or 'automatic device'}...", flush=True)
    model = load_model(args.model, args.device, args.download_root)
    mel = prepare_mel(audio_file, model)
    output = args.output or Path(f"results/condition_{args.condition}_seed_{args.seed}.jsonl")
    rows = decode_samples(
        model=model,
        mel=mel,
        audio_file=audio_file,
        condition=args.condition,
        temperatures=temperatures,
        n_samples=args.n_samples,
        seed=args.seed,
        prompt=prompt,
        model_name=args.model,
        best_of=args.best_of,
        sample_len=args.sample_len,
        suppress_blank=args.suppress_blank,
        suppress_tokens=args.suppress_tokens,
        audio_duration=audio_duration,
    )
    count = write_jsonl(output, rows)
    print(json.dumps({"output": str(output), "samples": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
