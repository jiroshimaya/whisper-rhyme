#!/usr/bin/env python3
"""Test whether /a/ placement in nonsense CV audio biases Whisper output rhyme."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from whisper_rap.audio import AudioConfig, save_condition
from whisper_rap.decode import decode_samples, load_model, prepare_mel
from whisper_rap.rhyme import (
    VOWEL_CONDITIONS,
    build_vowel_condition_events,
    paired_condition_comparison,
    rhyme_measurements,
    summarize_vowel_conditions,
    vowel_condition_config,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prompt-file", type=Path, required=True)
    result.add_argument("--prefix-file", type=Path)
    result.add_argument("--audio-dir", type=Path, default=Path("generated_audio/vowel_rhyme"))
    result.add_argument("--output", type=Path, default=Path("results/vowel_rhyme.jsonl"))
    result.add_argument("--summary", type=Path, default=Path("results/vowel_rhyme_summary.json"))
    result.add_argument("--conditions", nargs="+", choices=VOWEL_CONDITIONS, default=VOWEL_CONDITIONS)
    result.add_argument("--duration", type=float, default=15.0)
    result.add_argument("--bpm", type=float, default=90.0)
    result.add_argument("--seed", type=int, default=2468)
    result.add_argument("--temperature", type=float, default=0.6)
    result.add_argument("--n-samples", type=int, default=20)
    result.add_argument("--sample-len", type=int, default=64)
    result.add_argument("--model", default="large-v3")
    result.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    result.add_argument("--download-root")
    return result


def main() -> None:
    args = parser().parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    prefix = args.prefix_file.read_text(encoding="utf-8").strip() if args.prefix_file else None
    config = vowel_condition_config(
        AudioConfig(duration=args.duration, bpm=args.bpm, seed=args.seed, voice_mode="cv")
    )
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    audio_files: dict[str, Path] = {}
    for condition in args.conditions:
        events = build_vowel_condition_events(condition=condition, seed=args.seed)
        output = args.audio_dir / f"{condition}.wav"
        save_condition(output, "C", config, events)
        audio_files[condition] = output

    print(f"Loading {args.model} on {args.device}...", flush=True)
    model = load_model(args.model, args.device, args.download_root)
    rows: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for condition, audio_file in audio_files.items():
            mel = prepare_mel(audio_file, model)
            info = sf.info(audio_file)
            decoded_rows = decode_samples(
                model=model,
                mel=mel,
                audio_file=audio_file,
                condition="D",
                temperatures=[args.temperature],
                n_samples=args.n_samples,
                seed=args.seed,
                prompt=prompt,
                prefix=prefix,
                model_name=args.model,
                best_of=1,
                sample_len=args.sample_len,
                suppress_blank=False,
                suppress_tokens="-1",
                audio_duration=info.frames / info.samplerate,
            )
            for row in decoded_rows:
                row["vowel_condition"] = condition
                row.update(rhyme_measurements(row))
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "condition": condition,
                            "sample": row["temperature_sample_id"],
                            "last_1_rate": row["target_last_1_rate"],
                            "last_2_rate": row["target_last_2_rate"],
                            "text": row["text"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    result = {
        "experiment": {
            "target": "aa",
            "seed": args.seed,
            "temperature": args.temperature,
            "n_samples_per_condition": args.n_samples,
            "prompt": prompt,
            "prefix": prefix,
        },
        "by_vowel_condition": summarize_vowel_conditions(rows),
        "paired_comparisons": {
            "a_final_minus_a_shuffled": paired_condition_comparison(
                rows, left="a_final", right="a_shuffled", seed=args.seed
            ),
            "a_heavy_minus_balanced": paired_condition_comparison(
                rows, left="a_heavy", right="balanced", seed=args.seed
            ),
        },
    }
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
