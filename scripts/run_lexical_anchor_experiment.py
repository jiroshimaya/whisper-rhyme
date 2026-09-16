#!/usr/bin/env python3
"""Run a 3x2 clear-word-anchor and phrase-final-vowel experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import soundfile as sf

from whisper_rap.audio import AudioConfig, generate_condition
from whisper_rap.decode import decode_samples, load_model, prepare_mel
from whisper_rap.lexical import (
    lexical_measurements,
    overlay_word_anchors,
    paired_cell_comparison,
    summarize_lexical_cells,
)
from whisper_rap.rhyme import build_vowel_condition_events, rhyme_measurements

ANCHOR_MODES = ("none", "neutral", "theme")
RHYME_MODES = ("off", "on")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--anchor-manifest", type=Path, required=True)
    result.add_argument("--prompt-file", type=Path, required=True)
    result.add_argument("--prefix-file", type=Path)
    result.add_argument("--audio-dir", type=Path, default=Path("generated_audio/lexical_anchor"))
    result.add_argument("--output", type=Path, default=Path("results/lexical_anchor.jsonl"))
    result.add_argument("--summary", type=Path, default=Path("results/lexical_anchor_summary.json"))
    result.add_argument("--duration", type=float, default=15.0)
    result.add_argument("--bpm", type=float, default=90.0)
    result.add_argument("--seed", type=int, default=3579)
    result.add_argument("--temperature", type=float, default=0.6)
    result.add_argument("--n-samples", type=int, default=20)
    result.add_argument("--sample-len", type=int, default=64)
    result.add_argument("--model", default="large-v3")
    result.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    result.add_argument("--download-root")
    return result


def load_anchor_sets(path: Path) -> dict[str, list[dict[str, str]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sets = manifest.get("sets")
    if not isinstance(sets, dict) or not all(mode in sets for mode in ("neutral", "theme")):
        raise ValueError("anchor manifest must contain neutral and theme sets")
    for mode in ("neutral", "theme"):
        if len(sets[mode]) != 4:
            raise ValueError(f"anchor set {mode} must contain exactly four words")
    return sets


def create_audio_cells(args: argparse.Namespace) -> dict[tuple[str, str], Path]:
    anchor_sets = load_anchor_sets(args.anchor_manifest)
    config = AudioConfig(
        duration=args.duration,
        bpm=args.bpm,
        seed=args.seed,
        voice_mode="cv",
        voice_gain=0.72,
        beat_gain=0.16,
    )
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple[str, str], Path] = {}
    for rhyme_mode in RHYME_MODES:
        vowel_condition = "a_final" if rhyme_mode == "on" else "balanced"
        events = build_vowel_condition_events(condition=vowel_condition, seed=args.seed)
        base, _ = generate_condition("C", config, events)
        for anchor_mode in ANCHOR_MODES:
            audio = base
            if anchor_mode != "none":
                entries = anchor_sets[anchor_mode]
                audio = overlay_word_anchors(
                    base,
                    sample_rate=config.sample_rate,
                    anchor_files=[entry["audio_file"] for entry in entries],
                    starts=[0.18, 3.68, 7.18, 10.68],
                )
            output = args.audio_dir / f"{anchor_mode}_rhyme_{rhyme_mode}.wav"
            sf.write(output, audio, config.sample_rate, subtype="PCM_16")
            cells[(anchor_mode, rhyme_mode)] = output
    return cells


def comparisons(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    lexical_metrics = (
        "anchor_copy_sample",
        "present_anchor_copy_sample",
        "theme_term_sample",
        "related_term_sample",
        "anchor_copy_count",
        "present_anchor_copy_count",
        "theme_term_count",
        "related_distinct_count",
        "related_terms_per_100_chars",
    )
    rhyme_metrics = ("target_last_1_rate", "target_last_2_rate")
    return {
        "theme_minus_neutral_rhyme_off": paired_cell_comparison(
            rows, left=("theme", "off"), right=("neutral", "off"), metrics=lexical_metrics, seed=seed
        ),
        "theme_minus_neutral_rhyme_on": paired_cell_comparison(
            rows, left=("theme", "on"), right=("neutral", "on"), metrics=lexical_metrics, seed=seed
        ),
        "theme_rhyme_on_minus_off": paired_cell_comparison(
            rows, left=("theme", "on"), right=("theme", "off"), metrics=rhyme_metrics, seed=seed
        ),
        "neutral_rhyme_on_minus_off": paired_cell_comparison(
            rows, left=("neutral", "on"), right=("neutral", "off"), metrics=rhyme_metrics, seed=seed
        ),
    }


def main() -> None:
    args = parser().parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    prefix = args.prefix_file.read_text(encoding="utf-8").strip() if args.prefix_file else None
    audio_cells = create_audio_cells(args)
    print(f"Loading {args.model} on {args.device}...", flush=True)
    model = load_model(args.model, args.device, args.download_root)
    rows: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for (anchor_mode, rhyme_mode), audio_file in audio_cells.items():
            mel = prepare_mel(audio_file, model)
            info = sf.info(audio_file)
            decoded = decode_samples(
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
                length_penalty=1.0,
                suppress_blank=False,
                suppress_tokens="-1",
                audio_duration=info.frames / info.samplerate,
            )
            for row in decoded:
                row["anchor_mode"] = anchor_mode
                row["rhyme_mode"] = rhyme_mode
                row.update(lexical_measurements(row["text"], row.get("reading")))
                if anchor_mode == "theme":
                    row["present_anchor_copy_count"] = row["anchor_copy_count"]
                    row["present_anchor_copy_distinct"] = row["anchor_copy_distinct"]
                    row["present_anchor_copy_sample"] = row["anchor_copy_sample"]
                elif anchor_mode == "neutral":
                    row["present_anchor_copy_count"] = row["neutral_anchor_copy_count"]
                    row["present_anchor_copy_distinct"] = row["neutral_anchor_copy_distinct"]
                    row["present_anchor_copy_sample"] = row["neutral_anchor_copy_sample"]
                else:
                    row["present_anchor_copy_count"] = 0
                    row["present_anchor_copy_distinct"] = 0
                    row["present_anchor_copy_sample"] = False
                row.update(rhyme_measurements(row))
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "cell": [anchor_mode, rhyme_mode],
                            "sample": row["temperature_sample_id"],
                            "anchor_copy": row["anchor_copy_count"],
                            "present_anchor_copy": row["present_anchor_copy_count"],
                            "related": row["related_term_count"],
                            "last_1": row["target_last_1_rate"],
                            "text": row["text"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    result = {
        "experiment": {
            "seed": args.seed,
            "temperature": args.temperature,
            "n_samples_per_cell": args.n_samples,
            "prompt": prompt,
            "prefix": prefix,
            "anchor_words": {
                mode: [entry["word"] for entry in entries]
                for mode, entries in load_anchor_sets(args.anchor_manifest).items()
            },
        },
        "by_cell": summarize_lexical_cells(rows),
        "paired_comparisons": comparisons(rows, args.seed),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
