#!/usr/bin/env python3
"""Compare acoustic steps before adding decoder context, with frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

from whisper_rap.article import FINAL_WORDS, article_audio, clear_word_measurements
from whisper_rap.audio import AudioConfig
from whisper_rap.comparison import diagnostic_summary, output_diagnostics
from whisper_rap.decode import decode_samples, load_model, prepare_mel
from whisper_rap.lexical import lexical_measurements
from whisper_rap.rhyme import (
    paired_condition_comparison,
    rhyme_measurements,
)

CELLS = (
    ("beat", "none"),
    ("cv_balanced", "none"),
    ("cv_final", "none"),
    ("clear_final", "none"),
    ("clear_final", "prompt"),
    ("clear_final", "prompt_prefix"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    by_cell = {}
    for audio, context in CELLS:
        key = f"{audio}|{context}"
        selected = [row for row in rows if row["vowel_condition"] == key]
        if not selected:
            continue
        phrases = sum(row["recognized_phrase_count"] for row in selected)
        by_cell[key] = {
            **diagnostic_summary(selected),
            "recognized_phrase_count": phrases,
            **{field: sum(row[field] for row in selected) / len(selected) for field in (
                "clear_word_sample", "clear_word_reading_sample", "related_term_sample",
            )},
            **{f"target_last_{n}_rate": (
                sum(row[f"target_last_{n}_hits"] for row in selected) / phrases
                if phrases else None
            ) for n in (1, 2)},
        }
    contrasts = (
        ("cv_final|none", "cv_balanced|none"),
        ("clear_final|none", "cv_final|none"),
        ("clear_final|prompt", "clear_final|none"),
        ("clear_final|prompt_prefix", "clear_final|prompt"),
    )
    return {
        "completed": len(rows), "by_cell": by_cell,
        "paired_comparisons": {
            f"{left}_minus_{right}": paired_condition_comparison(
                rows, left=left, right=right,
                metrics=("character_count", "target_last_1_rate", "target_last_2_rate",
                         "clear_word_sample", "related_term_sample"), seed=20261028,
            ) for left, right in contrasts
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/lexical_anchor_context.txt"))
    parser.add_argument("--prefix-file", type=Path, default=Path("prompts/vowel_rhyme_prefix.txt"))
    parser.add_argument("--model", choices=("tiny", "small", "large-v3"), default="large-v3")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3579)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--sample-len", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n_samples < 1 or args.threads < 1 or not 1 <= args.sample_len <= 224:
        parser.error("positive sample/thread counts and sample-len in [1, 224] required")
    if not 0 <= args.temperature <= 1:
        parser.error("temperature must be in [0, 1]")

    import torch
    import whisper

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    prompt = args.prompt_file.read_text().strip()
    prefix = args.prefix_file.read_text().strip()
    if not prompt or not prefix:
        parser.error("prompt and prefix must be nonempty")
    anchors = json.loads(args.anchor_manifest.read_text())
    entries = anchors["sets"]["rhyme_final"]
    if [entry["word"] for entry in entries] != list(FINAL_WORDS):
        parser.error("clear words must match the phrase-final measurement dictionary")
    config = AudioConfig(duration=12, bpm=90, seed=args.seed, voice_gain=0.72, beat_gain=0.16)
    input_dir = args.output_dir / "audio"
    manifest_path = args.output_dir / "manifest.json"
    samples_path = args.output_dir / "samples.jsonl"
    summary_path = args.output_dir / "summary.json"
    if not args.resume and args.output_dir.exists():
        parser.error("output directory already exists; choose another or use --resume")
    if args.resume and not manifest_path.exists():
        parser.error("resume requires a saved manifest")
    input_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        cells, metadata = article_audio(config, [Path(entry["audio_file"]) for entry in entries])
        for name, audio in cells.items():
            sf.write(input_dir / f"{name}.wav", audio, config.sample_rate, subtype="PCM_16")
        (input_dir / "events.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
    audio_inputs = {
        name: {"path": str((input_dir / f"{name}.wav").resolve()),
               "sha256": sha256(input_dir / f"{name}.wav")}
        for name in dict.fromkeys(audio for audio, _ in CELLS)
    }
    manifest = {
        "experiment": "acoustic steps, then context", "cells": CELLS,
        "config": asdict(config), "model": args.model, "n_samples": args.n_samples,
        "seed": args.seed, "temperature": args.temperature, "sample_len": args.sample_len,
        "prompt": prompt, "prefix": prefix, "best_of": 1, "length_penalty": 1.0,
        "device": "cpu", "fp16": False, "threads": args.threads,
        "suppress_blank": False, "suppress_tokens": "model-specific non-speech and controls",
        "whisper_version": whisper.__version__, "torch_version": torch.__version__,
        "python_version": platform.python_version(), "audio": audio_inputs,
        "events_sha256": sha256(input_dir / "events.json"),
        "anchors": {"speaker": anchors.get("speaker"), "words": [
            {"word": entry["word"], "sha256": sha256(Path(entry["audio_file"]))}
            for entry in entries
        ]},
        "source_sha256": {str(path.resolve().relative_to(Path.cwd())): sha256(path) for path in (
            Path(__file__).resolve(), *sorted(Path("src/whisper_rap").glob("*.py")),
        )},
    }
    # Normalize tuples before equality checking the on-disk JSON.
    manifest = json.loads(json.dumps(manifest))
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error("resume requires identical inputs, source, options and environment")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    rows = [json.loads(line) for line in samples_path.read_text().splitlines()] \
        if samples_path.exists() else []
    done = {(row["audio_mode"], row["context_mode"], row["temperature_sample_id"]) for row in rows}
    expected = {(*cell, i) for cell in CELLS for i in range(args.n_samples)}
    if len(done) != len(rows) or not done <= expected:
        parser.error("saved rows have duplicate or unexpected keys")
    if done == expected:
        summary_path.write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2) + "\n")
        return
    model = load_model(args.model, "cpu")
    with samples_path.open("a") as handle:
        for audio, source in audio_inputs.items():
            pending = [(context, i) for name, context in CELLS if name == audio
                       for i in range(args.n_samples) if (audio, context, i) not in done]
            if not pending:
                continue
            mel = prepare_mel(source["path"], model)
            with torch.inference_mode():
                features = model.encoder(mel.unsqueeze(0)).squeeze(0)
            for context, index in pending:
                started = time.perf_counter()
                row = next(decode_samples(
                    model=model, mel=features, audio_file=source["path"],
                    condition="C" if context == "none" else "D",
                    temperatures=[args.temperature], n_samples=1, seed=args.seed + index,
                    prompt=prompt if context != "none" else None,
                    prefix=prefix if context == "prompt_prefix" else None,
                    model_name=args.model, sample_len=args.sample_len, best_of=1,
                    length_penalty=1.0, audio_duration=config.duration,
                ))
                if row.get("mora_error"):
                    raise RuntimeError(row["mora_error"])
                row.update({
                    "condition": "A" if audio == "beat" else row["condition"],
                    "seed": args.seed, "sample_id": index, "temperature_sample_id": index,
                    "audio_mode": audio, "context_mode": context,
                    "vowel_condition": f"{audio}|{context}", "audio_sha256": source["sha256"],
                    "decode_and_metrics_seconds": time.perf_counter() - started,
                })
                row.update(lexical_measurements(row["text"], row["reading"]))
                row.update(clear_word_measurements(row["text"], row["reading"]))
                row.update(rhyme_measurements(row))
                row.update(output_diagnostics(row))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
                print(json.dumps({"completed": len(rows), "expected": len(expected),
                                  "cell": row["vowel_condition"], "sample": index,
                                  "text": row["text"]}, ensure_ascii=False), flush=True)
            summary_path.write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
