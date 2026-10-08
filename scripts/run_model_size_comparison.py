#!/usr/bin/env python3
"""Compare multilingual Whisper sizes on six frozen lexical-anchor input WAVs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import soundfile as sf

from whisper_rap.comparison import output_diagnostics, summarize_models
from whisper_rap.decode import decode_samples, load_model, prepare_mel
from whisper_rap.lexical import lexical_measurements
from whisper_rap.rhyme import rhyme_measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("tiny", "small", "large-v3"),
                        default=["tiny", "small", "large-v3"])
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/lexical_anchor_context.txt"))
    parser.add_argument("--prefix-file", type=Path, default=Path("prompts/vowel_rhyme_prefix.txt"))
    parser.add_argument("--seed", type=int, default=3579)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--sample-len", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n_samples < 1 or args.threads < 1 or not 0 <= args.temperature <= 1:
        parser.error("need positive sample/thread counts and temperature in [0, 1]")
    if not 1 <= args.sample_len <= 224 or len(args.models) != len(set(args.models)):
        parser.error("need unique models and sample-len in [1, 224]")

    import torch
    import whisper

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    prefix = args.prefix_file.read_text(encoding="utf-8").strip()
    cells = {}
    for rhyme in ("off", "on"):
        for anchor in ("none", "neutral", "theme"):
            audio = args.audio_dir / f"{anchor}_rhyme_{rhyme}.wav"
            info = sf.info(audio)
            duration = info.frames / info.samplerate
            if not 0 < duration <= 30:
                parser.error(f"input must be 0 < duration <= 30 seconds: {audio}")
            cells[(anchor, rhyme)] = {
                "path": str(audio.resolve()),
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "duration": duration,
            }
    manifest = {
        "models": args.models, "seed": args.seed, "temperature": args.temperature,
        "n_samples": args.n_samples, "sample_len": args.sample_len, "best_of": 1,
        "length_penalty": 1.0, "prompt": prompt, "prefix": prefix,
        "device": args.device, "fp16": args.device == "cuda", "threads": args.threads,
        "whisper_version": whisper.__version__, "torch_version": torch.__version__,
        "python_version": platform.python_version(), "encoder_cached": True,
        "suppression": "model-specific non-speech, language and no-timestamps tokens",
        "audio": {"|".join(key): value for key, value in cells.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    jsonl_path = args.output_dir / "samples.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            parser.error("resume requires an identical saved manifest (audio, options, environment)")
    else:
        if any(path.exists() for path in (manifest_path, jsonl_path, summary_path)):
            parser.error("output already exists; choose a new directory or use --resume")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()] if jsonl_path.exists() else []
    done = {(r["model"], r["anchor_mode"], r["rhyme_mode"], r["temperature_sample_id"]) for r in rows}
    expected = {(model, *cell, i) for model in args.models for cell in cells for i in range(args.n_samples)}
    if len(done) != len(rows) or not done <= expected:
        parser.error("saved rows have duplicate or unexpected sample keys")

    def save_summary() -> None:
        summary_path.write_text(json.dumps({
            "manifest": manifest, "completed": len(rows), "expected": len(expected),
            "by_model": summarize_models(rows),
        }, ensure_ascii=False, indent=2) + "\n")

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for name in args.models:
            if all(key in done for key in expected if key[0] == name):
                continue
            print(f"Loading {name} on {args.device}", flush=True)
            model = load_model(name, args.device)
            for (anchor, rhyme), audio in cells.items():
                pending = [i for i in range(args.n_samples) if (name, anchor, rhyme, i) not in done]
                if not pending:
                    continue
                mel = prepare_mel(audio["path"], model)
                # Official decode accepts encoder features. Compute once per frozen WAV.
                if args.device == "cuda":
                    mel = mel.half()
                started = time.perf_counter()
                with torch.inference_mode():
                    features = model.encoder(mel.unsqueeze(0)).squeeze(0)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                encoder_seconds = time.perf_counter() - started
                for index in pending:
                    started = time.perf_counter()
                    row = next(decode_samples(
                        model=model, mel=features, audio_file=audio["path"], condition="D",
                        temperatures=[args.temperature], n_samples=1, seed=args.seed + index,
                        prompt=prompt, prefix=prefix, model_name=name, best_of=1,
                        sample_len=args.sample_len, length_penalty=1.0,
                        suppress_blank=False, suppress_tokens="-1", audio_duration=audio["duration"],
                    ))
                    row.update({
                        "seed": args.seed, "sample_id": index, "temperature_sample_id": index,
                        "anchor_mode": anchor, "rhyme_mode": rhyme,
                        "audio_sha256": audio["sha256"], "encoder_seconds": encoder_seconds,
                        "decode_and_metrics_seconds": time.perf_counter() - started,
                    })
                    if row.get("mora_error"):
                        raise RuntimeError(row["mora_error"])
                    row.update(lexical_measurements(row["text"], row["reading"]))
                    anchor_key = "neutral_anchor_copy" if anchor == "neutral" else "anchor_copy"
                    for suffix in ("count", "distinct", "sample"):
                        row[f"present_anchor_copy_{suffix}"] = (
                            row[f"{anchor_key}_{suffix}"] if anchor != "none" else 0
                        )
                    row.update(rhyme_measurements(row))
                    row.update(output_diagnostics(row))
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    rows.append(row)
                    done.add((name, anchor, rhyme, index))
                    print(json.dumps({"completed": len(rows), "expected": len(expected),
                                      "model": name, "cell": [anchor, rhyme], "sample": index,
                                      "text": row["text"]}, ensure_ascii=False), flush=True)
                save_summary()
                del mel, features
            del model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    save_summary()


if __name__ == "__main__":
    main()
