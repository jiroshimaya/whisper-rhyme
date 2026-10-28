#!/usr/bin/env python3
"""Compare phrase-initial and phrase-final clear words against frozen article results."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from whisper_rap.article import article_audio
from whisper_rap.audio import AudioConfig
from whisper_rap.comparison import diagnostic_summary, output_diagnostics
from whisper_rap.decode import decode_samples, load_model, prepare_mel
from whisper_rap.onset import onset_words
from whisper_rap.partial import mora_count_summary, replace_phrase_regions
from whisper_rap.rhyme import paired_condition_comparison, rhyme_measurements


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def pcm(audio: np.ndarray, rate: int) -> np.ndarray:
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return sf.read(buffer, dtype="int16")[0]


def measurements(row: dict, words: tuple[str, ...], readings: tuple[str, ...]) -> dict:
    return {
        "present_clear_word_sample": any(word in row["text"] for word in words),
        "present_clear_word_reading_sample": any(word in row["reading"] for word in readings),
        "both_clear_words_sample": all(word in row["text"] for word in words),
        "both_clear_words_reading_sample": all(word in row["reading"] for word in readings),
    }


def summary(rows: list[dict]) -> dict:
    groups = {}
    for key in dict.fromkeys(row["vowel_condition"] for row in rows):
        selected = [row for row in rows if row["vowel_condition"] == key]
        phrases = sum(row["recognized_phrase_count"] for row in selected)
        groups[key] = {
            **diagnostic_summary(selected), "mora_counts": mora_count_summary(selected),
            **{field: sum(row[field] for row in selected) / len(selected) for field in (
                "present_clear_word_sample", "present_clear_word_reading_sample",
                "both_clear_words_sample", "both_clear_words_reading_sample",
            )},
            "recognized_phrase_count": phrases,
            "target_last_2_hits": sum(row["target_last_2_hits"] for row in selected),
            "target_last_2_rate": (
                sum(row["target_last_2_hits"] for row in selected) / phrases if phrases else None
            ),
        }
    return {
        "by_cell": groups,
        "paired_comparisons": {
            name: paired_condition_comparison(
                rows, left=f"{name}_onset|none", right=f"{name}_final|none",
                metrics=("present_clear_word_sample", "present_clear_word_reading_sample",
                         "mora_count", "target_last_2_rate"), seed=20261028,
            ) for name in ("partial", "clear")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--partial-dir", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch
    import whisper

    baseline = json.loads((args.baseline_dir / "manifest.json").read_text())
    partial = json.loads((args.partial_dir / "manifest.json").read_text())
    matched = ("config", "model", "seed", "temperature", "n_samples", "sample_len", "best_of",
               "length_penalty", "device", "fp16", "threads", "suppress_blank", "suppress_tokens",
               "whisper_version", "torch_version", "python_version")
    if any(baseline[key] != partial[key] for key in matched):
        parser.error("frozen final conditions do not have identical decoding settings")
    if (baseline["device"] != "cpu" or baseline["fp16"]
            or baseline["whisper_version"] != whisper.__version__
            or baseline["torch_version"] != torch.__version__
            or baseline["python_version"] != platform.python_version()):
        parser.error("use the same CPU/float32 environment as the frozen final runs")
    for name, digest in baseline["source_sha256"].items():
        if name.startswith("src/") and sha256(Path(name)) != digest:
            parser.error(f"frozen inference/audio source changed: {name}")
    for recording in (baseline, partial):
        for name, item in recording["audio"].items():
            if sha256(Path(item["path"])) != item["sha256"]:
                parser.error(f"frozen audio changed: {name}")
    config = AudioConfig(**baseline["config"])
    anchors = json.loads(args.anchor_manifest.read_text())
    entries = anchors["sets"]["rhyme_final"]
    if [entry["word"] for entry in entries] != ["朝", "花", "山", "坂"]:
        parser.error("expected original four clear words")
    anchor_files = [Path(entry["audio_file"]) for entry in entries]
    if [sha256(path) for path in anchor_files] != [
        item["sha256"] for item in baseline["anchors"]["words"]
    ]:
        parser.error("clear word recordings changed")
    cells, metadata = article_audio(config, anchor_files)
    for name in ("beat", "cv_final", "clear_final"):
        frozen, rate = sf.read(baseline["audio"][name]["path"], dtype="int16")
        if rate != config.sample_rate or not np.array_equal(pcm(cells[name], rate), frozen):
            parser.error(f"reconstructed PCM does not exactly match frozen {name}")
    cv = pcm(cells["cv_final"], config.sample_rate)
    clear = pcm(cells["clear_final"], config.sample_rate)
    frozen_partial, rate = sf.read(partial["audio"]["partial_final"]["path"], dtype="int16")
    if not np.array_equal(
        replace_phrase_regions(cv, clear, sample_rate=rate, bpm=config.bpm), frozen_partial,
    ):
        parser.error("partial final PCM is not the matched baseline")
    controls = []
    for directory, audio in ((args.baseline_dir, "clear_final"), (args.partial_dir, "partial_final")):
        selected = [row for row in read_rows(directory / "samples.jsonl")
                    if row["audio_mode"] == audio and row["context_mode"] == "none"]
        if sorted(row["trial_seed"] for row in selected) != list(
            range(baseline["seed"], baseline["seed"] + baseline["n_samples"])
        ):
            parser.error("frozen control seeds are missing or duplicated")
        for row in selected:
            if row["prompt"] is not None or row["prefix"] is not None:
                parser.error("final controls must not have a context or prefix")
            words = ("花", "坂") if audio == "partial_final" else ("朝", "花", "山", "坂")
            readings = ("ハナ", "サカ") if audio == "partial_final" else ("アサ", "ハナ", "ヤマ", "サカ")
            controls.append({**row, **measurements(row, words, readings)})
    manifest_path = args.output_dir / "manifest.json"
    samples_path = args.output_dir / "samples.jsonl"
    if not args.resume and args.output_dir.exists():
        parser.error("output exists; choose another or resume")
    if args.resume and not manifest_path.exists():
        parser.error("resume requires an existing manifest")
    audio_inputs = {}
    placements = {}
    for name, phrases in (("partial_onset", (2, 4)), ("clear_onset", (1, 2, 3, 4))):
        audio, placements[name] = onset_words(
            cells["cv_final"], cells["beat"], config, anchor_files,
            shared_scale=metadata["shared_scale"], phrases=phrases,
        )
        encoded = pcm(audio, config.sample_rate)
        modified = np.zeros(len(cv), dtype=bool)
        for placement in placements[name]:
            start = round(placement["start"] * config.sample_rate)
            end = start + round(placement["duration"] * config.sample_rate)
            modified[start:end] = True
        if not np.array_equal(encoded[~modified], cv[~modified]):
            parser.error("onset changes samples outside the replacement regions")
        path = args.output_dir / "audio" / f"{name}.wav"
        if not args.resume:
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, encoded, config.sample_rate, subtype="PCM_16")
        elif not np.array_equal(sf.read(path, dtype="int16")[0], encoded):
            parser.error("saved onset audio changed")
        audio_inputs[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    manifest = {
        "experiment": "same clear words at phrase starts; no decoder context or prefix",
        **{key: baseline[key] for key in matched}, "config": asdict(config),
        "prompt": None, "prefix": None, "placements": placements, "audio": audio_inputs,
        "shared_scale": metadata["shared_scale"],
        "frozen_files": {str(path.resolve()): sha256(path) for path in (
            args.baseline_dir / "manifest.json", args.baseline_dir / "samples.jsonl",
            args.partial_dir / "manifest.json", args.partial_dir / "samples.jsonl",
            args.anchor_manifest, *anchor_files,
        )},
        "source_sha256": {str(path): sha256(path) for path in (
            Path("scripts/run_onset_experiment.py"), *sorted(Path("src/whisper_rap").glob("*.py")),
        )},
    }
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error("resume requires identical source, inputs and options")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    rows = read_rows(samples_path) if samples_path.exists() else []
    done = {(row["audio_mode"], row["temperature_sample_id"]) for row in rows}
    expected = {(name, index) for name in audio_inputs for index in range(baseline["n_samples"])}
    if len(done) != len(rows) or not done <= expected:
        parser.error("unexpected or duplicate saved samples")
    torch.set_num_threads(baseline["threads"])
    torch.set_num_interop_threads(1)
    if done != expected:
        model = load_model(baseline["model"], "cpu")
        with samples_path.open("a") as handle:
            for name, item in audio_inputs.items():
                pending = [i for i in range(baseline["n_samples"]) if (name, i) not in done]
                if not pending:
                    continue
                mel = prepare_mel(item["path"], model)
                with torch.inference_mode():
                    features = model.encoder(mel.unsqueeze(0)).squeeze(0)
                for index in pending:
                    started = time.perf_counter()
                    row = next(decode_samples(
                        model=model, mel=features, audio_file=item["path"], condition="C",
                        temperatures=[baseline["temperature"]], n_samples=1,
                        seed=baseline["seed"] + index, prompt=None, prefix=None,
                        model_name=baseline["model"], sample_len=baseline["sample_len"],
                        best_of=baseline["best_of"], length_penalty=baseline["length_penalty"],
                        audio_duration=config.duration,
                    ))
                    if row.get("mora_error"):
                        raise RuntimeError(row["mora_error"])
                    words = ("花", "坂") if name == "partial_onset" else ("朝", "花", "山", "坂")
                    readings = (("ハナ", "サカ") if name == "partial_onset"
                                else ("アサ", "ハナ", "ヤマ", "サカ"))
                    row.update({
                        "seed": baseline["seed"], "sample_id": index, "temperature_sample_id": index,
                        "audio_mode": name, "context_mode": "none", "vowel_condition": f"{name}|none",
                        "audio_sha256": item["sha256"], "clear_words": words,
                        "decode_and_metrics_seconds": time.perf_counter() - started,
                    })
                    row.update(measurements(row, words, readings))
                    row.update(rhyme_measurements(row))
                    row.update(output_diagnostics(row))
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    rows.append(row)
                    print(json.dumps({"completed": len(rows), "expected": len(expected),
                                      "cell": name, "seed": row["trial_seed"], "text": row["text"]},
                                     ensure_ascii=False), flush=True)
                (args.output_dir / "summary.json").write_text(
                    json.dumps(summary(controls + rows), ensure_ascii=False, indent=2) + "\n",
                )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary(controls + rows), ensure_ascii=False, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
