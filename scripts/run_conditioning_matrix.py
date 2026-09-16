#!/usr/bin/env python3
"""Run the paired 2x2x2 rap-conditioning experiment with one model load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from whisper_rap.decode import decode_samples, load_model, prepare_mel

BOILERPLATE = ("チャンネル登録", "ご視聴ありがとうございました", "字幕", "高評価")
SOURCE_ECHO = ("足らない", "絡まい", "皿ない", "狭間", "重なり", "飛び出し", "離さない", "放さない")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cv-audio", type=Path, required=True)
    result.add_argument("--tts-audio", type=Path, required=True)
    result.add_argument("--prompt-file", type=Path, required=True)
    result.add_argument("--prefix-file", type=Path, required=True)
    result.add_argument("--temperature", type=float, nargs="+", default=[0.6, 0.8])
    result.add_argument("--n-samples", type=int, default=20)
    result.add_argument("--sample-len", type=int, default=64)
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument("--model", default="large-v3")
    result.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    result.add_argument("--download-root")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--resume", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    prefix = args.prefix_file.read_text(encoding="utf-8").strip()
    audio_files = {"cv": args.cv_audio, "degraded_tts": args.tts_audio}
    completed: set[tuple[str, str, float, int]] = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed.add(
                (
                    row["acoustic_mode"],
                    row["conditioning_mode"],
                    float(row["temperature"]),
                    int(row["temperature_sample_id"]),
                )
            )

    print(f"Loading {args.model} on {args.device}...", flush=True)
    model = load_model(args.model, args.device, args.download_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for acoustic_mode, audio_file in audio_files.items():
            mel = prepare_mel(audio_file, model)
            info = sf.info(audio_file)
            duration = info.frames / info.samplerate
            for conditioning_mode, active_prefix in (
                ("prompt", None),
                ("prompt_prefix", prefix),
            ):
                for temperature in args.temperature:
                    cell_completed = {
                        sample
                        for acoustic, conditioning, saved_temperature, sample in completed
                        if acoustic == acoustic_mode
                        and conditioning == conditioning_mode
                        and saved_temperature == float(temperature)
                    }
                    if len(cell_completed) >= args.n_samples:
                        continue
                    rows = decode_samples(
                        model=model,
                        mel=mel,
                        audio_file=audio_file,
                        condition="D",
                        temperatures=[temperature],
                        n_samples=args.n_samples,
                        seed=args.seed,
                        prompt=prompt,
                        prefix=active_prefix,
                        model_name=args.model,
                        best_of=1,
                        sample_len=args.sample_len,
                        suppress_blank=False,
                        suppress_tokens="-1",
                        audio_duration=duration,
                    )
                    for row in rows:
                        key = (
                            acoustic_mode,
                            conditioning_mode,
                            float(temperature),
                            int(row["temperature_sample_id"]),
                        )
                        if key in completed:
                            continue
                        text = row["text"]
                        row["acoustic_mode"] = acoustic_mode
                        row["conditioning_mode"] = conditioning_mode
                        row["boilerplate"] = next(
                            (phrase for phrase in BOILERPLATE if phrase in text), None
                        )
                        row["source_echo_count"] = sum(phrase in text for phrase in SOURCE_ECHO)
                        row["usable_candidate"] = bool(
                            row["character_count"] >= 8
                            and row["japanese_character_ratio"] >= 0.75
                            and (row["mora_count"] or 0) >= 6
                            and row["boilerplate"] is None
                            and row["source_echo_count"] < 2
                            and "<|" not in text
                        )
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        completed.add(key)
                        written += 1
                        print(
                            json.dumps(
                                {
                                    "completed": len(completed),
                                    "cell": [acoustic_mode, conditioning_mode, temperature],
                                    "sample": row["temperature_sample_id"],
                                    "usable": row["usable_candidate"],
                                    "text": text,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
    print(json.dumps({"output": str(args.output), "new_rows": written, "total": len(completed)}))


if __name__ == "__main__":
    main()
