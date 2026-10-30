#!/usr/bin/env python3
"""Export the six article conditions and a common-seed example from saved results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The article uses cv_final, not the earlier balanced-vowel control.
CONDITIONS = (
    ("BGMのみ", "baseline", "beat", "none"),
    ("BGM＋曖昧CV（語尾 /a a/）", "baseline", "cv_final", "none"),
    ("2・4フレーズ末尾に明瞭語", "partial", "partial_final", "none"),
    ("全4フレーズ末尾に明瞭語", "baseline", "clear_final", "none"),
    ("全4フレーズ冒頭に明瞭語", "onset", "clear_onset", "none"),
    ("2・4フレーズ末尾に明瞭語＋押韻文脈", "context", "partial_final", "prompt"),
)
MATCHED = (
    "model", "seed", "n_samples", "temperature", "sample_len", "best_of", "length_penalty",
    "device", "fp16", "threads", "python_version", "whisper_version", "torch_version", "config",
)


def summarize(directory: Path, example_seed: int) -> dict:
    manifests, samples = {}, {}
    for stage in dict.fromkeys(item[1] for item in CONDITIONS):
        manifests[stage] = json.loads((directory / stage / "manifest.json").read_text())
        samples[stage] = [json.loads(line) for line in (
            directory / stage / "samples.jsonl"
        ).read_text().splitlines()]
    baseline = manifests["baseline"]
    if baseline["n_samples"] < 1:
        raise ValueError("baseline: n_samples must be positive")
    for stage, manifest in manifests.items():
        if any(manifest[key] != baseline[key] for key in MATCHED):
            raise ValueError(f"{stage}: decoding settings or environment differ from baseline")
    cells = []
    for label, stage, audio, context in CONDITIONS:
        rows = [row for row in samples[stage]
                if row["audio_mode"] == audio and row["context_mode"] == context]
        if sorted(row["trial_seed"] for row in rows) != list(
            range(baseline["seed"], baseline["seed"] + baseline["n_samples"])
        ):
            raise ValueError(f"{stage}/{audio}/{context}: missing or duplicate sample seeds")
        expected_prompt = (
            (Path(__file__).resolve().parents[1] / "prompts/rap_aa_four_phrases.txt")
            .read_text().strip() if context == "prompt" else None
        )
        if any(row["prompt"] != expected_prompt or row["prefix"] is not None for row in rows):
            raise ValueError(f"{stage}/{audio}/{context}: unexpected prompt or prefix")
        phrases = sum(row["recognized_phrase_count"] for row in rows)
        hits = sum(row["target_last_2_hits"] for row in rows)
        examples = [row["text"] for row in rows if row["trial_seed"] == example_seed]
        cells.append({
            "label": label, "stage": stage, "audio_mode": audio, "context_mode": context,
            "n": len(rows),
            "mean_character_count": sum(row["character_count"] for row in rows) / len(rows),
            "mean_mora_count": sum(row["mora_count"] for row in rows) / len(rows),
            "mora_48_count": sum(row["mora_count"] == 48 for row in rows),
            "boilerplate_count": sum(row["boilerplate"] for row in rows),
            "target_last_2_hits": hits, "recognized_phrase_count": phrases,
            "target_last_2_rate": hits / phrases if phrases else None,
            "example_text": examples[0] if examples else None,
        })
    return {"settings": {key: baseline[key] for key in MATCHED},
            "example_seed": example_seed, "conditions": cells}


def markdown(result: dict) -> str:
    settings = result["settings"]
    lines = [
        "# 記事の6条件の実行結果", "",
        (f"モデル: {settings['model']} / 各{settings['n_samples']}回 / "
         f"seed {settings['seed']}〜{settings['seed'] + settings['n_samples'] - 1}"), "",
        "語尾率は母音のある出力句を合算した値。意味の自然さや発音時刻の一致は評価しない。",
        "",
        "|条件|平均文字数|平均モーラ数|48モーラ一致|定型句|語尾 /a a/|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in result["conditions"]:
        rate = cell["target_last_2_rate"]
        percent = "—" if rate is None else f"{rate:.1%}"
        lines.append(
            f"|{cell['label']}|{cell['mean_character_count']:.1f}|{cell['mean_mora_count']:.1f}|"
            f"{cell['mora_48_count']}/{cell['n']}|{cell['boilerplate_count']}/{cell['n']}|{percent}|"
        )
    lines += ["", f"## 共通seed {result['example_seed']}の出力", "",
              "改行を追加せず、生出力を表示する。該当seedを実行していない場合はその旨を示す。", ""]
    for cell in result["conditions"]:
        lines += [f"### {cell['label']}", ""]
        text = cell["example_text"]
        if text is None:
            lines += ["該当seedの出力なし。", ""]
        else:
            # A model output may itself contain backticks.
            fence = "```"
            while fence in text:
                fence += "`"
            lines += [fence + "text", text, fence, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--example-seed", type=int, default=3585)
    args = parser.parse_args()
    try:
        result = summarize(args.directory, args.example_seed)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    (args.directory / "article_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    report = args.directory / "article_summary.md"
    report.write_text(markdown(result), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
