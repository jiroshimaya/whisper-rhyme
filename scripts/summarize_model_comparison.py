"""Recompute descriptive diagnostics from frozen model-size comparison results."""

import argparse
import json
from pathlib import Path

from whisper_rap.comparison import summarize_models
from whisper_rap.metrics import analyze_text
from whisper_rap.rhyme import rhyme_measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.directory / "samples.jsonl").read_text().splitlines()]
    manifest = json.loads((args.directory / "manifest.json").read_text())
    for row in rows:
        cell = f"{row['anchor_mode']}|{row['rhyme_mode']}"
        row.update(analyze_text(row["text"], manifest["audio"][cell]["duration"]))
        if row.get("mora_error"):
            raise RuntimeError(row["mora_error"])
        row.update(rhyme_measurements(row))
    result = {
        "manifest": manifest,
        "completed": len(rows),
        "expected": len(manifest["models"]) * len(manifest["audio"]) * manifest["n_samples"],
        "diagnostic_definitions": {
            "vowel_metrics": "Recomputed from frozen text with the current mora-to-vowel map.",
            "repetitive": "After NFKC and whitespace removal: a 1-32 character unit repeats >=3 "
                          "times consecutively, covers >=12 characters and >=50% of the output.",
            "prompt_echo": "At least 50% of normalized output characters belong to 4-grams "
                           "that also occur in the normalized decoder prompt.",
            "candidate": ">=8 visible characters, >=75% Japanese, >=6 moras, no dominant "
                         "repetition/prompt echo, boilerplate, replacement character or control token. "
                         "Exploratory surface filter, not a semantic quality or hallucination label.",
        },
        "by_model": summarize_models(rows),
    }
    output = args.directory / "analysis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
