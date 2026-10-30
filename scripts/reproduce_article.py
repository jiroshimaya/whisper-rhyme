#!/usr/bin/env python3
"""Run the Qiita experiments in dependency order without changing inference code."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def experiment_commands(args: argparse.Namespace) -> list[tuple[Path, list[str]]]:
    output = args.output_dir.resolve()
    baseline, partial = output / "baseline", output / "partial"
    common = [
        "--anchor-manifest", str(args.anchor_manifest.resolve()),
        "--model", args.model, "--n-samples", str(args.n_samples),
        "--seed", str(args.seed), "--threads", str(args.threads),
    ]
    article = [sys.executable, str(ROOT / "scripts/run_article_experiment.py")]
    stages = [
        (baseline, article + common),
        (partial, article + common + ["--frozen-baseline-dir", str(baseline)]),
        (output / "context", article + common + [
            "--frozen-baseline-dir", str(baseline), "--rhyming-context-only",
            "--prompt-file", str(ROOT / "prompts/rap_aa_four_phrases.txt"),
        ]),
        (output / "onset", [sys.executable, str(ROOT / "scripts/run_onset_experiment.py"),
                            "--baseline-dir", str(baseline), "--partial-dir", str(partial),
                            "--anchor-manifest", str(args.anchor_manifest.resolve())]),
    ]
    return [(directory, command + ["--output-dir", str(directory)] + (
        ["--resume"] if args.resume and directory.exists() else []
    )) for directory, command in stages]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qiita"))
    parser.add_argument("--model", choices=("tiny", "small", "large-v3"), default="large-v3")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3579)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = parser.parse_args()
    if args.n_samples < 1 or args.threads < 1 or args.seed < 0:
        parser.error("positive sample/thread counts and a nonnegative seed are required")
    stages = experiment_commands(args)
    if not args.dry_run:
        if not args.anchor_manifest.is_file():
            parser.error("synthesize the rhyme-final anchors first; see docs/reproduction.md")
        for directory, _ in stages:
            if directory.exists() and not args.resume:
                parser.error(f"output exists: {directory}; choose a new directory or use --resume")
            if directory.exists() and not (directory / "manifest.json").is_file():
                parser.error(f"no manifest in {directory}; preserve it and choose a new directory")
    commands = [command for _, command in stages] + [[
        sys.executable, str(ROOT / "scripts/summarize_article.py"), str(args.output_dir.resolve()),
    ]]
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
