#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from whisper_rap.metrics import aggregate_records


def load_records(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{file}:{line_number}: {exc}") from exc
    return rows


def create_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[str] = []
    dimensions = [
        ("condition", "character_count", "Condition", "Mean character count"),
        ("temperature", "character_count", "Temperature", "Mean character count"),
        ("condition", "japanese_character_ratio", "Condition", "Japanese character ratio"),
        ("condition", "mora_per_second", "Condition", "Mean mora/sec"),
        ("acoustic_mode", "character_count", "Acoustic mode", "Mean character count"),
        ("conditioning_mode", "character_count", "Conditioning mode", "Mean character count"),
    ]
    for group_field, metric, xlabel, ylabel in dimensions:
        grouped: dict[Any, list[float]] = defaultdict(list)
        for row in rows:
            if row.get(group_field) is None:
                continue
            value = row.get(metric)
            if isinstance(value, (int, float)):
                grouped[row.get(group_field)].append(float(value))
        if not grouped:
            continue
        labels = sorted(grouped, key=str)
        values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar([str(label) for label in labels], values)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=f"{ylabel} by {xlabel.lower()}")
        fig.tight_layout()
        filename = output_dir / f"{metric}_by_{group_field}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        plots.append(str(filename))

    by_temperature: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_temperature[row.get("temperature")].append(row)
    if by_temperature:
        labels = sorted(by_temperature, key=str)
        values = [aggregate_records(by_temperature[label])["unique_output_ratio"] for label in labels]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar([str(label) for label in labels], values)
        axis.set(xlabel="Temperature", ylabel="Unique output ratio", ylim=(0, 1))
        fig.tight_layout()
        filename = output_dir / "unique_output_ratio_by_temperature.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        plots.append(str(filename))
    return plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one JSONL file or a result directory")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    args = parser.parse_args()
    rows = load_records(args.path)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_temperature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_acoustic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_conditioning: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_matrix_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition"))].append(row)
        by_temperature[str(row.get("temperature"))].append(row)
        if row.get("acoustic_mode") is not None:
            acoustic = str(row["acoustic_mode"])
            conditioning = str(row.get("conditioning_mode"))
            by_acoustic[acoustic].append(row)
            by_conditioning[conditioning].append(row)
            cell = f"{acoustic}|{conditioning}|t={row.get('temperature')}"
            by_matrix_cell[cell].append(row)
    summary = {
        "overall": aggregate_records(rows),
        "by_condition": {key: aggregate_records(value) for key, value in sorted(by_condition.items())},
        "by_temperature": {
            key: aggregate_records(value) for key, value in sorted(by_temperature.items())
        },
        "by_acoustic_mode": {
            key: aggregate_records(value) for key, value in sorted(by_acoustic.items())
        },
        "by_conditioning_mode": {
            key: aggregate_records(value) for key, value in sorted(by_conditioning.items())
        },
        "by_matrix_cell": {
            key: aggregate_records(value) for key, value in sorted(by_matrix_cell.items())
        },
    }
    if args.plot_dir:
        summary["plots"] = create_plots(rows, args.plot_dir)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
