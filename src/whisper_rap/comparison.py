"""Descriptive output diagnostics for a matched model-size comparison."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from .lexical import paired_cell_comparison, summarize_lexical_cells

BOILERPLATE = ("チャンネル登録", "ご視聴", "ご清聴", "字幕", "高評価", "お疲れ様でした")


def output_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row["text"])
    normalized = "".join(unicodedata.normalize("NFKC", text).split())
    longest_run = max(
        (len(match.group()) for match in re.finditer(r"(.{1,32}?)\1{2,}", normalized)),
        default=0,
    )
    repeat_share = longest_run / len(normalized) if normalized else 0.0
    repetitive = longest_run >= 12 and repeat_share >= 0.5
    boilerplate = any(phrase in text for phrase in BOILERPLATE)
    context = "".join(unicodedata.normalize("NFKC", str(row.get("prompt") or "")).split())
    context_ngrams = {context[i:i + 4] for i in range(max(0, len(context) - 3))}
    echoed_positions = set()
    for index in range(max(0, len(normalized) - 3)):
        if normalized[index:index + 4] in context_ngrams:
            echoed_positions.update(range(index, index + 4))
    prompt_echo_share = len(echoed_positions) / len(normalized) if normalized else 0.0
    prompt_echo = prompt_echo_share >= 0.5
    japanese_output = bool(row["mora_count"])
    # This is a surface filter, not a fluency, semantic-quality or hallucination score.
    candidate = (
        len(normalized) >= 8
        and row["japanese_character_ratio"] >= 0.75
        and (row["mora_count"] or 0) >= 6
        and not repetitive
        and not boilerplate
        and not prompt_echo
        and "\ufffd" not in text
        and "<|" not in text
    )
    return {
        "japanese_output": japanese_output,
        "repetitive": repetitive,
        "repeat_share": repeat_share,
        "boilerplate": boilerplate,
        "prompt_echo": prompt_echo,
        "prompt_echo_share": prompt_echo_share,
        "nontrivial_japanese_candidate": candidate,
        "token_limit_reached": len(row["tokens"]) >= row["sample_len"],
    }


def diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["text"] for row in rows)
    fields = (
        "japanese_output", "repetitive", "boilerplate", "prompt_echo",
        "nontrivial_japanese_candidate", "token_limit_reached",
    )
    return {
        "sample_count": len(rows),
        "mean_character_count": sum(row["character_count"] for row in rows) / len(rows),
        "unique_output_ratio": len(counts) / len(rows),
        **{f"{field}_rate": sum(bool(row[field]) for row in rows) / len(rows) for field in fields},
        "top_outputs": [{"text": text, "count": count} for text, count in counts.most_common(5)],
    }


def summarize_models(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Recompute surface diagnostics so saved inference rows can be audited without rerunning ASR.
    rows = [{**row, **output_diagnostics(row)} for row in rows]
    result = {}
    for model in dict.fromkeys(row["model"] for row in rows):
        items = [row for row in rows if row["model"] == model]
        cells = summarize_lexical_cells(items)
        for cell, values in cells.items():
            selected = [
                row for row in items if f"{row['anchor_mode']}|{row['rhyme_mode']}" == cell
            ]
            values.update(diagnostic_summary(selected))
        result[model] = {
            "overall": diagnostic_summary(items),
            "by_cell": cells,
            "theme_rhyme_on_minus_off": paired_cell_comparison(
                items, left=("theme", "on"), right=("theme", "off"),
                metrics=("target_last_1_rate", "target_last_2_rate"), seed=3579,
            ),
            "none_rhyme_on_minus_off": paired_cell_comparison(
                items, left=("none", "on"), right=("none", "off"),
                metrics=("target_last_1_rate", "target_last_2_rate"), seed=3579,
            ),
        }
    return result
