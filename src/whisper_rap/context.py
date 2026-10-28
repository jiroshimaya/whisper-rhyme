"""Check the length and rhyme pattern of a decoder context example."""

from __future__ import annotations

from .metrics import analyze_text


def validate_rhyming_context(text: str) -> list[dict]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 4:
        raise ValueError("rhyming context must have four phrases")
    analyses = [analyze_text(line) for line in lines]
    for index, row in enumerate(analyses, 1):
        if row.get("mora_error") or row["mora_count"] != 12 or row["vowels"][-2:] != ["a", "a"]:
            raise ValueError(f"context phrase {index} must have 12 moras and end in /a a/")
    return analyses
