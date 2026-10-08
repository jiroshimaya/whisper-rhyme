"""Text and mora-level metrics for generated Japanese hypotheses."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々〆ヵヶ]")
PHRASE_SPLIT_RE = re.compile(r"[\s、。！？!?；;]+")
KATAKANA_MORA_RE = re.compile(r"[ァ-ヺー]+")

VOWEL_KANA = {
    "a": set("ァアカガサザタダナハバパマャヤラヮワ"),
    "i": set("ィイキギシジチヂニヒビピミリヰ"),
    "u": set("ゥウクグスズツヅヌフブプムュユルヴ"),
    "e": set("ェエケゲセゼテデネヘベペメレヱ"),
    "o": set("ォオコゴソゾトドノホボポモョヨロヲ"),
}


def japanese_character_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    return sum(bool(JAPANESE_RE.fullmatch(char)) for char in visible) / len(visible)


def to_katakana_reading(text: str) -> str:
    """Convert kanji/kana text to a katakana reading with pykakasi."""
    from pykakasi import kakasi

    converted = kakasi().convert(text)
    return "".join(item["kana"] for item in converted)


def split_moras(reading: str) -> list[str]:
    """Split a kana reading according to jamorasep's mora definition."""
    import jamorasep

    parsed = jamorasep.parse(reading, output_format="katakana")
    if isinstance(parsed, str):
        parsed = parsed.split()
    return [str(part) for part in parsed if KATAKANA_MORA_RE.fullmatch(str(part))]


def moras_to_vowels(moras: Iterable[str]) -> list[str]:
    """Return the pronounced vowel of each vocalic mora; carry long marks forward."""
    result: list[str] = []
    for mora in moras:
        if mora == "ー":
            if result:
                result.append(result[-1])
            continue
        vowel = next(
            (name for char in reversed(mora) for name, chars in VOWEL_KANA.items() if char in chars),
            None,
        )
        if vowel is not None:
            result.append(vowel)
    return result


def analyze_text(text: str, audio_duration: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "character_count": len(text),
        "japanese_character_ratio": japanese_character_ratio(text),
    }
    try:
        reading = to_katakana_reading(text)
        moras = split_moras(reading)
        phrases = [phrase.strip() for phrase in PHRASE_SPLIT_RE.split(text) if phrase.strip()]
        phrase_moras = [split_moras(to_katakana_reading(phrase)) for phrase in phrases]
        vowels = moras_to_vowels(moras)
        phrase_vowels = [moras_to_vowels(items) for items in phrase_moras]
        result.update(
            {
                "reading": reading,
                "moras": moras,
                "mora_count": len(moras),
                "phrase_mora_counts": [len(items) for items in phrase_moras],
                "phrase_end_moras": [
                    {"last_1": items[-1:], "last_2": items[-2:], "last_3": items[-3:]}
                    for items in phrase_moras
                ],
                "vowels": vowels,
                "phrase_end_vowels": [
                    {"last_1": items[-1:], "last_2": items[-2:], "last_3": items[-3:]}
                    for items in phrase_vowels
                ],
            }
        )
        if audio_duration and audio_duration > 0:
            result["mora_per_second"] = len(moras) / audio_duration
    except (ImportError, RuntimeError, ValueError) as exc:
        result.update({"reading": None, "moras": [], "mora_count": None, "mora_error": str(exc)})
    return result


def aggregate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    texts = [str(row.get("text", "")) for row in rows]
    counts = Counter(texts)
    nonempty = [row for row in rows if str(row.get("text", "")).strip()]
    usable = [row for row in rows if row.get("usable_candidate") is True]
    boilerplate = [row for row in rows if row.get("boilerplate")]

    def mean(field: str, source: list[dict[str, Any]] = rows) -> float:
        values = [row[field] for row in source if isinstance(row.get(field), (int, float))]
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "sample_count": len(rows),
        "nonempty_count": len(nonempty),
        "unique_count": len(counts),
        "unique_output_ratio": len(counts) / len(rows) if rows else 0.0,
        "usable_candidate_count": len(usable),
        "usable_candidate_ratio": len(usable) / len(rows) if rows else 0.0,
        "boilerplate_count": len(boilerplate),
        "boilerplate_ratio": len(boilerplate) / len(rows) if rows else 0.0,
        "mean_character_count": mean("character_count"),
        "mean_japanese_character_ratio": mean("japanese_character_ratio"),
        "mean_mora_count": mean("mora_count"),
        "mean_mora_per_second": mean("mora_per_second"),
        "top_outputs": [
            {"text": text, "count": count, "ratio": count / len(rows)}
            for text, count in counts.most_common(10)
        ]
        if rows
        else [],
    }
