from __future__ import annotations

import pytest

from whisper_rap.context import validate_rhyming_context


def test_context_checks_each_phrase_length_and_rhyme() -> None:
    text = "\n".join(["むかしのゆめをだいたまま"] * 4)
    rows = validate_rhyming_context(text)
    assert len(rows) == 4
    assert all(row["mora_count"] == 12 for row in rows)
    assert all(row["vowels"][-2:] == ["a", "a"] for row in rows)


def test_context_rejects_missing_phrase() -> None:
    with pytest.raises(ValueError, match="four phrases"):
        validate_rhyming_context("\n".join(["むかしのゆめをだいたまま"] * 3))


@pytest.mark.parametrize("last", ["むかしのゆめをだいた", "むかしのゆめをだいたとき"])
def test_context_rejects_wrong_length_or_final_vowels(last) -> None:
    with pytest.raises(ValueError, match="phrase 4"):
        validate_rhyming_context("\n".join(["むかしのゆめをだいたまま"] * 3 + [last]))
