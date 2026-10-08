from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from whisper_rap.decode import _suppress_tokens_for_model, load_model, write_jsonl


@pytest.mark.parametrize("name", ["tiny", "small", "large-v3"])
@pytest.mark.parametrize("prefix", ["", "openai/whisper-"])
def test_load_model_aliases(monkeypatch, name, prefix) -> None:
    calls = []
    stub = SimpleNamespace(load_model=lambda *a, **k: calls.append((a, k)))
    monkeypatch.setitem(sys.modules, "whisper", stub)
    load_model(prefix + name, device="cpu", download_root="cache")
    assert calls == [((name,), {"device": "cpu", "download_root": "cache"})]


@pytest.mark.parametrize("name", ["tiny.en", "small.en", "large-v3-turbo", "unknown"])
def test_unsupported_models_fail_before_loading(monkeypatch, name) -> None:
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace())
    with pytest.raises(ValueError, match="multilingual"):
        load_model(name)


@pytest.mark.parametrize("languages", [99, 100])
def test_suppression_uses_model_tokenizer(monkeypatch, languages) -> None:
    def get_tokenizer(multilingual, **kwargs):
        assert multilingual
        assert kwargs == {"num_languages": languages, "language": "ja", "task": "transcribe"}
        return SimpleNamespace(all_language_tokens=(1, 2), no_timestamps=50264 + languages)

    monkeypatch.setitem(
        sys.modules, "whisper", SimpleNamespace(tokenizer=SimpleNamespace(get_tokenizer=get_tokenizer))
    )
    model = SimpleNamespace(is_multilingual=True, num_languages=languages)
    result = _suppress_tokens_for_model(model, "-1")
    assert result == f"-1,1,2,{50264 + languages}"


def test_explicit_suppression_is_preserved() -> None:
    assert _suppress_tokens_for_model(None, None) == []
    assert _suppress_tokens_for_model(None, [7, 8]) == [7, 8]
    assert _suppress_tokens_for_model(None, "7,8") == "7,8"


def test_write_jsonl(tmp_path) -> None:
    output = tmp_path / "result.jsonl"
    count = write_jsonl(output, [{"text": "夜"}, {"text": "朝"}])
    assert count == 2
    assert [json.loads(line)["text"] for line in output.read_text().splitlines()] == ["夜", "朝"]
