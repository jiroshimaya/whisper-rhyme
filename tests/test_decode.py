from __future__ import annotations

import json

from whisper_rap.decode import ALLOWED_MODELS, write_jsonl


def test_large_v3_only() -> None:
    assert ALLOWED_MODELS == {"large-v3", "openai/whisper-large-v3"}
    assert "large-v3-turbo" not in ALLOWED_MODELS


def test_write_jsonl(tmp_path) -> None:
    output = tmp_path / "result.jsonl"
    count = write_jsonl(output, [{"text": "夜"}, {"text": "朝"}])
    assert count == 2
    assert [json.loads(line)["text"] for line in output.read_text().splitlines()] == ["夜", "朝"]
