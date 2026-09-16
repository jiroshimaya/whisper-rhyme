#!/usr/bin/env python3
"""Create an original nonsense-utterance source with a local VOICEVOX engine."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_TEXT = (
    "タラナイ、カラマイ、サラダイ。"
    "ナガレル、ハザマデ、カサナリ。"
    "マダミナイ、アサヘ、トビダシ。"
    "ララライ、カラマイ、ハナサナイ。"
)


def post(url: str, data: bytes | None = None, content_type: str | None = None) -> bytes:
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib.request.Request(url, data=data or b"", headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:50021")
    parser.add_argument("--speaker", type=int, default=2)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    query_url = f"{args.url.rstrip('/')}/audio_query?{urllib.parse.urlencode({'text': args.text, 'speaker': args.speaker})}"
    query = json.loads(post(query_url))
    query.update(
        {
            "speedScale": 1.08,
            "pitchScale": -0.02,
            "intonationScale": 0.85,
            "prePhonemeLength": 0.10,
            "postPhonemeLength": 0.35,
            "outputSamplingRate": 16_000,
            "outputStereo": False,
        }
    )
    synthesis_url = f"{args.url.rstrip('/')}/synthesis?{urllib.parse.urlencode({'speaker': args.speaker})}"
    wave = post(
        synthesis_url,
        json.dumps(query, ensure_ascii=False).encode(),
        "application/json",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(wave)
    print(json.dumps({"output": str(args.output), "speaker": args.speaker, "text": args.text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
