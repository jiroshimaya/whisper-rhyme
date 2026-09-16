#!/usr/bin/env python3
"""Synthesize matched theme and neutral word anchors with a local VOICEVOX engine."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

ANCHOR_SETS = {
    "theme": ("夜", "街", "影", "鼓動"),
    "neutral": ("これ", "ただ", "いま", "ことば"),
}


def post(url: str, data: bytes | None = None, content_type: str | None = None) -> bytes:
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib.request.Request(url, data=data or b"", headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def synthesize(text: str, *, url: str, speaker: int) -> bytes:
    query_url = f"{url.rstrip('/')}/audio_query?{urllib.parse.urlencode({'text': text, 'speaker': speaker})}"
    query = json.loads(post(query_url))
    query.update(
        {
            "speedScale": 1.12,
            "pitchScale": -0.02,
            "intonationScale": 0.8,
            "prePhonemeLength": 0.06,
            "postPhonemeLength": 0.10,
            "outputSamplingRate": 16_000,
            "outputStereo": False,
        }
    )
    synthesis_url = f"{url.rstrip('/')}/synthesis?{urllib.parse.urlencode({'speaker': speaker})}"
    return post(synthesis_url, json.dumps(query, ensure_ascii=False).encode(), "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:50021")
    parser.add_argument("--speaker", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered_sets: dict[str, list[dict[str, object]]] = {}
    manifest: dict[str, object] = {"speaker": args.speaker, "sets": rendered_sets}
    for set_name, words in ANCHOR_SETS.items():
        entries = []
        for index, word in enumerate(words):
            output = args.output_dir / f"{set_name}_{index}.wav"
            output.write_bytes(synthesize(word, url=args.url, speaker=args.speaker))
            entries.append({"word": word, "audio_file": str(output.resolve())})
        rendered_sets[set_name] = entries
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
