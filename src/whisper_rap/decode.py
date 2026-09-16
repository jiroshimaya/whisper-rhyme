"""Direct Whisper decoder runner without transcribe-time filtering."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import analyze_text

ALLOWED_MODELS = {"large-v3", "openai/whisper-large-v3"}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, device: str | None = None, download_root: str | None = None) -> Any:
    import whisper

    if model_name not in ALLOWED_MODELS:
        raise ValueError("this experiment only permits Whisper large-v3 (not turbo)")
    official_name = "large-v3" if model_name == "openai/whisper-large-v3" else model_name
    return whisper.load_model(official_name, device=device, download_root=download_root)


def prepare_mel(audio_file: str | Path, model: Any) -> Any:
    import whisper

    audio = whisper.load_audio(str(audio_file))
    audio = whisper.pad_or_trim(audio)
    return whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)


def decode_samples(
    *,
    model: Any,
    mel: Any,
    audio_file: str | Path,
    condition: str,
    temperatures: Iterable[float],
    n_samples: int,
    seed: int,
    prompt: str | None,
    prefix: str | None = None,
    model_name: str = "large-v3",
    best_of: int | None = 1,
    sample_len: int | None = None,
    length_penalty: float | None = None,
    suppress_blank: bool = False,
    suppress_tokens: str | list[int] | None = "-1",
    audio_duration: float = 30.0,
) -> Iterable[dict[str, Any]]:
    """Yield raw DecodingResult data plus derived metrics for every draw."""
    import torch
    import whisper

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if best_of is not None and best_of < 1:
        raise ValueError("best_of must be at least 1")
    fp16 = getattr(model, "device", torch.device("cpu")).type == "cuda"
    tokenizer = whisper.tokenizer.get_tokenizer(
        getattr(model, "is_multilingual", True), language="ja", task="transcribe"
    )
    effective_suppress_tokens = [] if suppress_tokens is None else suppress_tokens
    if isinstance(suppress_tokens, str) and "-1" in suppress_tokens.split(","):
        control_tokens = [*tokenizer.all_language_tokens, tokenizer.no_timestamps]
        effective_suppress_tokens = ",".join(
            [suppress_tokens, *(str(token) for token in control_tokens if token is not None)]
        )
    sample_id = 0
    for temperature in temperatures:
        if not 0 <= temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        for index in range(n_samples):
            trial_seed = seed + sample_id
            _seed_everything(trial_seed)
            option_values: dict[str, Any] = {
                "language": "ja",
                "task": "transcribe",
                "temperature": float(temperature),
                "without_timestamps": True,
                "fp16": fp16,
                "prompt": prompt or None,
                "prefix": prefix or None,
                "sample_len": sample_len,
                "length_penalty": length_penalty,
                "suppress_blank": suppress_blank,
                "suppress_tokens": effective_suppress_tokens,
            }
            if temperature > 0 and best_of is not None:
                option_values["best_of"] = best_of
            options = whisper.DecodingOptions(**option_values)
            with torch.inference_mode():
                decoded = model.decode(mel, options)
            text = decoded.text.strip()
            row: dict[str, Any] = {
                "condition": condition.upper(),
                "model": model_name,
                "language": "ja",
                "task": "transcribe",
                "without_timestamps": True,
                "seed": seed,
                "trial_seed": trial_seed,
                "temperature": float(temperature),
                "sample_id": sample_id,
                "temperature_sample_id": index,
                "audio_file": str(audio_file),
                "prompt": prompt,
                "prefix": prefix,
                "best_of": best_of if temperature > 0 else None,
                "sample_len": sample_len,
                "length_penalty": length_penalty,
                "suppress_blank": suppress_blank,
                "suppress_tokens": effective_suppress_tokens,
                "text": text,
                "full_text": f"{prefix or ''}{text}",
                "tokens": [int(token) for token in decoded.tokens],
                "avg_logprob": float(decoded.avg_logprob),
                "no_speech_prob": float(decoded.no_speech_prob),
                "compression_ratio": float(decoded.compression_ratio),
            }
            row.update(analyze_text(text, audio_duration))
            yield row
            sample_id += 1


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
    return count
