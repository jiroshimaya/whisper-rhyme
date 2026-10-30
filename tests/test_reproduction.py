from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce = load_script("reproduce_article")
export = load_script("summarize_article")


def test_resume_reuses_started_stages_and_starts_missing_dependencies(tmp_path):
    output = tmp_path / "results with spaces"
    (output / "baseline").mkdir(parents=True)
    args = argparse.Namespace(
        output_dir=output, anchor_manifest=tmp_path / "words.json", model="tiny",
        n_samples=1, seed=3579, threads=2, resume=True,
    )
    stages = reproduce.experiment_commands(args)
    assert [path.name for path, _ in stages] == ["baseline", "partial", "context", "onset"]
    assert "--resume" in stages[0][1]
    assert all("--resume" not in command for _, command in stages[1:])
    for _, command in stages[1:3]:
        assert command[command.index("--frozen-baseline-dir") + 1] == str(output / "baseline")
        assert command[command.index("--model") + 1] == "tiny"
    onset = stages[3][1]
    assert onset[onset.index("--partial-dir") + 1] == str(output / "partial")


def test_failed_stage_stops_pipeline(tmp_path, monkeypatch):
    import subprocess
    import sys

    anchor = tmp_path / "words.json"
    anchor.write_text("{}")
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(sys, "argv", ["reproduce_article.py", "--anchor-manifest", str(anchor),
                                     "--output-dir", str(tmp_path / "results")])
    monkeypatch.setattr(reproduce.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        reproduce.main()
    assert len(calls) == 1


@pytest.mark.parametrize("resume,manifest", [(False, True), (True, False)])
def test_existing_outputs_are_preserved(tmp_path, monkeypatch, resume, manifest):
    import sys

    anchor = tmp_path / "words.json"
    anchor.write_text("{}")
    output = tmp_path / "results"
    (output / "baseline").mkdir(parents=True)
    saved = output / "baseline" / ("manifest.json" if manifest else "keep.txt")
    saved.write_text("preserve me")
    monkeypatch.setattr(sys, "argv", ["reproduce_article.py", "--anchor-manifest", str(anchor),
                                     "--output-dir", str(output)] + (["--resume"] if resume else []))
    monkeypatch.setattr(reproduce.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(SystemExit) as exc:
        reproduce.main()
    assert exc.value.code == 2
    assert saved.read_text() == "preserve me"


@pytest.fixture
def saved_run(tmp_path):
    manifest = dict.fromkeys(export.MATCHED)
    manifest.update(model="large-v3", seed=3584, n_samples=2)
    prompt = (ROOT / "prompts/rap_aa_four_phrases.txt").read_text().strip()
    for stage in ("baseline", "partial", "context", "onset"):
        directory = tmp_path / stage
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps(manifest))
        rows = []
        for _, row_stage, audio, context in export.CONDITIONS:
            if row_stage != stage:
                continue
            for i in range(2):
                rows.append({
                    "audio_mode": audio, "context_mode": context, "trial_seed": 3584 + i,
                    "text": "a```b" if i else "first", "character_count": 10 + 10 * i,
                    "mora_count": 48 if i else 24, "boilerplate": bool(i),
                    "target_last_2_hits": 1 if i == 0 else 0,
                    "recognized_phrase_count": 1 if i == 0 else 3,
                    "prompt": prompt if context == "prompt" else None, "prefix": None,
                })
        # Unrelated controls must not be included in the article's six conditions.
        rows.append({"audio_mode": "unused", "context_mode": "prompt_prefix"})
        (directory / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return tmp_path


def test_article_export_uses_six_conditions_pooled_phrases_and_common_seed(saved_run):
    result = export.summarize(saved_run, 3585)
    assert len(result["conditions"]) == 6
    for cell in result["conditions"]:
        assert cell["n"] == 2
        assert cell["mean_character_count"] == 15
        assert cell["mean_mora_count"] == 36
        assert cell["mora_48_count"] == cell["boilerplate_count"] == 1
        assert cell["target_last_2_rate"] == 0.25  # Phrase-pooled, not mean of output rates.
        assert cell["example_text"] == "a```b"
    assert "````text\na```b\n````" in export.markdown(result)
    missing = export.summarize(saved_run, 9999)
    assert all(cell["example_text"] is None for cell in missing["conditions"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "model", "prompt"])
def test_article_export_rejects_incomplete_or_mixed_runs(saved_run, change):
    directory = saved_run / "context"
    if change == "model":
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["model"] = "tiny"
        path.write_text(json.dumps(manifest))
    else:
        path = directory / "samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "missing":
            rows.pop(0)
        elif change == "duplicate":
            rows.insert(0, rows[0])
        else:
            rows[0]["prompt"] = "old context"
        path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        export.summarize(saved_run, 3585)
