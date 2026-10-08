"""Verify relocated artifacts without accepting the wrong SFT reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

import verify as V

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.mark.parametrize("root,base", [
    ("/content/lab22", "/content/lab22/models/sft-merged"),
    ("D:/old/lab22", "D:\\old\\lab22\\models\\sft-merged"),
])
def test_copied_reference_requires_explicit_original_root(root, base):
    assert not V.reference_matches(base)
    assert V.reference_matches(base, V.training_root_path(root))


@pytest.mark.parametrize("base", [
    "", "/content/lab22/sft-merged", "/another/lab22/models/sft-merged",
    "/content/lab22/models/base", "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    "/content/lab22/other/../models/sft-merged",
])
def test_relocation_rejects_wrong_reference(base):
    assert not V.reference_matches(base, V.training_root_path("/content/lab22"))


@pytest.mark.parametrize("root", ["lab22", "D:lab22", "/content/../lab22"])
def test_original_root_must_be_unambiguous_and_absolute(root):
    with pytest.raises(argparse.ArgumentTypeError):
        V.training_root_path(root)


def test_current_reference_still_works():
    assert V.reference_matches(str(V.REPO / "models" / "sft-merged"))


@pytest.fixture
def copied_submission(tmp_path, monkeypatch):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    for name in V.NOTEBOOKS:
        write(f"notebooks/{name}.py", "notebook evidence")
    write("adapters/sft-mini/adapter_config.json", {})
    write("models/sft-merged/config.json", {})
    write("data/pref/train.parquet", "train bytes")
    write("data/pref/eval.parquet", "held-out bytes")
    write("adapters/dpo/adapter_config.json", {"base_model_name_or_path": "/content/lab22/models/sft-merged"})
    write("adapters/dpo/dpo_metrics.json", {
        "end_reward_gap": 0.08, "eval_reward_accuracy": 0.62, "diagnosis": "INTENDED",
    })
    write("adapters/dpo/split.json", {
        split: hashlib.sha256((tmp_path / f"data/pref/{split}.parquet").read_bytes()).hexdigest()
        for split in ("train", "eval")
    })
    write("data/eval/side_by_side.jsonl", {"outputs": "test fixture"})
    write("data/eval/judge_summary.json", {
        "outputs_sha256": hashlib.sha256((tmp_path / "data/eval/side_by_side.jsonl").read_bytes()).hexdigest(),
        "heldout": {"n": 50}, "sanity_accuracy": 1.0,
    })
    write("submission/REFLECTION.md", "completed reflection fixture")
    for name in V.CORE_SCREENSHOTS:
        write(f"submission/screenshots/{name}.png", "image fixture")
    monkeypatch.setattr(V, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify.py", "--training-root", "/content/lab22"])
    return tmp_path


def test_relocation_cli_passes_without_rewriting_config(copied_submission, monkeypatch, capsys):
    config = copied_submission / "adapters/dpo/adapter_config.json"
    before = config.read_bytes()
    assert V.main() == 0
    assert config.read_bytes() == before
    monkeypatch.setattr(sys, "argv", ["verify.py"])
    assert V.main() == 1
    assert "WRONG REF" in capsys.readouterr().out


def test_relocation_still_rejects_changed_splits_and_stale_judgments(copied_submission, capsys):
    (copied_submission / "data/pref/train.parquet").write_bytes(b"changed split")
    (copied_submission / "data/eval/side_by_side.jsonl").write_bytes(b"changed outputs")
    assert V.main() == 1
    output = capsys.readouterr().out
    assert "SPLIT" in output
    assert "STALE" in output
