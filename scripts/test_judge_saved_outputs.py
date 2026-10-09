"""CPU-only tests for API judging, resume safety, and cross-judge summaries."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import judge_saved_outputs as S


def inputs():
    records = [
        {"id": "e0", "category": "heldout", "prompt": "p", "sft": "s", "dpo": "d"},
        {"id": "h1", "category": "helpfulness", "prompt": "q", "sft": "same", "dpo": "same"},
    ]
    rm = {"judge": "rm-panel:test", "outputs_sha256": "digest", "records": [
        {**r, "winner": "sft" if r["id"] == "e0" else "tie"} for r in records
    ]}
    return records, rm


def spec():
    return {"judge": "openai:test-model", "outputs_sha256": "digest", "seed": 42}


def test_inputs_reject_stale_hash_changed_text_and_duplicate_ids():
    records, rm = inputs()
    S.validate_inputs(records, rm, "digest")
    with pytest.raises(ValueError, match="SHA256"):
        S.validate_inputs(records, rm, "other")
    wrong = deepcopy(rm)
    wrong["records"][0]["prompt"] = "different"
    with pytest.raises(ValueError, match="text"):
        S.validate_inputs(records, wrong, "digest")
    with pytest.raises(ValueError, match="duplicate"):
        S.validate_inputs(records + [records[0]], rm, "digest")


def test_resume_preserves_judgments_and_does_not_recall_completed_pairs():
    records, _ = inputs()
    replies = iter(['{"winner":"B"}', '{"winner":"A"}'])
    checkpoints, seen = [], []

    def call(system, user):
        seen.append(user)
        return next(replies)

    judged = S.judge_remaining(records, {}, call, checkpoints.append, limit=1)
    assert judged[0]["winner"] == "dpo"
    assert seen[0].index("Câu trả lời A]\ns") < seen[0].index("Câu trả lời B]\nd")
    assert seen[1].index("Câu trả lời A]\nd") < seen[1].index("Câu trả lời B]\ns")
    saved = {"run": spec(), "records": checkpoints[-1]}
    done = S.resume_records(saved, spec(), records)
    new_calls = []
    finished = S.judge_remaining(records, done, lambda s, u: new_calls.append(u) or "tie", checkpoints.append)
    assert len(new_calls) == 2  # only h1, still checked in both orders even when answers are identical
    assert finished[0] == judged[0]
    assert finished[1]["winner"] == "tie"


def test_resume_rejects_changed_model_and_tampered_verdict():
    records, _ = inputs()
    row = {**records[0], **S.J.verdict_record("B", "A")}
    saved = {"run": spec(), "records": [row]}
    with pytest.raises(ValueError, match="different"):
        S.resume_records(saved, {**spec(), "judge": "openai:other"}, records)
    row["winner"] = "sft"
    with pytest.raises(ValueError, match="verdict"):
        S.resume_records(saved, spec(), records)


def test_summary_separates_overall_and_changed_agreement_without_mutating_rm():
    records, rm = inputs()
    original = deepcopy(rm)
    rows = [
        {**records[0], **S.J.verdict_record("B", "A")},
        {**records[1], **S.J.verdict_record("tie", "tie")},
    ]
    summary = S.build_summary(rows, rm, spec())
    assert summary["cross_judge"] == {"other_judge": rm["judge"], "n": 2, "agreement": 0.5}
    assert summary["cross_judge_heldout_changed"]["n"] == 1
    assert summary["cross_judge_heldout_changed"]["agreement"] == 0
    assert summary["heldout"]["dpo_win_rate"] == 1
    assert summary["overall"]["position_consistency"] == 1
    assert rm == original


def test_failed_and_position_inconsistent_verdicts_keep_lab_semantics():
    records, rm = inputs()
    rows = [
        {**records[0], **S.J.verdict_record("B", "B")},
        {**records[1], **S.J.verdict_record("invalid", "tie")},
    ]
    summary = S.build_summary(rows, rm, spec())
    assert summary["overall"]["n"] == 1
    assert summary["overall"]["n_failed"] == 1
    assert summary["overall"]["position_consistency"] == 0
    assert summary["cross_judge"]["n"] == 1


def test_cli_writes_api_results_but_preserves_rm_statistics(monkeypatch):
    records, rm = inputs()
    text = "\n".join(json.dumps(r) for r in records) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    rm["outputs_sha256"] = digest
    original = {
        "judge": rm["judge"], "outputs_sha256": digest,
        "sanity_accuracy": 1.0, "heldout": {"n": 1, "dpo_win_rate": 0.0},
    }
    files = {
        "side_by_side.jsonl": text,
        "judge_results_rm.json": json.dumps(rm),
        "judge_summary.json": json.dumps(original),
    }
    monkeypatch.setattr(Path, "read_bytes", lambda p: files[p.name].encode())
    monkeypatch.setattr(Path, "read_text", lambda p, **kw: files[p.name])
    monkeypatch.setattr(Path, "exists", lambda p: False)
    monkeypatch.setattr(S.J, "has_judge_key", lambda provider: True)
    answers = iter(["B", "A", "tie", "tie"])
    monkeypatch.setattr(S.J, "make_caller", lambda provider, model: lambda s, u: next(answers))
    writes = {}
    monkeypatch.setattr(S, "write_json", lambda path, value: writes.update({path.name: deepcopy(value)}))
    monkeypatch.setattr(S.sys, "argv", ["judge_saved_outputs.py", "--provider", "openai", "--model", "test-model"])
    assert S.main() == 0
    assert len(writes["judge_results_api.json"]["records"]) == 2
    assert writes["judge_summary_api.json"]["heldout"]["dpo_win_rate"] == 1
    for key, value in original.items():
        assert writes["judge_summary.json"][key] == value
    assert writes["judge_summary.json"]["cross_judge"]["other_judge"] == "openai:test-model"
