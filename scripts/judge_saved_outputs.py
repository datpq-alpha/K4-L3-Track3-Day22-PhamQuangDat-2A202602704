#!/usr/bin/env python3
"""Judge saved NB4 answers through an API, without loading a GPU model.

python -X utf8 scripts/judge_saved_outputs.py --check
python -X utf8 scripts/judge_saved_outputs.py --limit 2
python -X utf8 scripts/judge_saved_outputs.py

Uses the provider/model/key configured in .env or the environment. Checkpoints
each pair under .cache/ and refuses to mix different inputs or judge settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lab22 import config as C
from lab22 import judge as J

FIELDS = ("id", "category", "prompt", "sft", "dpo")


def validate_inputs(records: list[dict], rm: dict, digest: str) -> None:
    if not records or any(not isinstance(r.get(k), str) or not r[k] for r in records for k in FIELDS):
        raise ValueError("Saved answers must have nonempty id/category/prompt/sft/dpo fields.")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Saved answers contain duplicate IDs.")
    if rm.get("outputs_sha256") != digest:
        raise ValueError("RM judgments were produced for different saved answers (SHA256 mismatch).")
    indexed = {r["id"]: r for r in rm["records"]}
    if len(indexed) != len(rm["records"]) or set(indexed) != {r["id"] for r in records}:
        raise ValueError("RM and saved answers must have the same unique IDs.")
    if any(indexed[r["id"]].get(k) != r[k] for r in records for k in FIELDS):
        raise ValueError("RM input text differs from the saved answers.")


def resume_records(saved: dict, spec: dict, records: list[dict]) -> dict[str, dict]:
    if not saved:
        return {}
    if saved.get("run") != spec:
        raise ValueError("Checkpoint has different inputs/model/settings; use a separate --checkpoint path.")
    original = {r["id"]: r for r in records}
    done = {}
    for row in saved["records"]:
        identifier = row["id"]
        if identifier in done or identifier not in original:
            raise ValueError("Checkpoint contains an unknown or duplicate ID.")
        if any(row.get(k) != original[identifier][k] for k in FIELDS):
            raise ValueError("Checkpoint answer text differs from the saved answers.")
        expected = J.verdict_record(row["order_sft_first"], row["order_dpo_first"])
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("Checkpoint verdict does not match its two A/B orders.")
        done[identifier] = row
    return done


def judge_remaining(records, done, call, save, limit=None, retry_failed=False):
    remaining = [r for r in records if r["id"] not in done or (
        retry_failed and done[r["id"]]["winner"] == "failed"
    )]
    for row in remaining[:limit]:
        done[row["id"]] = {**row, **J.judge_pair(row["prompt"], row["sft"], row["dpo"], call)}
        save([done[r["id"]] for r in records if r["id"] in done])
        print(f"[{len(done)}/{len(records)}] {row['id']}: {done[row['id']]['winner']}", flush=True)
    return [done[r["id"]] for r in records if r["id"] in done]


def build_summary(judged: list[dict], rm: dict, spec: dict) -> dict:
    def cross(rows):
        return {"other_judge": rm["judge"], **J.agreement(rows, rm["records"])}

    changed = [r for r in judged if r["sft"] != r["dpo"]]
    return {
        "judge": spec["judge"], "outputs_sha256": spec["outputs_sha256"],
        "sanity_accuracy": None, "sanity": None,
        "overall": J.summarize(judged, seed=spec["seed"]),
        **{category: J.summarize([r for r in judged if r["category"] == category], seed=spec["seed"])
           for category in ("heldout", "helpfulness", "safety")},
        "cross_judge": cross(judged),
        "cross_judge_heldout": cross([r for r in judged if r["category"] == "heldout"]),
        "cross_judge_changed": cross(changed),
        "cross_judge_heldout_changed": cross([r for r in changed if r["category"] == "heldout"]),
        "run": spec,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate configuration/data; no API calls or writes")
    parser.add_argument("--limit", type=int, help="judge at most this many new pairs, then save progress")
    parser.add_argument("--retry-failed", action="store_true", help="retry pairs previously marked failed")
    parser.add_argument("--provider", default=C.JUDGE_PROVIDER, choices=tuple(J.API_KEYS))
    parser.add_argument("--model", default=C.JUDGE_MODEL)
    parser.add_argument("--eval-dir", type=Path, default=REPO / "data/eval")
    parser.add_argument("--checkpoint", type=Path, default=REPO / ".cache/api-judge-progress.json")
    args = parser.parse_args()
    if args.provider not in J.API_KEYS:
        parser.error("Set JUDGE_PROVIDER to an API provider in .env or pass --provider.")
    if not args.model:
        parser.error("Set JUDGE_MODEL in .env or pass --model.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    outputs = args.eval_dir / "side_by_side.jsonl"
    digest = hashlib.sha256(outputs.read_bytes()).hexdigest()
    records = [json.loads(line) for line in outputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    rm = json.loads((args.eval_dir / "judge_results_rm.json").read_text(encoding="utf-8"))
    validate_inputs(records, rm, digest)
    # Keep the main summary as RM evidence for the existing reflection/diagnostics.
    main_path = args.eval_dir / "judge_summary.json"
    main_summary = json.loads(main_path.read_text(encoding="utf-8"))
    if main_summary.get("judge") != rm["judge"] or main_summary.get("outputs_sha256") != digest:
        raise ValueError("Main summary must contain the matching RM results before adding cross-judge metrics.")
    spec = {
        "judge": f"{args.provider}:{args.model}", "outputs_sha256": digest,
        "seed": C.SEED, "max_completion_tokens": 200,
        "judge_prompt_sha256": hashlib.sha256((J.JUDGE_SYSTEM + J.JUDGE_TEMPLATE).encode()).hexdigest(),
        "orders_per_pair": 2, "invalid_verdict_retries_per_order": 1,
        "temperature": "provider default (same caller as NB4)",
    }
    saved = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint.exists() else {}
    done = resume_records(saved, spec, records)
    remaining = sum(r["id"] not in done or (
        args.retry_failed and done[r["id"]]["winner"] == "failed"
    ) for r in records)
    print(f"Judge: {spec['judge']}")
    print(f"Answers: {len(records)} pairs; held-out: {sum(r['category'] == 'heldout' for r in records)}")
    print(f"RM match: OK; SHA256: {digest}")
    print(f"Checkpoint: {len(done)} completed; {remaining} pairs remaining")
    print(f"{J.API_KEYS[args.provider]}: {'present' if J.has_judge_key(args.provider) else 'missing'}")
    if args.check:
        print("Configuration/data check finished. No API calls made.")
        return 0
    if remaining and not J.has_judge_key(args.provider):
        raise ValueError(f"Set {J.API_KEYS[args.provider]} in the ignored .env file, then rerun.")
    call = J.make_caller(args.provider, args.model) if remaining else None
    judged = judge_remaining(
        records, done, call,
        lambda rows: write_json(args.checkpoint, {"run": spec, "records": rows}),
        limit=args.limit, retry_failed=args.retry_failed,
    )
    if len(judged) < len(records):
        print("Partial run saved in checkpoint. Rerun without --limit to finish; final summaries are not written yet.")
        return 0
    summary = build_summary(judged, rm, spec)
    write_json(args.eval_dir / "judge_results_api.json", {
        "judge": spec["judge"], "outputs_sha256": digest,
        "records": judged, "per_judge": {}, "run": spec,
    })
    write_json(args.eval_dir / "judge_summary_api.json", summary)
    for key in ("cross_judge", "cross_judge_heldout", "cross_judge_changed", "cross_judge_heldout_changed"):
        main_summary[key] = {**summary[key], "other_judge": spec["judge"]}
    main_summary["api_summary_file"] = "judge_summary_api.json"
    write_json(main_path, main_summary)
    print(json.dumps({key: summary[key] for key in (
        "judge", "overall", "heldout", "cross_judge", "cross_judge_heldout_changed",
    )}, ensure_ascii=False, indent=2))
    print("Saved judge_results_api.json and judge_summary_api.json; added cross-judge metrics to the RM summary.")
    return 1 if summary["overall"]["n_failed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as error:
        # API exception messages can contain credential fragments; print type/status only.
        print(f"Request stopped: {type(error).__name__}; HTTP {getattr(error, 'status_code', 'n/a')}. "
              "Progress is saved; check credentials, quota or network, then rerun.", file=sys.stderr)
        raise SystemExit(1) from None
