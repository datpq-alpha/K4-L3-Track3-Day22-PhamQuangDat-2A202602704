#!/usr/bin/env python3
"""Reproduce the post-hoc analyses in REFLECTION section 4 from saved NB4 data.

Usage: python -X utf8 scripts/analyze_eval_diagnostics.py
Requires scipy. Reads existing results; does not generate or judge answers.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lab22 import judge as J


def main() -> None:
    path = REPO / "data/eval/judge_results_rm.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = json.loads((path.parent / "judge_summary.json").read_text(encoding="utf-8"))
    rows = data["records"]
    heldout = [r for r in rows if r["category"] == "heldout"]
    changed = [r for r in heldout if r["sft"] != r["dpo"]]
    names = summary["judge_agreement"]["judges"]
    qwen = next(n for n in names if "Qwen3" in n)
    llama = next(n for n in names if "Llama" in n)
    judges = {n: {r["id"]: r for r in data["per_judge"][n]} for n in names}
    score = {"dpo": 1.0, "tie": 0.5, "sft": 0.0}
    out = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for category in ("overall", "heldout", "helpfulness", "safety"):
        subset = rows if category == "overall" else [r for r in rows if r["category"] == category]
        assert J.summarize(subset, seed=42) == summary[category]
        out[category] = {"n": len(subset), "changed": sum(r["sft"] != r["dpo"] for r in subset)}
    out["heldout"].update({
        "winners": dict(Counter(r["winner"] for r in heldout)),
        "changed_winners": dict(Counter(r["winner"] for r in changed)),
        "tie_adjusted_win_rate": summary["heldout"]["dpo_win_rate"],
        "bootstrap_ci95": summary["heldout"]["win_rate_ci95"],
    })
    wins = sum(r["winner"] == "dpo" for r in changed)
    non_ties = [r for r in heldout if r["winner"] != "tie"]
    assert {r["id"] for r in changed} == {r["id"] for r in non_ties}
    test = binomtest(wins, len(non_ties), p=0.5, alternative="two-sided")
    out["non_tie_binomial_test"] = {"wins": wins, "n": len(non_ties), "pvalue": test.pvalue}
    differences = [score[judges[qwen][r["id"]]["winner"]] - score[judges[llama][r["id"]]["winner"]] for r in heldout]
    out["qwen_minus_llama_paired"] = {
        "mean": sum(differences) / len(differences),
        "bootstrap_ci95": J.bootstrap_ci(differences, seed=42),
        "changed_cross_tab": {f"{a}/{b}": count for (a, b), count in Counter(
            (judges[qwen][r["id"]]["winner"], judges[llama][r["id"]]["winner"]) for r in changed
        ).items()},
    }
    out["agreement"] = {}
    for category, subset in (("overall", rows), ("heldout", heldout)):
        changed_subset = [r for r in subset if r["sft"] != r["dpo"]]
        out["agreement"][category] = {
            label: {"n": len(group), "agree": sum(judges[qwen][r["id"]]["winner"] == judges[llama][r["id"]]["winner"] for r in group)}
            for label, group in (("all", subset), ("changed", changed_subset))
        }
    out["delta_reward_delta_chars_spearman"] = {}
    for name in names:
        out["delta_reward_delta_chars_spearman"][name] = {
            label: J.spearman(
                [judges[name][r["id"]]["dpo_score"] - judges[name][r["id"]]["sft_score"] for r in subset],
                [len(r["dpo"]) - len(r["sft"]) for r in subset],
            ) for label, subset in (("heldout_50", heldout), ("changed_16", changed))
        }
    out["sanity_12_of_12_clopper_pearson_ci95"] = list(binomtest(12, 12).proportion_ci(confidence_level=0.95, method="exact"))
    close = [r for r in heldout if max(len(r["sft"]), len(r["dpo"])) <= 1.2 * max(1, min(len(r["sft"]), len(r["dpo"]))) ]
    out["length_matched_heldout"] = {"n": len(close), "winners": dict(Counter(r["winner"] for r in close))}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
