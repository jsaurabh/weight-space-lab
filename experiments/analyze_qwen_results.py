"""Regrade saved texts against the published grader; make no model/API calls."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import random

from weight_space_lab.qwen import grade, save_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
grader_path = root / "third_party/qwen_p50c/published_grading.py"
spec = importlib.util.spec_from_file_location("published_qwen_grading", grader_path)
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
rows, disagreements = [], []
for condition in ["base", "published_edit"]:
    source = args.directory / f"{condition}_generations.jsonl"
    for line, content in enumerate(source.read_text().splitlines(), 1):
        record = json.loads(content)
        pred, _ = upstream.extract(record["text"])
        correct = upstream.is_correct(pred, record["answer"])
        local = grade(record["answer"], record["text"])
        if local != correct:
            disagreements.append({"id": record["id"], "sample": record["sample"], "condition": condition})
        rows.append({"condition": condition, "id": record["id"], "sample": record["sample"],
                     "correct": bool(correct), "original_remote_correct": record["correct"],
                     "generated_tokens": record["generated_tokens"], "hit_token_limit": record["hit_token_limit"],
                     "thinking_closed": record["thinking_closed"], "source_file": source.name, "source_line": line})
save_json(args.directory / "grader_validation.json", {
    "published_grader_source_sha256": hashlib.sha256(grader_path.read_bytes()).hexdigest(),
    "comparisons": len(rows), "disagreements": disagreements,
    "remote_score_changes": sum(r["correct"] != r["original_remote_correct"] for r in rows),
    "method": "published extraction and math-verify equivalence on saved texts; no regeneration",
    "software": {name: importlib.metadata.version(name) for name in ["math-verify", "sympy", "latex2sympy2-extended"]},
})
if disagreements:
    raise SystemExit("Local grader differs from published grader; see grader_validation.json")
with (args.directory / "graded_scores.jsonl").open("w") as target:
    for row in rows:
        target.write(json.dumps(row) + "\n")
summary = {}
for condition in ["base", "published_edit"]:
    group = [r for r in rows if r["condition"] == condition]
    summary[condition] = {"correct": sum(r["correct"] for r in group), "total": len(group),
                          "accuracy": sum(r["correct"] for r in group) / len(group),
                          "hit_token_limit": sum(r["hit_token_limit"] for r in group),
                          "mean_generated_tokens": sum(r["generated_tokens"] for r in group) / len(group)}
problems = json.loads((args.directory / "problems.json").read_text())
deltas = []
for problem in problems:
    rates = {}
    for condition in ["base", "published_edit"]:
        group = [r for r in rows if r["id"] == problem["unique_id"] and r["condition"] == condition]
        rates[condition] = sum(r["correct"] for r in group) / len(group)
    deltas.append(rates["published_edit"] - rates["base"])
rng = random.Random(20261102)
bootstrap = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000))
summary.update({"mean_accuracy_difference": sum(deltas) / len(deltas),
                "problem_cluster_bootstrap_95_interval": [bootstrap[249], bootstrap[9749]],
                "bootstrap_replicates": 10000,
                "problem_accuracy_differences": deltas,
                "problem_subjects": dict(Counter(r["subject"] for r in problems))})
base = {(r["id"], r["sample"]): r for r in rows if r["condition"] == "base"}
edited = {(r["id"], r["sample"]): r for r in rows if r["condition"] == "published_edit"}
assert base.keys() == edited.keys()
summary["paired_sample_wins"] = sum(not base[k]["correct"] and edited[k]["correct"] for k in base)
summary["paired_sample_losses"] = sum(base[k]["correct"] and not edited[k]["correct"] for k in base)
save_json(args.directory / "analysis.json", summary)
print(json.dumps(summary, indent=2))
