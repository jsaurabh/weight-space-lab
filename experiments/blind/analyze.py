"""Audit and regrade the frozen blind trial without making model or API calls."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    frozen = json.loads((directory / "frozen.json").read_text())
    assert frozen["test_data_present_at_freeze"] is False
    seal = json.loads((directory / "test_seal.json").read_text())
    test_path = directory / "sealed_test.json"
    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == seal["sha256"]
    problems = json.loads(test_path.read_text())
    ids = {p["unique_id"] for p in problems}
    assert len(ids) == len(problems) == 32
    root = Path(__file__).resolve().parents[2]
    grader_path = root / "third_party/qwen_p50c/published_grading.py"
    spec = importlib.util.spec_from_file_location("postfreeze_grader", grader_path)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    all_rows, changes, summary = [], [], {}
    for mode in ["base", "current"]:
        data = json.loads((directory / ("final_" + mode + ".json")).read_text())
        assert data["max_tokens"] == 32768
        records = data["records"]
        assert len(records) == 64
        assert {(r["id"], r["sample"]) for r in records} == {(i, s) for i in ids for s in [0, 1]}
        for r in records:
            pred, _ = grader.extract(r["text"])
            score = bool(grader.is_correct(pred, r["answer"]))
            row = {k: v for k, v in r.items() if k not in {"text", "question", "answer", "correct"}}
            row.update(condition=mode, correct=score, original_remote_correct=r["correct"],
                       closed_and_correct=score and r["thinking_closed"])
            all_rows.append(row)
            if score != r["correct"]:
                changes.append({"condition": mode, "id": r["id"], "sample": r["sample"],
                                "remote": r["correct"], "audited": score})
        group = [r for r in all_rows if r["condition"] == mode]
        summary[mode] = {"correct": sum(r["correct"] for r in group), "total": len(group),
            "accuracy": statistics.mean(r["correct"] for r in group),
            "closed_and_correct": sum(r["closed_and_correct"] for r in group),
            "thinking_closed": sum(r["thinking_closed"] for r in group),
            "hit_token_limit": sum(r["hit_token_limit"] for r in group),
            "mean_tokens": statistics.mean(r["tokens"] for r in group),
            "generation_seconds": data["seconds"]}
    per_problem = []
    for problem in problems:
        row = {"id": problem["unique_id"], "subject": problem["subject"]}
        for mode in ["base", "current"]:
            row[mode] = sum(r["correct"] for r in all_rows if r["condition"] == mode and r["id"] == row["id"]) / 2
        row["difference"] = row["current"] - row["base"]
        per_problem.append(row)
    differences = [r["difference"] for r in per_problem]
    rng = random.Random(20261103)
    boot = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    base = {(r["id"], r["sample"]): r for r in all_rows if r["condition"] == "base"}
    current = {(r["id"], r["sample"]): r for r in all_rows if r["condition"] == "current"}
    summary.update(mean_accuracy_difference=statistics.mean(differences),
        problem_cluster_bootstrap_95_interval=[boot[249], boot[9749]], bootstrap_replicates=10000,
        problem_subjects=dict(Counter(p["subject"] for p in problems)),
        paired_sample_wins=sum(not base[k]["correct"] and current[k]["correct"] for k in base),
        paired_sample_losses=sum(base[k]["correct"] and not current[k]["correct"] for k in base))
    write(directory / "analysis.json", summary)
    write(directory / "per_problem.json", per_problem)
    write(directory / "grader_validation.json", {"regraded_after_freeze": True, "comparisons": len(all_rows),
        "remote_score_changes": changes, "published_grader_source_sha256": hashlib.sha256(grader_path.read_bytes()).hexdigest()})
    (directory / "graded_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
