"""Audit the frozen three-condition comparison using saved texts only."""
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
    root = args.directory
    protocol = json.loads((root / "protocol.json").read_text())
    state = json.loads((root / "run.json").read_text())
    assert state["status"] == "completed" and state["g4_stopped"]
    audit = {}
    for label in protocol["models"]:
        frozen = json.loads((root / label / "frozen.json").read_text())
        assert not frozen["test_data_present_at_freeze"]
        assert hashlib.sha256((root / label / "selected.safetensors").read_bytes()).hexdigest() == frozen["selected_file_sha256"]
        responses = [json.loads(p.read_text()) for p in sorted((root / label / "turns").glob("*/response.json"))]
        assert all(r["created_at"] <= frozen["frozen_at_unix"] for r in responses)
        audit[label] = {"frozen": frozen, "api_requests": len(responses), "api_requests_after_freeze": 0}
    first = {}
    for label in protocol["models"]:
        request = json.loads((root / label / "turns/001/request.json").read_text())
        first[label] = {k: v for k, v in request.items() if k != "model"}
    assert first["astra"] == first["gpt55"], "Initial editor context or tool budgets differed"
    evaluation = root / "evaluation"
    problems_path = evaluation / "problems.json"
    assert hashlib.sha256(problems_path.read_bytes()).hexdigest() == protocol["test"]["sha256"]
    problems = json.loads(problems_path.read_text())
    grader_path = Path(__file__).resolve().parents[2] / "third_party/qwen_p50c/published_grading.py"
    spec = importlib.util.spec_from_file_location("comparison_grader", grader_path)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    scores, summary, changes = [], {}, []
    expected = {(p["unique_id"], s) for p in problems for s in range(2)}
    for condition in ["base", "astra", "gpt55"]:
        data = json.loads((evaluation / (condition + ".json")).read_text())
        assert {(r["id"], r["sample"]) for r in data["records"]} == expected
        group = []
        for r in data["records"]:
            prediction, _ = grader.extract(r["text"])
            correct = bool(grader.is_correct(prediction, r["answer"]))
            row = {k: v for k, v in r.items() if k not in {"text", "question", "answer", "correct"}}
            row.update(condition=condition, correct=correct,
                       finished_and_correct=correct and not r["hit_token_limit"],
                       closed_and_correct=correct and r["thinking_closed"])
            group.append(row)
            if correct != r["correct"]:
                changes.append({"condition": condition, "id": r["id"], "sample": r["sample"],
                                "original": r["correct"], "audited": correct})
        scores.extend(group)
        summary[condition] = {"correct": sum(r["correct"] for r in group), "total": len(group),
            "accuracy": statistics.mean(r["correct"] for r in group),
            "finished_and_correct": sum(r["finished_and_correct"] for r in group),
            "thinking_closed": sum(r["thinking_closed"] for r in group),
            "hit_token_limit": sum(r["hit_token_limit"] for r in group),
            "mean_tokens": statistics.mean(r["tokens"] for r in group), "seconds": data["seconds"]}
    pair_results = {}
    for before, after in [("base", "astra"), ("base", "gpt55"), ("astra", "gpt55")]:
        a = {(r["id"], r["sample"]): r for r in scores if r["condition"] == before}
        b = {(r["id"], r["sample"]): r for r in scores if r["condition"] == after}
        differences = [statistics.mean(int(b[(p["unique_id"], s)]["correct"]) - int(a[(p["unique_id"], s)]["correct"]) for s in range(2)) for p in problems]
        rng = random.Random(20261104)
        boot = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
        pair_results[after + "_minus_" + before] = {"accuracy_difference": statistics.mean(differences),
            "problem_cluster_bootstrap_95_interval": [boot[249], boot[9749]],
            "paired_rescues": sum(not a[k]["correct"] and b[k]["correct"] for k in a),
            "paired_losses": sum(a[k]["correct"] and not b[k]["correct"] for k in a)}
    write(root / "analysis.json", {"conditions": summary, "comparisons": pair_results,
        "problem_subjects": dict(Counter(p["subject"] for p in problems)), "bootstrap_replicates": 10000})
    write(root / "blinding_audit.json", {"identical_initial_inputs_and_tools": True, "editors": audit,
        "test_uploaded_after_both_frozen": True, "prior_solution_provided": False})
    write(root / "grader_validation.json", {"comparisons": len(scores), "changes": changes,
        "grader_sha256": hashlib.sha256(grader_path.read_bytes()).hexdigest(), "regraded_after_both_frozen": True})
    (root / "graded_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    print(json.dumps(summary, indent=2))
    print(json.dumps(pair_results, indent=2))


if __name__ == "__main__":
    main()
