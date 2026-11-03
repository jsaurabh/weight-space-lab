"""Integrity checks for the completed matched editor experiment."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "results/comparison/qwen-astra-vs-gpt55-20261103"


def read(path):
    return json.loads(path.read_text())


@unittest.skipUnless((SAVED / "analysis.json").exists(), "Completed editor comparison not present")
class SavedQwenComparison(unittest.TestCase):
    def test_same_starting_inputs_and_sealed_evaluation(self):
        protocol, state = read(SAVED / "protocol.json"), read(SAVED / "run.json")
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["g4_stopped"])
        uploaded = datetime.fromisoformat(state["heldout_uploaded_utc"]).timestamp()
        starts = []
        for label, model in protocol["models"].items():
            frozen = read(SAVED / label / "frozen.json")
            self.assertFalse(frozen["test_data_present_at_freeze"])
            self.assertLess(frozen["frozen_at_unix"], uploaded)
            selected = SAVED / label / "selected.safetensors"
            self.assertEqual(hashlib.sha256(selected.read_bytes()).hexdigest(), frozen["selected_file_sha256"])
            first = read(SAVED / label / "turns/001/request.json")
            self.assertEqual(first.pop("model"), model)
            starts.append(first)
            for response in (SAVED / label / "turns").glob("*/response.json"):
                self.assertLessEqual(read(response)["created_at"], frozen["frozen_at_unix"])
        self.assertEqual(starts[0], starts[1])
        test = SAVED / "evaluation/problems.json"
        self.assertEqual(hashlib.sha256(test.read_bytes()).hexdigest(), protocol["test"]["sha256"])
        ids = {p["unique_id"] for p in read(test)}
        self.assertEqual(len(ids), 32)
        self.assertTrue(ids.isdisjoint(protocol["test"]["excluded_prior_pilot_ids"]))

    def test_all_conditions_have_matching_samples_and_summary_scores(self):
        analysis = read(SAVED / "analysis.json")
        scores = [json.loads(line) for line in (SAVED / "graded_scores.jsonl").read_text().splitlines()]
        self.assertEqual(len(scores), 192)
        expected = {(p["unique_id"], sample) for p in read(SAVED / "evaluation/problems.json") for sample in range(2)}
        accuracies = {}
        for label in ["base", "astra", "gpt55"]:
            group = [r for r in scores if r["condition"] == label]
            self.assertEqual(len(group), 64)
            self.assertEqual({(r["id"], r["sample"]) for r in group}, expected)
            raw = read(SAVED / "evaluation" / (label + ".json"))["records"]
            self.assertEqual(len(raw), 64)
            self.assertEqual({(r["id"], r["sample"]) for r in raw}, expected)
            summary = analysis["conditions"][label]
            for metric in ["correct", "finished_and_correct", "hit_token_limit", "thinking_closed"]:
                self.assertEqual(sum(r[metric] for r in group), summary[metric])
            accuracies[label] = sum(r["correct"] for r in group) / 64
        for before, after in [("base", "astra"), ("base", "gpt55"), ("astra", "gpt55")]:
            self.assertAlmostEqual(accuracies[after] - accuracies[before], analysis["comparisons"][after + "_minus_" + before]["accuracy_difference"])

    def test_saved_evidence_checksums(self):
        manifest = read(SAVED / "evidence_manifest.json")
        self.assertEqual(len(manifest["files"]), manifest["file_count"])
        for name, info in manifest["files"].items():
            path = SAVED / name
            self.assertEqual(path.stat().st_size, info["bytes"], name)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), info["sha256"], name)


if __name__ == "__main__":
    unittest.main()
