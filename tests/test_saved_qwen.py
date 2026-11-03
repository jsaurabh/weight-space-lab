import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "results/qwen/g4-pilot"


class SavedQwen(unittest.TestCase):
    def test_completed_g4_run_has_frozen_sources_and_complete_paired_samples(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["colab_accelerator_requested"], "G4")
        self.assertEqual(manifest["sampling"]["max_tokens"], 32768)
        for name, digest in manifest["code_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "source" / name).read_bytes()).hexdigest(), digest)
        pairs = []
        for condition in ["base", "published_edit"]:
            records = [json.loads(line) for line in (ROOT / f"{condition}_generations.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 64)
            self.assertEqual({r["id"] for r in records}, set(manifest["selected_ids"]))
            self.assertTrue(all({r["sample"] for r in records if r["id"] == identifier} == {0, 1, 2, 3}
                                for identifier in manifest["selected_ids"]))
            pairs.append([(r["id"], r["sample"]) for r in records])
        self.assertEqual(*pairs)
        self.assertTrue(json.loads((ROOT / "execution_record.json").read_text())["session_stopped"])

    def test_audited_scores_and_summary_agree(self):
        records = [json.loads(line) for line in (ROOT / "graded_scores.jsonl").read_text().splitlines()]
        summary = json.loads((ROOT / "analysis.json").read_text())
        for condition, expected in [("base", 4), ("published_edit", 28)]:
            group = [r for r in records if r["condition"] == condition]
            correct = sum(r["correct"] for r in group)
            self.assertEqual(correct, expected)
            self.assertEqual(summary[condition]["correct"], correct)
            self.assertEqual(summary[condition]["total"], len(group))
        validation = json.loads((ROOT / "grader_validation.json").read_text())
        self.assertEqual(validation["comparisons"], 128)
        self.assertEqual(validation["disagreements"], [])
        self.assertEqual(validation["remote_score_changes"], 1)


if __name__ == "__main__":
    unittest.main()
