"""Consistency checks on the completed, independently frozen discovery trial."""
import hashlib
import json
from pathlib import Path
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "results/qwen/blind-discovery"


@unittest.skipUnless((SAVED / "analysis.json").exists(), "Completed blind trial not present")
class SavedBlindQwen(unittest.TestCase):
    def test_initial_vm_payload_contains_only_generic_tools_and_practice(self):
        manifest = json.loads((SAVED / "payload_manifest.json").read_text())
        with tarfile.open(SAVED / "initial_payload.tar.gz") as archive:
            self.assertEqual(set(archive.getnames()), {"backend.py", "protocol.py", "practice.json"})
            for name, expected in manifest.items():
                self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((SAVED / "practice.json").read_bytes()).hexdigest(), manifest["practice.json"])
        evidence = json.loads((SAVED / "evidence_manifest.json").read_text())
        self.assertEqual(len(evidence["files"]), evidence["file_count"])
        for name, info in evidence["files"].items():
            self.assertEqual(hashlib.sha256((SAVED / name).read_bytes()).hexdigest(), info["sha256"], name)

    def test_complete_trial_keeps_test_sealed_until_freeze(self):
        run = json.loads((SAVED / "run.json").read_text())
        frozen = json.loads((SAVED / "frozen.json").read_text())
        seal = json.loads((SAVED / "test_seal.json").read_text())
        self.assertEqual(run["status"], "completed")
        self.assertTrue(run["g4_stopped"])
        self.assertFalse(run["prior_conversation_provided"])
        self.assertFalse(run["external_repair_files_provided"])
        self.assertFalse(frozen["test_data_present_at_freeze"])
        self.assertEqual(hashlib.sha256((SAVED / "selected.safetensors").read_bytes()).hexdigest(), frozen["selected_file_sha256"])
        self.assertEqual(hashlib.sha256((SAVED / "sealed_test.json").read_bytes()).hexdigest(), seal["sha256"])
        held = json.loads((SAVED / "sealed_test.json").read_text())
        pilot = json.loads((ROOT / "results/qwen/g4-pilot/problems.json").read_text())
        self.assertTrue({p["unique_id"] for p in held}.isdisjoint(p["unique_id"] for p in pilot))
        first = json.loads((SAVED / "turns/001/request.json").read_text())
        self.assertEqual([r["role"] for r in first["input"]], ["system", "user"])
        self.assertEqual({t["name"] for t in first["tools"]}, {"practice", "generate", "probe", "python_analysis", "finish"})
        for marker in ["p50c", "RikiyaT", "layers.15", "layers.19", "130188"]:
            self.assertNotIn(marker, json.dumps(first))
        for path in (SAVED / "turns").glob("*/response.json"):
            response = json.loads(path.read_text())
            self.assertLessEqual(response["created_at"], frozen["frozen_at_unix"])

    def test_paired_outputs_and_audited_summary_agree(self):
        summary = json.loads((SAVED / "analysis.json").read_text())
        rows = [json.loads(line) for line in (SAVED / "graded_scores.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 128)
        keys = {}
        for mode in ["base", "current"]:
            group = [r for r in rows if r["condition"] == mode]
            keys[mode] = {(r["id"], r["sample"]) for r in group}
            self.assertEqual(len(keys[mode]), 64)
            self.assertEqual(sum(r["correct"] for r in group), summary[mode]["correct"])
            self.assertEqual(sum(r["hit_token_limit"] for r in group), summary[mode]["hit_token_limit"])
            self.assertEqual(sum(r["closed_and_correct"] for r in group), summary[mode]["closed_and_correct"])
            self.assertEqual(sum(r["finished_and_correct"] for r in group), summary[mode]["finished_and_correct"])
        self.assertEqual(keys["base"], keys["current"])
        self.assertAlmostEqual(summary["current"]["accuracy"] - summary["base"]["accuracy"], summary["mean_accuracy_difference"])


if __name__ == "__main__":
    unittest.main()
