import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from weight_space_lab.discovery import Budget, run_task, split_examples, task_prompt
from weight_space_lab.evaluate import state_digest
from weight_space_lab.programs import parity, addition
from weight_space_lab.proposals import load_proposal


def numeric_proposal(model):
    return {"name": "offline_test_fixture", "explanation": "Known fixture for protocol tests only.",
            "patches": [{"tensor": name, "axes": [list(range(n)) for n in tensor.shape],
                         "values": tensor.flatten().tolist()} for name, tensor in model.named_parameters()]}


class Discovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_numeric_roundtrip_both_architectures(self):
        for task, factory in (("parity", parity), ("addition", addition)):
            reference = factory()
            loaded = load_proposal(task, numeric_proposal(reference))
            self.assertEqual(state_digest(reference), state_digest(loaded))

    def test_broadcast_and_cartesian_order(self):
        proposal = {"name": "test", "explanation": "test", "patches": [
            {"tensor": "query.weight", "axes": [[0, 2], [1, 2]], "values": [1, 2, 3, 4]},
            {"tensor": "readout.bias", "axes": [[0, 1]], "values": [5]},
        ]}
        model = load_proposal("parity", proposal)
        self.assertEqual(model.query.weight.tolist(), [[0, 1, 2], [0, 0, 0], [0, 3, 4]])
        self.assertEqual(model.readout.bias.tolist(), [5, 5, 0, 0])

    def test_malformed_and_nonfinite_submissions_rejected(self):
        base = {"name": "test", "explanation": "test", "patches": [
            {"tensor": "query.weight", "axes": [[0], [0]], "values": [1]}]}
        mutations = [
            ("tensor", "__import__('os').system('false')"),
            ("axes", [[-1], [0]]), ("axes", [[3], [0]]),
            ("axes", [[0, 0], [0]]), ("axes", [[True], [0]]),
            ("axes", [[]]), ("axes", [[0]]),
            ("values", [float("nan")]), ("values", [float("inf")]),
            ("values", [1e7]), ("values", [True]), ("values", ["1+1"]),
            ("values", []), ("values", [1, 2]),
        ]
        for key, value in mutations:
            proposal = copy.deepcopy(base)
            proposal["patches"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                load_proposal("parity", proposal)

    def test_split_disjoint_complete_reproducible(self):
        for task, size in (("parity", 2046), ("addition", 10000)):
            dev, holdout = split_examples(task, 42)
            self.assertEqual(len(dev), 200)
            self.assertEqual(len(dev) + len(holdout), size)
            self.assertFalse({r[0] for r in dev} & {r[0] for r in holdout})
            self.assertEqual((dev, holdout), split_examples(task, 42))

    def test_prompt_contains_forward_but_no_reference_solution(self):
        for task in ("parity", "addition"):
            prompt = task_prompt(task)
            self.assertIn("def forward", prompt)
            self.assertIn("ASSIGNMENT FORMAT", prompt)
            self.assertNotIn("def parity", prompt)
            self.assertNotIn("def addition", prompt)
            self.assertNotIn("zigzag", prompt)
            self.assertNotIn("floor(sum", prompt)

    def test_cost_bound_and_usage_accounting(self):
        request = {"model": "gpt-6-astra", "max_output_tokens": 1000}
        with self.assertRaises(ValueError):
            Budget(0.001).check_request(request)
        budget = Budget(1)
        self.assertLess(budget.check_request(request), 1)
        self.assertEqual(budget.record("gpt-6-astra", {"input_tokens": 1000, "output_tokens": 1000}), 0.06)
        self.assertAlmostEqual(budget.spent, 0.06)
        cached = {"input_tokens": 1000, "output_tokens": 1000,
                  "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 300}}
        self.assertEqual(Budget(1).record("gpt-6-astra", cached), 0.05895)
        with self.assertRaises(ValueError):
            budget.record("gpt-6-astra", None)

    def test_protocol_freezes_candidate_and_saves_final_without_feedback_leak(self):
        proposal = numeric_proposal(parity())
        response_data = {"id": "offline-fixture", "model": "gpt-6-astra", "status": "completed",
                         "usage": {"input_tokens": 100, "output_tokens": 200}}
        response = SimpleNamespace(**response_data, output_text=json.dumps(proposal),
                                   model_dump=lambda **kwargs: response_data)
        requests = []

        def create(**request):
            requests.append(copy.deepcopy(request))
            return response

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "parity"
            result = run_task(client, "parity", destination, Budget(1), model_name="gpt-6-astra",
                              attempts=3, max_output_tokens=1000, seed=42)
            self.assertEqual(result["full_correct"], 2046)
            self.assertEqual(result["holdout_total"], 1846)
            self.assertEqual(result["attempt_count"], 1)
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(requests[0]["input"]), 2)
            self.assertNotIn("tools", requests[0])
            self.assertFalse(requests[0]["store"])
            self.assertTrue((destination / "selected.json").is_file())
            selection = json.loads((destination / "selection.json").read_text())
            final = json.loads((destination / "final_holdout.json").read_text())
            self.assertEqual(selection["state_sha256"], final["state_sha256"])


if __name__ == "__main__":
    unittest.main()
