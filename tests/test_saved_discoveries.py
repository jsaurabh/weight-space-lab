"""Regression checks on the actual, unedited API-generated numerical programs."""

import json
from pathlib import Path
import unittest

import torch

from weight_space_lab.evaluate import encode, evaluate, state_digest
from weight_space_lab.proposals import load_proposal
from weight_space_lab.specs import VOCABS

ROOT = Path(__file__).resolve().parents[1] / "results/discovery/astra-initial"


class SavedDiscoveries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def load(self, task):
        return load_proposal(task, json.loads((ROOT / task / "selected.json").read_text()))

    def test_saved_api_proposals_replay_entire_domains_with_identical_digests(self):
        for task in ("parity", "addition"):
            with self.subTest(task=task):
                model = self.load(task)
                selection = json.loads((ROOT / task / "selection.json").read_text())
                report = evaluate(model, task, provenance="saved_api_regression")
                self.assertEqual(report["correct"], report["total"])
                self.assertEqual(state_digest(model), selection["state_sha256"])

    def test_api_parity_explanation_matches_intermediate_channels(self):
        model = self.load("parity")
        for count in range(11):
            prompt = ("1" * count).rjust(10, "_") + "="
            _, trace = model(encode([prompt], VOCABS["parity"]), trace=True)
            self.assertAlmostEqual(trace["after_attention"][0, -1, 1].item(), count)
            self.assertAlmostEqual(trace["after_mlp"][0, -1, 2].item(), 2 * (count % 2) - 1)

    def test_api_addition_explanation_matches_remainder_and_three_active_units(self):
        model = self.load("addition")
        tokens = encode(["89+76="], VOCABS["addition"])
        for remainder, digit in ((165, 1), (65, 6), (5, 5)):
            logits, trace = model(tokens, trace=True)
            self.assertAlmostEqual(trace["after_attention"][0, -1, 8].item(), remainder)
            self.assertEqual(torch.count_nonzero(trace["hidden"][0, -1]).item(), 1)
            self.assertEqual(torch.count_nonzero(trace["hidden"][0, -1, 3:]).item(), 0)
            predicted = logits[:, -1].argmax(-1, keepdim=True)
            self.assertEqual(predicted.item(), digit)
            tokens = torch.cat((tokens, predicted), dim=1)

    def test_api_addition_requires_feedback_from_earlier_output_digits(self):
        model = self.load("addition")
        with torch.no_grad():
            model.attention_out.weight[8, 16] = 0
            model.attention_out.weight[8, 24] = 0
        answer = model.generate(encode(["89+76="], VOCABS["addition"]), 3)[0, -3:]
        self.assertEqual(answer.tolist(), [1, 9, 9])


if __name__ == "__main__":
    unittest.main()
