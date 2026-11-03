import tempfile
from pathlib import Path
import unittest

import torch

from weight_space_lab.evaluate import encode, evaluate, examples, state_digest
from weight_space_lab.model import Config, TinyTransformer
from weight_space_lab.programs import ADDITION_VOCAB, PARITY_VOCAB, addition, parity


class Experiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parity_entire_domain(self):
        report = evaluate(parity(), "parity")
        self.assertEqual((report["correct"], report["total"]), (2046, 2046))
        self.assertEqual(len(report["groups"]), 10)

    def test_addition_entire_domain(self):
        report = evaluate(addition(), "addition")
        self.assertEqual((report["correct"], report["total"]), (10000, 10000))
        self.assertEqual(set(report["groups"]), {"carry", "no_ones_carry"})

    def test_parity_mechanism(self):
        model = parity()
        for count in range(11):
            prompt = ("1" * count).rjust(10, "_") + "="
            _, trace = model(encode([prompt], PARITY_VOCAB), trace=True)
            self.assertAlmostEqual(trace["after_attention"][0, -1, 1].item(), count)
            self.assertAlmostEqual(trace["after_mlp"][0, -1, 2].item(), count % 2)

    def test_addition_routes_digits_and_extracts_place_values(self):
        model = addition()
        tokens = encode(["89+76="], ADDITION_VOCAB)
        for expected_digit in (1, 6, 5):
            logits, trace = model(tokens, trace=True)
            routed = trace["after_attention"][0, -1, 20:24]
            torch.testing.assert_close(routed, torch.tensor([8., 9., 7., 6.], dtype=torch.float64))
            self.assertAlmostEqual(trace["after_mlp"][0, -1, 24].item(), expected_digit)
            token = logits[:, -1].argmax(-1, keepdim=True)
            self.assertEqual(token.item(), expected_digit)
            tokens = torch.cat((tokens, token), dim=1)

    def test_ablating_count_path_breaks_parity(self):
        model = parity()
        with torch.no_grad():
            model.attention_out.weight.zero_()
        report = evaluate(model, "parity")
        self.assertEqual(report["correct"], 1023)

    def test_ablating_digit_route_breaks_addition(self):
        model = addition()
        with torch.no_grad():
            model.attention_out.weight[23].zero_()
        prediction = model.generate(encode(["00+01="], ADDITION_VOCAB), 3)[0, -3:]
        self.assertEqual(prediction.tolist(), [0, 0, 0])

    def test_future_tokens_cannot_change_earlier_logits(self):
        model = addition()
        prefix = encode(["12+34="], ADDITION_VOCAB)
        longer = encode(["12+34=99"], ADDITION_VOCAB)
        torch.testing.assert_close(model(prefix), model(longer)[:, :6])

    def test_saved_weights_round_trip_and_full_storage_accounting(self):
        model = addition()
        inventory = model.inventory()
        self.assertEqual(inventory["buffer_elements"], 0)
        self.assertEqual(inventory["stored_elements"], sum(v.numel() for v in model.state_dict().values()))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "weights.pt"
            torch.save({"config": inventory["config"], "state_dict": model.state_dict()}, target)
            saved = torch.load(target, map_location="cpu", weights_only=True)
            restored = TinyTransformer(Config(**saved["config"]))
            restored.load_state_dict(saved["state_dict"])
            self.assertEqual(state_digest(model), state_digest(restored))

    def test_domain_and_batch_validation(self):
        self.assertEqual(len(list(examples("addition"))), 10000)
        with self.assertRaises(ValueError):
            evaluate(parity(), "parity", batch_size=0)
        with self.assertRaises(ValueError):
            list(examples("unknown"))
        with self.assertRaises(ValueError):
            parity()(torch.zeros((1, 12), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
