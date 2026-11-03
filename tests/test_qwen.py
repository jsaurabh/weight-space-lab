import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from weight_space_lab.qwen import PresencePenalty, grade, last_boxed, read_edit, rebuild_tensor, tensor_sha


class QwenEdit(unittest.TestCase):
    def test_sparse_edit_exact_bfloat16_bytes(self):
        base = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        edited = base.clone()
        edited[0, 1] = 9
        edited[1, 2] = -4
        bits = edited.view(torch.uint16).numpy().reshape(-1)
        spec = {"form": "sparse", "shape": [2, 3], "base_sha256": tensor_sha(base),
                "edited_sha256": tensor_sha(edited), "changed_entries": 2}
        archive = {"t0_idx": np.array([1, 5], dtype=np.int64), "t0_val": bits[[1, 5]]}
        actual = rebuild_tensor(base, spec, archive, 0)
        torch.testing.assert_close(actual, edited)
        self.assertEqual(tensor_sha(actual), tensor_sha(edited))
        bad = copy.deepcopy(spec)
        bad["base_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            rebuild_tensor(base, bad, archive, 0)
        bad = copy.deepcopy(spec)
        bad["edited_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            rebuild_tensor(base, bad, archive, 0)
        with self.assertRaises(ValueError):
            rebuild_tensor(base.float(), spec, archive, 0)
        with self.assertRaises(ValueError):
            rebuild_tensor(base, spec, {"t0_idx": np.array([1, 1]), "t0_val": bits[[1, 1]]}, 0)

    def test_published_archive_checksum_and_changed_counts(self):
        root = Path(__file__).resolve().parents[1] / "third_party/qwen_p50c"
        manifest, archive = read_edit(root)
        self.assertEqual(sum(t["changed_entries"] for t in manifest["tensors"]), 130188)
        self.assertEqual(len(manifest["tensors"]), 2)
        for index, spec in enumerate(manifest["tensors"]):
            self.assertEqual(archive[f"t{index}_idx"].size, spec["changed_entries"])
        archive.close()

    def test_presence_penalty_counts_generated_types_once_and_ignores_prompt(self):
        tokens = torch.tensor([[1, 2, 2, 3, 3, 4]])
        scores = torch.zeros(1, 6)
        actual = PresencePenalty(3)(tokens, scores)
        torch.testing.assert_close(actual, torch.tensor([[0., 0., 0., -1.5, -1.5, 0.]]))
        torch.testing.assert_close(scores, torch.zeros_like(scores))

    def test_extract_last_boxed_with_nested_braces(self):
        self.assertEqual(last_boxed(r"First \boxed{1}, finally \boxed{\frac{1}{2}}."), r"\boxed{\frac{1}{2}}")
        self.assertIsNone(last_boxed(r"unfinished \boxed{\frac{1}{2}"))
        self.assertIsNone(last_boxed("No final answer"))

    @unittest.skipUnless(importlib.util.find_spec("math_verify"), "optional qwen dependencies not installed")
    def test_math_grading_equivalence_and_incomplete_reasoning(self):
        self.assertTrue(grade("1/2", r"Answer: \boxed{\frac{1}{2}}"))
        self.assertFalse(grade("1/2", r"Answer: \boxed{2}"))
        self.assertFalse(grade("42", "I think it is 42 but no boxed answer yet"))


if __name__ == "__main__":
    unittest.main()
