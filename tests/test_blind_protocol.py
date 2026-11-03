import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/blind"
sys.path.insert(0, str(DIRECTORY))
from protocol import SYSTEM, TOOLS, validate_code
from driver import check_budget, usage_cost
from backend import Service


class BlindProtocol(unittest.TestCase):
    def test_editor_instructions_and_tools_have_no_prior_solution_hints(self):
        text = SYSTEM + str(TOOLS)
        for marker in ["p50c", "RikiyaT", "131072", "130188", "layers.15", "layers.19", "rank-one", "loop direction"]:
            self.assertNotIn(marker, text)
        self.assertEqual({t["name"] for t in TOOLS}, {"practice", "python_analysis", "generate", "probe", "finish"})
        self.assertNotIn("final_evaluate", text)

    def test_analysis_accepts_algebra_but_rejects_external_access_and_training(self):
        validate_code('import numpy as np\nx = weight(names[0]); print(np.linalg.norm(x))')
        for code in ['import os', 'open("file")', 'np.load("file")', 'x.__class__',
                     'np.linalg.lstsq(a,b)', 'x.backward()', 'while True: pass']:
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate_code(code)

    def test_edits_preserve_shape_reject_global_gain_and_allow_local_change(self):
        service = Service.__new__(Service)
        service.frozen = False
        service.edits = {}
        name = "model.language_model.layers.0.test.weight"
        before = torch.arange(1, 5, dtype=torch.bfloat16).reshape(2, 2)
        service.headers = {name: {"shape": [2, 2]}}
        service.base_tensor = lambda key: before
        with self.assertRaises(ValueError):
            service.save_edit(name, before.float().numpy() * 2)
        changed = before.float().numpy().copy()
        changed[0, 0] += 1
        result = service.save_edit(name, changed)
        self.assertEqual(result["changed_entries"], 1)
        service.frozen = True
        with self.assertRaises(ValueError):
            service.save_edit(name, changed)

    def test_usage_cost_accounts_for_cache_and_long_context(self):
        usage = {"input_tokens": 1000, "output_tokens": 1000,
                 "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 300}}
        self.assertAlmostEqual(usage_cost(usage), 0.05895)
        request = {"max_output_tokens": 16000}
        with self.assertRaises(ValueError):
            check_budget(0., 0.01, request, None, 0)
        self.assertLess(check_budget(0., 25., request, usage, 1000), 25.)


if __name__ == "__main__":
    unittest.main()
