import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/blind"
sys.path.insert(0, str(DIRECTORY))
from protocol import SYSTEM, TOOLS, validate_code
from driver import check_budget, usage_cost
from backend import Service
import backend


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

    @unittest.skipUnless(importlib.util.find_spec('safetensors'), 'Install the qwen extra to check native checkpoint transport')
    def test_native_reload_uses_complete_checkpoint_and_restores_base(self):
        from safetensors.torch import load_file, save_file
        class Engine:
            def __init__(self):
                self.calls = []
            def collective_rpc(self, method, kwargs):
                self.calls.append((method, kwargs, load_file(Path(kwargs['weights_path']) / 'model.safetensors')))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / 'base'
            base.mkdir()
            original = {'edited': torch.ones(2, dtype=torch.bfloat16), 'fixed': torch.zeros(2, dtype=torch.bfloat16)}
            save_file(original, base / 'model.safetensors')
            (base / 'config.json').write_text('{}')
            engine = Engine()
            service = Service.__new__(Service)
            service.edits = {'edited': original['edited'] + 1}
            service.mapping = {name: 'model.safetensors' for name in original}
            service.engine_signature = ()
            service.get_llm = lambda: engine
            with patch.object(backend, 'BASE', base), patch.object(backend, 'ROOT', root):
                service.apply_generator('current')
                service.apply_generator('current')
                service.apply_generator('base')
            self.assertEqual(len(engine.calls), 2)
            self.assertTrue(all(c[0] == 'reload_weights' for c in engine.calls))
            self.assertTrue(torch.equal(engine.calls[0][2]['edited'], service.edits['edited']))
            self.assertTrue(torch.equal(engine.calls[0][2]['fixed'], original['fixed']))
            self.assertTrue(all(torch.equal(engine.calls[1][2][k], v) for k, v in original.items()))


if __name__ == "__main__":
    unittest.main()
