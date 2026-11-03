import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/qwen_compare"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


protocol = module("comparison_protocol", DIRECTORY / "protocol.py")
with patch.dict(sys.modules, {"protocol": protocol}):
    backend = module("comparison_backend", DIRECTORY / "backend.py")
    driver = module("comparison_driver", DIRECTORY / "driver.py")


class QwenComparison(unittest.TestCase):
    def test_recovery_worker_import_does_not_start_service_in_spawned_child(self):
        fake_backend = MagicMock()
        namespace = {"__name__": "__mp_main__"}
        source = (DIRECTORY / "resume_service.py").read_text()
        with patch.dict(sys.modules, {"backend": fake_backend}):
            exec(compile(source, "resume_service.py", "exec"), namespace)
        fake_backend.Service.assert_not_called()

    def test_recovery_executes_prefetched_response_without_second_api_charge(self):
        class Bridge:
            def __init__(self, session, label):
                self.calls = []
            def rpc(self, name, arguments, **kwargs):
                self.calls.append(name)
                return {"ok": True, "result": {"config": {}, "sources": {}, "practice_count": 0}}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            original_metadata = {"initial": True}
            (output / "metadata.json").write_text(json.dumps(original_metadata))
            (output / "run.json").write_text(json.dumps({"created_utc": driver.datetime.now(driver.timezone.utc).isoformat(),
                "estimated_api_cost_usd": 0.12, "status": "stopped_on_error", "error_type": "BadRequestError"}))
            history = [{"role": "user", "content": "Continue the saved math experiment."}]
            request = {"model": driver.MODELS["astra"], "input": json.loads(json.dumps(history)),
                "tools": protocol.TOOLS, "parallel_tool_calls": False, "store": False,
                "service_tier": "default", "include": ["reasoning.encrypted_content"],
                "reasoning": {"effort": "xhigh"}, "max_output_tokens": 16000}
            usage = {"input_tokens": 1000, "output_tokens": 1000}
            response = {"status": "completed", "usage": usage, "output": [{"type": "function_call",
                "name": "finish", "arguments": '{"checkpoint_label":"base","rationale":"done"}', "call_id": "call_test"}]}
            accounting = {"usage": usage, "estimated_api_cost_usd": 0.06, "api_wall_seconds": 3.0}
            client = MagicMock()
            client.responses.create.side_effect = AssertionError("Prefetched call must not be billed twice")
            with patch.object(driver, "Bridge", Bridge), patch.object(driver, "OpenAI", return_value=client):
                bridge = driver.editor("test", "astra", output, "unused", resume={"history": history, "first_turn": 11,
                    "prefetched": {"request": request, "response": response, "accounting": accounting}})
            self.assertEqual(bridge.calls, ["metadata", "finish"])
            client.responses.create.assert_not_called()
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["status"], "frozen")
            self.assertAlmostEqual(run["estimated_api_cost_usd"], 0.18)
            self.assertEqual(json.loads((output / "turns/011/accounting.json").read_text()), accounting)
            self.assertEqual(json.loads((output / "metadata.json").read_text()), original_metadata)

    def test_sparse_column_edits_are_allowed_and_global_gains_are_rejected(self):
        service = backend.Service.__new__(backend.Service)
        service.frozen = False
        service.edits = {}
        name = "model.language_model.layers.0.test.weight"
        # The old 8,192-element stride inspected only even columns here.
        before = torch.ones((128, 128), dtype=torch.bfloat16)
        service.headers = {name: {"shape": [128, 128]}}
        service.base_tensor = lambda key: before
        sparse = before.float().numpy().copy()
        sparse[:, 1] += 0.125
        self.assertEqual(service.save_edit(name, sparse)["changed_entries"], 128)
        scalar = before.float().numpy() * 1.25
        with self.assertRaises(ValueError):
            service.save_edit(name, scalar)
        np.testing.assert_array_equal(service.weight(name, current=True), sparse)

    def test_equal_long_generation_budget_and_no_published_solution(self):
        self.assertEqual(protocol.MAX_FORWARD_CALLS, 10)
        tools = {tool["name"]: tool for tool in protocol.TOOLS}
        self.assertEqual(set(tools), {"practice", "python_analysis", "generate", "probe", "finish"})
        self.assertNotIn("max_tokens", tools["generate"]["parameters"]["properties"])
        self.assertIn("32768", tools["generate"]["description"])
        self.assertIn("32,768", protocol.SYSTEM)
        for marker in ["p50c", "layers.15", "layers.19", "130188", "Householder"]:
            self.assertNotIn(marker, protocol.SYSTEM + str(protocol.TOOLS))

    def test_separate_model_costs(self):
        usage = {"input_tokens": 1000, "output_tokens": 1000}
        self.assertAlmostEqual(driver.usage_cost("astra", usage), 0.06)
        self.assertAlmostEqual(driver.usage_cost("gpt55", usage), 0.035)
        with self.assertRaises(ValueError):
            driver.usage_cost("astra", {**usage, "input_tokens_details": {"cached_tokens": 1001}})

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "Install the qwen extra")
    def test_finish_requires_full_budget_and_test_requires_freeze(self):
        service = backend.Service.__new__(backend.Service)
        service.frozen = False
        service.forward_calls = 0
        with self.assertRaises(ValueError):
            service.finish("", "early")
        with self.assertRaises(ValueError):
            service.evaluate_conditions([], "invalid")

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "Install the qwen extra")
    def test_three_conditions_use_frozen_bytes_and_clear_base(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results"
            output.mkdir()
            test = root / "sealed_test.json"
            test.write_text(json.dumps([{"unique_id": "held", "problem": "2+2", "answer": "4"}]))
            conditions = []
            for index, label in enumerate(["astra", "gpt55"], 1):
                checkpoint = root / (label + ".safetensors")
                save_file({"writer": torch.tensor([index], dtype=torch.bfloat16)}, checkpoint)
                manifest = root / (label + ".json")
                manifest.write_text(json.dumps({"test_data_present_at_freeze": False,
                    "selected_file_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
                conditions.append({"label": label, "checkpoint": str(checkpoint), "manifest": str(manifest)})
            service = backend.Service.__new__(backend.Service)
            service.frozen = True
            service.edits = {"old": torch.ones(1)}
            service.update_status = lambda phase: None
            seen = []
            def generate(rows, which, max_tokens, samples, seed):
                seen.append({k: float(v[0]) for k, v in service.edits.items()})
                self.assertEqual((which, max_tokens, samples, seed), ("current", 32768, 2, 0))
                return {"records": [], "total": 2, "correct": 0}
            service.generate_rows = generate
            with patch.object(backend, "ROOT", root), patch.object(backend, "OUT", output):
                summary = service.evaluate_conditions(conditions, hashlib.sha256(test.read_bytes()).hexdigest())
                self.assertEqual(set(summary), {"base", "astra", "gpt55"})
                self.assertEqual(seen, [{}, {"writer": 1.0}, {"writer": 2.0}])
                (root / "astra.safetensors").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    service.evaluate_conditions(conditions, hashlib.sha256(test.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
