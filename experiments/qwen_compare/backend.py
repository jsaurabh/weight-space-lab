"""Clean VM service: base weights, training practice and numerical tools only."""
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import tarfile
import time
import traceback

import numpy as np
import torch

from protocol import BASE_ID, BASE_REVISION, MAX_FORWARD_CALLS, MAX_ANALYSIS_CALLS, validate_code

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
BASE = Path(os.environ.get("QWEN_BASE_DIRECTORY", str(ROOT / "base")))
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def extract_box(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    start, key = text.rfind("\\boxed"), "\\boxed"
    if start < 0:
        start, key = text.rfind("\\fbox"), "\\fbox"
    if start < 0:
        return None
    tail = text[start + len(key):]
    content = tail.lstrip()
    if content.startswith("{"):
        depth = 0
        for i, char in enumerate(content):
            depth += (char == "{") - (char == "}")
            if depth == 0:
                answer = content[1:i].strip()
                return answer or None
    elif key == "\\boxed" and tail.startswith(" "):
        return tail.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def correct(gold, text):
    from math_verify import parse, verify
    pred = extract_box(text)
    if pred is None:
        return False
    try:
        g = parse("\\boxed{" + str(gold) + "}", parsing_timeout=5)
        p = parse("\\boxed{" + pred + "}", parsing_timeout=5)
        return bool(g and p and verify(g, p, timeout_seconds=5))
    except Exception:
        return False


class Service:
    def __init__(self):
        from huggingface_hub import snapshot_download
        torch.set_num_threads(4)
        OUT.mkdir(parents=True, exist_ok=True)
        for name in ["requests", "responses"]:
            (ROOT / name).mkdir(exist_ok=True)
        snapshot_download(BASE_ID, revision=BASE_REVISION, local_dir=BASE,
                          allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"])
        self.mapping = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        for shard in set(self.mapping.values()):
            with (BASE / shard).open("rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size))
                self.headers.update({name: value for name, value in header.items() if name != "__metadata__"})
        self.config = json.loads((BASE / "config.json").read_text())
        self.practice_rows = json.loads((ROOT / "practice.json").read_text())
        self.practice_map = {r["id"]: r for r in self.practice_rows}
        self.base_cache, self.edits, self.artifacts = {}, {}, {}
        self.llm = self.hf = self.tokenizer = None
        self.engine_applied, self.hf_applied = set(), set()
        self.forward_calls = self.analysis_calls = 0
        self.frozen = False
        self.started = time.monotonic()
        self.namespace = None
        self.engine_signature = ()
        self.maintenance_notice = None
        snapshot = ROOT / "maintenance_snapshot.json"
        if snapshot.exists():
            saved = json.loads(snapshot.read_text())
            self.restore(saved["checkpoint_label"])
            self.forward_calls = saved["successful_forward_calls"]
            self.analysis_calls = saved["analysis_calls"]
            self.started -= saved["elapsed_seconds"]
            for descriptor in sorted((OUT / "artifacts").glob("*.json")):
                array_path = descriptor.with_suffix(".npz")
                arrays = np.load(array_path, allow_pickle=False) if array_path.exists() else None

                def decode_artifact(value):
                    if isinstance(value, dict) and set(value) == {"array", "shape", "dtype"}:
                        return np.array(arrays[value["array"]], copy=True)
                    if isinstance(value, dict):
                        return {k: decode_artifact(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [decode_artifact(v) for v in value]
                    return value

                self.artifacts[descriptor.stem] = decode_artifact(json.loads(descriptor.read_text()))
                if arrays is not None:
                    arrays.close()
            self.maintenance_notice = (
                "Infrastructure maintenance fixed candidate-weight transport using the native reload API. "
                "Your exact current edits, checkpoints and all generation/probe artifacts are preserved. "
                "Temporary CPU analysis variables were cleared; recreate them from your transcript/artifacts. "
                "Failed weight-transfer requests that ran no inference were refunded. No repair choices changed."
            )
        source = OUT / "source"
        source.mkdir(exist_ok=True)
        for name in ["backend.py", "protocol.py"]:
            (source / name).write_bytes((ROOT / name).read_bytes())
        write(OUT / "environment.json", {
            "model": BASE_ID, "revision": BASE_REVISION,
            "practice_count": len(self.practice_rows), "practice_sha256": hashlib.sha256((ROOT / "practice.json").read_bytes()).hexdigest(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["backend.py", "protocol.py"]},
            "payload_contains_only_base_and_practice": True,
        })
        self.update_status("ready")

    def base_tensor(self, name):
        from safetensors import safe_open
        if name not in self.headers:
            raise ValueError("Unknown tensor name")
        if name not in self.base_cache:
            with safe_open(BASE / self.mapping[name], framework="pt", device="cpu") as source:
                self.base_cache[name] = source.get_tensor(name)
        return self.base_cache[name]

    def weight(self, name, current=False):
        tensor = self.edits.get(name) if current else None
        if tensor is None:
            tensor = self.base_tensor(name)
        return np.array(tensor.float().numpy(), copy=True)

    def save_edit(self, name, array):
        if self.frozen:
            raise ValueError("Final weights are frozen")
        if not name.startswith("model.language_model.layers.") or name not in self.headers:
            raise ValueError("Only internal text-layer tensors are editable")
        array = np.asarray(array)
        base = self.base_tensor(name)
        if list(array.shape) != list(base.shape) or array.dtype.kind not in "fiu":
            raise ValueError("Replacement must be a numeric array of the original shape")
        if not np.isfinite(array).all() or np.max(np.abs(array)) > 1e6:
            raise ValueError("Replacement contains invalid values")
        tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(base.dtype)
        changed = int(torch.count_nonzero(tensor != base))
        if changed == 0:
            self.edits.pop(name, None)
            return {"name": name, "changed_entries": 0}
        # Reject a whole-tensor scalar gain without prohibiting local interventions.
        # A strided sample can miss an entire edited column. Compare all entries
        # so legitimate sparse interventions cannot look like the identity gain.
        old = base.float().flatten().numpy().astype(np.float64)
        new = np.asarray(array).reshape(-1).astype(np.float64)
        norm = float(old @ old)
        if norm:
            gain = float(new @ old / norm)
            if abs(gain - 1.0) > 1e-6 and np.linalg.norm(new - gain * old) <= 1e-6 * max(np.linalg.norm(new), 1e-20):
                raise ValueError("Whole-tensor scalar rescaling is not an admissible edit")
        candidate = dict(self.edits)
        candidate[name] = tensor.contiguous()
        if len(candidate) > 64 or sum(t.numel() for t in candidate.values()) > 160000000:
            raise ValueError("Candidate storage budget exceeded")
        self.edits = candidate
        return {"name": name, "changed_entries": changed, "edited_sha256": digest(tensor)}

    def list_edits(self):
        return [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype),
                 "changed_entries": int(torch.count_nonzero(t != self.base_tensor(name))),
                 "base_sha256": digest(self.base_tensor(name)), "edited_sha256": digest(t)}
                for name, t in sorted(self.edits.items())]

    def clear_edits(self):
        if self.frozen:
            raise ValueError("Final weights are frozen")
        self.edits.clear()

    def remove_edit(self, name):
        if self.frozen:
            raise ValueError("Final weights are frozen")
        self.edits.pop(name, None)

    def checkpoint(self, label):
        from safetensors.torch import save_file
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", label):
            raise ValueError("Checkpoint label must be 1–40 letters, digits, underscore or hyphen")
        path = OUT / "checkpoints" / (label + ".safetensors")
        path.parent.mkdir(exist_ok=True)
        save_file(self.edits, path)
        write(OUT / "candidate_manifests" / (label + ".json"), self.list_edits())
        return {"label": label, "edits": self.list_edits()}

    def restore(self, label):
        from safetensors.torch import load_file
        if self.frozen or not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", label):
            raise ValueError("Unavailable checkpoint operation")
        self.edits = load_file(OUT / "checkpoints" / (label + ".safetensors"), device="cpu")
        return self.list_edits()

    def get_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
        return self.tokenizer

    def tokenize(self, text):
        return self.get_tokenizer().encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.get_tokenizer().decode(list(map(int, ids)), skip_special_tokens=False)

    def python_analysis(self, code):
        if self.frozen or self.analysis_calls >= MAX_ANALYSIS_CALLS:
            raise ValueError("Analysis budget exhausted or weights frozen")
        validate_code(code)
        self.analysis_calls += 1
        if self.namespace is None:
            import builtins
            allowed = ["abs", "all", "any", "bool", "dict", "enumerate", "Exception", "float", "int", "isinstance",
                       "len", "list", "max", "min", "next", "print", "reversed", "round", "set", "sorted", "str",
                       "repr", "sum", "tuple", "type", "ValueError", "zip"]
            safe = {name: getattr(builtins, name) for name in allowed}

            def limited_range(*args):
                values = range(*args)
                if len(values) > 100000:
                    raise ValueError("Analysis range too large")
                return values

            def restricted_import(name, *args, **kwargs):
                if name not in {"numpy", "math"} and not name.startswith("numpy."):
                    raise ValueError("Unavailable module")
                return __import__(name, *args, **kwargs)

            safe.update(range=limited_range, __import__=restricted_import)
            self.namespace = {"__builtins__": safe, "np": np, "math": math,
                "names": sorted(self.headers), "shapes": {n: h["shape"] for n, h in self.headers.items()},
                "config": copy.deepcopy(self.config), "weight": self.weight, "save_edit": self.save_edit,
                "remove_edit": self.remove_edit, "clear_edits": self.clear_edits, "list_edits": self.list_edits,
                "checkpoint": self.checkpoint, "restore": self.restore,
                "artifact": lambda key: self.artifacts[key], "tokenize": self.tokenize, "decode": self.decode}
            self.namespace.update(practice_ids=[r["id"] for r in self.practice_rows],
                practice_row=lambda identifier: copy.deepcopy(self.practice_map[identifier]))
        output = io.StringIO()

        def timeout_handler(*args):
            raise TimeoutError("CPU analysis exceeded 120 seconds")

        previous = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(code, "<editor-analysis>", "exec"), self.namespace)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        return {"stdout": output.getvalue()[:14000], "truncated": len(output.getvalue()) > 14000,
                "edits": self.list_edits()}

    def get_llm(self):
        if self.llm is None:
            from vllm import LLM
            self.update_status("initializing_generator")
            self.llm = LLM(model=str(BASE), dtype="bfloat16", max_model_len=34816,
                max_num_seqs=64, max_num_batched_tokens=8192, gpu_memory_utilization=0.70,
                enable_prefix_caching=False, seed=0, trust_remote_code=False,
                limit_mm_per_prompt={"image": 0, "video": 0})
        return self.llm

    def apply_generator(self, which):
        from safetensors.torch import load_file, save_file
        from shutil import copyfile
        llm = self.get_llm()
        target = self.edits if which == "current" else {}
        signature = tuple((name, digest(t)) for name, t in sorted(target.items()))
        if signature != self.engine_signature:
            directory = BASE
            if target:
                directory = ROOT / "runtime-candidate"
                directory.mkdir(exist_ok=True)
                for source in BASE.iterdir():
                    if source.is_file() and not source.name.endswith(".safetensors"):
                        copyfile(source, directory / source.name)
                for shard in sorted(set(self.mapping.values())):
                    tensors = load_file(BASE / shard, device="cpu")
                    tensors.update({name: value for name, value in target.items() if self.mapping[name] == shard})
                    save_file(tensors, directory / shard, metadata={"format": "pt"})
            # String RPC + a complete checkpoint avoids serializing executable
            # callbacks and preserves vLLM's default serialization protections.
            llm.collective_rpc("reload_weights", kwargs={"weights_path": str(directory), "is_checkpoint_format": True})
            self.engine_signature = signature
        self.engine_applied = set(target)

    def smoke_reload(self):
        self.apply_generator("base")
        before = self.llm.collective_rpc("compute_weight_checksums")
        self.apply_generator("current")
        after = self.llm.collective_rpc("compute_weight_checksums")
        changes = [name for name in before[0] if before[0][name] != after[0][name]]
        self.apply_generator("base")
        restored = self.llm.collective_rpc("compute_weight_checksums")
        result = {"changed_gpu_parameters": changes, "base_roundtrip_exact": before == restored,
                  "candidate_tensors": self.list_edits(), "did_not_run_inference": True}
        if not result["base_roundtrip_exact"] or (self.edits and not changes):
            raise ValueError("Native reload round-trip check failed")
        write(OUT / "native_reload_validation.json", result)
        return result

    def put_artifact(self, data, prefix):
        identifier = f"{prefix}-{len(self.artifacts) + 1:03d}"
        self.artifacts[identifier] = data
        arrays = {}

        def encode(value, key="root"):
            if isinstance(value, np.ndarray):
                arrays[key] = value
                return {"array": key, "shape": list(value.shape), "dtype": str(value.dtype)}
            if isinstance(value, dict):
                return {k: encode(v, key + "/" + k) for k, v in value.items()}
            if isinstance(value, list):
                return [encode(v, key + "/" + str(i)) for i, v in enumerate(value)]
            if isinstance(value, np.generic):
                return value.item()
            return value

        write(OUT / "artifacts" / (identifier + ".json"), encode(data))
        if arrays:
            np.savez_compressed(OUT / "artifacts" / (identifier + ".npz"), **arrays)
        return identifier

    def generate_rows(self, rows, which, max_tokens, samples=1, seed=0):
        from vllm import SamplingParams
        self.apply_generator(which)
        tokenizer = self.get_tokenizer()
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"] + SUFFIX}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True) for row in rows]
        started = time.perf_counter()
        outputs = self.llm.generate(prompts, SamplingParams(n=samples, temperature=0.6, top_p=0.95,
            top_k=20, presence_penalty=1.5, max_tokens=max_tokens, seed=seed), use_tqdm=True)
        records = []
        for row, request in zip(rows, outputs, strict=True):
            for sample, output in enumerate(request.outputs):
                records.append({"id": row["id"], "sample": sample, "question": row["question"],
                    "answer": row["answer"], "text": output.text, "correct": correct(row["answer"], output.text),
                    "tokens": len(output.token_ids), "hit_token_limit": output.finish_reason == "length",
                    "thinking_closed": "</think>" in output.text})
        return {"records": records, "correct": sum(r["correct"] for r in records), "total": len(records),
                "hit_token_limit": sum(r["hit_token_limit"] for r in records),
                "seconds": time.perf_counter() - started, "max_tokens": max_tokens}

    def take_pass(self):
        if self.frozen or self.forward_calls >= MAX_FORWARD_CALLS:
            raise ValueError("Forward budget exhausted or weights frozen")
        self.forward_calls += 1

    def generate(self, ids, which, max_tokens):
        if not ids or len(set(ids)) != len(ids) or any(i not in self.practice_map for i in ids):
            raise ValueError("Provide distinct valid practice IDs")
        if which not in {"base", "current", "both"} or not 1 <= max_tokens <= 32768:
            raise ValueError("Invalid generation request")
        if max_tokens != 32768:
            raise ValueError("Comparison practice generation uses the fixed 32768-token cap")
        if len(ids) * (2 if which == "both" else 1) > 64:
            raise ValueError("At most 64 total practice generations per call")
        self.take_pass()
        data, summaries = {}, {}
        for mode in (["base", "current"] if which == "both" else [which]):
            self.update_status("generating_" + mode)
            group = self.generate_rows([self.practice_map[i] for i in ids], mode, max_tokens)
            data[mode] = group["records"]
            summaries[mode] = {k: v for k, v in group.items() if k != "records"}
            summaries[mode]["examples"] = [{k: v for k, v in row.items() if k != "text"} |
                {"text_start": row["text"][:1000], "text_end": row["text"][-1500:]} for row in group["records"][:2]]
        identifier = self.put_artifact(data, "generation")
        result = {"artifact": identifier, "summaries": summaries}
        if which == "both":
            before = {r["id"]: r for r in data["base"]}
            delta = np.array([int(r["correct"]) - int(before[r["id"]]["correct"]) for r in data["current"]])
            result["paired"] = {"accuracy_difference": float(delta.mean()),
                "rescues": int(np.count_nonzero(delta > 0)), "losses": int(np.count_nonzero(delta < 0)),
                "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None}
        return result

    def probe(self, which, texts, modules, positions):
        if which not in {"base", "current"} or not 1 <= len(texts) <= 16 or not 1 <= len(positions) <= 32 or len(modules) > 8:
            raise ValueError("Probe limits: 16 texts, 32 positions, 8 modules")
        lengths = [len(self.tokenize(t)) if isinstance(t, str) else 32769 for t in texts]
        if max(lengths) > 32768 or sum(lengths) > 131072:
            raise ValueError("Probe limit: 32768 tokens per text and 131072 tokens total")
        self.take_pass()
        from transformers import Qwen3_5ForConditionalGeneration
        if self.hf is None:
            self.update_status("initializing_probe_model")
            self.hf = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True).requires_grad_(False).eval().to("cuda")
        target = self.edits if which == "current" else {}
        with torch.no_grad():
            for name in set(target) | self.hf_applied:
                parameter = self.hf.get_parameter(name)
                parameter.copy_(target.get(name, self.base_tensor(name)).to(parameter.device, parameter.dtype))
        self.hf_applied = set(target)
        module_map = dict(self.hf.named_modules())
        if any(name not in module_map for name in modules):
            raise ValueError("Unknown module name")
        records = []
        for text in texts:
            tokens = self.get_tokenizer()(text, return_tensors="pt").to("cuda")
            n = tokens.input_ids.shape[1]
            selected = [p if p >= 0 else n + p for p in positions]
            if not n or any(not 0 <= p < n for p in selected):
                raise ValueError("Probe position outside text")
            captured, handles = {}, []
            for name in modules:
                captured[name] = {}

                def before(module, inputs, key=name):
                    value = inputs[0]
                    if isinstance(value, torch.Tensor) and value.ndim == 3:
                        captured[key]["input"] = value[0, selected].detach().float().cpu().numpy()

                def after(module, inputs, output, key=name):
                    value = output[0] if isinstance(output, tuple) else output
                    if isinstance(value, torch.Tensor) and value.ndim == 3:
                        captured[key]["output"] = value[0, selected].detach().float().cpu().numpy()

                handles.extend([module_map[name].register_forward_pre_hook(before), module_map[name].register_forward_hook(after)])
            try:
                with torch.inference_mode():
                    output = self.hf(**tokens, output_hidden_states=True,
                        logits_to_keep=torch.tensor(selected, device="cuda"), use_cache=False)
                hidden = np.stack([state[0, selected].detach().float().cpu().numpy() for state in output.hidden_states])
                logprobs = output.logits[0].float().log_softmax(-1)
                top_logprobs, top_ids = logprobs.topk(20, dim=-1)
                next_ids = np.array([int(tokens.input_ids[0, p + 1]) if p + 1 < n else -1 for p in selected])
                next_logprobs = np.array([float(logprobs[j, token]) if token >= 0 else np.nan for j, token in enumerate(next_ids)])
                predictions = {"top_ids": top_ids.cpu().numpy(), "top_logprobs": top_logprobs.cpu().numpy(),
                               "next_ids": next_ids, "next_logprobs": next_logprobs}
                del output, logprobs, top_ids, top_logprobs
            finally:
                for handle in handles:
                    handle.remove()
            records.append({"text": text, "tokens": tokens.input_ids[0].cpu().numpy(),
                            "positions": selected, "hidden": hidden, "modules": captured, **predictions})
        identifier = self.put_artifact({"texts": records, "which": which}, "probe")
        return {"artifact": identifier, "texts": len(records), "hidden_shape": list(records[0]["hidden"].shape), "modules": modules}

    def finish(self, checkpoint_label, rationale, force=False):
        from safetensors.torch import save_file
        if self.frozen:
            return json.loads((OUT / "frozen.json").read_text())
        if self.forward_calls < MAX_FORWARD_CALLS and not force:
            raise ValueError("Use all 10 successful forward calls before selecting final weights")
        if checkpoint_label:
            self.restore(checkpoint_label)
        save_file(self.edits, OUT / "selected.safetensors")
        result = {"rationale": rationale, "checkpoint_label": checkpoint_label, "edits": self.list_edits(),
                  "selected_file_sha256": hashlib.sha256((OUT / "selected.safetensors").read_bytes()).hexdigest(),
                  "forward_calls": self.forward_calls, "analysis_calls": self.analysis_calls, "host_forced": force,
                  "frozen_at_unix": time.time(), "test_data_present_at_freeze": (ROOT / "sealed_test.json").exists()}
        if result["test_data_present_at_freeze"]:
            raise ValueError("Blindness violation: test data was present before freeze")
        write(OUT / "frozen.json", result)
        self.frozen = True
        return result

    def final_evaluate(self):
        if not self.frozen:
            raise ValueError("Final evaluation requires frozen weights")
        held = json.loads((ROOT / "sealed_test.json").read_text())
        write(OUT / "sealed_test.json", held)
        rows = [{"id": r["unique_id"], "question": r["problem"], "answer": r["answer"]} for r in held]
        results = {}
        for mode in ["base", "current"]:
            self.update_status("final_evaluation_" + mode)
            results[mode] = self.generate_rows(rows, mode, 32768, samples=2, seed=0)
            write(OUT / ("final_" + mode + ".json"), results[mode])
        summary = {mode: {k: v for k, v in value.items() if k != "records"} for mode, value in results.items()}
        write(OUT / "final_summary.json", summary)
        return summary

    def evaluate_conditions(self, conditions, test_sha256):
        """Host-only evaluation after every candidate is frozen; never an editor tool."""
        from safetensors.torch import load_file
        if not self.frozen:
            raise ValueError("Final editor must freeze before any held-out evaluation")
        test = ROOT / "sealed_test.json"
        if hashlib.sha256(test.read_bytes()).hexdigest() != test_sha256:
            raise ValueError("Held-out seal mismatch")
        verified = []
        for item in conditions:
            label = item["label"]
            if not re.fullmatch(r"[a-z0-9_-]+", label):
                raise ValueError("Invalid evaluation label")
            frozen = json.loads(Path(item["manifest"]).read_text())
            checkpoint = Path(item["checkpoint"])
            if frozen["test_data_present_at_freeze"] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != frozen["selected_file_sha256"]:
                raise ValueError("Candidate was not independently frozen or its bytes changed")
            verified.append((label, checkpoint, frozen))
        output = OUT / "evaluation"
        output.mkdir(exist_ok=True)
        held = json.loads(test.read_text())
        write(output / "problems.json", held)
        write(output / "candidates.json", [{"label": label, "frozen": f} for label, _, f in verified])
        rows = [{"id": r["unique_id"], "question": r["problem"], "answer": r["answer"]} for r in held]
        summaries = {}
        for label, checkpoint in [("base", None)] + [(label, path) for label, path, _ in verified]:
            self.update_status("final_evaluation_" + label)
            self.edits = {} if checkpoint is None else load_file(checkpoint, device="cpu")
            result = self.generate_rows(rows, "current", 32768, samples=2, seed=0)
            write(output / (label + ".json"), result)
            summaries[label] = {k: v for k, v in result.items() if k != "records"}
            write(output / "summary.json", summaries)
        return summaries

    def export(self):
        self.update_status("exporting")
        archive = ROOT / "results.tar.gz"
        with tarfile.open(archive, "w:gz") as target:
            for path in OUT.rglob("*"):
                if path.is_file() and "checkpoints" not in path.relative_to(OUT).parts:
                    target.add(path, arcname=ROOT.name + "/" + str(path.relative_to(OUT)))
        return {"archive": str(archive)}

    def update_status(self, phase):
        write(OUT / "status.json", {"phase": phase, "forward_calls": self.forward_calls,
              "analysis_calls": self.analysis_calls, "frozen": self.frozen,
              "edited_tensors": len(self.edits), "elapsed_seconds": time.monotonic() - self.started})

    def dispatch(self, name, args):
        if name == "metadata":
            ranges = {}
            for i, row in enumerate(self.practice_rows):
                source = row.get("source", "practice")
                if source not in ranges:
                    ranges[source] = {"offset": i, "count": 0}
                ranges[source]["count"] += 1
            return {"config": self.config, "tensors": self.headers, "practice_count": len(self.practice_rows), "sources": ranges}
        if name == "practice":
            offset, count = args["offset"], args["count"]
            if not 0 <= offset < len(self.practice_rows) or not 1 <= count <= 16:
                raise ValueError("Practice lookup: valid offset and count 1–16")
            return self.practice_rows[offset:offset + count]
        if name == "python_analysis":
            return self.python_analysis(args["code"])
        if name == "generate":
            return self.generate(args["ids"], args["which"], 32768)
        if name == "probe":
            return self.probe(args["which"], args["texts"], args["modules"], args["positions"])
        if name == "finish":
            return self.finish(args["checkpoint_label"], args["rationale"])
        if name == "force_finish":
            return self.finish("", args["rationale"], force=True)
        if name == "evaluate_conditions":
            return self.evaluate_conditions(args["conditions"], args["test_sha256"])
        if name == "final_evaluate":
            return self.final_evaluate()
        if name == "export":
            return self.export()
        if name == "smoke_reload":
            return self.smoke_reload()
        raise ValueError("Unknown operation")

    def run(self):
        print("Clean discovery service ready", flush=True)
        while time.monotonic() - self.started < 8 * 3600:
            pending = sorted((ROOT / "requests").glob("*.json"))
            for path in pending:
                response_path = ROOT / "responses" / path.name
                if response_path.exists():
                    continue
                try:
                    request = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                started = time.perf_counter()
                self.update_status("tool_" + request["name"])
                previous_forward_calls = self.forward_calls
                try:
                    result = {"ok": True, "result": self.dispatch(request["name"], request["arguments"])}
                except Exception as exc:
                    if request["name"] in {"generate", "probe"}:
                        self.forward_calls = previous_forward_calls
                    result = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:3000]}
                    print(traceback.format_exc(), flush=True)
                if self.maintenance_notice and request["name"] in {"practice", "python_analysis", "generate", "probe", "finish"}:
                    result["maintenance_notice"] = self.maintenance_notice
                    self.maintenance_notice = None
                result["remaining_forward_calls"] = MAX_FORWARD_CALLS - self.forward_calls
                result["remaining_analysis_calls"] = MAX_ANALYSIS_CALLS - self.analysis_calls
                result["seconds"] = time.perf_counter() - started
                with (OUT / "tool_log.jsonl").open("a") as log:
                    log.write(json.dumps({"request": request, "response": result}) + "\n")
                write(response_path, result)
                self.update_status("frozen" if self.frozen else "waiting_for_editor")
            time.sleep(0.5)
        self.export()


if __name__ == "__main__":
    Service().run()
