"""Run isolated Astra/GPT-5.5 editors of Qwen on one G4, then a shared sealed test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tarfile
import time
import uuid

from dotenv import dotenv_values
from openai import OpenAI

from protocol import BASE_ID, BASE_REVISION, SYSTEM, TOOLS, MAX_FORWARD_CALLS

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"astra": "gpt-6-astra", "gpt55": "gpt-5.5-2026-04-23"}
RATES = {"astra": {"input": 10, "cached": 1, "write": 12.5, "output": 50},
         "gpt55": {"input": 5, "cached": 0.5, "write": 5, "output": 30}}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def usage_cost(label, usage):
    detail = usage.get("input_tokens_details") or {}
    cached, writes = detail.get("cached_tokens", 0) or 0, detail.get("cache_write_tokens", 0) or 0
    ordinary = usage["input_tokens"] - cached - writes
    if min(ordinary, cached, writes) < 0:
        raise ValueError("Inconsistent token accounting")
    r = RATES[label]
    long = usage["input_tokens"] > 272000
    return ((ordinary * r["input"] + cached * r["cached"] + writes * r["write"]) * (2 if long else 1)
            + usage["output_tokens"] * r["output"] * (1.5 if long else 1)) / 1e6


def strip_none(value):
    if isinstance(value, dict):
        return {k: strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [strip_none(v) for v in value]
    return value


class Bridge:
    def __init__(self, session, label):
        self.session, self.label, self.number = session, label, 0
        self.remote = "/content/qwen-comparison/" + label
        self.local = ROOT / ".artifacts/qwen-compare/bridge" / label
        self.local.mkdir(parents=True, exist_ok=True)
        self.prefix = uuid.uuid4().hex[:12]

    def command(self, arguments, check=True):
        result = subprocess.run(["colab", *arguments], capture_output=True, text=True, timeout=600)
        if check and result.returncode:
            # CLI errors can contain signed runtime-proxy URLs. Never put
            # their bodies into experiment logs or public exception records.
            raise RuntimeError(f"Colab {arguments[0]} failed with exit code {result.returncode}")
        return result

    def upload(self, local, remote):
        self.command(["upload", "-s", self.session, str(local), remote])

    def rpc(self, name, arguments, timeout=3600):
        self.number += 1
        filename = f"{self.prefix}-{self.number:06d}.json"
        request, response = self.local / ("request-" + filename), self.local / ("response-" + filename)
        write(request, {"id": self.number, "name": name, "arguments": arguments})
        self.upload(request, self.remote + "/requests/" + filename)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.command(["download", "-s", self.session, self.remote + "/responses/" + filename, str(response)], check=False)
            if result.returncode == 0 and response.exists():
                try:
                    return json.loads(response.read_text())
                except json.JSONDecodeError:
                    pass
            time.sleep(5)
        raise TimeoutError("Remote operation timed out: " + name)

    def launch(self):
        control = self.local / "control.json"
        write(control, {"label": self.label})
        self.upload(control, "/content/qwen-comparison/control.json")
        result = self.command(["exec", "-s", self.session, "-f", str(Path(__file__).with_name("launch.py")), "--timeout", "120"])
        return result.stdout

    def stop_worker(self):
        self.command(["exec", "-s", self.session, "-f", str(Path(__file__).with_name("stop_worker.py")), "--timeout", "60"])

    def export(self, output):
        output.mkdir(parents=True, exist_ok=True)
        result = self.rpc("export", {}, timeout=600)
        if not result["ok"]:
            raise RuntimeError("Worker export failed")
        archive = self.local / "results.tar.gz"
        self.command(["download", "-s", self.session, result["result"]["archive"], str(archive)])
        extracted = self.local / "exported"
        extracted.mkdir(exist_ok=True)
        with tarfile.open(archive) as source:
            source.extractall(extracted, filter="data")
        import shutil
        shutil.copytree(extracted / self.label, output, dirs_exist_ok=True)
        for filename in ["service.log", "runtime.json"]:
            self.command(["download", "-s", self.session, self.remote + "/" + filename, str(output / filename)])
        return {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "downloaded": True}


def editor(session, label, output, key, max_turns=120, resume=None):
    bridge = Bridge(session, label)
    if resume is None:
        (output / "launch.log").write_text(bridge.launch())
    metadata = bridge.rpc("metadata", {}, timeout=600)
    if not metadata["ok"]:
        raise RuntimeError("Numerical service failed to initialize")
    write(output / ("metadata_resume.json" if resume is not None else "metadata.json"), metadata)
    task = {"target": BASE_ID, "revision": BASE_REVISION,
        "text_architecture": metadata["result"]["config"].get("text_config", metadata["result"]["config"]),
        "practice_sources": metadata["result"]["sources"], "practice_count": metadata["result"]["practice_count"],
        "goal": "Find an evidence-supported internal weight repair for math accuracy; compare base and edited behavior on practice, then select a final checkpoint."}
    history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(task, sort_keys=True)}]
    client = OpenAI(api_key=key, max_retries=0, timeout=600)
    spent, frozen, started = 0., False, time.monotonic()
    run = {"label": label, "model": MODELS[label], "status": "discovering", "api_cost_ceiling_usd": None,
           "reasoning_effort": "xhigh", "max_api_turns": max_turns, "max_forward_calls": MAX_FORWARD_CALLS,
           "created_utc": datetime.now(timezone.utc).isoformat()}
    first_turn = 1
    if resume is not None:
        history = resume["history"]
        run = json.loads((output / "run.json").read_text())
        run.update(status="discovering", recovery="recovery_001")
        for field in ["error_type", "http_status"]:
            run.pop(field, None)
        spent = run["estimated_api_cost_usd"]
        first_turn = resume["first_turn"]
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(run["created_utc"])).total_seconds()
        started = time.monotonic() - elapsed
    try:
        for turn in range(first_turn, max_turns + 1):
            if time.monotonic() - started > 4 * 3600:
                run["stop_reason"] = "four-hour editor wall limit"
                break
            request = {"model": MODELS[label], "input": history, "tools": TOOLS, "parallel_tool_calls": False,
                "store": False, "service_tier": "default", "include": ["reasoning.encrypted_content"],
                "reasoning": {"effort": "xhigh"}, "max_output_tokens": 16000}
            folder = output / "turns" / f"{turn:03d}"
            write(folder / "request.json", request)
            print(f"EDITOR {label} turn={turn}/{max_turns} api_cost=${spent:.3f}", flush=True)
            api_started = time.perf_counter()
            prefetched = resume.get("prefetched") if resume is not None and turn == first_turn else None
            if prefetched is not None:
                assert request == prefetched["request"], "Recovery request changed after its API response"
                raw = prefetched["response"]
            else:
                response = client.responses.create(**request)
                raw = response.model_dump(mode="json")
            write(folder / "response.json", raw)
            if not raw.get("usage"):
                raise RuntimeError("API usage missing")
            cost = usage_cost(label, raw["usage"])
            spent += cost
            write(folder / "accounting.json", prefetched["accounting"] if prefetched is not None else {
                "usage": raw["usage"], "estimated_api_cost_usd": cost,
                "api_wall_seconds": time.perf_counter() - api_started})
            run.update(api_turns=turn, estimated_api_cost_usd=round(spent, 6))
            write(output / "run.json", run)
            if raw["status"] != "completed":
                history.append({"role": "user", "content": "Your last response did not complete. Be concise and use a tool."})
                continue
            history.extend(strip_none(raw["output"]))
            calls = [item for item in raw["output"] if item["type"] == "function_call"]
            for call in calls:
                arguments = json.loads(call["arguments"])
                print("TOOL", label, call["name"], str(arguments.get("reason", arguments.get("rationale", "")))[:250], flush=True)
                if call["name"] not in {tool["name"] for tool in TOOLS}:
                    result = {"ok": False, "error": "Unknown editor tool"}
                else:
                    result = bridge.rpc(call["name"], arguments, timeout=3600 if call["name"] in {"generate", "probe"} else 300)
                write(folder / (call["call_id"] + ".json"), {"name": call["name"], "arguments": arguments, "response": result})
                history.append({"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result)})
                print("RESULT", label, call["name"], "ok=", result["ok"], "forward_remaining=", result.get("remaining_forward_calls"), flush=True)
                if call["name"] == "finish" and result["ok"]:
                    frozen = True
                    write(output / "editor_selection.json", result["result"])
                    break
            if frozen:
                break
            if not calls:
                history.append({"role": "user", "content": "Continue using tools. Use all forward calls, then finish to select final weights."})
            if turn >= max_turns - 5:
                history.append({"role": "user", "content": "The API-turn budget is nearly exhausted. Select final weights soon; the host will freeze current weights at the limit."})
        if not frozen:
            result = bridge.rpc("force_finish", {"rationale": "Host froze current weights at the declared API-turn or wall limit."})
            if not result["ok"]:
                raise RuntimeError("Unable to freeze candidate")
            write(output / "editor_selection.json", result["result"])
        run["status"] = "frozen"
    except Exception as exc:
        run.update(status="stopped_on_error", error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
        print("EDITOR ERROR", label, type(exc).__name__, flush=True)
        raise
    finally:
        client.close()
        run.update(estimated_api_cost_usd=round(spent, 6), wall_seconds=time.monotonic() - started)
        write(output / "run.json", run)
    return bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="weight-space-qwen-compare")
    parser.add_argument("--output", type=Path, default=ROOT / "results/comparison/qwen-astra-vs-gpt55-20261103")
    args = parser.parse_args()
    output = args.output
    if (output / "protocol.json").exists():
        raise SystemExit("This comparison already has a protocol; do not overwrite it")
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(ROOT / ".env.local").get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("No authorized project key")
    order = list(MODELS)
    random.Random(20261103).shuffle(order)
    protocol = {"protocol": "paired_qwen_editor_pilot_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": MODELS, "order": order, "order_seed": 20261103, "target": BASE_ID, "revision": BASE_REVISION,
        "reasoning_effort": "xhigh", "max_output_tokens_per_api_call": 16000, "api_cost_ceiling_usd": None,
        "api_retries": 0, "max_api_turns_per_editor": 120, "forward_calls_per_editor": 10,
        "cpu_calls_per_editor": 100, "generation_cap": 32768, "max_total_generations_per_call": 64,
        "max_probe_texts": 16, "max_probe_tokens_per_text": 32768, "practice_examples": 15458,
        "prior_repair_provided": False, "other_editor_outputs_provided": False,
        "test_uploaded_after_both_frozen": True, "test": json.loads((output / "test_seal.json").read_text()),
        "evaluation_conditions": ["base", "astra", "gpt55"], "samples_per_test_problem": 2,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "seed": 0},
        "price_rates_usd_per_million": RATES,
        "price_sources": ["https://developers.openai.com/api/docs/models/gpt-6-astra", "https://developers.openai.com/api/docs/models/gpt-5.5"],
        "prices_verified_utc": "2026-11-03", "repetitions_per_editor": 1,
        "scope": "Small matched Qwen-repair pilot; not the paper's 50-pass discovery protocol or an equivalence test."}
    write(output / "protocol.json", protocol)
    (output / "initial_payload.tar.gz").write_bytes((ROOT / ".artifacts/qwen-compare/payload.tar.gz").read_bytes())
    source = output / "source"
    source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        (source / path.name).write_bytes(path.read_bytes())
    active, completed = None, []
    state = {"status": "discovery", "session": args.session, "order": order, "completed_editors": []}
    try:
        for label in order:
            destination = output / label
            destination.mkdir(exist_ok=False)
            active = Bridge(args.session, label)
            # editor() creates an identical bridge endpoint; keep this fallback
            # for exporting the active service if its API conversation fails.
            state["active_editor"] = label
            write(output / "run.json", state)
            active = editor(args.session, label, destination, key)
            write(destination / "export.json", active.export(destination))
            completed.append(label)
            state["completed_editors"] = completed.copy()
            write(output / "run.json", state)
            if label != order[-1]:
                active.stop_worker()
        # Both choices exist and are immutable before the test is placed on VM.
        for label in order:
            assert json.loads((output / label / "frozen.json").read_text())["test_data_present_at_freeze"] is False
        test = ROOT / ".artifacts/qwen-compare/sealed_test.json"
        assert hashlib.sha256(test.read_bytes()).hexdigest() == protocol["test"]["sha256"]
        active.upload(test, active.remote + "/sealed_test.json")
        state.update(status="sealed_evaluation", active_editor=None,
                     heldout_uploaded_utc=datetime.now(timezone.utc).isoformat())
        write(output / "run.json", state)
        conditions = [{"label": label, "checkpoint": f"/content/qwen-comparison/{label}/results/selected.safetensors",
                       "manifest": f"/content/qwen-comparison/{label}/results/frozen.json"} for label in MODELS]
        result = active.rpc("evaluate_conditions", {"conditions": conditions, "test_sha256": protocol["test"]["sha256"]}, timeout=7200)
        write(output / "final_evaluation.json", result)
        if not result["ok"]:
            raise RuntimeError("Held-out evaluation failed")
        write(output / order[-1] / "export.json", active.export(output / order[-1]))
        import shutil
        shutil.copytree(output / order[-1] / "evaluation", output / "evaluation", dirs_exist_ok=True)
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="stopped_on_error", error_type=type(exc).__name__)
        print("COMPARISON ERROR", type(exc).__name__, str(exc)[:300], flush=True)
        if active is not None:
            try:
                active.export(output / active.label)
            except Exception as exported:
                state["export_error_type"] = type(exported).__name__
    finally:
        stopped = subprocess.run(["colab", "stop", "-s", args.session], capture_output=True, text=True, timeout=120)
        state["g4_stopped"] = stopped.returncode == 0
        write(output / "run.json", state)
    print("RUN", json.dumps(state), flush=True)
    if state["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
