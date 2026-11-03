"""Local API editor loop. Credentials stay here; only explicit RPC files go to G4."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import uuid

from dotenv import dotenv_values
from openai import OpenAI

from protocol import BASE_ID, BASE_REVISION, SYSTEM, TOOLS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Bridge:
    def __init__(self, session, directory):
        self.session, self.directory, self.number = session, directory, 0
        self.prefix = uuid.uuid4().hex[:12]
        directory.mkdir(parents=True, exist_ok=True)

    def command(self, args, *, check=True):
        result = subprocess.run(["colab", *args], capture_output=True, text=True, timeout=120)
        if check and result.returncode:
            raise RuntimeError("Colab transport failed: " + result.stderr[-1000:] + result.stdout[-1000:])
        return result

    def upload(self, local, remote):
        self.command(["upload", "-s", self.session, str(local), remote])

    def rpc(self, name, arguments, timeout=1200):
        self.number += 1
        filename = f"{self.prefix}-{self.number:06d}.json"
        request = self.directory / ("request-" + filename)
        response = self.directory / ("response-" + filename)
        write(request, {"id": self.number, "name": name, "arguments": arguments})
        self.upload(request, "/content/qwen-blind/requests/" + filename)
        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            result = self.command(["download", "-s", self.session,
                                   "/content/qwen-blind/responses/" + filename, str(response)], check=False)
            if result.returncode == 0 and response.is_file():
                return json.loads(response.read_text())
            attempt += 1
            time.sleep(1 if attempt < 3 else 5)
        raise TimeoutError(f"Remote operation {name} timed out")


def usage_cost(usage):
    details = usage.get("input_tokens_details") or {}
    cached, writes = details.get("cached_tokens", 0) or 0, details.get("cache_write_tokens", 0) or 0
    ordinary = usage["input_tokens"] - cached - writes
    if min(ordinary, cached, writes) < 0:
        raise ValueError("Invalid token usage accounting")
    long = usage["input_tokens"] > 272000
    return ((ordinary * 10 + cached * 1 + writes * 12.5) * (2 if long else 1)
            + usage["output_tokens"] * 50 * (1.5 if long else 1)) / 1e6


def check_budget(spent, ceiling, request, previous_usage, new_text_bytes):
    # Replayed encrypted reasoning is charged as its underlying tokens, not as
    # ciphertext bytes. Bound new input from the preceding actual usage.
    if previous_usage:
        input_bound = previous_usage["input_tokens"] + previous_usage["output_tokens"] + new_text_bytes + 4096
    else:
        input_bound = len(json.dumps(request).encode()) + 4096
    long = input_bound > 272000
    reserve = (input_bound * 12.5 * (2 if long else 1) + request["max_output_tokens"] * 50 * (1.5 if long else 1)) / 1e6
    if spent + reserve > ceiling:
        raise ValueError("Next request would exceed the conservative estimated API ceiling")
    return reserve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="weight-space-qwen-blind")
    parser.add_argument("--output", type=Path, default=Path("results/qwen/blind-discovery"))
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-cost-usd", type=float, default=25.)
    parser.add_argument("--skip-final-evaluation", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(Path.cwd() / ".env.local").get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("No authorized project API key available")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    bridge = Bridge(args.session, Path(".artifacts/blind-bridge"))
    client = OpenAI(api_key=key, max_retries=0, timeout=600)
    manifest = {"protocol": "blind_qwen_discovery_v1", "model": "gpt-6-astra", "reasoning_effort": "max",
        "max_api_turns": args.max_turns, "max_forward_calls": 20, "estimated_api_ceiling_usd": args.max_cost_usd,
        "prior_conversation_provided": False, "external_repair_files_provided": False,
        "sealed_test_uploaded_during_discovery": False, "status": "initializing", "api_cost_usd_estimated": 0.,
        "session": args.session, "requested_accelerator": "G4", "created_utc": datetime.now(timezone.utc).isoformat()}
    write(output / "run.json", manifest)
    spent, previous_usage, new_bytes = 0., None, 0
    frozen = False
    started = time.monotonic()
    try:
        metadata = bridge.rpc("metadata", {}, timeout=240)
        if not metadata["ok"]:
            raise RuntimeError("Clean service metadata unavailable")
        write(output / "base_metadata.json", metadata)
        info = metadata["result"]
        architecture = info["config"].get("text_config", info["config"])
        task = {"target": BASE_ID, "revision": BASE_REVISION, "text_architecture": architecture,
                "tensor_catalog": "names, shapes and config are available in python_analysis",
                "practice_examples_available": info["practice_count"],
                "goal": "Improve math answer accuracy by discovering and applying a useful internal weight repair. Inspect behavior and weights, run justified experiments, and finalize your selected checkpoint. Use comparable practice IDs when evaluating candidates."}
        history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(task)}]
        manifest["status"] = "discovering"
        for turn in range(1, args.max_turns + 1):
            if time.monotonic() - started > 3 * 3600:
                manifest["stop_reason"] = "three-hour discovery wall limit"
                break
            request = {"model": "gpt-6-astra", "input": history, "tools": TOOLS,
                "parallel_tool_calls": False, "store": False, "service_tier": "default",
                "include": ["reasoning.encrypted_content"], "reasoning": {"effort": "max"}, "max_output_tokens": 16000}
            try:
                reserved = check_budget(spent, args.max_cost_usd, request, previous_usage, new_bytes)
            except ValueError as exc:
                manifest["stop_reason"] = str(exc)
                break
            folder = output / "turns" / f"{turn:03d}"
            folder.mkdir(parents=True)
            write(folder / "request.json", request)
            print(f"EDITOR turn {turn}/{args.max_turns}; estimated spend ${spent:.3f}", flush=True)
            response = client.responses.create(**request)
            raw = response.model_dump(mode="json")
            write(folder / "response.json", raw)
            if not raw.get("usage"):
                raise RuntimeError("API usage missing; stopping cost accounting")
            cost = usage_cost(raw["usage"])
            spent += cost
            previous_usage = raw["usage"]
            manifest.update(api_cost_usd_estimated=round(spent, 6), api_turns=turn)
            write(folder / "accounting.json", {"usage": raw["usage"], "cost_usd_estimated": cost,
                                              "reserved_estimate_usd": reserved})
            if response.status != "completed":
                history.append({"role": "user", "content": "The last response did not complete. Make the next response concise and use a tool."})
                new_bytes = 300
                write(output / "run.json", manifest)
                continue
            items = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
            history.extend(items)
            calls = [item for item in response.output if item.type == "function_call"]
            new_bytes = 0
            for call in calls:
                arguments = json.loads(call.arguments)
                print(f"TOOL {call.name}: " + str(arguments.get("reason", arguments.get("rationale", "")))[:250], flush=True)
                if call.name not in {tool["name"] for tool in TOOLS}:
                    result = {"ok": False, "error": "Unknown editor tool"}
                else:
                    result = bridge.rpc(call.name, arguments, timeout=1500 if call.name in {"generate", "probe"} else 240)
                write(folder / (call.call_id + ".json"), {"name": call.name, "arguments": arguments, "response": result})
                text = json.dumps(result)
                history.append({"type": "function_call_output", "call_id": call.call_id, "output": text})
                new_bytes += len(text.encode())
                print("RESULT", call.name, "ok=", result["ok"], "forward remaining=", result.get("remaining_forward_calls"), flush=True)
                if call.name == "finish" and result["ok"]:
                    frozen = True
                    write(output / "editor_selection.json", result["result"])
                    break
            if frozen:
                break
            if not calls:
                text = "Use a tool to continue the experiment, or use finish to select and freeze your final checkpoint."
                history.append({"role": "user", "content": text})
                new_bytes += len(text)
            if turn >= args.max_turns - 2:
                text = "The API turn budget is almost exhausted. Select your final checkpoint with finish now."
                history.append({"role": "user", "content": text})
                new_bytes += len(text)
            write(output / "run.json", manifest)
        if not frozen:
            result = bridge.rpc("finish", {"checkpoint_label": "", "rationale": "Host froze the current candidate at the declared API/time ceiling; editor did not submit finish."})
            if not result["ok"]:
                raise RuntimeError("Unable to freeze the final candidate")
            write(output / "editor_selection.json", result["result"])
            manifest["selection_was_editor_finalized"] = False
            frozen = True
        else:
            manifest["selection_was_editor_finalized"] = True
        manifest["status"] = "frozen"
        write(output / "run.json", manifest)
        if not args.skip_final_evaluation:
            test = Path(".artifacts/blind/sealed_test.json")
            seal = json.loads(Path(".artifacts/blind/test_seal.json").read_text())
            if hashlib.sha256(test.read_bytes()).hexdigest() != seal["sha256"]:
                raise RuntimeError("Sealed test digest changed")
            write(output / "test_seal.json", seal)
            bridge.upload(test, "/content/qwen-blind/sealed_test.json")
            manifest["status"] = "independent_final_evaluation"
            write(output / "run.json", manifest)
            print("EDITOR FROZEN. Starting independent test evaluation; no further API calls.", flush=True)
            result = bridge.rpc("final_evaluate", {}, timeout=3000)
            write(output / "final_evaluation.json", result)
            if not result["ok"]:
                raise RuntimeError("Final evaluation failed")
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "stopped_on_error"
        manifest["error_type"] = type(exc).__name__
        print("Stopped:", type(exc).__name__, str(exc)[:500], flush=True)
    finally:
        client.close()
        manifest["api_cost_usd_estimated"] = round(spent, 6)
        manifest["wall_seconds"] = time.monotonic() - started
        try:
            exported = bridge.rpc("export", {}, timeout=240)
            if exported["ok"]:
                archive = Path(".artifacts/blind_results.tar.gz")
                bridge.command(["download", "-s", args.session, exported["result"]["archive"], str(archive)])
                extracted = Path(".artifacts/blind-extracted")
                extracted.mkdir(exist_ok=True)
                with tarfile.open(archive) as source:
                    source.extractall(extracted, filter="data")
                manifest["remote_archive_downloaded"] = True
                manifest["remote_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        except Exception as exc:
            manifest["export_error_type"] = type(exc).__name__
        try:
            bridge.command(["download", "-s", args.session, "/content/blind-service.log", str(output / "service.log")], check=False)
        finally:
            stopped = bridge.command(["stop", "-s", args.session], check=False)
            manifest["g4_stopped"] = stopped.returncode == 0
        write(output / "run.json", manifest)
    print("RUN", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
