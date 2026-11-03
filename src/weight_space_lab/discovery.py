"""Bounded API discovery with numeric-only submissions and CPU evaluation."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random

from dotenv import dotenv_values
from openai import OpenAI
import torch

from . import model as model_module
from .evaluate import evaluate, examples, state_digest, write_report
from .model import TinyTransformer
from .proposals import PROPOSAL_SCHEMA, load_proposal
from .specs import CONFIGS, DESCRIPTIONS, PROMPT_EXAMPLES, VOCABS

# Standard text prices checked 2026-11-02. Estimates are not an invoice or an
# account-level spending limit. No automatic API retries or optional paid tools.
RATES = {"gpt-6-astra": {"input": 10.0, "cached_input": 1.0, "cache_write": 12.5, "output": 50.0}}
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-6-astra"

SYSTEM = (
    "You are programming a tiny transformer by setting its weights directly. "
    "Solve the specified bounded task using the unchanged architecture. "
    "There is no training, fitting, optimizer, gradient computation, pretrained target, "
    "external computation of answers, or ability to change the forward function. "
    "You have no tools or repository access. Return the structured numeric proposal. "
    "Every attempt starts from all-zero weights, so each proposal must be complete. "
    "Explain your mechanism concisely. Do not merely describe weights: supply them."
)


def task_prompt(task):
    model = TinyTransformer(CONFIGS[task])
    return (
        f"TASK\n{DESCRIPTIONS[task]}\n"
        f"Vocabulary indices: {dict(enumerate(VOCABS[task]))}\n"
        f"Fixed config: {json.dumps(asdict(CONFIGS[task]))}\n"
        f"Tensor names and shapes: {json.dumps(model.inventory()['tensors'])}\n\n"
        "EXACT NETWORK IMPLEMENTATION\n" + inspect.getsource(model_module) + "\n\n"
        "ASSIGNMENT FORMAT\n"
        "All tensors start at zero. Each patch names a tensor and supplies one list "
        "of indices per tensor axis. The selected elements are the Cartesian product "
        "of these lists, in row-major order (last axis changes fastest). "
        "values contains either one number, broadcast to that product, or exactly "
        "one number per selected element. A vector uses one axis; a matrix uses two. "
        "Patches apply in order; later assignments overwrite earlier ones. "
        "Omitted elements remain zero. Linear weight matrices have shape (out, in). "
        "Use only finite numeric constants, absolute value at most 1e6. "
        "Index lists must be nonempty and contain no duplicates. "
        "Example of syntax only: {\"tensor\":\"query.weight\","
        "\"axes\":[[0,1],[0,1]],\"values\":[1,0,0,1]} assigns a 2x2 identity block. "
        "You may use dense blocks or broadcast assignments to keep your JSON compact. "
        "The evaluator uses CPU float64, causal masking and greedy token selection. "
        "Only development scores and at most five failures are available between attempts. "
        "The final candidate is frozen before evaluation on the unseen remainder."
    )


@dataclass
class Budget:
    limit: float
    spent: float = 0.0

    def check_request(self, request):
        rates = RATES[request["model"]]
        # UTF-8 bytes conservatively upper-bound ordinary BPE text tokens;
        # allowance covers message framing and schema. Reserve cache-write premium.
        input_bound = len(json.dumps(request, ensure_ascii=False).encode()) + 2048
        worst = (input_bound * rates["input"] * 1.25 + request["max_output_tokens"] * rates["output"]) / 1e6
        if self.spent + worst > self.limit:
            raise ValueError("Estimated cost ceiling would be exceeded; no request sent")
        return round(worst, 6)

    def record(self, model, usage):
        if usage is None:
            raise ValueError("Missing API usage; stopping because cost cannot be accounted for")
        rates = RATES[model]
        details = usage.get("input_tokens_details") or {}
        cached = details.get("cached_tokens", 0) or 0
        writes = details.get("cache_write_tokens", 0) or 0
        uncached = usage["input_tokens"] - cached - writes
        if min(cached, writes, uncached) < 0:
            raise ValueError("Inconsistent input token accounting")
        cost = (uncached * rates["input"] + cached * rates["cached_input"] +
                writes * rates["cache_write"] + usage["output_tokens"] * rates["output"]) / 1e6
        self.spent += cost
        return round(cost, 6)


def split_examples(task, seed, size=200):
    rows = list(examples(task))
    development = random.Random(seed).sample(rows, min(size, len(rows) - 1))
    development_prompts = {row[0] for row in development}
    visible = development_prompts | set(PROMPT_EXAMPLES[task])
    holdout = [row for row in rows if row[0] not in visible]
    return development, holdout


def run_task(client, task, destination, budget, *, model_name, attempts, max_output_tokens, seed):
    destination.mkdir(parents=True, exist_ok=False)
    development, holdout = split_examples(task, seed)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_prompt(task)}]
    write_report({"development_prompts": [r[0] for r in development],
                  "prompt_demonstrations": list(PROMPT_EXAMPLES[task]),
                  "holdout_count": len(holdout), "seed": seed}, destination / "split.json")
    records, best = [], None
    for number in range(1, attempts + 1):
        folder = destination / f"attempt-{number:02d}"
        folder.mkdir()
        request = {
            "model": model_name, "input": messages.copy(), "store": False,
            "service_tier": "default", "reasoning": {"effort": "high"},
            "max_output_tokens": max_output_tokens,
            "text": {"format": {"type": "json_schema", "name": "weight_proposal",
                                "strict": True, "schema": PROPOSAL_SCHEMA}},
        }
        record = {"attempt": number}
        try:
            record["reserved_cost_upper_estimate_usd"] = budget.check_request(request)
        except ValueError as exc:
            record["not_sent"] = str(exc)
            records.append(record)
            break
        write_report(request, folder / "request.json")
        print(f"{task}: requesting attempt {number}/{attempts}", flush=True)
        try:
            response = client.responses.create(**request)
        except Exception as exc:
            # Do not write exception bodies, headers, credentials, or request URLs.
            record["api_error"] = {"type": type(exc).__name__,
                                   "status": getattr(exc, "status_code", None),
                                   "code": getattr(exc, "code", None)}
            records.append(record)
            write_report({"attempts": records}, destination / "attempts.json")
            raise RuntimeError(f"API request failed: {type(exc).__name__}; see attempts.json") from None
        raw = response.model_dump(mode="json")
        write_report(raw, folder / "response.json")
        usage = raw.get("usage")
        record["usage"] = usage
        record["estimated_cost_usd"] = budget.record(model_name, usage)
        record["response_id"] = response.id
        record["returned_model"] = response.model
        record["response_status"] = response.status
        output = response.output_text
        try:
            if response.status != "completed":
                raise ValueError(f"Response did not complete: {response.status}")
            proposal = json.loads(output)
            candidate = load_proposal(task, proposal)
            write_report(proposal, folder / "proposal.json")
            score = evaluate(candidate, task, rows=development, provenance="api_discovery_development")
            write_report(score, folder / "development.json")
            record.update({"development_correct": score["correct"], "development_total": score["total"],
                           "state_sha256": score["state_sha256"]})
            if best is None or score["correct"] > best[0]:
                best = (score["correct"], number, proposal, candidate)
            feedback = {"correct": score["correct"], "total": score["total"],
                        "failures": score["failures_first_10"][:5]}
            print(f"{task}: development {score['correct']}/{score['total']}; "
                  f"estimated cumulative API cost ${budget.spent:.4f}", flush=True)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            record["proposal_error"] = str(exc)
            feedback = {"proposal_error": str(exc)}
            print(f"{task}: proposal rejected: {exc}", flush=True)
        records.append(record)
        write_report({"attempts": records}, destination / "attempts.json")
        if "development_correct" in record and record["development_correct"] == len(development):
            break
        # Only prior proposal text and DEVELOPMENT feedback enter the next request.
        messages.append({"role": "assistant", "content": output or "No complete proposal was returned."})
        messages.append({"role": "user", "content": "Development feedback: " + json.dumps(feedback) +
                         "\nRevise your construction. Submit a complete replacement starting from zero."})

    if best is None:
        result = {"task": task, "status": "no_valid_candidate", "attempt_count": len(records)}
        write_report(result, destination / "summary.json")
        return result
    _, chosen_attempt, proposal, candidate = best
    # Freeze the selection using development scores alone, BEFORE any final score.
    write_report(proposal, destination / "selected.json")
    write_report({"attempt": chosen_attempt, "selection_rule": "highest development accuracy; earliest tie",
                  "state_sha256": state_digest(candidate)}, destination / "selection.json")
    torch.save({"state_dict": candidate.state_dict(), "config": asdict(candidate.config),
                "task": task, "provenance": "api_discovery"}, destination / "weights.pt")
    full = evaluate(candidate, task, provenance="api_discovery")
    unseen = evaluate(candidate, task, rows=holdout, provenance="api_discovery_holdout")
    write_report(full, destination / "final_all.json")
    write_report(unseen, destination / "final_holdout.json")
    baselines = {}
    for label in ("zero", "random_seed_1337"):
        baseline = TinyTransformer(CONFIGS[task])
        if label != "zero":
            generator = torch.Generator(device="cpu").manual_seed(1337)
            with torch.no_grad():
                for parameter in baseline.parameters():
                    parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * 0.1)
        baseline_report = evaluate(baseline, task, rows=holdout, provenance=f"baseline_{label}")
        write_report(baseline_report, destination / f"baseline_{label}.json")
        baselines[label] = baseline_report["accuracy"]
    result = {
        "task": task, "status": "evaluated", "selected_attempt": chosen_attempt,
        "attempt_count": len(records), "development_count": len(development),
        "full_correct": full["correct"], "full_total": full["total"],
        "holdout_correct": unseen["correct"], "holdout_total": unseen["total"],
        "stored_elements": full["stored_elements"], "nonzero_elements": full["nonzero_elements"],
        "state_sha256": full["state_sha256"], "baseline_holdout_accuracy": baselines,
    }
    write_report(result, destination / "summary.json")
    print(f"{task}: FINAL all={full['correct']}/{full['total']}; "
          f"unseen={unseen['correct']}/{unseen['total']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--task", choices=[*CONFIGS, "all"], default="all")
    run.add_argument("--model", choices=RATES, default="gpt-6-astra")
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--max-output-tokens", type=int, default=16000)
    run.add_argument("--max-cost-usd", type=float, default=5.0)
    run.add_argument("--seed", type=int, default=20261102)
    run.add_argument("--output", type=Path)
    replay = commands.add_parser("replay")
    replay.add_argument("proposal", type=Path)
    replay.add_argument("--task", choices=CONFIGS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "replay":
        candidate = load_proposal(args.task, json.loads(args.proposal.read_text()))
        print(json.dumps(evaluate(candidate, args.task, provenance="replayed_numeric_proposal"), indent=2))
        return
    if not 1 <= args.max_attempts <= 10 or not 1 <= args.max_output_tokens <= 32768:
        parser.error("Use 1–10 attempts and 1–32768 output tokens")
    if not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0:
        parser.error("Cost ceiling must be positive and finite")
    # Read only the named local secret; never put dotenv values into logs.
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(Path.cwd() / ".env.local").get("OPENAI_API_KEY")
    if not key:
        parser.error("No OPENAI_API_KEY configured")
    destination = args.output or Path("results/discovery") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True, exist_ok=False)
    budget = Budget(args.max_cost_usd)
    client = OpenAI(api_key=key, max_retries=0, timeout=240)
    manifest = {
        "protocol": "numeric_only_fixed_architecture_v1", "model": args.model,
        "reasoning_effort": "high", "max_attempts_per_task": args.max_attempts,
        "max_output_tokens_per_attempt": args.max_output_tokens,
        "estimated_cost_ceiling_usd": args.max_cost_usd,
        "price_source": PRICE_SOURCE, "price_rates_per_million": RATES[args.model],
        "price_checked_date": "2026-11-02", "seed": args.seed,
        "reference_weights_provided": False, "tools_provided": False,
        "final_holdout_policy": "exclude development examples and explicit prompt demonstrations",
        "model_source_sha256": hashlib.sha256(inspect.getsource(model_module).encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(json.dumps(PROPOSAL_SCHEMA, sort_keys=True).encode()).hexdigest(),
        "tasks": [], "status": "running",
    }
    write_report(manifest, destination / "manifest.json")
    try:
        for task in CONFIGS if args.task == "all" else [args.task]:
            manifest["tasks"].append(run_task(
                client, task, destination / task, budget, model_name=args.model,
                attempts=args.max_attempts, max_output_tokens=args.max_output_tokens, seed=args.seed))
            manifest["estimated_cost_usd"] = round(budget.spent, 6)
            write_report(manifest, destination / "manifest.json")
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "stopped_on_error"
        manifest["error_type"] = type(exc).__name__
        print(f"Discovery stopped: {type(exc).__name__}; see saved attempt metadata", flush=True)
    finally:
        client.close()
        manifest["estimated_cost_usd"] = round(budget.spent, 6)
        write_report(manifest, destination / "manifest.json")
    print(f"Reports: {destination}; estimated API cost ${budget.spent:.4f}", flush=True)
    success = manifest["status"] == "completed" and all(
        r.get("full_correct", -1) == r.get("full_total") for r in manifest["tasks"])
    raise SystemExit(0 if success else 1)
