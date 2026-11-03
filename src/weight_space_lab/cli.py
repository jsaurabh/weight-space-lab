import argparse
import json
from pathlib import Path

import torch

from .evaluate import encode, evaluate, write_report
from .programs import BUILDERS, VOCABS


def main():
    parser = argparse.ArgumentParser(description="Run CPU-only weight-space reference experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Exhaustively evaluate reference constructions")
    run.add_argument("--task", choices=[*BUILDERS, "all"], default="all")
    run.add_argument("--output", type=Path, default=Path("results/reference"))
    run.add_argument("--batch-size", type=int, default=256)
    inspect = commands.add_parser("inspect", help="Inspect one example's internal computation")
    inspect.add_argument("task", choices=BUILDERS)
    inspect.add_argument("prompt", help="Parity: 1–10 bits. Addition: e.g. 12+34=")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "run":
        tasks = list(BUILDERS) if args.task == "all" else [args.task]
        passed = True
        for task in tasks:
            model = BUILDERS[task]()
            report = evaluate(model, task, args.batch_size)
            destination = args.output / task
            destination.mkdir(parents=True, exist_ok=True)
            write_report(report, destination / "report.json")
            torch.save({"state_dict": model.state_dict(), "config": report["config"],
                        "task": task, "provenance": report["provenance"]}, destination / "weights.pt")
            print(f"{task}: {report['correct']}/{report['total']} correct; "
                  f"{report['stored_elements']} stored values; {report['seconds']:.3f}s on CPU")
            passed &= report["correct"] == report["total"]
        raise SystemExit(0 if passed else 1)
    model = BUILDERS[args.task]()
    prompt = args.prompt
    if args.task == "parity":
        if not 1 <= len(prompt) <= 10 or set(prompt) - {"0", "1"}:
            parser.error("Parity requires 1–10 binary digits")
        prompt = prompt.rjust(10, "_") + "="
    elif len(prompt) != 6 or prompt[2] != "+" or prompt[5] != "=" or any(c not in "0123456789" for c in prompt[:2] + prompt[3:5]):
        parser.error("Addition requires a prompt such as 12+34=")
    tokens = encode([prompt], VOCABS[args.task])
    steps = 1 if args.task == "parity" else 3
    trace_steps = []
    with torch.inference_mode():
        for _ in range(steps):
            logits, trace = model(tokens, trace=True)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            trace_steps.append({"next_token": VOCABS[args.task][int(next_token.item())],
                                "attention_last_position": trace["attention"][0, :, -1].tolist(),
                                "residual_after_attention": trace["after_attention"][0, -1].tolist(),
                                "residual_after_mlp": trace["after_mlp"][0, -1].tolist()})
            tokens = torch.cat((tokens, next_token), dim=1)
    print(json.dumps({"prompt": prompt, "steps": trace_steps}, indent=2))
