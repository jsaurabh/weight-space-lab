"""Replay the published Qwen edit, verifying raw bfloat16 hashes before use."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

BASE_ID = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
UPSTREAM_REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def tensor_sha(tensor):
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Edit verification requires original bfloat16 tensors")
    bits = tensor.detach().cpu().contiguous().view(torch.uint16).numpy()
    return hashlib.sha256(bits.tobytes()).hexdigest()


def rebuild_tensor(base, spec, archive, index):
    if spec["form"] != "sparse" or list(base.shape) != spec["shape"]:
        raise ValueError("Only the published sparse p50c edit with matching shapes is supported")
    if tensor_sha(base) != spec["base_sha256"]:
        raise ValueError("Base tensor hash differs from the published revision")
    bits = base.detach().cpu().contiguous().view(torch.uint16).numpy().copy()
    indices, values = archive[f"t{index}_idx"], archive[f"t{index}_val"]
    if indices.ndim != 1 or values.shape != indices.shape or values.dtype != np.uint16:
        raise ValueError("Malformed sparse diff")
    if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= bits.size):
        raise ValueError("Sparse index out of bounds")
    if np.unique(indices).size != indices.size:
        raise ValueError("Duplicate sparse indices")
    bits.reshape(-1)[indices] = values
    edited = torch.from_numpy(bits).view(torch.bfloat16)
    if tensor_sha(edited) != spec["edited_sha256"]:
        raise ValueError("Rebuilt tensor does not match the evaluated published edit")
    changed = int(torch.count_nonzero(base.cpu().view(torch.uint16) != edited.view(torch.uint16)))
    if changed != spec["changed_entries"]:
        raise ValueError("Changed-entry count differs from manifest")
    return edited


def read_edit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    data = (directory / "edit.npz").read_bytes()
    if hashlib.sha256(data).hexdigest() != provenance["sha256"]:
        raise ValueError("Published diff archive checksum mismatch")
    return manifest, np.load(directory / "edit.npz", allow_pickle=False)


def verify_checkpoint(base_directory, edit_directory):
    from safetensors import safe_open
    manifest, archive = read_edit(edit_directory)
    index = json.loads((base_directory / "model.safetensors.index.json").read_text())["weight_map"]
    results = []
    for i, spec in enumerate(manifest["tensors"]):
        with safe_open(base_directory / index[spec["name"]], framework="pt", device="cpu") as source:
            base = source.get_tensor(spec["name"])
        edited = rebuild_tensor(base, spec, archive, i)
        results.append({"name": spec["name"], "base_sha256": tensor_sha(base),
                        "edited_sha256": tensor_sha(edited), "changed_entries": spec["changed_entries"]})
    archive.close()
    return {"base_model": BASE_ID, "base_revision": BASE_REVISION,
            "edit_upstream_revision": UPSTREAM_REVISION, "verified": True,
            "changed_entries": sum(row["changed_entries"] for row in results), "tensors": results}


@torch.no_grad()
def apply_to_model(model, edit_directory):
    manifest, archive = read_edit(edit_directory)
    for i, spec in enumerate(manifest["tensors"]):
        parameter = model.get_parameter(spec["name"])
        edited = rebuild_tensor(parameter, spec, archive, i)
        parameter.copy_(edited.to(device=parameter.device))
    archive.close()


class PresencePenalty:
    """vLLM-style presence penalty on generated tokens, excluding the prompt."""

    def __init__(self, prompt_length, penalty=1.5):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        for row in range(input_ids.shape[0]):
            seen = input_ids[row, self.prompt_length:].unique()
            scores[row, seen] -= self.penalty
        return scores


def last_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 1
    for position in range(start + 7, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position + 1]
    return None


def grade(answer, generated):
    from math_verify import parse, verify
    boxed = last_boxed(generated)
    if boxed is None:
        return False
    try:
        gold = parse("\\boxed{" + str(answer) + "}")
        predicted = parse(boxed)
        return bool(gold and predicted and verify(gold, predicted))
    except Exception:
        return False


def repetition_fraction(text):
    words = text.split()
    if len(words) < 4:
        return 0.0
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1 - len(Counter(grams)) / len(grams)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def compare_hf(args):
    from transformers import AutoTokenizer, LogitsProcessorList, Qwen3_5ForConditionalGeneration
    verification = verify_checkpoint(args.base, args.edit)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    rows = random.Random(args.seed).sample(rows, args.limit)
    args.output.mkdir(parents=True, exist_ok=False)
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                "max_new_tokens": args.max_new_tokens, "samples_per_problem": 1, "enable_thinking": True}
    manifest = {"experiment": "published_edit_replay", "backend": "transformers",
                "device": args.device, "dtype": "bfloat16", "seed": args.seed,
                "model": BASE_ID, "revision": BASE_REVISION, "sampling": sampling,
                "verification": verification, "problems": len(rows), "status": "running",
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "selected_ids": [row.get("unique_id", str(i)) for i, row in enumerate(rows)],
                "deviations_from_published_benchmark": [
                    "Transformers instead of vLLM", "one sample per problem instead of four",
                    f"{len(rows)} MATH-500 problems instead of 500",
                    *([] if args.max_new_tokens == 32768 else [f"{args.max_new_tokens} token cap instead of 32768"])]}
    save_json(args.output / "manifest.json", manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    started = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    model.requires_grad_(False).eval()
    model.to(args.device)
    print(f"Model loaded on {args.device} in {time.perf_counter() - started:.1f}s", flush=True)
    save_json(args.output / "problems.json", rows)
    results = []
    try:
        for condition in ("base", "published_edit"):
            if condition != "base":
                apply_to_model(model, args.edit)
                print("Published edit applied; both tensor hashes match", flush=True)
            for number, row in enumerate(rows):
                identifier = row.get("unique_id", str(number))
                print(f"{condition}: problem {number + 1}/{len(rows)} ({identifier})", flush=True)
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["problem"] + SUFFIX}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
                length = inputs["input_ids"].shape[1]
                torch.manual_seed(args.seed + number)
                if args.device == "mps":
                    torch.mps.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                        max_new_tokens=args.max_new_tokens, use_cache=True, logits_to_keep=1,
                        pad_token_id=tokenizer.eos_token_id,
                        logits_processor=LogitsProcessorList([PresencePenalty(length)]))
                if args.device == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - started
                tokens = output[0, length:].cpu().tolist()
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                correct = grade(row["answer"], text)
                result = {"condition": condition, "id": identifier, "answer": row["answer"],
                          "text": text, "correct": correct, "generated_tokens": len(tokens),
                          "seconds": elapsed, "tokens_per_second": len(tokens) / elapsed,
                          "hit_token_limit": len(tokens) >= args.max_new_tokens and tokens[-1] != tokenizer.eos_token_id,
                          "thinking_closed": "</think>" in text,
                          "repetition_4gram_fraction": repetition_fraction(text)}
                results.append(result)
                with (args.output / "generations.jsonl").open("a") as target:
                    target.write(json.dumps(result) + "\n")
                print(f"  correct={correct}, tokens={len(tokens)}, {len(tokens) / elapsed:.1f} tokens/s", flush=True)
        summaries = {}
        for condition in ("base", "published_edit"):
            group = [r for r in results if r["condition"] == condition]
            summaries[condition] = {"correct": sum(r["correct"] for r in group), "total": len(group),
                                    "hit_token_limit": sum(r["hit_token_limit"] for r in group),
                                    "mean_tokens": sum(r["generated_tokens"] for r in group) / len(group),
                                    "tokens_per_second": sum(r["generated_tokens"] for r in group) / sum(r["seconds"] for r in group)}
        manifest["results"] = summaries
        manifest["status"] = "completed"
        print(json.dumps(summaries, indent=2), flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        save_json(args.output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--base", type=Path, default=Path(".artifacts/qwen-base"))
    prepare.add_argument("--dataset-directory", type=Path, default=Path(".artifacts/math500"))
    verify = commands.add_parser("verify")
    compare = commands.add_parser("compare")
    for command in (verify, compare):
        command.add_argument("--base", type=Path, default=Path(".artifacts/qwen-base"))
        command.add_argument("--edit", type=Path, default=Path("third_party/qwen_p50c"))
    compare.add_argument("--dataset", type=Path, default=Path(".artifacts/math500/test.jsonl"))
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--limit", type=int, default=1)
    compare.add_argument("--max-new-tokens", type=int, default=128)
    compare.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    compare.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "prepare":
        from huggingface_hub import hf_hub_download, snapshot_download
        snapshot_download(BASE_ID, revision=BASE_REVISION, local_dir=args.base,
                          allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"])
        for filename in ["test.jsonl", "README.md"]:
            hf_hub_download("HuggingFaceH4/MATH-500", repo_type="dataset", filename=filename,
                            revision=DATASET_REVISION, local_dir=args.dataset_directory)
        save_json(args.dataset_directory / "provenance.json",
                  {"dataset": "HuggingFaceH4/MATH-500", "revision": DATASET_REVISION})
        print("Pinned base model and MATH-500 dataset ready")
    elif args.command == "verify":
        print(json.dumps(verify_checkpoint(args.base, args.edit), indent=2))
    else:
        if args.limit < 1 or not 1 <= args.max_new_tokens <= 32768:
            parser.error("Use a positive problem count and 1–32768 new tokens")
        compare_hf(args)
