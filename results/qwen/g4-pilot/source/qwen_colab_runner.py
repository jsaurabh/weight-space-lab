"""Bounded paired MATH-500 pilot on a Colab GPU, using the published edit."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def save(name, value):
    (RESULTS / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def archive():
    with tarfile.open("/content/qwen_results.tar.gz", "w:gz") as target:
        target.add(RESULTS, arcname="qwen-colab")
    print("Result archive updated", flush=True)


def prepare():
    import qwen_edit_helper as Q
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from shutil import copyfile
    base = ROOT / "base"
    snapshot_download(Q.BASE_ID, revision=Q.BASE_REVISION, local_dir=base,
                      allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"])
    verification = Q.verify_checkpoint(base, ROOT / "edit")
    save("verification.json", verification)
    edited = ROOT / "published-edit"
    edited.mkdir(exist_ok=True)
    for source in base.iterdir():
        if source.is_file() and not source.name.endswith(".safetensors"):
            copyfile(source, edited / source.name)
    manifest, diff = Q.read_edit(ROOT / "edit")
    mapping = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in sorted(set(mapping.values())):
        tensors = load_file(base / shard, device="cpu")
        for index, spec in enumerate(manifest["tensors"]):
            if spec["name"] in tensors:
                tensors[spec["name"]] = Q.rebuild_tensor(tensors[spec["name"]], spec, diff, index)
        save_file(tensors, edited / shard, metadata={"format": "pt"})
    diff.close()
    print("Base and bit-exact edited checkpoints ready", flush=True)


def condition(name):
    import qwen_edit_helper as Q
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    problems = json.loads((RESULTS / "problems.json").read_text())
    directory = ROOT / ("base" if name == "base" else "published-edit")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": row["problem"] + Q.SUFFIX}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True) for row in problems]
    started = time.perf_counter()
    model = LLM(
        model=str(directory), dtype="bfloat16", max_model_len=34816,
        max_num_seqs=64, max_num_batched_tokens=8192, gpu_memory_utilization=0.85,
        enable_prefix_caching=False, seed=0, trust_remote_code=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    initialization_seconds = time.perf_counter() - started
    sampling = SamplingParams(n=4, temperature=0.6, top_p=0.95, top_k=20,
                              presence_penalty=1.5, max_tokens=32768, seed=0)
    started = time.perf_counter()
    outputs = model.generate(prompts, sampling, use_tqdm=True)
    generation_seconds = time.perf_counter() - started
    records = []
    for problem, request in zip(problems, outputs, strict=True):
        for sample, output in enumerate(request.outputs):
            records.append({
                "condition": name, "id": problem["unique_id"], "sample": sample, "answer": problem["answer"],
                "text": output.text, "correct": Q.grade(problem["answer"], output.text),
                "generated_tokens": len(output.token_ids), "finish_reason": output.finish_reason,
                "hit_token_limit": output.finish_reason == "length",
                "thinking_closed": "</think>" in output.text,
                "repetition_4gram_fraction": Q.repetition_fraction(output.text),
            })
    with (RESULTS / f"{name}_generations.jsonl").open("w") as target:
        for record in records:
            target.write(json.dumps(record) + "\n")
    summary = {
        "correct": sum(r["correct"] for r in records), "total": len(records),
        "hit_token_limit": sum(r["hit_token_limit"] for r in records),
        "mean_generated_tokens": sum(r["generated_tokens"] for r in records) / len(records),
        "initialization_seconds": initialization_seconds, "generation_seconds": generation_seconds,
        "aggregate_tokens_per_second": sum(r["generated_tokens"] for r in records) / generation_seconds,
    }
    save(f"{name}_summary.json", summary)
    print(name, json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["base", "published_edit"])
    args = parser.parse_args()
    if args.condition:
        condition(args.condition)
        return
    RESULTS.mkdir(exist_ok=True)
    source_directory = RESULTS / "source"
    source_directory.mkdir(exist_ok=True)
    for name in ["qwen_colab_runner.py", "qwen_edit_helper.py"]:
        (source_directory / name).write_bytes((ROOT / name).read_bytes())
    dataset = ROOT / "test.jsonl"
    problems = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    selected = random.Random(20261102).sample(problems, 16)
    save("problems.json", selected)
    manifest = {
        "experiment": "published_qwen_p50c_edit_replay", "status": "preparing",
        "dataset": "HuggingFaceH4/MATH-500", "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "dataset_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "selection_seed": 20261102, "problems": 16,
        "selected_ids": [row["unique_id"] for row in selected],
        "sampling": {"enable_thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "presence_penalty": 1.5, "max_tokens": 32768, "n": 4, "seed": 0},
        "software": {name: importlib.metadata.version(name) for name in ["vllm", "transformers", "torch", "math-verify"]},
        "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                                              "--format=csv,noheader"], text=True).strip(),
        "deviations_from_published_benchmark": ["16 problems instead of 500",
                                                "full multimodal checkpoint with text-only prompts instead of text-only export"],
        "condition_time_limit_seconds": 1200, "colab_accelerator_requested": "G4",
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ["qwen_colab_runner.py", "qwen_edit_helper.py"]},
    }
    save("manifest.json", manifest)
    started = time.perf_counter()
    try:
        prepare()
        archive()
        for name in ["base", "published_edit"]:
            manifest["status"] = "running_" + name
            save("manifest.json", manifest)
            process = subprocess.Popen([sys.executable, "-u", __file__, "--condition", name], start_new_session=True)
            try:
                code = process.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError(f"{name} exceeded the 20-minute condition limit") from None
            if code:
                raise RuntimeError(f"{name} subprocess exited with status {code}")
            archive()
        base = [json.loads(line) for line in (RESULTS / "base_generations.jsonl").read_text().splitlines()]
        edited = [json.loads(line) for line in (RESULTS / "published_edit_generations.jsonl").read_text().splitlines()]
        assert [(r["id"], r["sample"]) for r in base] == [(r["id"], r["sample"]) for r in edited]
        wins = sum(not b["correct"] and e["correct"] for b, e in zip(base, edited, strict=True))
        losses = sum(b["correct"] and not e["correct"] for b, e in zip(base, edited, strict=True))
        deltas = []
        for problem in selected:
            b = [r["correct"] for r in base if r["id"] == problem["unique_id"]]
            e = [r["correct"] for r in edited if r["id"] == problem["unique_id"]]
            deltas.append(sum(e) / len(e) - sum(b) / len(b))
        rng = random.Random(20261102)
        bootstrap = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000))
        summary = {"base": json.loads((RESULTS / "base_summary.json").read_text()),
                   "published_edit": json.loads((RESULTS / "published_edit_summary.json").read_text()),
                   "paired_sample_wins": wins, "paired_sample_losses": losses,
                   "mean_accuracy_difference": sum(deltas) / len(deltas),
                   "problem_cluster_bootstrap_95_interval": [bootstrap[249], bootstrap[9749]],
                   "bootstrap_replicates": 10000, "problem_accuracy_differences": deltas}
        save("summary.json", summary)
        manifest["status"] = "completed"
        print("FINAL", json.dumps(summary), flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        save("manifest.json", manifest)
        archive()


if __name__ == "__main__":
    main()
