"""Rebuild the public training pool, sealed test and clean G4 payload."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New experiment result directory")
    parser.add_argument("--verify-only", action="store_true", help="Check the original prepared inputs without changing them")
    args = parser.parse_args()
    cache = ROOT / ".artifacts/qwen-compare"
    cache.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        args.output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], {}
    for name in ["gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"]:
        path = cache / name
        url = f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{REVISION}/exp8_math/practice/{name}"
        if not path.exists():
            path.write_bytes(urlopen(url, timeout=120).read())
        data = path.read_bytes()
        original = [json.loads(line) for line in data.splitlines() if line.strip()]
        sources[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "count": len(original)}
        for r in original:
            rows.append({"id": r["id"], "source": r["source"], "question": r["problem"],
                "answer": str(r["answer"]), "solution": r.get("solution", ""),
                **{k: r[k] for k in ["subject", "level", "year"] if k in r}})
    assert len(rows) == len({r["id"] for r in rows}) == 15458
    dataset = ROOT / ".artifacts/math500/test.jsonl"
    if not dataset.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download("HuggingFaceH4/MATH-500", repo_type="dataset", filename="test.jsonl",
            revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", local_dir=dataset.parent)
    all_test = [json.loads(line) for line in dataset.read_bytes().splitlines() if line.strip()]
    exclude = {p["unique_id"] for p in json.loads((ROOT / "results/qwen/g4-pilot/problems.json").read_text())}
    exclude |= {p["unique_id"] for p in json.loads((ROOT / "results/qwen/blind-discovery/sealed_test.json").read_text())}
    rng, held = random.Random(20261104), []
    for index, subject in enumerate(sorted({r["subject"] for r in all_test})):
        candidates = [r for r in all_test if r["subject"] == subject and r["unique_id"] not in exclude]
        held.extend(rng.sample(candidates, 5 if index < 4 else 4))
    rng.shuffle(held)
    practice_bytes = (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode()
    test_bytes = (json.dumps(held, indent=2, sort_keys=True) + "\n").encode()
    seal = {"sha256": hashlib.sha256(test_bytes).hexdigest(), "count": 32, "samples_per_problem": 2,
        "seed": 20261104, "subjects": dict(Counter(r["subject"] for r in held)),
        "excluded_prior_pilot_ids": sorted(exclude), "editor_access": False}
    if args.verify_only:
        assert (cache / "practice.json").read_bytes() == practice_bytes
        assert (cache / "sealed_test.json").read_bytes() == test_bytes
        assert json.loads((args.output / "test_seal.json").read_text()) == seal
        assert json.loads((args.output / "practice_sources.json").read_text()) == sources
        print("Original training inputs and sealed test reproduced exactly")
        return
    (cache / "practice.json").write_bytes(practice_bytes)
    (cache / "sealed_test.json").write_bytes(test_bytes)
    write(args.output / "test_seal.json", seal)
    write(args.output / "practice_sources.json", sources)
    files = [(Path(__file__).with_name("backend.py"), "backend.py"),
             (Path(__file__).with_name("protocol.py"), "protocol.py"), (cache / "practice.json", "practice.json")]
    with tarfile.open(cache / "payload.tar.gz", "w:gz") as target:
        for path, name in files:
            target.add(path, arcname=name)
    write(args.output / "payload_manifest.json", {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files})
    print("Prepared clean payload and sealed test. No API or GPU calls were made.")


if __name__ == "__main__":
    main()
