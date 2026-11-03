"""Materialize the frozen editor candidate from saved tensors; no API or GPU needed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from safetensors.torch import load_file, save_file
import torch

from protocol import BASE_ID, BASE_REVISION


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("results/qwen/blind-discovery"))
    parser.add_argument("--base", type=Path, required=True, help="Local copy of the pinned base checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the complete edited checkpoint")
    args = parser.parse_args()
    frozen = json.loads((args.result / "frozen.json").read_text())
    selected = args.result / "selected.safetensors"
    if hashlib.sha256(selected.read_bytes()).hexdigest() != frozen["selected_file_sha256"]:
        raise ValueError("Selected checkpoint checksum differs from the frozen choice")
    edits = load_file(selected)
    manifest = {e["name"]: e for e in frozen["edits"]}
    if edits.keys() != manifest.keys():
        raise ValueError("Selected tensor names differ from the frozen manifest")
    mapping = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    # Validate the edited base tensors before creating output. The model ID and
    # revision are documented requirements; hashes bind every replaced tensor.
    for shard in sorted({mapping[name] for name in edits}):
        tensors = load_file(args.base / shard)
        for name, edited in edits.items():
            if mapping[name] != shard:
                continue
            info = manifest[name]
            base = tensors[name]
            if digest(base) != info["base_sha256"] or digest(edited) != info["edited_sha256"]:
                raise ValueError("Tensor checksum mismatch: " + name)
            if base.shape != edited.shape or base.dtype != edited.dtype:
                raise ValueError("Tensor shape or dtype mismatch: " + name)
        del tensors
    args.output.mkdir(parents=True, exist_ok=False)
    for path in args.base.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors"):
            shutil.copyfile(path, args.output / path.name)
    for shard in sorted(set(mapping.values())):
        tensors = load_file(args.base / shard)
        tensors.update({name: value for name, value in edits.items() if mapping[name] == shard})
        save_file(tensors, args.output / shard, metadata={"format": "pt"})
    print(json.dumps({"model": BASE_ID, "revision": BASE_REVISION, "output": str(args.output),
                      "edited_tensors": len(edits), "verified_frozen_bytes": True}))


if __name__ == "__main__":
    main()
