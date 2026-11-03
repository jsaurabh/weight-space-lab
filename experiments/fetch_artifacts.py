"""Restore public Hugging Face model artifacts at a pinned revision, verifying SHA-256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen


def matches(path, info):
    if not path.is_file() or path.stat().st_size != info["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == info["sha256"]


def download(repository, revision, remote, target, info):
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{quote(remote, safe='/')}"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(Request(url, headers={"User-Agent": "weight-space-lab-artifact-fetch"}), timeout=120) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > info["bytes"]:
                        raise ValueError("Downloaded artifact exceeds its recorded size: " + remote)
                    output.write(chunk)
        if not matches(temporary, info):
            raise ValueError("Artifact checksum or size mismatch: " + remote)
        os.replace(temporary, target)
        print("DOWNLOADED", remote)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore_records(archive, root):
    root = root.resolve()
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            destination = (root / member.name).resolve()
            if not member.isfile() or not destination.is_relative_to(root / "results"):
                raise ValueError("Record archive contains an unexpected path: " + member.name)
        source.extractall(root, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--group", choices=["all", "tiny", "qwen-published", "qwen-blind", "qwen-comparison"], default="all")
    parser.add_argument("--verify-only", action="store_true", help="Check local files without downloading")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "model_artifacts.json").read_text())
    repository, revision = manifest["repo_id"], manifest["revision"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected a Hugging Face model repository and immutable commit revision")
    missing = []
    for name, info in manifest["files"].items():
        if args.group != "all" and info["group"] != args.group:
            continue
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Artifact path escapes the project directory")
        if matches(target, info):
            print("VERIFIED", name)
            continue
        if args.verify_only:
            missing.append(name)
            continue
        download(repository, revision, name, target, info)
    for name, info in manifest.get("record_archives", {}).items():
        if args.group != "all" and args.group not in info["groups"]:
            continue
        target = (root / ".artifacts" / name).resolve()
        if not target.is_relative_to(root / ".artifacts" / "records"):
            raise ValueError("Record archive path escapes its cache directory")
        if not matches(target, info):
            if args.verify_only:
                missing.append(name)
                continue
            download(repository, revision, name, target, info)
        if args.verify_only:
            print("VERIFIED", name)
        else:
            restore_records(target, root)
            print("RESTORED", name)
    if missing:
        raise SystemExit("Missing or changed artifacts: " + ", ".join(missing))


if __name__ == "__main__":
    main()
