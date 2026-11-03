"""Build the explicit public-data payload, excluding credentials and model weights."""
from pathlib import Path
import hashlib
import tarfile

root = Path(__file__).resolve().parents[1]
files = [
    (root / "experiments/qwen_colab_runner.py", "qwen_colab_runner.py"),
    (root / "src/weight_space_lab/qwen.py", "qwen_edit_helper.py"),
    (root / ".artifacts/math500/test.jsonl", "test.jsonl"),
]
files.extend((root / "third_party/qwen_p50c" / name, "edit/" + name)
             for name in ["manifest.json", "edit.npz", "provenance.json", "LICENSE.upstream"])
for source, _ in files:
    if not source.is_file():
        raise SystemExit(f"Missing {source}; run weight-space-qwen prepare first")
destination = root / ".artifacts/qwen_payload.tar.gz"
destination.parent.mkdir(exist_ok=True)
with tarfile.open(destination, "w:gz") as target:
    for source, name in files:
        target.add(source, arcname=name)
print("Payload:", destination)
print("SHA-256:", hashlib.sha256(destination.read_bytes()).hexdigest())
