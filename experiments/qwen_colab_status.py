"""Read-only progress snapshot from the remote evaluator."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess

print("Snapshot UTC:", datetime.now(timezone.utc).isoformat())
print("GPU:", subprocess.check_output([
    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"
], text=True).strip())

pid_path = Path("/content/qwen-run.pid")
if pid_path.exists():
    pid = int(pid_path.read_text())
    try:
        os.kill(pid, 0)
        print("Evaluator process exists:", pid)
    except ProcessLookupError:
        print("Evaluator process finished")
manifest = Path("/content/qwen-work/results/manifest.json")
if manifest.exists():
    data = json.loads(manifest.read_text())
    print("Status:", data["status"])
    for condition in ["base", "published_edit"]:
        summary = manifest.parent / f"{condition}_summary.json"
        if summary.exists():
            print(condition, summary.read_text())
log = Path("/content/qwen-run.log")
if log.exists():
    lines = log.read_text(errors="replace").replace("\r", "\n").splitlines()
    print("\n".join(lines[-14:]))
