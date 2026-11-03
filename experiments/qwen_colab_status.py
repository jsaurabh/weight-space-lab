"""Read-only progress snapshot from the remote evaluator."""
from pathlib import Path
import json
import os

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
log = Path("/content/qwen-run.log")
if log.exists():
    print(log.read_text(errors="replace")[-6000:])
