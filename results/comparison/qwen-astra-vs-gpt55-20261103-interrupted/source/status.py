"""Read numerical-worker progress without supplying anything to either editor."""
from pathlib import Path
import json
import subprocess

shared = Path("/content/qwen-comparison")
control = json.loads((shared / "control.json").read_text())
root = shared / control["label"]
status = root / "results/status.json"
print("WORKSPACE", root.name)
print(status.read_text() if status.exists() else "initializing")
print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"], text=True))
log = root / "service.log"
if log.exists():
    print("\n".join(log.read_text(errors="replace").replace("\r", "\n").splitlines()[-8:]))
