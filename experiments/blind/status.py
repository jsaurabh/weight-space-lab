"""Read-only service status; never supplies hints to the editor."""
from pathlib import Path
import json
import subprocess

path = Path("/content/qwen-blind/results/status.json")
print(path.read_text() if path.exists() else "Service initializing")
for condition in ["base", "current"]:
    saved = path.parent / ("final_" + condition + ".json")
    if saved.exists():
        result = json.loads(saved.read_text())
        print(condition, json.dumps({k: v for k, v in result.items() if k != "records"}))
print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"], text=True))
log = Path("/content/blind-service.log")
if log.exists():
    print("\n".join(log.read_text(errors="replace").replace("\r", "\n").splitlines()[-12:]))
