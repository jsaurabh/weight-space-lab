"""Read-only service status; never supplies hints to the editor."""
from pathlib import Path
import subprocess

path = Path("/content/qwen-blind/results/status.json")
print(path.read_text() if path.exists() else "Service initializing")
print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"], text=True))
log = Path("/content/blind-service.log")
if log.exists():
    print("\n".join(log.read_text(errors="replace").replace("\r", "\n").splitlines()[-12:]))
