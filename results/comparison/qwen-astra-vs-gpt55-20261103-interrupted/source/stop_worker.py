"""Stop only the current numerical worker, retaining its VM files for export."""
from pathlib import Path
import json
import os
import signal
import time

shared = Path("/content/qwen-comparison")
control = json.loads((shared / "control.json").read_text())
root = shared / control["label"]
pid = int((root / "pid").read_text())
try:
    os.killpg(pid, signal.SIGTERM)
except ProcessLookupError:
    pass
for _ in range(30):
    try:
        finished, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        break
    if finished:
        break
    time.sleep(1)
print("Stopped numerical worker:", control["label"])
