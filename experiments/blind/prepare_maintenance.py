"""Preserve recorded observations and the editor's exact checkpoint for a transport restart."""
import json
import os
from pathlib import Path
import shutil
import signal
import time

root = Path("/content/qwen-blind")
out = root / "results"
status_file = out / "status.json"
status = json.loads(status_file.read_text())
records = [json.loads(line) for line in (out / "tool_log.jsonl").read_text().splitlines()]
successful = sum(r["request"]["name"] in {"generate", "probe"} and r["response"]["ok"] for r in records)
saved = {"checkpoint_label": "transport_backup", "successful_forward_calls": successful,
         "analysis_calls": status["analysis_calls"],
         "elapsed_seconds": status["elapsed_seconds"] + time.time() - status_file.stat().st_mtime,
         "refunded_failed_forward_requests": status["forward_calls"] - successful,
         "reason": "Replace rejected executable-callback transport with native disk checkpoint reload; no candidate changes"}
(root / "maintenance_snapshot.json").write_text(json.dumps(saved, indent=2) + "\n")
history = out / "source_before_maintenance"
history.mkdir(exist_ok=True)
for path in (out / "source").iterdir():
    if path.is_file(): shutil.copyfile(path, history / path.name)
shutil.copyfile(out / "environment.json", history / "environment.json")
shutil.copyfile(root / "maintenance_snapshot.json", out / "maintenance_snapshot.json")
pid = int(Path("/content/blind-service.pid").read_text())
os.killpg(pid, signal.SIGTERM)
print("Service stopped for transport maintenance; exact editor checkpoint and artifacts retained")
print(json.dumps(saved))
