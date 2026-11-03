"""Record the independently provisioned VM; not sent to the editor."""
from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys

data = {"python": sys.version, "requested_accelerator": "G4",
        "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                                             "--format=csv,noheader"], text=True).strip(),
        "software": {name: importlib.metadata.version(name) for name in ["torch", "vllm", "transformers", "math-verify"]}}
path = Path("/content/qwen-blind/results/runtime.json")
path.write_text(json.dumps(data, indent=2) + "\n")
print("G4 runtime metadata recorded")
