"""Provision numerical evaluation libraries on the separately allocated VM."""
import shutil
import subprocess
import sys

print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                               "--format=csv,noheader"], text=True), flush=True)
packages = ["vllm==0.33.0", "transformers==5.20.2", "math-verify==0.9.0"]
uv = shutil.which("uv")
command = [uv, "pip", "install", "--system", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"]
subprocess.check_call(command + packages)
print("Discovery evaluation environment ready", flush=True)
