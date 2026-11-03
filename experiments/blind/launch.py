"""Start the clean service from the explicitly prepared payload."""
import os
from pathlib import Path
import subprocess
import sys
import tarfile

root = Path("/content/qwen-blind")
root.mkdir(exist_ok=True)
with tarfile.open("/content/blind_payload.tar.gz") as payload:
    payload.extractall(root, filter="data")
env = os.environ.copy()
env.update(VLLM_WORKER_MULTIPROC_METHOD="spawn", TOKENIZERS_PARALLELISM="false",
           OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
log = Path("/content/blind-service.log").open("w")
process = subprocess.Popen([sys.executable, "-u", str(root / "backend.py")], stdout=log,
    stderr=subprocess.STDOUT, env=env, start_new_session=True)
Path("/content/blind-service.pid").write_text(str(process.pid))
print("Clean discovery service started", process.pid)
