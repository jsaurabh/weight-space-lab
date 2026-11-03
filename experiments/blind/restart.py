import os
from pathlib import Path
import subprocess
import sys

root = Path("/content/qwen-blind")
env = os.environ.copy()
env.update(VLLM_WORKER_MULTIPROC_METHOD="spawn", TOKENIZERS_PARALLELISM="false",
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="4")
log = Path("/content/blind-service.log").open("a")
process = subprocess.Popen([sys.executable, "-u", str(root / "backend.py")], stdout=log,
    stderr=subprocess.STDOUT, env=env, start_new_session=True)
Path("/content/blind-service.pid").write_text(str(process.pid))
print("Numerical service resumed", process.pid)
