"""Launch an isolated background evaluator; only the explicit payload is sent."""
from pathlib import Path
import os
import subprocess
import sys
import tarfile

root = Path("/content/qwen-work")
root.mkdir(exist_ok=True)
with tarfile.open("/content/qwen_payload.tar.gz") as source:
    source.extractall(root, filter="data")
environment = os.environ.copy()
environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
environment["TOKENIZERS_PARALLELISM"] = "false"
log = Path("/content/qwen-run.log").open("w")
process = subprocess.Popen([sys.executable, "-u", str(root / "qwen_colab_runner.py")],
                           stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
Path("/content/qwen-run.pid").write_text(str(process.pid))
print("Evaluator started; PID", process.pid)
