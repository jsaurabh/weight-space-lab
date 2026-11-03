"""Install the published evaluator versions on the already allocated Colab VM."""
import subprocess
import sys

print(subprocess.check_output([
    "nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"
], text=True), flush=True)
print("Installing pinned evaluation dependencies", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "vllm==0.33.0", "transformers==5.20.2", "math-verify==0.9.0",
])
print("Colab evaluation dependencies ready", flush=True)
