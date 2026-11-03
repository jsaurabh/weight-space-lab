"""Restore only Astra's own observations after the recorded API interruption."""
from pathlib import Path
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile

shared = Path('/content/qwen-comparison')
root = shared / 'astra'
root.mkdir(exist_ok=False)
for name in ['requests', 'responses']:
    (root / name).mkdir()
for archive in ['payload.tar.gz', 'resume.tar.gz']:
    with tarfile.open(shared / archive) as source:
        source.extractall(root, filter='data')
env = os.environ.copy()
env.update(QWEN_BASE_DIRECTORY=str(shared / 'base'), VLLM_WORKER_MULTIPROC_METHOD='spawn',
           TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='4')
process = subprocess.Popen([sys.executable, '-u', str(root / 'resume_service.py')],
    stdout=(root / 'service.log').open('w'), stderr=subprocess.STDOUT, env=env, start_new_session=True)
(root / 'pid').write_text(str(process.pid))
(shared / 'control.json').write_text(json.dumps({'label': 'astra'}))
runtime = {'python': sys.version, 'requested_accelerator': 'G4', 'recovery': 'recovery_001',
    'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version', '--format=csv,noheader'], text=True).strip(),
    'software': {p: importlib.metadata.version(p) for p in ['torch', 'vllm', 'transformers', 'math-verify']}}
(root / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
print(json.dumps({'label': 'astra', 'pid': process.pid, 'runtime': runtime}))
