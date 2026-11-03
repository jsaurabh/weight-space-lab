"""Replay only recorded CPU edits on the same pinned Colab software stack."""
from pathlib import Path
import os
import subprocess
import sys
import tarfile
from huggingface_hub import snapshot_download
shared = Path('/content/qwen-comparison');shared.mkdir(exist_ok=True)
with tarfile.open('/content/paused-portable.tar.gz') as source:source.extractall(shared/'paused-portable',filter='data')
snapshot_download('Qwen/Qwen3.5-0.8B',revision='2fc06364715b967f1860aea9cf38778875588b17',local_dir=shared/'base',allow_patterns=['*.json','*.jinja','*.txt','*.safetensors','LICENSE'])
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='4')
subprocess.run([sys.executable,str(shared/'paused-portable/code/rebuild_paused.py'),'--base',str(shared/'base'),'--input',str(shared/'paused-portable/input'),'--output',str(shared/'astra')],check=True,env=env)
print('Paused state rebuilt and verified on Colab CPU; no new target inference.')
