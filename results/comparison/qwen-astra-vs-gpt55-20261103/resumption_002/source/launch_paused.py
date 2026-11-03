"""Start the verified reconstructed worker with the original numerical backend."""
from pathlib import Path
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile

shared=Path('/content/qwen-comparison');root=shared/'astra'
assert json.loads((root/'verification.json').read_text())['all_hashes_match']
with tarfile.open(shared/'payload.tar.gz') as source:source.extractall(root,filter='data')
for name in ['requests','responses']:(root/name).mkdir()
control=json.loads((shared/'continue_control.json').read_text())
snapshot=json.loads((root/'maintenance_snapshot.json').read_text());snapshot['elapsed_seconds']=control['elapsed_seconds']
(root/'maintenance_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
(shared/'control.json').write_text(json.dumps({'label':'astra'}))
env=os.environ.copy();env.update(QWEN_BASE_DIRECTORY=str(shared/'base'),VLLM_WORKER_MULTIPROC_METHOD='spawn',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='4')
process=subprocess.Popen([sys.executable,'-u',str(root/'resume_service.py')],stdout=(root/'service.log').open('w'),stderr=subprocess.STDOUT,env=env,start_new_session=True)
(root/'pid').write_text(str(process.pid))
runtime={'python':sys.version,'requested_accelerator':'G4','recovery':'resumption_002','hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,compute_cap,driver_version','--format=csv,noheader'],text=True).strip(),'software':{p:importlib.metadata.version(p) for p in ['torch','vllm','transformers','math-verify','numpy']}}
(root/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
print(json.dumps({'worker_started':True,'pid':process.pid,'runtime':runtime}))
