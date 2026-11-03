"""Repair the recovery entry point while preserving the existing numerical state."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

shared = Path('/content/qwen-comparison')
root = shared / 'astra'
assert not (root / 'sealed_test.json').exists()
assert not (shared / 'gpt55').exists()
control = json.loads((shared / 'launcher_repair.json').read_text())
checkpoint = root / 'results/checkpoints/host_before_launcher_fix.safetensors'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == control['current_checkpoint_sha256']
archive = root / 'results/launcher_repair'
archive.mkdir(exist_ok=False)
for name in ['resume_service.py', 'service.log', 'runtime.json', 'maintenance_snapshot.json']:
    shutil.copyfile(root / name, archive / ('before_' + name))
shutil.copyfile(root / 'results/status.json', archive / 'before_status.json')
shutil.copyfile(shared / 'guarded_resume_service.py', root / 'resume_service.py')
snapshot = {'checkpoint_label': 'host_before_launcher_fix',
    'successful_forward_calls': control['successful_editor_forward_calls'],
    'analysis_calls': control['counted_editor_cpu_calls'], 'elapsed_seconds': control['elapsed_seconds']}
(root / 'maintenance_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
control.update(backend_unchanged=True, protocol_unchanged=True, editor_weights_unchanged=True,
    peer_or_test_data_provided=False,
    corrected_entry_sha256=hashlib.sha256((root / 'resume_service.py').read_bytes()).hexdigest(),
    original_entry_sha256=hashlib.sha256((archive / 'before_resume_service.py').read_bytes()).hexdigest())
(archive / 'event.json').write_text(json.dumps(control, indent=2) + '\n')
env = os.environ.copy()
env.update(QWEN_BASE_DIRECTORY=str(shared / 'base'), VLLM_WORKER_MULTIPROC_METHOD='spawn',
           TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='4')
process = subprocess.Popen([sys.executable, '-u', str(root / 'resume_service.py')],
    stdout=(root / 'service.log').open('w'), stderr=subprocess.STDOUT, env=env, start_new_session=True)
(root / 'pid').write_text(str(process.pid))
runtime = json.loads((root / 'runtime.json').read_text())
runtime['launcher_repaired'] = True
(root / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
print(json.dumps({'worker_restarted': True, 'pid': process.pid, 'snapshot': snapshot}))
