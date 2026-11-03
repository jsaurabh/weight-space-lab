from pathlib import Path
import hashlib,json,shutil
r=Path('/content/qwen-comparison/astra/results')
shutil.copyfile(r/'checkpoints/l11_h12_half.safetensors',r/'checkpoints/host_before_launcher_fix.safetensors')
shutil.copyfile(r/'candidate_manifests/l11_h12_half.json',r/'candidate_manifests/host_before_launcher_fix.json')
print(json.dumps({'host_checkpoint_sha256':hashlib.sha256((r/'checkpoints/host_before_launcher_fix.safetensors').read_bytes()).hexdigest()}))
