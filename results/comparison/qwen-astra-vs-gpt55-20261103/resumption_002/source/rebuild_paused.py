"""Reconstruct Astra's paused weights from its recorded CPU edits; never infer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from safetensors import safe_open

import backend
ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

RESULT = ROOT / 'results/comparison/qwen-astra-vs-gpt55-20261103'


def restore_artifacts(directory):
    restored = {}
    for descriptor in sorted(directory.glob('*.json')):
        arrays_path = descriptor.with_suffix('.npz')
        arrays = np.load(arrays_path, allow_pickle=False) if arrays_path.exists() else None
        def decode(v):
            if isinstance(v, dict) and set(v) == {'array', 'shape', 'dtype'}:
                return np.array(arrays[v['array']], copy=True)
            if isinstance(v, dict): return {k: decode(x) for k, x in v.items()}
            if isinstance(v, list): return [decode(x) for x in v]
            return v
        restored[descriptor.stem] = decode(json.loads(descriptor.read_text()))
        if arrays is not None: arrays.close()
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / '.artifacts/qwen-base')
    parser.add_argument('--output', type=Path, default=ROOT / '.artifacts/qwen-compare/paused-rebuild')
    parser.add_argument('--input', type=Path, help='Portable archive of own prior observations and recorded CPU cells')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output / 'results'; out.mkdir()
    if args.input:
        shutil.copytree(args.input / 'artifacts', out / 'artifacts')
        replay = json.loads((args.input / 'cells.json').read_text())
        counts = json.loads((args.input / 'counts.json').read_text())
        practice_path = args.input / 'practice.json'
    else:
        shutil.copytree(ROOT / '.artifacts/qwen-compare/launcher-fix-export/artifacts', out / 'artifacts')
        replay = {t: json.loads(next((RESULT / 'astra/turns' / t).glob('call_*.json')).read_text()) for t in ['014','029','031','032','034']}
        practice_path = ROOT / '.artifacts/qwen-compare/practice.json'
    for turn, artifact in [('028','generation-004'),('030','generation-005'),('033','generation-006')]:
        if args.input: continue
        call = json.loads(next((RESULT / 'astra/turns' / turn).glob('call_*.json')).read_text())
        write(out / 'artifacts' / (artifact + '.json'), {'unavailable_full_records': True,
            'reason': 'Original full generation records lost with Colab VM; original returned summaries retained.',
            'original_turn': int(turn), 'summaries': call['response']['result']['summaries']})
    backend.BASE = args.base
    backend.ROOT = args.output
    backend.OUT = out
    service = backend.Service.__new__(backend.Service)
    service.mapping = json.loads((args.base / 'model.safetensors.index.json').read_text())['weight_map']
    service.headers = {}
    for shard in set(service.mapping.values()):
        with safe_open(args.base / shard, framework='pt', device='cpu') as source:
            for name in source.keys():
                service.headers[name] = {'shape': source.get_slice(name).get_shape()}
    service.config = json.loads((args.base / 'config.json').read_text())
    service.practice_rows = json.loads(practice_path.read_text())
    service.practice_map = {row['id']: row for row in service.practice_rows}
    service.base_cache, service.edits = {}, {}
    service.artifacts = restore_artifacts(out / 'artifacts')
    service.namespace = None; service.frozen = False; service.analysis_calls = 0
    service.tokenizer = None
    torch.set_num_threads(4)
    service.checkpoint('base')
    checks = []
    for turn in ['014', '029', '031', '032', '034']:
        call = replay[turn]
        code = call['arguments']['code']
        if turn == '029': code = code[code.index("restore('base')"):]
        service.python_analysis(code)
        if turn in ['014', '029', '032', '034']:
            actual = service.list_edits()
            expected = call['response']['result']['edits']
            matches = actual == expected
            checks.append({'turn':int(turn), 'tensors':len(actual), 'matches_recorded_manifest':matches,
                'mismatches':[e['name'] for e in actual if e not in expected]})
            write(args.output / 'verification.json', {'checks':checks, 'all_hashes_match':all(c['matches_recorded_manifest'] for c in checks)})
            print(json.dumps(checks[-1]), flush=True)
            if not matches: raise SystemExit('Reconstructed tensor hashes differ; do not resume from these weights.')
    if not args.input: counts = {'forward':0, 'analysis':0}
    for p in ([] if args.input else sorted((RESULT / 'astra/turns').glob('*/call_*.json'))):
        c = json.loads(p.read_text())
        if c['name'] in {'generate','probe'} and c['response']['ok']: counts['forward'] += 1
        if c['name'] == 'python_analysis':
            try: backend.validate_code(c['arguments']['code']); counts['analysis'] += 1
            except (ValueError,SyntaxError): pass
    assert counts['forward'] == 6
    write(args.output / 'maintenance_snapshot.json', {'checkpoint_label':'loop_projection_full',
        'successful_forward_calls':counts['forward'], 'analysis_calls':counts['analysis'], 'elapsed_seconds':0})
    shutil.copyfile(Path(__file__).with_name('resume_service.py'), args.output / 'resume_service.py')
    write(args.output / 'verification.json', {'checks':checks,'all_hashes_match':True,
        'restored_forward_calls':counts['forward'],'restored_analysis_calls':counts['analysis'],
        'missing_full_generation_records':['generation-004','generation-005','generation-006'],
        'read_only_inspection_skipped_in_turn_029':True,'new_target_forward_passes':0})
    print('Exact paused checkpoints reconstructed; no new target inference.',flush=True)


if __name__ == '__main__': main()
