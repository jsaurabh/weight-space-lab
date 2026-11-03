"""Resume the declared Astra continuation, then run the original sealed evaluation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from dotenv import dotenv_values
from safetensors.torch import save_file
from driver import Bridge, MODELS, ROOT, editor, write

OUTPUT = ROOT / 'results/comparison/qwen-astra-vs-gpt55-20261103'
SESSION = 'weight-space-qwen-compare'


def prepare_restore():
    recovery = OUTPUT / 'recovery_001'
    expected_payload = json.loads((OUTPUT / 'payload_manifest.json').read_text())
    with tarfile.open(ROOT / '.artifacts/qwen-compare/payload.tar.gz') as source:
        assert set(source.getnames()) == {'backend.py', 'protocol.py', 'practice.json'}
        for name, expected in expected_payload.items():
            assert hashlib.sha256(source.extractfile(name).read()).hexdigest() == expected
    prefetched = json.loads((recovery / 'prefetched.json').read_text())
    assert prefetched['response']['status'] == 'completed'
    assert any(item['type'] == 'function_call' for item in prefetched['response']['output'])
    assert not any(part.get('type') == 'refusal' for item in prefetched['response']['output']
                   for part in item.get('content', []))
    source = recovery / 'numerical_state'
    state = json.loads((recovery / 'status.json').read_text())
    assert state['forward_calls'] == 1 and state['edited_tensors'] == 0
    staging = ROOT / '.artifacts/qwen-compare/resume-files'
    staging.mkdir(exist_ok=False)
    results = staging / 'results'
    results.mkdir()
    for name in ['artifacts', 'candidate_manifests']:
        shutil.copytree(source / name, results / name)
    shutil.copyfile(source / 'tool_log.jsonl', results / 'tool_log.jsonl')
    (results / 'checkpoints').mkdir()
    assert json.loads((results / 'candidate_manifests/base.json').read_text()) == []
    save_file({}, results / 'checkpoints/base.safetensors')
    write(staging / 'maintenance_snapshot.json', {'checkpoint_label': 'base',
        'successful_forward_calls': state['forward_calls'], 'analysis_calls': state['analysis_calls'],
        'elapsed_seconds': state['elapsed_seconds']})
    shutil.copyfile(Path(__file__).with_name('resume_service.py'), staging / 'resume_service.py')
    archive = ROOT / '.artifacts/qwen-compare/resume.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        for path in sorted(staging.rglob('*')):
            if path.is_file():
                target.add(path, arcname=str(path.relative_to(staging)))
    write(recovery / 'restore_manifest.json', {'files': {str(p.relative_to(staging)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(staging.rglob('*')) if p.is_file()},
        'peer_artifacts_included': False, 'test_included': False,
        'backend_and_protocol_unchanged': True, 'checkpoint': 'base', 'forward_calls_restored': 1})
    host_source = recovery / 'source'
    host_source.mkdir()
    for name in ['driver.py', 'resume.py', 'resume_launch.py', 'resume_service.py', 'restore_peer.py', 'recover_api.py', 'analyze.py']:
        shutil.copyfile(Path(__file__).with_name(name), host_source / name)
    return archive, prefetched


def main():
    archive, prefetched = prepare_restore()
    active = Bridge(SESSION, 'astra')
    state = {'status': 'discovery', 'session': SESSION, 'order': ['gpt55', 'astra'],
        'completed_editors': ['gpt55'], 'active_editor': 'astra', 'recovery': 'recovery_001', 'g4_stopped': False}
    write(OUTPUT / 'run.json', state)
    try:
        active.upload(ROOT / '.artifacts/qwen-compare/payload.tar.gz', '/content/qwen-comparison/payload.tar.gz')
        active.upload(archive, '/content/qwen-comparison/resume.tar.gz')
        launch = active.command(['exec', '-s', SESSION, '-f', str(Path(__file__).with_name('resume_launch.py')), '--timeout', '120'])
        (OUTPUT / 'astra/resume_launch.log').write_text(launch.stdout)
        context = json.loads((OUTPUT / 'recovery_001/history.json').read_text())
        context['prefetched'] = prefetched
        key = os.environ.get('OPENAI_API_KEY') or dotenv_values(ROOT / '.env.local').get('OPENAI_API_KEY')
        active = editor(SESSION, 'astra', OUTPUT / 'astra', key, resume=context)
        write(OUTPUT / 'astra/export.json', active.export(OUTPUT / 'astra'))
        state['completed_editors'] = ['gpt55', 'astra']
        for label in MODELS:
            assert not json.loads((OUTPUT / label / 'frozen.json').read_text())['test_data_present_at_freeze']
        # The peer artifact is unavailable on the recovery VM until Astra freezes.
        active.command(['exec', '-s', SESSION, '-f', str(Path(__file__).with_name('restore_peer.py')), '--timeout', '60'])
        for filename in ['selected.safetensors', 'frozen.json']:
            active.upload(OUTPUT / 'gpt55' / filename, '/content/qwen-comparison/gpt55/results/' + filename)
        test = ROOT / '.artifacts/qwen-compare/sealed_test.json'
        protocol = json.loads((OUTPUT / 'protocol.json').read_text())
        assert hashlib.sha256(test.read_bytes()).hexdigest() == protocol['test']['sha256']
        active.upload(test, active.remote + '/sealed_test.json')
        state.update(status='sealed_evaluation', active_editor=None,
            heldout_uploaded_utc=datetime.now(timezone.utc).isoformat())
        write(OUTPUT / 'run.json', state)
        conditions = [{'label': label, 'checkpoint': '/content/qwen-comparison/' + label + '/results/selected.safetensors',
                       'manifest': '/content/qwen-comparison/' + label + '/results/frozen.json'} for label in MODELS]
        result = active.rpc('evaluate_conditions', {'conditions': conditions, 'test_sha256': protocol['test']['sha256']}, timeout=7200)
        write(OUTPUT / 'final_evaluation.json', result)
        if not result['ok']:
            raise RuntimeError('Held-out evaluation failed')
        write(OUTPUT / 'astra/export.json', active.export(OUTPUT / 'astra'))
        shutil.copytree(OUTPUT / 'astra/evaluation', OUTPUT / 'evaluation', dirs_exist_ok=True)
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='stopped_on_error', error_type=type(exc).__name__)
        print('RECOVERY ERROR', type(exc).__name__, flush=True)
        try:
            active.export(OUTPUT / 'astra')
        except Exception as exported:
            state['export_error_type'] = type(exported).__name__
    finally:
        stopped = subprocess.run(['colab', 'stop', '-s', SESSION], capture_output=True, text=True, timeout=120)
        state['g4_stopped'] = stopped.returncode == 0
        write(OUTPUT / 'run.json', state)
    print('RUN', json.dumps(state), flush=True)
    if state['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
