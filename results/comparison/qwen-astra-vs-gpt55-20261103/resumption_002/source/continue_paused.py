"""Continue the explicitly paused Astra run without resetting its budget."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from dotenv import dotenv_values
from driver import Bridge, ROOT, editor, strip_none, write
from resume import evaluate_frozen

OUTPUT=ROOT/'results/comparison/qwen-astra-vs-gpt55-20261103'
SESSION='weight-space-qwen-compare'


def main():
    started=time.monotonic()
    record=OUTPUT/'resumption_002';record.mkdir(exist_ok=False)
    for name in ['metadata_resume.json','service.log','runtime.json','environment.json','tool_log.jsonl','status.json']:
        if (OUTPUT/'astra'/name).exists():shutil.copyfile(OUTPUT/'astra'/name,record/('before_'+name))
    pause=json.loads((OUTPUT/'pause_001/manifest.json').read_text())
    oldrun=json.loads((OUTPUT/'astra/run.json').read_text())
    used=(datetime.fromisoformat(pause['paused_utc'])-datetime.fromisoformat(oldrun['created_utc'])).total_seconds()
    verified=json.loads((ROOT/'.artifacts/qwen-compare/paused-remote-verification.json').read_text())
    assert verified['all_hashes_match'] and verified['restored_forward_calls']==6
    write(record/'reconstruction.json',verified)
    source=record/'source';source.mkdir()
    for name in ['driver.py','continue_paused.py','launch_paused.py','resume.py','resume_service.py','rebuild_paused.py','rebuild_on_colab.py']:
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    state={'status':'discovery','session':SESSION,'order':['gpt55','astra'],'completed_editors':['gpt55'],
           'active_editor':'astra','recovery':'recovery_001','resumption':'resumption_002','g4_stopped':False}
    write(OUTPUT/'run.json',state)
    active=Bridge(SESSION,'astra')
    try:
        active.upload(ROOT/'.artifacts/qwen-compare/payload.tar.gz','/content/qwen-comparison/payload.tar.gz')
        active.upload(Path(__file__).with_name('resume_service.py'),'/content/qwen-comparison/astra/resume_service.py')
        control=ROOT/'.artifacts/qwen-compare/continue_control.json';write(control,{'elapsed_seconds':used+time.monotonic()-started})
        active.upload(control,'/content/qwen-comparison/continue_control.json')
        launched=active.command(['exec','-s',SESSION,'-f',str(Path(__file__).with_name('launch_paused.py')),'--timeout','120'])
        (record/'launch.log').write_text(launched.stdout)
        metadata=active.rpc('metadata',{},timeout=600);assert metadata['ok'] and metadata['remaining_forward_calls']==4
        write(record/'metadata.json',metadata)
        native=active.rpc('smoke_reload',{},timeout=900);write(record/'native_validation.json',native)
        assert native['ok'] and native['result']['base_roundtrip_exact'] and native['remaining_forward_calls']==4
        request=json.loads((OUTPUT/'astra/turns/035/request.json').read_text())
        response=json.loads((OUTPUT/'astra/turns/035/response.json').read_text())
        calls=[item for item in response['output'] if item['type']=='function_call'];assert len(calls)==1 and calls[0]['name']=='generate'
        call=calls[0];arguments=json.loads(call['arguments'])
        print('Re-running unresolved GPU feedback call 35; no additional editor API request.',flush=True)
        result=active.rpc(call['name'],arguments,timeout=3600)
        assert result['ok'] and result['remaining_forward_calls']==3
        write(OUTPUT/'astra/turns/035'/(call['call_id']+'.json'),{'name':call['name'],'arguments':arguments,'response':result})
        write(OUTPUT/'astra/live_export.json',active.export(OUTPUT/'astra'));active.backup_checkpoints(OUTPUT/'astra')
        history=request['input']+strip_none(response['output'])
        history.append({'type':'function_call_output','call_id':call['call_id'],'output':json.dumps(result)})
        notice=('Resuming after the user pause and loss of the old Colab VM. All reconstructed checkpoint tensor hashes '
          'match your recorded weights exactly; numerical tools and decoding are unchanged. Your original baseline and '
          'two probe artifacts are restored. Full texts of generation-004, generation-005 and generation-006 were lost; '
          'their artifact objects now explicitly retain only the original returned summaries. CPU variables are cleared. '
          'The unresolved call 35 was rerun and its new result is above. Seven successful feedback calls are now used, '
          'leaving three. The user pause is excluded from the remaining wall budget. No peer edit, published repair or '
          'held-out test data has been supplied. Continue your investigation and select final weights after all ten calls.')
        history.append({'role':'user','content':notice})
        write(record/'manifest.json',{'resumed_utc':datetime.now(timezone.utc).isoformat(),'paused_record':'pause_001/manifest.json',
          'reconstruction_hashes_match':True,'pending_call_reissued_without_extra_api_request':35,
          'history_notice':notice,'restored_completed_forward_calls':6,'remaining_after_pending_result':3,
          'lost_full_generation_artifacts':['generation-004','generation-005','generation-006'],
          'durable_exports_after_every_future_forward_call':True,'user_pause_excluded_from_wall_budget':True,
          'peer_or_test_data_provided':False})
        write(record/'history.json',{'history':history,'first_turn':36})
        key=os.environ.get('OPENAI_API_KEY') or dotenv_values(ROOT/'.env.local').get('OPENAI_API_KEY')
        active=editor(SESSION,'astra',OUTPUT/'astra',key,resume={'history':history,'first_turn':36,
                    'elapsed_seconds':used+time.monotonic()-started,'durable':True,'resumption':'resumption_002'})
        write(OUTPUT/'astra/export.json',active.export(OUTPUT/'astra'))
        active.backup_checkpoints(OUTPUT/'astra')
        evaluate_frozen(active,state)
    except BaseException as exc:
        state.update(status='stopped_on_error',error_type=type(exc).__name__)
        print('RESUMPTION ERROR',type(exc).__name__,flush=True)
        try:active.export(OUTPUT/'astra')
        except Exception as exported:state['export_error_type']=type(exported).__name__
    finally:
        stopped=subprocess.run(['colab','stop','-s',SESSION],capture_output=True,text=True,timeout=120)
        state['g4_stopped']=stopped.returncode==0;write(OUTPUT/'run.json',state)
    print('RUN',json.dumps(state),flush=True)
    if state['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
