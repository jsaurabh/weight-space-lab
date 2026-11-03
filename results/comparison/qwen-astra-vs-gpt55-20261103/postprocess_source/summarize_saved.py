"""Summarize recorded editor tools and preserve raw logs for a completed run."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args();root=args.directory
    state=json.loads((root/'run.json').read_text());assert state['status']=='completed' and state['g4_stopped']
    for label in ['astra','gpt55']:
        directory=root/label;calls=[];events=[];ids=set();tokens=0;complete=0;cells=[]
        for p in sorted((directory/'turns').glob('*/call_*.json')):
            call=json.loads(p.read_text());calls.append(call)
            if call['name']=='python_analysis':
                cells.append('\n# Turn '+p.parent.name+'; tool success: '+str(call['response']['ok'])+'\n')
                cells.extend('# '+line+'\n' for line in call['arguments']['reason'].splitlines())
                cells.append('\n'.join(s.rstrip() for s in call['arguments']['code'].splitlines())+'\n')
            if call['name']=='generate' and call['response']['ok']:
                result=call['response']['result'];identifier=result['artifact'];artifact=json.loads((directory/'artifacts'/(identifier+'.json')).read_text())
                available=not artifact.get('unavailable_full_records',False)
                if available:
                    tokens+=sum(row['tokens'] for rows in artifact.values() for row in rows);complete+=1
                ids.update(call['arguments']['ids'])
                events.append({'turn':int(p.parent.name),'reason':call['arguments']['reason'],'ids':call['arguments']['ids'],
                    'which':call['arguments']['which'],'artifact':identifier,'full_records_available':available,
                    'summaries':{mode:{k:v for k,v in summary.items() if k!='examples'} for mode,summary in result['summaries'].items()},'paired':result.get('paired')})
        write(directory/'practice_summary.json',{'scope':'Practice feedback during discovery; not the sealed test.',
            'successful_tool_calls':dict(Counter(c['name'] for c in calls if c['response']['ok'])),
            'failed_tool_calls':dict(Counter(c['name'] for c in calls if not c['response']['ok'])),
            'unique_generated_practice_questions':len(ids),'total_practice_generations':sum(s['total'] for e in events for s in e['summaries'].values()),
            'generated_tokens_in_retained_full_records':tokens,'generation_calls_with_full_records':complete,
            'generation_calls_missing_full_records':len(events)-complete,
            'generation_seconds':sum(s['seconds'] for e in events for s in e['summaries'].values()),'events':events})
        (directory/'editor_analysis_transcript.py').write_text('# Exact editor CPU cells in execution order.\n# Inspection transcript; helpers and persistent state were supplied by the backend.\n'+''.join(cells))
    formats={}
    for path in sorted(root.rglob('*.log')):
        if path.name=='service.log' and (path.parent/'service_raw.log.gz').exists():
            raw_path=path.parent/'service_raw.log.gz';raw=gzip.decompress(raw_path.read_bytes())
        else:
            raw_path=path.with_name(path.name+'.raw.gz')
            if raw_path.exists():raw=gzip.decompress(raw_path.read_bytes())
            else:
                raw=path.read_bytes();raw_path.write_bytes(gzip.compress(raw,mtime=0))
        normalized=('\n'.join(s.rstrip() for s in raw.decode().replace('\r\n','\n').replace('\r','\n').splitlines())+'\n').encode()
        path.write_bytes(normalized)
        formats[str(path.relative_to(root))]={'raw_file':str(raw_path.relative_to(root)),
            'raw_sha256':hashlib.sha256(raw).hexdigest(),'normalized_sha256':hashlib.sha256(normalized).hexdigest()}
    write(root/'log_normalization.json',{'normalization':'Expand carriage returns; remove trailing whitespace. Original bytes retained in gzip files.','files':formats})
    print('Saved search summaries, CPU transcripts and lossless raw logs.')


if __name__=='__main__':main()
