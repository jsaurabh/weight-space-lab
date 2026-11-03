# Exact editor CPU cells in execution order.
# Helpers and persistent state were supplied by the recorded backend;
# this transcript is for inspection, not a standalone program.

# Turn 002; tool success: True
# Inspect available weight names and shapes, make base checkpoint.
print('num weights', len(names))
for i,n in enumerate(names[:80]):
    print(i,n,shapes[n])
print('...last')
for i,n in enumerate(names[-20:], len(names)-20): print(i,n,shapes[n])
checkpoint('base')
print('practice ids', len(practice_ids), practice_ids[:3], practice_ids[-3:])

# Turn 007; tool success: False
# List sample practice IDs by source/subject to choose valid generation batches.
# print sample IDs around each source and subject counts
from collections import Counter
print('first math IDs', practice_ids[7469:7485])
print('some math IDs offsets')
for off in [7469,7569,7969,8469,9469,10469,11469,12469,13469,14469]:
    print(off, practice_ids[off:off+5])
print('aime first', practice_ids[14744:14760])
print('gsm samples', [practice_ids[i] for i in [0,1,2,10,100,500,1000,2000,3000,4000,5000,6000,7000]])
# subject counts from IDs prefix
cnt=Counter()
for pid in practice_ids[7469:14744]:
    parts=pid.split('_')
    cnt['_'.join(parts[2:-1])] += 1
print(cnt)
for subj in sorted(cnt):
    ids=[pid for pid in practice_ids[7469:14744] if ('_'+subj+'_') in pid]
    print(subj, ids[:3], ids[-3:])

# Turn 008; tool success: True
# List sample practice IDs by source/subject to choose valid generation batches.
print('first math IDs', practice_ids[7469:7485])
print('some math IDs offsets')
for off in [7469,7569,7969,8469,9469,10469,11469,12469,13469,14469]:
    print(off, practice_ids[off:off+5])
print('aime first', practice_ids[14744:14760])
print('gsm samples', [practice_ids[i] for i in [0,1,2,10,100,500,1000,2000,3000,4000,5000,6000,7000]])
# subject counts from IDs prefix
cnt={}
for pid in practice_ids[7469:14744]:
    parts=pid.split('_')
    subj='_'.join(parts[2:-1])
    cnt[subj]=cnt.get(subj,0)+1
print('counts', cnt)
for subj in sorted(cnt):
    ids=[pid for pid in practice_ids[7469:14744] if ('_'+subj+'_') in pid]
    print(subj, ids[:3], ids[-3:])

# Turn 010; tool success: True
# Inspect generation artifact keys and summarize per example length/correct/loop endings for baseline.
art=artifact('generation-001')
print(type(art), art.keys())
# summarize structure
for k,v in art.items():
    print(k, type(v))
    if isinstance(v, list): print('list len',len(v), 'first keys', v[0].keys())
    elif isinstance(v, dict): print('dict keys', v.keys())
# Try access texts
print('artifact repr shallow', str(art)[:1000])

# Turn 011; tool success: True
# Summarize baseline batch: correct, token limits, loop patterns, text starts/ends, and source distribution.
art=artifact('generation-001')['base']
for i,e in enumerate(art):
    tail=e['text'][-400:].replace('\n','\\n')
    head=e['text'][:100].replace('\n','\\n')
    # simple repeated word counts tail
    print(i,e['id'],'ans',e['answer'],'corr',e['correct'],'tok',e['tokens'],'limit',e['hit_token_limit'],'closed',e['thinking_closed'])
    print('  head:',head)
    print('  tail:',tail[:500])
# Count phrases
for phrase in ['Wait', 'Okay', 'boxed', '</think>', 'I need']:
    print('phrase',phrase, sum(e['text'].count(phrase) for e in art))

# Turn 012; tool success: True
# Inspect tokenizer IDs for loop and thinking markers; locate repeated tokens in a generated completion.
phrases=['Wait',' Wait','\n    Wait','Okay',' Okay','</think>','\\boxed{10}','boxed','\n']
for p in phrases:
    ids=tokenize(p)
    print(repr(p), ids, decode(ids))
art=artifact('generation-001')['base']
e=art[1]
ids=tokenize(e['text'])
print('completion tokens', len(ids), 'meta tokens reported', e['tokens'])
for tid in tokenize('Wait'):
    print('Wait token',tid)
# show last 80 token ids and decoded pieces individually
last=ids[-100:]
print('last decoded:', decode(last))
print('pieces:')
for tid in last[-40:]: print(tid, repr(decode([tid])))

# Turn 013; tool success: True
# Find editable language model final norm/lm_head names and exact layer names.
for n in names:
    if 'lm_head' in n or 'norm' in n or 'layers.23' in n:
        if ('visual' not in n): print(n, shapes[n])

# Turn 015; tool success: True
# Analyze probe-002: top next tokens at loop/nonloop contexts and MLP input activations to select candidate anti-loop columns.
p=artifact('probe-002')
print(p.keys())
print('num texts', len(p['texts']))
print('text entry keys', p['texts'][0].keys())
for i,t in enumerate(p['texts']):
    print('--- text',i,'tokens',len(t['tokens']))
    # positions stored? top ids shape
    print('keys',t.keys())
    print('pos?', t.get('positions','no'))
    # top ids/logprobs for first position
    ids=t['top_ids'][0]
    lps=t['top_logprobs'][0]
    toks=[decode([int(x)]) for x in ids[:10]]
    print('top', list(zip(toks, [round(float(x),3) for x in lps[:10]])))
    print('next id/lp', t.get('next_ids'), t.get('next_logprobs'))
    print('end',repr(t['text'][-60:]))
# modules arrays shapes
for m in p['texts'][0]['modules']:
    arr=p['texts'][0]['modules'][m]
    print(m, arr.keys() if isinstance(arr,dict) else type(arr))
    try:
        print(' input shape', np.array(arr['input']).shape, 'output', np.array(arr['output']).shape)
    except Exception as e: print(e)
    break

# Turn 016; tool success: True
# Compute loop-vs-normal MLP activations and direct down_proj column alignment with Wait/Okay/close token directions for candidate edits.
p=artifact('probe-002')['texts']
loop_idx=list(range(8)); norm_idx=list(range(8,12))
# embeddings / final norm for target directions
E=weight('model.language_model.embed_tokens.weight')  # noneditable but inspect
fn=weight('model.language_model.norm.weight')
# Token ids
tids={
 'Wait_nospace': tokenize('Wait')[0],
 'Wait_space': tokenize(' Wait')[0],
 'Okay_space': tokenize(' Okay')[0],
 'The_space': tokenize(' The')[0],
 'Therefore': tokenize('Therefore')[0],
 'close': tokenize('</think>')[0],
 'boxed': tokenize('boxed')[0],
}
print('tids',tids, {k:decode([v]) for k,v in tids.items()})
# target g for logit Wait avg, with norm weight applied (rough)
ewait=(E[tids['Wait_space']]+E[tids['Wait_nospace']])/2
esafe=(E[tids['The_space']]+E[tids['Okay_space']]+E[tids['Therefore']]+E[tids['close']])/4
g_wait=(ewait-esafe)*fn
# close vs wait
g_close=(E[tids['close']]-ewait)*fn
for m in p[0]['modules'].keys():
    X=np.stack([np.array(t['modules'][m]['input'][0],dtype=np.float32) for t in p],axis=0)
    meanL=X[loop_idx].mean(axis=0); meanN=X[norm_idx].mean(axis=0)
    std=X.std(axis=0)+1e-6
    score=(meanL-meanN)/std
    # column alignment
    W=weight(m+'.weight') # [1024,3584]
    prom=W.T @ g_wait  # positive columns promote Wait over safe
    close=W.T @ g_close
    # candidate scores: high active loop diff and promotes Wait
    cand=score*prom
    top=np.argsort(-cand)[:8]
    print('\n',m)
    print('act mean loop max/min', meanL.max(), meanL.min(), 'norm max', meanN.max())
    for j in top:
        print(' j',int(j),'cand',float(cand[j]),'score',float(score[j]),'meanL',float(meanL[j]),'meanN',float(meanN[j]),'promW',float(prom[j]),'closeProj',float(close[j]))
    # candidates high loop activation not norm and high prom absolute
    top2=np.argsort(-(meanL*np.maximum(prom,0)))[:5]
    print(' top meanL*prom', [(int(j), float(meanL[j]), float(meanN[j]), float(prom[j])) for j in top2])

# Turn 017; tool success: True
# Calibrate logit lens from probe hidden states: reproduce top logprobs and derivative scale for last-layer residual shifts.
p=artifact('probe-002')['texts']
E=weight('model.language_model.embed_tokens.weight')
fn=weight('model.language_model.norm.weight')
for i in [0,1,5,8]:
    H=np.array(p[i]['hidden'],dtype=np.float32)[:,0,:]
    h=H[-1]  # likely final hidden after layer 24? before final norm? let's try
    hn=h*fn/np.sqrt(np.mean(h*h)+1e-6)
    # logits for top ids
    ids=np.array(p[i]['top_ids'][0],dtype=int)
    logits=E[ids]@hn
    # logprobs over full vocab (expensive but ok 248k*1024?) do for few
    all_logits=E@hn
    mx=all_logits.max(); lp=all_logits-(mx+np.log(np.exp(all_logits-mx).sum()))
    print('text',i,'rms h',np.sqrt(np.mean(h*h)),'artifact top',[(decode([int(t)]),round(float(l),3)) for t,l in zip(p[i]['top_ids'][0][:5],p[i]['top_logprobs'][0][:5])])
    print('computed top ids/lp', [(decode([int(j)]), round(float(lp[j]),3)) for j in ids[:5]])
    top=np.argsort(-lp)[:5]
    print('our top', [(decode([int(j)]), round(float(lp[j]),3)) for j in top])
# derivative norm for shift along g direction at text 1
i=1; h=np.array(p[i]['hidden'],dtype=np.float32)[-1,0]
r=np.sqrt(np.mean(h*h)+1e-6); y=fn*h/r
ids=[tokenize(' Wait')[0],tokenize('Wait')[0], tokenize(' The')[0], tokenize('</think>')[0]]
print('ids',ids,[decode([x]) for x in ids])
# derivative of logit diff Wait vs The with residual shift delta: (fn/r)*(e_diff - y? due rms correction). formula d = fn/r*e_diff - h*(?); compute numerical for random directions
ediff=E[ids[0]]-E[ids[2]]
g=fn*ediff/r
# correction term subtract h * sum(ediff*fn*h)/(1024*r^3)?
corr=h*(np.sum(ediff*fn*h)/(1024*(r**3)))
g_exact=fn*ediff/r - corr
print('g norm approx/exact',np.linalg.norm(g),np.linalg.norm(g_exact),'dot h',np.dot(g_exact,h))
W=weight('model.language_model.layers.23.mlp.down_proj.weight')
projs=W.T@g_exact
print('proj stats exact',projs.min(),projs.max(),projs.std(), 'top abs', np.argsort(-np.abs(projs))[:5], projs[np.argsort(-np.abs(projs))[:5]])

# Turn 018; tool success: True
# Check which hidden layer index reproduces probe top logits best to identify hidden semantics.
p=artifact('probe-002')['texts']; E=weight('model.language_model.embed_tokens.weight'); fn=weight('model.language_model.norm.weight')
for i in [1,5]:
    ids=np.array(p[i]['top_ids'][0],dtype=int)
    art=np.array(p[i]['top_logprobs'][0],dtype=np.float32)
    print('TEXT',i, 'artifact', [(decode([int(ids[k])]),float(art[k])) for k in range(5)])
    best=[]
    for li in range(25):
        h=np.array(p[i]['hidden'],dtype=np.float32)[li,0]
        # try with and without final norm
        for normed in [True,False]:
            if normed:
                hn=h*fn/np.sqrt(np.mean(h*h)+1e-6)
            else:
                hn=h
            logits=E@hn
            mx=logits.max(); lp=logits-(mx+np.log(np.exp(logits-mx).sum()))
            err=np.mean((lp[ids[:10]]-art[:10])**2)
            rank_wait=np.where(np.argsort(-lp)==ids[0])[0][0] if False else -1
            best.append((err,li,normed,[(decode([int(j)]),round(float(lp[j]),2)) for j in np.argsort(-lp)[:3]]))
    best=sorted(best,key=lambda x:x[0])[:5]
    for b in best: print(' ',b[:3], b[3])

# Turn 019; tool success: True
# Quantify baseline false examples: whether correct answer occurs/boxed in generated thinking and main loop phrases, to see if closure/anti-loop could improve accuracy.
art=artifact('generation-001')['base']
# simple boxed extraction strings
imported=False
for i,e in enumerate(art):
    text=e['text']
    ans=e['answer']
    count_box=text.count('\\boxed')+text.count('boxed')
    contains=ans in text
    # last boxed content rough
    last=''
    idx=text.rfind('\\boxed')
    if idx!=-1: last=text[idx:idx+80].replace('\n',' ')
    print(i,e['id'],'corr',e['correct'],'limit',e['hit_token_limit'],'closed',e['thinking_closed'],'ansInText',contains,'boxCount',count_box,'lastbox',last)
    # first/last occurrence of answer
    if contains:
        idx=text.rfind(ans); print('  last ans ctx',repr(text[max(0,idx-80):idx+80].replace('\n','\\n')))

# Turn 020; tool success: True
# Print high loop-specific activations in late MLPs for signed anti-Wait edit choices.
p=artifact('probe-002')['texts']; loop_idx=np.arange(8); norm_idx=np.arange(8,12)
for m in ['model.language_model.layers.20.mlp.down_proj','model.language_model.layers.21.mlp.down_proj','model.language_model.layers.22.mlp.down_proj','model.language_model.layers.23.mlp.down_proj']:
    X=np.stack([np.array(t['modules'][m]['input'][0],dtype=np.float32) for t in p])
    meanL=X[loop_idx].mean(0); meanN=X[norm_idx].mean(0); std=X.std(0)+1e-4
    # score abs loop-specific contribution: abs(meanL)*(abs(meanL)-abs(meanN)) positive and z diff
    diff_abs=np.abs(meanL)-np.abs(meanN)
    score=diff_abs*np.abs(meanL)
    top=np.argsort(-score)[:12]
    print('\n',m)
    for j in top:
        print('j',int(j),'meanL',float(meanL[j]),'meanN',float(meanN[j]),'score',float(score[j]),'valsL',[round(float(x),3) for x in X[loop_idx,j]],'valsN',[round(float(x),3) for x in X[norm_idx,j]])

# Turn 021; tool success: True
# Compute Wait/close projection and activation-signed effects for high loop-specific neurons.
p=artifact('probe-002')['texts']; loop_idx=np.arange(8); norm_idx=np.arange(8,12)
E=weight('model.language_model.embed_tokens.weight')
# Since hidden top logits match E@hidden without final norm; use target vector in final hidden space.
wait_ids=[tokenize(' Wait')[0],tokenize('Wait')[0]]
safe_ids=[tokenize('</think>')[0],tokenize(' The')[0],tokenize(' Therefore')[0] if len(tokenize(' Therefore'))==1 else tokenize(' Therefore')[0],tokenize(' So')[0] if len(tokenize(' So'))==1 else tokenize(' So')[0],tokenize(' Okay')[0]]
print('safe toks',[decode([i]) for i in safe_ids])
gW=(E[wait_ids[0]]+E[wait_ids[1]])/2 - E[safe_ids].mean(axis=0)
gClose=E[tokenize('</think>')[0]] - (E[wait_ids[0]]+E[wait_ids[1]])/2
gBox=E[tokenize(' \\')[0] if len(tokenize(' \\'))==1 else tokenize(' \\')[0]] - (E[wait_ids[0]]+E[wait_ids[1]])/2
for m, js in [('model.language_model.layers.20.mlp.down_proj',[41,1111,14,52,117,0,4,31]),('model.language_model.layers.22.mlp.down_proj',[11,573,250,2,391,324,82,21]),('model.language_model.layers.23.mlp.down_proj',[11,113,84,36,26,72,2,28,693,107,67,108,346,477,3041])]:
    X=np.stack([np.array(t['modules'][m]['input'][0],dtype=np.float32) for t in p])
    W=weight(m+'.weight')
    print('\n',m)
    for j in js:
        col=W[:,j]
        print('j',j,'meanL',float(X[loop_idx,j].mean()),'meanN',float(X[norm_idx,j].mean()),'vals',[round(float(x),2) for x in X[:8,j]], 'projW',float(col@gW),'projClose',float(col@gClose),'colnorm',float(np.linalg.norm(col)))

# Turn 022; tool success: True
# Compute rank-one activation detectors (loop-vs-normal) for late MLPs and estimate scalar scores on probed contexts.
p=artifact('probe-002')['texts']; loop_idx=np.arange(8); norm_idx=np.arange(8,12)
for m in ['model.language_model.layers.20.mlp.down_proj','model.language_model.layers.21.mlp.down_proj','model.language_model.layers.22.mlp.down_proj','model.language_model.layers.23.mlp.down_proj']:
    X=np.stack([np.array(t['modules'][m]['input'][0],dtype=np.float32) for t in p])
    L=X[loop_idx].mean(0); N=X[norm_idx].mean(0)
    # min norm u with u.L=1, u.N=0
    A=np.stack([L,N],axis=0).astype(np.float64)
    y=np.array([1.0,0.0])
    u=A.T @ np.linalg.solve(A@A.T + 1e-6*np.eye(2), y)
    scores=X@u
    # Alternative whitened diff normalized mean loop=1
    d=L-N; u2=d/(np.dot(d,L)+1e-9)
    scores2=X@u2
    print('\n',m)
    print('u norm',np.linalg.norm(u),'scores', [round(float(s),3) for s in scores], 'means loop/norm',scores[loop_idx].mean(),scores[norm_idx].mean())
    print('u2 norm',np.linalg.norm(u2),'scores2', [round(float(s),3) for s in scores2], 'means',scores2[loop_idx].mean(),scores2[norm_idx].mean())
    # score margin
    print('score min loop max norm',scores[loop_idx].min(), scores[norm_idx].max())

# Turn 023; tool success: True
# Inspect token embedding norms and target vector norms to choose rank-one edit scale.
E=weight('model.language_model.embed_tokens.weight')
toks=[' Wait','Wait','</think>',' The',' Therefore',' So',' Okay',' \\',' boxed','boxed',' Final',' Answer']
for s in toks:
    ids=tokenize(s)
    print(repr(s),ids,[decode([i]) for i in ids], 'norms',[float(np.linalg.norm(E[i])) for i in ids])
wait=(E[tokenize(' Wait')[0]]+E[tokenize('Wait')[0]])/2
safe_ids=[tokenize('</think>')[0],tokenize(' The')[0],tokenize(' Therefore')[0],tokenize(' So')[0],tokenize(' Okay')[0]]
safe=E[safe_ids].mean(0)
t=wait-safe
print('wait norm',np.linalg.norm(wait),'safe norm',np.linalg.norm(safe),'t norm',np.linalg.norm(t), 't rms', np.sqrt(np.mean(t*t)))
# token directions pair dot stats
for s in ['</think>',' The',' Therefore',' So',' Okay',' \\']:
    e=E[tokenize(s)[0]]
    print('diff Wait-',s,'norm',float(np.linalg.norm(wait-e)), 'dot_t',float((wait-e)@t))

# Turn 024; tool success: True
# Inspect final norm weight scale to understand hidden RMS.
fn=weight('model.language_model.norm.weight')
print('final norm weight mean rms min max', float(fn.mean()), float(np.sqrt(np.mean(fn*fn))), float(fn.min()), float(fn.max()))
for l in [20,21,22,23]:
    w=weight(f'model.language_model.layers.{l}.post_attention_layernorm.weight')
    print(l, 'post ln weight rms',float(np.sqrt(np.mean(w*w))), 'mean',float(w.mean()))

# Turn 025; tool success: True
# Apply candidate rank-one anti-loop edit to final MLP: activation detector from loop-vs-normal probe times anti-Wait output direction.
restore('base')
ptexts=artifact('probe-002')['texts']
loop_idx=np.arange(8); norm_idx=np.arange(8,12)
layer=23
m=f'model.language_model.layers.{layer}.mlp.down_proj'
X=np.stack([np.array(t['modules'][m]['input'][0],dtype=np.float32) for t in ptexts])
L=X[loop_idx].mean(0); N=X[norm_idx].mean(0)
A=np.stack([L,N],axis=0).astype(np.float64); y=np.array([1.0,0.0])
u=(A.T @ np.linalg.solve(A@A.T + 1e-6*np.eye(2), y)).astype(np.float32)
# Target: lower Wait/ wait relative to natural continuation/closing tokens
E=weight('model.language_model.embed_tokens.weight')
wait=(E[tokenize(' Wait')[0]]+E[tokenize('Wait')[0]])/2
safe_ids=[tokenize('</think>')[0],tokenize(' The')[0],tokenize(' Therefore')[0],tokenize(' So')[0],tokenize(' Okay')[0],tokenize(' Thus')[0] if len(tokenize(' Thus'))==1 else tokenize(' Thus')[0]]
safe=E[safe_ids].mean(0)
t=(wait-safe).astype(np.float32)
alpha=8.0
W=weight(m+'.weight')
Wnew=W - alpha*np.outer(t,u).astype(np.float32)
save_edit(m+'.weight', Wnew)
checkpoint('antiwait_l23_a8')
print('saved edit',m,'alpha',alpha,'u norm',float(np.linalg.norm(u)),'t norm',float(np.linalg.norm(t)))
print('scores', [round(float(s),3) for s in X@u])
print('delta W frob',float(np.linalg.norm(Wnew-W)), 'base W frob', float(np.linalg.norm(W)))
print('edits',list_edits())

# Turn 027; tool success: False
# Compare base vs antiwait probe top logprobs and Wait ranks on loop/non-loop motifs.
base=artifact('probe-002')['texts']; cur=artifact('probe-003')['texts']
wait_ids=[tokenize(' Wait')[0],tokenize('Wait')[0]]
close_id=tokenize('</think>')[0]
for i,(b,c) in enumerate(zip(base,cur)):
    print('\ntext',i,'end',repr(b['text'][-35:]))
    for label,t in [('base',b),('cur',c)]:
        ids=list(map(int,t['top_ids'][0])); lps=list(map(float,t['top_logprobs'][0]))
        top=[(decode([ids[k]]),round(lps[k],3)) for k in range(8)]
        # get logprob of wait ids if in top or use? actual top only top20 includes likely
        wp=[]
        for wid in wait_ids+[close_id]:
            if wid in ids: wp.append((decode([wid]), round(lps[ids.index(wid)],3), ids.index(wid)+1))
            else: wp.append((decode([wid]), None, None))
        print(' ',label,'top',top,'special',wp)

# Turn 028; tool success: True
# Compare base vs antiwait probe top logprobs and Wait ranks on loop/non-loop motifs.
base=artifact('probe-002')['texts']; cur=artifact('probe-003')['texts']
wait_ids=[tokenize(' Wait')[0],tokenize('Wait')[0]]
close_id=tokenize('</think>')[0]
for i in range(len(base)):
    b=base[i]; c=cur[i]
    print('\ntext',i,'end',repr(b['text'][-35:]))
    for label,t in [('base',b),('cur',c)]:
        ids=[int(x) for x in t['top_ids'][0]]; lps=[float(x) for x in t['top_logprobs'][0]]
        top=[(decode([ids[k]]),round(lps[k],3)) for k in range(min(8,len(ids)))]
        wp=[]
        for wid in wait_ids+[close_id]:
            if wid in ids:
                idx=ids.index(wid); wp.append((decode([wid]), round(lps[idx],3), idx+1))
            else: wp.append((decode([wid]), None, None))
        print(' ',label,'top',top,'special',wp)

# Turn 030; tool success: True
# Inspect paired generation results for antiwait_l23_a8: rescues/losses, loop patterns, boxed answers, token limits.
art=artifact('generation-004')
base=art['base']; cur=art['current']
print('idx id basecorr curcorr baselim curlim curclosed baseclosed cur box/ans tail summary')
rescues=[]; losses=[]
for i,(b,c) in enumerate(zip(base,cur)):
    if (not b['correct']) and c['correct']: rescues.append(i)
    if b['correct'] and (not c['correct']): losses.append(i)
    print(i,b['id'],b['correct'],c['correct'],b['hit_token_limit'],c['hit_token_limit'],b['thinking_closed'],c['thinking_closed'],'curAnsIn',c['answer'] in c['text'],'curBox',c['text'].count('\\boxed'), 'curWait',c['text'].count('Wait'), 'curTherefore', c['text'].count('Therefore'))
print('rescues',rescues,[cur[i]['id'] for i in rescues])
print('losses',losses,[cur[i]['id'] for i in losses])
for i in rescues+losses:
    c=cur[i]
    print('\nID',i,c['id'],'ans',c['answer'],'correct',c['correct'],'limit',c['hit_token_limit'],'closed',c['thinking_closed'],'tokens',c['tokens'])
    print('start',repr(c['text'][:700]))
    print('tail',repr(c['text'][-1000:]))
    # rough last boxed
    idx=c['text'].rfind('\\boxed')
    print('last box ctx',repr(c['text'][max(0,idx-100):idx+150] if idx!=-1 else ''))

# Turn 031; tool success: True
# Create a weaker anti-Wait rank-one checkpoint (alpha 4) for strength comparison on the same paired batch.
restore('base')
ptexts=artifact('probe-002')['texts']; loop_idx=np.arange(8); norm_idx=np.arange(8,12)
layer=23; m=f'model.language_model.layers.{layer}.mlp.down_proj'
X=np.stack([np.array(t['modules'][m]['input'][0],dtype=np.float32) for t in ptexts])
L=X[loop_idx].mean(0); N=X[norm_idx].mean(0)
A=np.stack([L,N],axis=0).astype(np.float64); y=np.array([1.0,0.0])
u=(A.T @ np.linalg.solve(A@A.T + 1e-6*np.eye(2), y)).astype(np.float32)
E=weight('model.language_model.embed_tokens.weight')
wait=(E[tokenize(' Wait')[0]]+E[tokenize('Wait')[0]])/2
safe_ids=[tokenize('</think>')[0],tokenize(' The')[0],tokenize(' Therefore')[0],tokenize(' So')[0],tokenize(' Okay')[0],tokenize(' Thus')[0]]
safe=E[safe_ids].mean(0); t=(wait-safe).astype(np.float32)
alpha=4.0
W=weight(m+'.weight')
Wnew=W - alpha*np.outer(t,u).astype(np.float32)
save_edit(m+'.weight', Wnew)
checkpoint('antiwait_l23_a4')
print('checkpoint antiwait_l23_a4 delta frob',float(np.linalg.norm(Wnew-W)))

# Turn 033; tool success: True
# Create intermediate anti-Wait rank-one checkpoint alpha 6 for strength comparison; a4 and a8 both improved but with different losses.
restore('base')
ptexts=artifact('probe-002')['texts']; loop_idx=np.arange(8); norm_idx=np.arange(8,12)
m='model.language_model.layers.23.mlp.down_proj'
X=np.stack([np.array(t['modules'][m]['input'][0],dtype=np.float32) for t in ptexts])
L=X[loop_idx].mean(0); N=X[norm_idx].mean(0)
A=np.stack([L,N],axis=0).astype(np.float64); y=np.array([1.0,0.0])
u=(A.T @ np.linalg.solve(A@A.T + 1e-6*np.eye(2), y)).astype(np.float32)
E=weight('model.language_model.embed_tokens.weight')
wait=(E[tokenize(' Wait')[0]]+E[tokenize('Wait')[0]])/2
safe_ids=[tokenize('</think>')[0],tokenize(' The')[0],tokenize(' Therefore')[0],tokenize(' So')[0],tokenize(' Okay')[0],tokenize(' Thus')[0]]
t=(wait-E[safe_ids].mean(0)).astype(np.float32)
alpha=6.0
W=weight(m+'.weight')
save_edit(m+'.weight', W - alpha*np.outer(t,u).astype(np.float32))
checkpoint('antiwait_l23_a6')
print('checkpoint antiwait_l23_a6 saved')

# Turn 035; tool success: False
# Select a second independent mixed validation batch of 32 IDs from practice_ids, avoiding the original batch.
orig=set(['gsm8k_train_0','gsm8k_train_1','gsm8k_train_2','gsm8k_train_10','gsm8k_train_101','gsm8k_train_502','gsm8k_train_1002','gsm8k_train_2003','gsm8k_train_3003','gsm8k_train_4003','gsm8k_train_5003','gsm8k_train_6004','gsm8k_train_7004','math_train_algebra_0','math_train_algebra_100','math_train_counting_and_probability_0','math_train_geometry_0','math_train_intermediate_algebra_0','math_train_number_theory_0','math_train_prealgebra_0','math_train_precalculus_0','math_train_algebra_1008','math_train_geometry_595','math_train_number_theory_472','math_train_prealgebra_614','math_train_precalculus_445','math_train_intermediate_algebra_752','math_train_counting_and_probability_277','math_train_algebra_502','aime_1983-1','aime_1983-3','aime_1983-4'])
# choose spread by offsets: 14 gsm, 14 math, 4 aime
idxs=[50,150,750,1250,1750,2500,3500,4500,5500,6500,7200,300,900,6900,
      7500,7700,8200,8900,9400,9900,10400,10900,11400,11900,12400,13000,13700,14400,
      14760,14850,15000,15300]
ids=[]
for idx in idxs:
    pid=practice_ids[idx]
    if pid in orig:
        # advance until unused
        j=idx+1
        while practice_ids[j] in orig: j+=1
        pid=practice_ids[j]
    ids.append(pid)
print(len(ids), ids)
# print questions answers short source
for pid in ids[:5]:
    r=practice_row(pid); print(pid,r.get('source'),r['answer'],r['question'][:80].replace('\n',' '))

# Turn 036; tool success: False
# Select a second independent mixed validation batch of 32 valid IDs from practice_ids, avoiding the original batch.
orig_list=['gsm8k_train_0','gsm8k_train_1','gsm8k_train_2','gsm8k_train_10','gsm8k_train_101','gsm8k_train_502','gsm8k_train_1002','gsm8k_train_2003','gsm8k_train_3003','gsm8k_train_4003','gsm8k_train_5003','gsm8k_train_6004','gsm8k_train_7004','math_train_algebra_0','math_train_algebra_100','math_train_counting_and_probability_0','math_train_geometry_0','math_train_intermediate_algebra_0','math_train_number_theory_0','math_train_prealgebra_0','math_train_precalculus_0','math_train_algebra_1008','math_train_geometry_595','math_train_number_theory_472','math_train_prealgebra_614','math_train_precalculus_445','math_train_intermediate_algebra_752','math_train_counting_and_probability_277','math_train_algebra_502','aime_1983-1','aime_1983-3','aime_1983-4']
idxs=[50,150,750,1250,1750,2500,3500,4500,5500,6500,7200,300,900,6900,7500,7700,8200,8900,9400,9900,10400,10900,11400,11900,12400,13000,13700,14400,14760,14850,15000,15300]
ids=[]
for idx in idxs:
    pid=practice_ids[idx]
    if pid in orig_list:
        j=idx+1
        while practice_ids[j] in orig_list:
            j+=1
        pid=practice_ids[j]
    ids.append(pid)
print(len(ids))
print(ids)
for pid in ids[:8]:
    r=practice_row(pid); print(pid,r.get('source'),r['answer'],r['question'][:80].replace('\n',' '))

# Turn 037; tool success: True
# Print practice IDs at selected offsets to manually form validation batch.
idxs=np.array([50,150,750,1250,1750,2500,3500,4500,5500,6500,7200,300,900,6900,7500,7700,8200,8900,9400,9900,10400,10900,11400,11900,12400,13000,13700,14400,14760,14850,15000,15300])
for idx in idxs:
    print(int(idx), practice_ids[int(idx)])

# Turn 038; tool success: True
# Restore the best current checkpoint so far, antiwait_l23_a8, for independent validation.
restore('antiwait_l23_a8')
print(list_edits())

# Turn 040; tool success: True
# Summarize independent validation results for a8: rescues/losses and remaining loop categories.
art=artifact('generation-007'); base=art['base']; cur=art['current']
resc=[]; loss=[]
for i in range(len(base)):
    b=base[i]; c=cur[i]
    if (not b['correct']) and c['correct']: resc.append(i)
    if b['correct'] and (not c['correct']): loss.append(i)
print('base correct',sum(1 for e in base if e['correct']),'cur correct',sum(1 for e in cur if e['correct']),'rescues',resc,'losses',loss)
for i in loss+resc:
    c=cur[i]
    print(i,c['id'],'ans',c['answer'],'base',base[i]['correct'],'cur',c['correct'],'limit',c['hit_token_limit'],'closed',c['thinking_closed'],'tokens',c['tokens'],'curBox',c['text'].count('\\boxed'),'Wait',c['text'].count('Wait'),'Therefore',c['text'].count('Therefore'))
print('\nRemaining current wrong with answer-in-text and no box:')
for i,c in enumerate(cur):
    if not c['correct']:
        print(i,c['id'],'ansIn',c['answer'] in c['text'],'limit',c['hit_token_limit'],'closed',c['thinking_closed'],'box',c['text'].count('\\boxed'),'Wait',c['text'].count('Wait'),'tail',repr(c['text'][-250:].replace('\n','\\n')))

# Turn 041; tool success: True
# Create alternative anti-loop edit with same loop detector but target </think> specifically, to test whether closure-focused repair improves over a8 without globally suppressing thinking.
restore('base')
ptexts=artifact('probe-002')['texts']; loop_idx=np.arange(8); norm_idx=np.arange(8,12)
m='model.language_model.layers.23.mlp.down_proj'
X=np.stack([np.array(t['modules'][m]['input'][0],dtype=np.float32) for t in ptexts])
L=X[loop_idx].mean(0); N=X[norm_idx].mean(0)
A=np.stack([L,N],axis=0).astype(np.float64); y=np.array([1.0,0.0])
u=(A.T @ np.linalg.solve(A@A.T + 1e-6*np.eye(2), y)).astype(np.float32)
E=weight('model.language_model.embed_tokens.weight')
wait=(E[tokenize(' Wait')[0]]+E[tokenize('Wait')[0]])/2
close=E[tokenize('</think>')[0]]
t=(wait-close).astype(np.float32)
alpha=4.5
W=weight(m+'.weight')
save_edit(m+'.weight', W - alpha*np.outer(t,u).astype(np.float32))
checkpoint('antiwait_close_l23_a45')
print('saved antiwait_close_l23_a45 delta frob',float(np.linalg.norm(alpha*np.outer(t,u))))

# Turn 043; tool success: True
# Restore best antiwait_l23_a8 after testing closure variant; select another validation batch of IDs at new offsets.
restore('antiwait_l23_a8')
idxs=np.array([25,225,425,625,1025,1425,1825,2225,2625,3225,3825,4425,5025,5625,6225,6825,7425,7850,8350,9050,9550,10150,10650,11250,11750,12250,12750,13250,13950,14650,14790,14950])
for idx in idxs:
    print(int(idx), practice_ids[int(idx)])

# Turn 045; tool success: True
# Select final MATH/AIME-heavy validation batch and keep antiwait_l23_a8 active.
restore('antiwait_l23_a8')
idxs=np.array([7480,7520,7600,7800,8000,8300,8600,9000,9300,9600,10000,10300,10600,11000,11300,11600,12000,12300,12600,12900,13300,13600,14000,14500,14770,14800,14880,14920,14980,15080,15200,15400])
for idx in idxs:
    print(int(idx),practice_ids[int(idx)])

# Turn 047; tool success: True
# Final aggregate evidence summary and verify selected checkpoint is active before finish.
# Summarize paired a8 validations
for artid in ['generation-004','generation-007','generation-009','generation-010']:
    art=artifact(artid); b=art['base']; c=art['current']
    bc=sum(1 for e in b if e['correct']); cc=sum(1 for e in c if e['correct'])
    bl=sum(1 for e in b if e['hit_token_limit']); cl=sum(1 for e in c if e['hit_token_limit'])
    resc=sum(1 for i in range(len(b)) if (not b[i]['correct']) and c[i]['correct'])
    loss=sum(1 for i in range(len(b)) if b[i]['correct'] and (not c[i]['correct']))
    print(artid,'base/current',bc,cc,'net',cc-bc,'resc/loss',resc,loss,'limits',bl,cl)
# Independent only excluding original
for label, ids in [('independent',['generation-007','generation-009','generation-010']),('all_a8',['generation-004','generation-007','generation-009','generation-010'])]:
    bc=cc=bl=cl=resc=loss=tot=0
    for artid in ids:
        art=artifact(artid); b=art['base']; c=art['current']; tot+=len(b)
        bc+=sum(1 for e in b if e['correct']); cc+=sum(1 for e in c if e['correct'])
        bl+=sum(1 for e in b if e['hit_token_limit']); cl+=sum(1 for e in c if e['hit_token_limit'])
        resc+=sum(1 for i in range(len(b)) if (not b[i]['correct']) and c[i]['correct'])
        loss+=sum(1 for i in range(len(b)) if b[i]['correct'] and (not c[i]['correct']))
    print(label,'total',tot,'base',bc,'current',cc,'net',cc-bc,'resc/loss',resc,loss,'limits',bl,cl)
print('active edits',list_edits())
