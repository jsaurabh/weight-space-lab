"""Audit transcript of editor CPU cells, including failed cells; not a standalone replay.
Temporary namespace reset before turn 25. Use replay.py and selected.safetensors for exact replay.
"""

# API turn 001; tool result ok=True
# Inspect the editable tensor layout and available architecture metadata before choosing a diagnostic.
print(config)
print('tensors', len(names))
for n in names:
    print(n, shapes[n])

# API turn 005; tool result ok=True
# The baseline computes plausible answers but never exits repetitive reasoning. Inspect internal tensor magnitudes and normalization parameters for a localized source of this behavior.
prefix = 'model.language_model.layers.'
text_names = [n for n in names if n.startswith(prefix)]
print('Nonlayer text tensors:', [(n, shapes[n]) for n in names if 'language_model' in n and n not in text_names])
stat = {}
for n in text_names:
    w = weight(n)
    rms = float(np.sqrt(np.mean(w*w)))
    stat[n] = (rms, float(np.max(np.abs(w))), float(np.mean(w)), float(np.std(w)))
for i in range(24):
    print('\nLAYER', i)
    for n in text_names:
        if n.startswith(prefix+str(i)+'.'):
            print(n.split(prefix+str(i)+'.')[1], tuple(round(v,5) for v in stat[n]))
checkpoint('base')

# API turn 006; tool result ok=True
# Complete the late-layer inspection and read full baseline trace structure, because excessive deliberation may originate near the output-side reasoning-control features.
for i in range(17,24):
    print('\nLAYER', i)
    for n in text_names:
        if n.startswith(prefix+str(i)+'.'):
            print(n.split(prefix+str(i)+'.')[1], tuple(round(v,5) for v in stat[n]))
gen0 = artifact('generation-001')
print('Artifact type/keys', type(gen0), (list(gen0) if isinstance(gen0,dict) else ''))
print(str(gen0)[:1400])

# API turn 007; tool result ok=False
# Separate arithmetic failures from answer-finalization failures, and identify control-token representations without using example answers to fit any parameters.
for e in gen0['base']:
    print('\n', e['id'], list(e.keys()))
    print('START',e['text'][:500])
    print('END',e['text'][-700:])
for s in ['<think>', '</think>', 'Wait', ' Wait', 'But', ' But', 'Maybe', ' maybe', 'Therefore', ' Therefore', 'Thus', 'Final', '\\boxed', '<|im_end|>']:
    print(repr(s),tokenize(s))
print('final norm stats', np.quantile(weight('model.language_model.norm.weight'), [0,.01,.25,.5,.75,.99,1]))

# API turn 008; tool result ok=False
# Read all baseline endings and inspect the token IDs and fixed output normalization needed for a control-feature diagnostic.
for e in gen0['base']:
    print('\n', e['id'], list(e.keys()))
    print('START',e['text'][:260])
    print('END',e['text'][-500:])
for s in ['<think>', '</think>', 'Wait', ' Wait', 'But', ' But', 'Maybe', ' maybe', 'Therefore', ' Therefore', 'Thus', 'Final', '\\boxed', '<|im_end|>']:
    print(s,tokenize(s))
print('final norm stats', np.quantile(weight('model.language_model.norm.weight'), [0,.01,.25,.5,.75,.99,1]))

# API turn 009; tool result ok=True
# Retrieve trace endings and scalar/token metadata using serialization-safe list output.
for e in gen0['base']:
    print('\n', e['id'])
    print('START',e['text'][:230])
    print('END',e['text'][-450:])
for s in ['<think>', '</think>', 'Wait', ' Wait', 'But', ' But', 'Maybe', ' maybe', 'Therefore', ' Therefore', 'Thus', 'Final', '\\boxed', '<|im_end|>']:
    print(s, [int(x) for x in tokenize(s)])
fnorm=weight('model.language_model.norm.weight')
print('final norm stats', np.quantile(fnorm, [0,.01,.25,.5,.75,.99,1]).tolist())

# API turn 011; tool result ok=True
# Measure each late module's contribution along the fixed think-end versus think-start output direction, and inspect individual neurons rather than applying a blanket layer scale.
p2 = artifact('probe-002')
print('probe keys',list(p2))
print('text keys', list(p2['texts'][0]))
print('module keys', list(p2['texts'][0]['modules']['model.language_model.layers.23.mlp.down_proj']))
emb = weight('model.language_model.embed_tokens.weight')
end_id=int(tokenize('</think>')[0]); start_id=int(tokenize('<think>')[0])
control = (emb[end_id]-emb[start_id]) * (1.0+fnorm)
control_unit = control / np.linalg.norm(control)
print('control length',float(np.linalg.norm(control)))
for t in p2['texts']:
    h=np.asarray(t['hidden'])
    print('tokens end',[decode([int(x)]) for x in t['tokens'][-8:]], 'hidden rms',np.sqrt(np.mean(h[:,-1,:]**2,axis=1)).tolist())
    for m,d in t['modules'].items():
        out=np.asarray(d['output']); ins=np.asarray(d['input'])
        print(m.split('layers.')[1], 'shape',ins.shape,out.shape,'out proj',np.round(out@control_unit,3).tolist())


# API turn 012; tool result ok=True
# Check whether suppression of the think-ending direction in layer 23 is large enough to matter, and quantify its overlap with other vocabulary directions before editing.
print('embedding norms',[(s,float(np.linalg.norm(emb[int(tokenize(s)[0])]))) for s in ['<think>','</think>','Wait','Therefore']])
for ti,t in enumerate(p2['texts']):
    h=np.asarray(t['hidden'])
    print('TEXT',ti)
    for layer in [20,21,22,23,24]:
        x=h[layer,0]
        if layer<24: x=x/np.sqrt(np.mean(x*x)+1e-6)*(1+fnorm)
        logits=emb@x
        top=np.argsort(logits)[-8:][::-1]
        print(layer,[(decode([int(idx)]),round(float(logits[idx]),2)) for idx in top], 'end',round(float(logits[end_id]),2),'rank',int(np.sum(logits>logits[end_id])))
for m in ['model.language_model.layers.23.mlp.down_proj','model.language_model.layers.23.self_attn.o_proj']:
    w=weight(m+'.weight')
    row=control_unit@w
    print(m,'rank1 relative Frobenius',float(np.linalg.norm(row)/np.linalg.norm(w)), 'top neuron coords',np.argsort(np.abs(row))[-12:].tolist())

# API turn 013; tool result ok=True
# Test a minimal rank-one repair: prevent the final MLP and attention output from overwriting the existing think-ending-versus-think-start signal. This removes only one output direction, not an entire tensor scale.
clear_edits()
for m in ['model.language_model.layers.23.mlp.down_proj','model.language_model.layers.23.self_attn.o_proj']:
    w=weight(m+'.weight')
    repaired=w-np.outer(control_unit,control_unit@w)
    save_edit(m+'.weight',repaired)
checkpoint('end_control_l23')
print(list_edits())

# API turn 017; tool result ok=True
# Verify that the successful current probe actually used the rank-one repair, and measure the end-marker probability change without assuming the failed generation ran.
p3=artifact('probe-003')
for key,t in [('base',p2['texts'][0]),('current',p3['texts'][0])]:
    h=np.asarray(t['hidden'])[-1,0]
    logits=emb@h
    mx=float(np.max(logits)); logz=mx+float(np.log(np.sum(np.exp(logits-mx))))
    top=np.argsort(logits)[-8:][::-1]
    print(key,[(decode([int(idx)]),round(float(logits[idx]),3)) for idx in top], 'end_logprob',float(logits[end_id])-logz)
    for m,d in t['modules'].items():
        if '23.' in m:
            print(m,float(np.asarray(d['output'])[0]@control_unit))

# API turn 018; tool result ok=True
# Inspect whether the last-layer anti-termination component is concentrated in identifiable MLP neurons, and build a diagnostic from existing baseline prefixes without fitting any weights to their answers.
w23=weight(prefix+'23.mlp.down_proj.weight')
cu=control_unit@w23
order=np.argsort(np.abs(cu))[-8:][::-1]
for j in order:
    d=w23[:,j]*(1+fnorm)
    logits=emb@d
    top=np.argsort(logits)[-6:][::-1]
    bot=np.argsort(logits)[:4]
    acts=[float(np.asarray(t['modules'][prefix+'23.mlp.down_proj']['input'])[0,j]) for t in p2['texts']]
    print('neuron',int(j),'control',float(cu[j]),'act',acts,'top',[decode([int(x)]) for x in top],'bottom',[decode([int(x)]) for x in bot])
baseline_prefixes=[]
for e in gen0['base']:
    toks=tokenize(e['text'])
    txt=decode(toks[:600])
    boundary=txt.rfind('\n')
    txt=txt[:boundary+1] if boundary>0 else txt
    baseline_prefixes.append('<|im_start|>user\n'+e['question']+'<|im_end|>\n<|im_start|>assistant\n<think>\n'+txt)
    print(e['id'], 'prefix ending', txt[-350:])

# API turn 020; tool result ok=True
# Quantify termination specificity and the contribution of a weight-derived doubt-versus-conclusion direction. No direction is fitted to the diagnostic examples.
p4=artifact('probe-004')
doubt_words=['Wait',' Wait','Maybe',' maybe','But',' But','However',' However','Alternatively',' alternatively','Perhaps',' perhaps']
resolve_words=['Therefore',' Therefore','Thus',' Thus','Hence',' hence','Finally',' finally','So',' So']
doubt_ids=[int(tokenize(s)[0]) for s in doubt_words]
resolve_ids=[int(tokenize(s)[0]) for s in resolve_words]
doubt=(np.mean(emb[doubt_ids],axis=0)-np.mean(emb[resolve_ids],axis=0))*(1+fnorm)
doubt=doubt/np.linalg.norm(doubt)
print('doubt/control overlap',float(doubt@control_unit))
for i,t in enumerate(p4['texts']):
    x=np.asarray(t['hidden'])[-1,0]
    logits=emb@x
    mx=float(np.max(logits)); lz=mx+float(np.log(np.sum(np.exp(logits-mx))))
    top=np.argsort(logits)[-5:][::-1]
    print('TEXT',i,[(decode([int(j)]),round(float(logits[j]),2)) for j in top],'end logp',round(float(logits[end_id])-lz,2))
    print('module components',[ (m.split('layers.')[1],round(float(np.asarray(d['output'])[0]@control_unit),3),round(float(np.asarray(d['output'])[0]@doubt),3)) for m,d in t['modules'].items()])


# API turn 021; tool result ok=True
# Localize the final attention's anti-termination signal by head, checking whether it is selective for reasoning boundaries rather than a general numerical feature.
ow=weight(prefix+'23.self_attn.o_proj.weight')
cr=(control_unit@ow).reshape(8,256)
for i,t in enumerate(p4['texts']):
    z=np.asarray(t['modules'][prefix+'23.self_attn.o_proj']['input'])[0].reshape(8,256)
    vals=np.sum(cr*z,axis=1)
    print('text',i,'control by head',np.round(vals,4).tolist())
print('projection row norms per head',np.linalg.norm(cr,axis=1).tolist())
# Stronger but still rank-one candidate: reflect the suppressive final-layer component.
clear_edits()
for m in [prefix+'23.mlp.down_proj',prefix+'23.self_attn.o_proj']:
    w=weight(m+'.weight')
    save_edit(m+'.weight',w-2.0*np.outer(control_unit,control_unit@w))
checkpoint('end_control_reflection_l23')

# API turn 023; tool result ok=True
# Compare the reflection against base on all diagnostic states, including probability assigned to the unchanged numerical and formatting continuations.
p5=artifact('probe-005')
def logstats(t):
    h=np.asarray(t['hidden'])[-1,0]
    ls=emb@h
    mx=float(np.max(ls)); lz=mx+float(np.log(np.sum(np.exp(ls-mx))))
    return ls,lz
for i,(a,b) in enumerate(zip(p4['texts'],p5['texts'])):
    la,za=logstats(a); lb,zb=logstats(b)
    top=np.argsort(lb)[-4:][::-1]
    orig=int(np.argmax(la))
    print(i,'end logp',round(float(la[end_id])-za,3),round(float(lb[end_id])-zb,3), 'base top',decode([orig]), 'base top logp delta',round(float(lb[orig]-la[orig])-zb+za,3),'new top',[(decode([int(j)]),round(float(lb[j]),2)) for j in top])
# Inspect the mean vocabulary effect of the reflection's one residual direction.
end_direction=(1+fnorm)*control_unit
lex=emb@end_direction
top=np.argsort(lex)[-16:][::-1]
print('direction favors',[(decode([int(j)]),round(float(lex[j]),3)) for j in top])

# API turn 024; tool result ok=True
# Measure collateral changes over all captured arithmetic-token positions, not just terminal newlines, and check the available generation metadata for a faithful practice-style validation prompt.
print('generation example keys', list(gen0['base'][0]))
for i,(a,b) in enumerate(zip(p4['texts'],p5['texts'])):
    ha=np.asarray(a['hidden'])[-1]; hb=np.asarray(b['hidden'])[-1]
    scores=[]
    for j in range(8):
        la=emb@ha[j]; lb=emb@hb[j]
        xa=float(np.max(la)); xb=float(np.max(lb))
        za=xa+float(np.log(np.sum(np.exp(la-xa)))); zb=xb+float(np.log(np.sum(np.exp(lb-xb))))
        pa=np.exp(la-za)
        kl=float(np.sum(pa*((la-za)-(lb-zb))))
        pos=int(a['positions'][j]); pos=pos if pos>=0 else len(a['tokens'])+pos
        if pos+1<len(a['tokens']):
            nxt=int(a['tokens'][pos+1]); change=float(lb[nxt]-la[nxt])-zb+za
            scores.append((decode([nxt]),round(change,3),round(kl,3)))
    print('text',i,'next-token logprob change and KL',scores)


# API turn 026; tool result ok=True
# The late-only termination repair was insufficient in generation. Test a semantic rank-one ablation throughout residual writers to prevent repeated amplification of doubt while retaining orthogonal arithmetic features.
prefix='model.language_model.layers.'
emb=weight('model.language_model.embed_tokens.weight')
fnorm=weight('model.language_model.norm.weight')
doubt_words=['Wait',' Wait','Maybe',' maybe','But',' But','However',' However','Alternatively',' alternatively','Perhaps',' perhaps']
resolve_words=['Therefore',' Therefore','Thus',' Thus','Hence',' hence','Finally',' finally','So',' So']
doubt_ids=[int(tokenize(s)[0]) for s in doubt_words]
resolve_ids=[int(tokenize(s)[0]) for s in resolve_words]
doubt=(np.mean(emb[doubt_ids],axis=0)-np.mean(emb[resolve_ids],axis=0))*(1+fnorm)
doubt=doubt/np.linalg.norm(doubt)
clear_edits()
writers=[n for n in names if n.startswith(prefix) and (n.endswith('.mlp.down_proj.weight') or n.endswith('.linear_attn.out_proj.weight') or n.endswith('.self_attn.o_proj.weight'))]
changes=[]
for n in writers:
    w=weight(n); row=doubt@w
    save_edit(n,w-np.outer(doubt,row))
    changes.append(float(np.linalg.norm(row)/np.linalg.norm(w)))
checkpoint('doubt_direction_ablation')
print('edited writers',len(writers),'relative changes min mean max',float(np.min(changes)),float(np.mean(changes)),float(np.max(changes)))

# API turn 028; tool result ok=True
# The doubt ablation did not solve looping. Return to the localized termination mechanism and test a stronger one-dimensional counterbalance. Earlier probes showed completion boundaries receive much larger anti-termination contributions than unfinished arithmetic or final-answer mode.
end_id=int(tokenize('</think>')[0]); start_id=int(tokenize('<think>')[0])
control=(emb[end_id]-emb[start_id])*(1+fnorm)
control_unit=control/np.linalg.norm(control)
clear_edits()
for m in [prefix+'23.mlp.down_proj',prefix+'23.self_attn.o_proj']:
    w=weight(m+'.weight')
    save_edit(m+'.weight',w-8.0*np.outer(control_unit,control_unit@w))
checkpoint('end_control_counterbalance8_l23')
print('Two rank-one edits; fixed embeddings, output head and final norm remain unchanged.')

# API turn 030; tool result ok=True
# Inspect where the stronger repair places thinking delimiters and whether looping continues after them; this separates delimiter suppression from a deeper recurrent reasoning failure.
g8=artifact('generation-008')
for e in g8['current']:
    s=e['text']
    j=s.find('</think>')
    print('\n',e['id'],'closed',e['thinking_closed'],'close index',j,'chars',len(s))
    if j>=0: print('around marker',s[max(0,j-250):j+600])
    print('end',s[-850:])
# Reconstruct the available base probe statistics for broader mechanistic work.
p4=artifact('probe-004'); p2=artifact('probe-002')
print('hidden RMS at newline by layer for base tests')
for i in [0,2,3,4,6]:
    h=np.asarray(p4['texts'][i]['hidden'])[:,0,:]
    print(i,np.round(np.sqrt(np.mean(h*h,axis=1)),4).tolist())

# API turn 031; tool result ok=True
# The strong delimiter edit inserted markers inside sentences rather than ending reasoning. Inspect the actual looping trace to choose faithful probe positions instead of relying only on clean synthetic conclusions.
gen0=artifact('generation-001')
s=gen0['base'][0]['text']
print(s[:4800])
print('\n--- SODA ---\n',gen0['base'][7]['text'][:3000])

# API turn 033; tool result ok=True
# Construct an ambiguity-versus-direct-calculation axis solely from frozen vocabulary representations, then inspect how the real looping prefix writes along that axis.
ambig_words=[' ambiguity',' ambiguous',' contradiction',' riddle',' trick',' interpretation',' confusing',' unclear']
calc_words=[' arithmetic',' calculation',' straightforward',' calculate',' computed',' solve',' solution',' answer']
print('semantic tokens',[(s,[decode([int(x)]) for x in tokenize(s)]) for s in ambig_words+calc_words])
def wordmean(words):
    return np.mean(np.stack([np.mean(emb[[int(x) for x in tokenize(s)]],axis=0) for s in words]),axis=0)
ambig=(wordmean(ambig_words)-wordmean(calc_words))*(1+fnorm)
ambig=ambig/np.linalg.norm(ambig)
print('axis overlap with previous doubt/control',float(ambig@doubt),float(ambig@control_unit))
p9=artifact('probe-009')
for i,t in enumerate(p9['texts']):
    h=np.asarray(t['hidden'])
    print('text',i,'semantic projection per layer',np.round(h[:-1,0]@ambig,3).tolist())
    logits=emb@h[-1,0]; top=np.argsort(logits)[-10:][::-1]
    print('top',[(decode([int(j)]),round(float(logits[j]),2)) for j in top], 'end',float(logits[end_id]))
    print('module contributions',[(m.split('layers.')[1],round(float(np.asarray(d['output'])[0]@ambig),3),round(float(np.asarray(d['output'])[0]@control_unit),3)) for m,d in t['modules'].items()])

# API turn 034; tool result ok=True
# Check a data-free SwiGLU moment estimate: for centered isotropic input, E[silu(g·x)(u·x)] = (g·u)/2. This can identify persistent ambiguity-writing components without learning from example labels.
moments={}
for i in range(12,24):
    root=prefix+str(i)+'.'
    norm=1+weight(root+'post_attention_layernorm.weight')
    g=weight(root+'mlp.gate_proj.weight')*norm[None,:]
    up=weight(root+'mlp.up_proj.weight')*norm[None,:]
    b=0.5*np.sum(g*up,axis=1)
    down=weight(root+'mlp.down_proj.weight')
    c=ambig@down
    moments[i]=b
    pos=(b*c)>0
    print(i,'expected ambig',round(float(c@b),4),'positive contribution',round(float(c[pos]@b[pos]),4),'negative contribution',round(float(c[~pos]@b[~pos]),4),'positive count',int(np.sum(pos)))
    mod=root+'mlp.down_proj'
    if mod in p9['texts'][0]['modules']:
        for ti,t in enumerate(p9['texts']):
            a=np.asarray(t['modules'][mod]['input'])[0]
            print('  diagnostic',ti,'actual positive',round(float(c[pos]@a[pos]),4),'actual negative',round(float(c[~pos]@a[~pos]),4),'mean-feature cosine',round(float(a@b/np.linalg.norm(a)/np.linalg.norm(b)),3))


# API turn 035; tool result ok=True
# The real trace builds ambiguity-related activation before the final layer, whose MLP already suppresses that direction. Test removing its contextual buildup while preserving the final layer's lexical correction; the direction comes only from fixed word representations.
clear_edits()
for n in writers:
    if '.layers.23.' not in n:
        w=weight(n)
        save_edit(n,w-np.outer(ambig,ambig@w))
checkpoint('ambiguity_context_ablation')
print('Rank-one contextual ambiguity ablation in layers 0 through 22; layer 23 unchanged.')

# API turn 037; tool result ok=True
# Switch from lexical steering to the hybrid model's memory dynamics. Inspect baseline per-head decay timescales for extreme retention or forgetting that could make completed reasoning hard to retain.
for i in range(24):
    if i%4==3: continue
    a=weight(prefix+str(i)+'.linear_attn.A_log')
    dt=weight(prefix+str(i)+'.linear_attn.dt_bias')
    rate=np.exp(a)*np.logaddexp(0,dt)
    half=np.log(2)/rate
    print(i,'half lives',np.round(half,2).tolist(),'median',round(float(np.median(half)),2))

# API turn 038; tool result ok=True
# Test longer retention only in late linear-attention heads already specialized for long memory. Extend their baseline half-lives fourfold via an additive log-decay change; leave fast local heads and early layers unchanged.
clear_edits()
for i in range(12,24):
    if i%4==3: continue
    n=prefix+str(i)+'.linear_attn.A_log'
    a=weight(n)
    dt=weight(prefix+str(i)+'.linear_attn.dt_bias')
    half=np.log(2)/(np.exp(a)*np.logaddexp(0,dt))
    mask=half>64
    new=a.copy(); new[mask]-=np.log(4.0)
    save_edit(n,new)
    print(i,'modified heads',np.where(mask)[0].tolist())
checkpoint('late_long_memory4')

# API turn 040; tool result ok=False
# Localize the final attention head behavior behind repetitive reconsideration. Decompose its captured output by head and inspect vocabulary effects before choosing a structural head repair.
ow=weight(prefix+'23.self_attn.o_proj.weight')
for ti,t in enumerate(p9['texts']):
    z=np.asarray(t['modules'][prefix+'23.self_attn.o_proj']['input'])[0]
    print('TEXT',ti)
    for head in range(8):
        sl=slice(head*256,(head+1)*256)
        v=ow[:,sl]@z[sl]
        logits=emb@(v*(1+fnorm))
        top=np.argsort(logits)[-5:][::-1]
        print(head,'norm',round(float(np.linalg.norm(v)),3),'end',round(float(logits[end_id]),3),'ambiguity',round(float(v@ambig),3),'doubt',round(float(v@doubt),3),'top',[(decode([int(j)]),round(float(logits[j]),2)) for j in top])

# API turn 041; tool result ok=True
# Repeat the attention-head decomposition using supported NumPy indexing.
ow=weight(prefix+'23.self_attn.o_proj.weight')
for ti,t in enumerate(p9['texts']):
    z=np.asarray(t['modules'][prefix+'23.self_attn.o_proj']['input'])[0]
    print('TEXT',ti)
    for head in range(8):
        v=ow[:,head*256:(head+1)*256]@z[head*256:(head+1)*256]
        logits=emb@(v*(1+fnorm))
        top=np.argsort(logits)[-5:][::-1]
        print(head,'norm',round(float(np.linalg.norm(v)),3),'end',round(float(logits[end_id]),3),'ambiguity',round(float(v@ambig),3),'doubt',round(float(v@doubt),3),'top',[(decode([int(j)]),round(float(logits[j]),2)) for j in top])

# API turn 042; tool result ok=True
# All failed generations share the same verbose 'Here's a thinking process' opening. Test whether this boilerplate trajectory is the attractor: remove only the fixed output direction favoring 'Here' over a direct 'We' calculation opening, without changing decoding settings.
print('opening tokens',[(s,[int(x) for x in tokenize(s)]) for s in ['Here','We','Thinking']])
opening=(emb[int(tokenize('Here')[0])]-emb[int(tokenize('We')[0])])*(1+fnorm)
opening=opening/np.linalg.norm(opening)
clear_edits()
for n in writers:
    w=weight(n)
    save_edit(n,w-np.outer(opening,opening@w))
checkpoint('boilerplate_opening_ablation')
print('One fixed lexical opening direction removed from residual writers.')

# API turn 044; tool result ok=True
# Look for residual-channel concentration as a non-lexical source of repetitive behavior. Compare activation maxima and writer row norms rather than choosing another semantic steering direction.
for i in [0,3,7,11,15,19,22,23,24]:
    hh=np.concatenate([np.asarray(t['hidden'])[i] for t in p4['texts']+p9['texts']],axis=0)
    en=np.mean(hh*hh,axis=0)
    order=np.argsort(en)[-6:][::-1]
    print('layer',i,'max/total energy',float(en[order[0]]/np.sum(en)),'top dims',order.tolist(),'rms',np.sqrt(en[order]).tolist())
print('output writer largest row RMS relative to median')
for i in range(24):
    n=prefix+str(i)+'.mlp.down_proj.weight'; w=weight(n)
    rn=np.sqrt(np.mean(w*w,axis=1)); med=np.median(rn)
    j=np.argsort(rn)[-3:][::-1]
    print(i,[(int(k),round(float(rn[k]/med),2)) for k in j])

# API turn 045; tool result ok=True
# A non-lexical anomaly is consistent across layers: residual channel 0 carries roughly 10–17% of activation energy and is the largest MLP writer row in every layer. Test robust row-norm capping to reduce this concentrated sink without scaling whole tensors.
clear_edits()
for n in writers:
    w=weight(n)
    rn=np.sqrt(np.mean(w*w,axis=1))
    limit=1.5*np.median(rn)
    scales=np.minimum(1.0,limit/np.maximum(rn,1e-12))
    if np.any(scales<1):
        save_edit(n,w*scales[:,None])
        print(n.split('layers.')[1], 'capped',np.where(scales<1)[0].tolist(),'channel0 multiplier',round(float(scales[0]),3))
checkpoint('residual_row_norm_cap15')

# API turn 047; tool result ok=True
# Determine whether dominant residual channel 0 is a content feature or a normalization-control channel before considering the opposite intervention.
for i in [0,1,3,7,11,15,19,23]:
    il=weight(prefix+str(i)+'.input_layernorm.weight')
    pl=weight(prefix+str(i)+'.post_attention_layernorm.weight')
    print(i,'input effective gains 0 and median',float(1+il[0]),float(np.median(1+il)),'MLP gains 0 and median',float(1+pl[0]),float(np.median(1+pl)))
print('final gain channel0',float(1+fnorm[0]),'median',float(np.median(1+fnorm)))
for i in [0,7,15,23]:
    g=weight(prefix+str(i)+'.mlp.gate_proj.weight')
    print(i,'gate input column 0 / median',float(np.linalg.norm(g[:,0])/np.median(np.linalg.norm(g,axis=0))))

# API turn 048; tool result ok=True
# Row capping worsened interpretation, suggesting channel 0 is a stabilizing normalization carrier rather than noise. Test strengthening only this channel in late residual writers to damp runaway late-layer elaboration while leaving early parsing untouched.
clear_edits()
for n in writers:
    layer=int(n.split('layers.')[1].split('.')[0])
    if layer>=12:
        w=weight(n)
        w[0,:]*=2.0
        save_edit(n,w)
checkpoint('late_normalization_carrier2')
print('Only row 0 of late residual writers changed; all other coordinates and early layers fixed.')

# API turn 051; tool result ok=True
# Decompose the full-attention outputs by head along fixed reconsideration and termination directions; identify a common head-level source rather than fitting a new steering vector.
p15=artifact('probe-015')
wait_vec=emb[int(tokenize('Wait')[0])]*(1+fnorm)
for layer in [3,7,11,15,19,23]:
    mod=prefix+str(layer)+'.self_attn.o_proj'
    w=weight(mod+'.weight')
    print('LAYER',layer)
    for ti,t in enumerate(p15['texts']):
        z=np.asarray(t['modules'][mod]['input'])[0]
        vs=np.stack([w[:,j*256:(j+1)*256]@z[j*256:(j+1)*256] for j in range(8)])
        print('text',ti,'doubt',np.round(vs@doubt,3).tolist(),'wait',np.round(vs@wait_vec,3).tolist(),'control',np.round(vs@control_unit,3).tolist())


# API turn 052; tool result ok=True
# Test the localized causal candidate found by head decomposition: ablate final-layer attention head 6, which strongly writes reconsideration continuations at both completed calculations. Revert all previous edits.
clear_edits()
n=prefix+'23.self_attn.o_proj.weight'
w=weight(n)
w[:,6*256:7*256]=0.0
save_edit(n,w)
checkpoint('final_attention_head6_ablation')
print('Changed one 256-column head block in one internal matrix; all other tensors are base.')

# API turn 054; tool result ok=True
# Select the conservative norm-preserving two-matrix reflection based on diagnostic termination selectivity, while explicitly recording that practice generation showed no accuracy gain. Remove all later speculative edits.
restore('end_control_reflection_l23')
print('FINAL TENSORS',[(e['name'],e['changed_entries']) for e in list_edits()])
for aid in ['generation-001','generation-006','generation-007','generation-008','generation-010','generation-011','generation-012','generation-013','generation-014','generation-016']:
    a=artifact(aid)
    for which,es in a.items():
        print(aid,which,'correct',sum(int(e['correct']) for e in es),'total',len(es),'token_limits',sum(int(e['hit_token_limit']) for e in es))
checkpoint('selected_final_reflection')
