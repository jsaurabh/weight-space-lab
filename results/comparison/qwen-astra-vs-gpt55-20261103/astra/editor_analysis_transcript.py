# Exact editor CPU cells in execution order.
# Inspection transcript; helpers and persistent state were supplied by the backend.

# Turn 001; tool success: True
# Inspect allowed tensor names and preserve untouched base before diagnosing practice failures.
checkpoint('base')
print(len(names)); print(names[:35]); print(config)
print('IDs',practice_ids[:5],practice_ids[7469:7474],practice_ids[14744:14749])

# Turn 002; tool success: True
# Select a fixed, source-stratified diagnostic set without looking at answers or model success.
idx=[int(i) for i in np.linspace(31,7401,8)]+[int(i) for i in np.linspace(7488,14715,20)]+[14800,14979,15158,15337]
ids=[practice_ids[i] for i in idx]
for i,p in zip(idx,ids):
 r=practice_row(p); print(i,p,str(r)[:280])
print('BATCH',ids)

# Turn 004; tool success: True
# Inspect complete failed and successful outputs and scan internal weight statistics for plausible recurrent amplification mechanisms.
g=artifact('generation-001'); print(type(g));print(g.keys())
print(str(g)[:1800])
for n in names:
 if '.layers.' in n and ('layernorm' in n or '.norm.weight' in n or 'A_log' in n or 'dt_bias' in n):
  w=weight(n);print(n.replace('model.language_model.',''),w.shape,'range',np.round([w.min(),w.max(),np.mean(w),np.std(w)],3))

# Turn 005; tool success: True
# Characterize where reasoning degenerates and prepare faithful chat-formatted probes of long repetitions and completed reasoning.
rows=g['base']; print(rows[0].keys())
for j,r in enumerate(rows):
 print(j,r['id'],'correct',r['correct'],'tokens',r['tokens'],'closed',r.get('thinking_closed'), 'tail',r['text'][-180:].replace('\n',' '))
for j in [0,5,13,25]:
 print('\nDETAIL',j,rows[j]['text'][:6800])
print('SPECIAL',[(i,decode([i])) for i in range(248044,248054)])
print('self_attn',[(n,shapes[n]) for n in names if 'layers.19.self_attn' in n])

# Turn 006; tool success: True
# Check chat tokens and attention tensor shapes; locate a compact transition from valid calculation to repetitive verification.
print('SPECIAL',[(i,decode([i])) for i in range(248044,248054)])
print([(n,shapes[n]) for n in names if 'layers.19.self_attn' in n])
for j in [4,11,13]:
 r=rows[j];print('TEXT',j,'FIRST',r['text'][:5200])

# Turn 007; tool success: True
# Compare full-attention head scales and output projection head energies before selecting a causal ablation.
for l in [3,7,11,15,19,23]:
 p=f'model.language_model.layers.{l}.self_attn.'
 print('LAYER',l)
 for s in ['q_norm.weight','k_norm.weight']:
  w=weight(p+s);print(s, np.round([w.min(),w.max(),np.mean(w),np.std(w)],3))
 for s in ['q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight']:
  w=weight(p+s);print(s,'rms',round(float(np.sqrt(np.mean(w*w))),4),'max',round(float(np.max(np.abs(w))),4))
 o=weight(p+'o_proj.weight').reshape(1024,8,256);print('o_head_energy',np.round(np.sqrt(np.sum(o*o,axis=(0,2))),2))
print('factorial continuation 3400-7000:',rows[13]['text'][3400:7000])

# Turn 008; tool success: True
# Locate the exact earliest stable repeated loop, to probe a contiguous saved model continuation rather than fabricate a loop.
for j in [0,1,2,8,9,10,11,12,13]:
 t=rows[j]['text'];lines=t.splitlines();seen={}; hit=None;ch=0
 for line in lines:
  z=line.strip()
  if len(z)>12:
   seen[z]=seen.get(z,0)+1
   if seen[z]==5:hit=(ch,z);break
  ch+=len(line)+1
 print(j,'first 5th exact line',hit)
print('GEOM 5200-9000',rows[11]['text'][5200:9000])

# Turn 009; tool success: True
# Extract one entire contiguous failure prefix for a long-context head-activation probe.
print(rows[1]['text'][:8800])

# Turn 012; tool success: False
# Measure matched-token hidden drift and per-head output changes induced only by repetition, without fitting to math answers.
g=artifact('generation-001'); rows=g['base']; ids=[r['id'] for r in rows]
p=artifact('probe-002');print(p.keys());print(p['texts'][0].keys())
a=p['texts']
for i in [0,2]:
 h0=np.asarray(a[i]['hidden']);h1=np.asarray(a[i+1]['hidden']);print('pair',i,'hidden-relative-drift',np.round(np.linalg.norm(h1-h0,axis=(1,2))/np.linalg.norm(h0,axis=(1,2)),3))
mods=list(a[0]['modules']);scores=[]
for m in mods:
 w=weight(m+'.weight');nh=8 if 'self_attn' in m else 16;hd=w.shape[1]//nh
 vals=[]
 for i in [0,2]:
  x=np.asarray(a[i]['modules'][m]['input']);y=np.asarray(a[i+1]['modules'][m]['input']); d=y-x
  v=[]
  for h in range(nh):
   sl=slice(h*hd,(h+1)*hd); b=x[:,sl]@w[:,sl].T;c=y[:,sl]@w[:,sl].T
   v.append([np.linalg.norm(c-b),np.linalg.norm(b),np.linalg.norm(c)])
  vals.append(v)
 v=np.array(vals);diff=np.mean(v[:,:,0],0); rel=np.mean(v[:,:,0]/(v[:,:,1]+1e-8),0)
 print(m.split('layers.')[1],'diff',np.round(diff,2),'relative',np.round(rel,2),'growth',np.round(np.mean(v[:,:,2]/v[:,:,1],0),2))
 for h in range(nh): scores.append((float(diff[h]),float(rel[h]),m,h))
print('largest absolute drifts',sorted(scores,reverse=True)[:12])

# Turn 013; tool success: True
# Compute repetition-sensitive attention head contributions with supported NumPy indexing.
g=artifact('generation-001'); rows=g['base']; ids=[r['id'] for r in rows]
p=artifact('probe-002');a=p['texts'];print(p.keys());print(a[0].keys())
for i in [0,2]:
 h0=np.asarray(a[i]['hidden']);h1=np.asarray(a[i+1]['hidden']);print('pair',i,'hidden-relative-drift',np.round(np.linalg.norm(h1-h0,axis=(1,2))/np.linalg.norm(h0,axis=(1,2)),3))
mods=list(a[0]['modules']);scores=[];headvals={}
for m in mods:
 w=weight(m+'.weight');nh=8 if 'self_attn' in m else 16;hd=w.shape[1]//nh
 vals=[]
 for i in [0,2]:
  x=np.asarray(a[i]['modules'][m]['input']);y=np.asarray(a[i+1]['modules'][m]['input'])
  v=[]
  for h in range(nh):
   b=x[:,h*hd:(h+1)*hd]@w[:,h*hd:(h+1)*hd].T;c=y[:,h*hd:(h+1)*hd]@w[:,h*hd:(h+1)*hd].T
   v.append([np.linalg.norm(c-b),np.linalg.norm(b),np.linalg.norm(c)])
  vals.append(v)
 v=np.array(vals);headvals[m]=v;diff=np.mean(v[:,:,0],0); rel=np.mean(v[:,:,0]/(v[:,:,1]+1e-8),0)
 print(m.split('layers.')[1],'diff',np.round(diff,2),'relative',np.round(rel,2),'growth',np.round(np.mean(v[:,:,2]/v[:,:,1],0),2))
 for h in range(nh): scores.append((float(diff[h]),float(rel[h]),m,h))
print('largest absolute drifts',sorted(scores,reverse=True)[:12])

# Turn 014; tool success: True
# Create a head-local causal test: halve only layer-11 attention heads 1 and 2, which showed the largest repetition-induced amplification (2.06x and 2.55x), leaving other heads untouched.
restore('base')
edit_name='model.language_model.layers.11.self_attn.o_proj.weight'
w=weight(edit_name);w[:,256:768]*=0.5;save_edit(edit_name,w);checkpoint('l11_h12_half')
print(list_edits())

# Turn 016; tool success: True
# Check preserved checkpoints and artifacts after the generation worker startup failure before retrying the same experiment.
print('edits',list_edits())
print('baseline',len(artifact('generation-001')['base']))
print('probe keys',artifact('probe-002').keys())
restore('l11_h12_half');print('restored',list_edits())

# Turn 020; tool success: True
# Quantify the probe-level causal effect and preserve compact diagnostics while checking whether generation can be recovered.
a=artifact('probe-002')['texts'];b=artifact('probe-003')['texts']
for i in range(4):
 x=np.array(a[i]['next_logprobs']);y=np.array(b[i]['next_logprobs']);print('text',i,'base/current mean observed token logprob',np.round([x[np.isfinite(x)].mean(),y[np.isfinite(y)].mean()],4),'last top base',[(decode([int(t)]),round(float(v),3)) for t,v in zip(a[i]['top_ids'][-1][:5],a[i]['top_logprobs'][-1][:5])],'edited',[(decode([int(t)]),round(float(v),3)) for t,v in zip(b[i]['top_ids'][-1][:5],b[i]['top_logprobs'][-1][:5])])
print(list_edits())

# Turn 022; tool success: True
# Return active weights to the untouched base while generation is blocked; keep the candidate checkpoint only for a future causal accuracy test.
restore('base');print(list_edits())

# Turn 024; tool success: True
# Resume the checkpointed causal experiment after the interruption, preserving the untouched base checkpoint.
restore('l11_h12_half');print(list_edits())
g=artifact('generation-001');rows=g['base'];ids=[r['id'] for r in rows];print('baseline correct',sum(r['correct'] for r in rows),'n',len(rows))

# Turn 026; tool success: True
# Prepare a controlled probe battery with repetitive continuations and nonrepetitive arithmetic controls; restore base for its reference measurements.
restore('base');print('active edits',list_edits())
a=artifact('probe-002')['texts']
for i in [0,1,2,3]:
 q=np.asarray(a[i]['next_logprobs']);print('original probe',i,'mean observed-token logprob',round(float(q[np.isfinite(q)].mean()),5))

# Turn 027; tool success: True
# The service reports completed maintenance; restore the saved half-strength candidate for one renewed attempt at the required accuracy test.
restore('l11_h12_half');print(list_edits())

# Turn 029; tool success: True
# Reject attenuation: accuracy fell from 5/32 to 2/32, consistent with probes showing stronger loop likelihood. Test the opposite causal hypothesis: these heads are a compensatory anti-repetition response.
base=artifact('generation-001')['base'];half=artifact('generation-004')['current']
print('rescues',[b['id'] for b,h in zip(base,half) if h['correct'] and not b['correct']]);print('losses',[b['id'] for b,h in zip(base,half) if b['correct'] and not h['correct']])
restore('base');en='model.language_model.layers.11.self_attn.o_proj.weight';w=weight(en);w[:,256:768]*=2.0;save_edit(en,w);checkpoint('l11_h12_double');print(list_edits())

# Turn 031; tool success: True
# Both gains on the initially implicated heads worsened accuracy; reject that hypothesis. Examine whether the matched-token repetition drift has a shared residual direction, suitable for a selective geometric rather than whole-head intervention.
restore('base');a=artifact('probe-002')['texts'];dirs=[]
for l in range(1,25):
 ds=[]
 for i in [0,2]:
  h0=np.asarray(a[i]['hidden'])[l];h1=np.asarray(a[i+1]['hidden'])[l]
  d=np.mean(h1-h0,axis=0);ds.append(d)
 c=np.dot(ds[0],ds[1])/(np.linalg.norm(ds[0])*np.linalg.norm(ds[1]));d=ds[0]/np.linalg.norm(ds[0])+ds[1]/np.linalg.norm(ds[1]);d/=np.linalg.norm(d);dirs.append(d)
 print(l,'cross-question cosine',round(float(c),3),'mean drift norms',np.round([np.linalg.norm(v) for v in ds],3),'late projections',np.round([np.mean(np.asarray(a[i]['hidden'])[l]@d) for i in range(4)],3))
print('head net alignment with local shared drift')
for m in a[0]['modules']:
 l=int(m.split('layers.')[1].split('.')[0]);w=weight(m+'.weight');nh=8 if 'self_attn' in m else 16;hd=w.shape[1]//nh;d=dirs[l]
 av=[]
 for h in range(nh):
  v=[]
  for i in [0,2]:
   dx=np.mean(np.asarray(a[i+1]['modules'][m]['input'])-np.asarray(a[i]['modules'][m]['input']),axis=0)
   v.append(float(d@(w[:,h*hd:(h+1)*hd]@dx[h*hd:(h+1)*hd])))
  av.append(v)
 print(m.split('layers.')[1],np.round(np.array(av).T,3))

# Turn 032; tool success: True
# Create a selective alternative to head scaling: remove half of the cross-question repetition-drift component from middle-layer residual writes. Directions come only from matched-token hidden differences, not answers or stop-token vectors.
restore('base');a=artifact('probe-002')['texts'];loopdirs={}
for l in range(7,19):
 ds=[]
 for i in [0,2]:
  d=np.mean(np.asarray(a[i+1]['hidden'])[l+1]-np.asarray(a[i]['hidden'])[l+1],axis=0);ds.append(d/np.linalg.norm(d))
 d=ds[0]+ds[1];d/=np.linalg.norm(d);loopdirs[l]=d
stats=[]
for l,d in loopdirs.items():
 suffix='self_attn.o_proj.weight' if l%4==3 else 'linear_attn.out_proj.weight'
 for s in [suffix,'mlp.down_proj.weight']:
  n=f'model.language_model.layers.{l}.{s}';w=weight(n);v=d@w;delta=np.outer(d,v);stats.append(float(np.linalg.norm(delta)/np.linalg.norm(w)));save_edit(n,w-0.5*delta)
checkpoint('loop_projection_half');print('edited tensors',len(list_edits()),'full projection relative norms min/mean/max',np.round([np.min(stats),np.mean(stats),np.max(stats)],4))

# Turn 034; tool success: True
# The half projection did not improve accuracy (4/32); compare the natural full-projection strength before rejecting this mechanism.
restore('base')
for l,d in loopdirs.items():
 suffix='self_attn.o_proj.weight' if l%4==3 else 'linear_attn.out_proj.weight'
 for s in [suffix,'mlp.down_proj.weight']:
  n=f'model.language_model.layers.{l}.{s}';w=weight(n);save_edit(n,w-np.outer(d,d@w))
checkpoint('loop_projection_full');print('full-projection tensors',len(list_edits()))

# Turn 036; tool success: True
# Start a distinct causal test of sharp attention feedback, not stopping behavior: inspect and cap only above-unit query-normalization channels in later full-attention layers, preserving low-gain channels and all value/content projections.
restore('base');capstats=[]
for l in [7,11,15,19,23]:
 n=f'model.language_model.layers.{l}.self_attn.q_norm.weight';w=weight(n);v=np.minimum(w,0.25);save_edit(n,v);capstats.append((l,int(np.sum(w>0.25)),np.round(np.quantile(w,[0,0.25,0.5,0.75,1]),3).tolist()))
checkpoint('query_gain_cap025');print(capstats)

# Turn 038; tool success: True
# The 1.25 effective-gain cap worsened accuracy to 1/32. Compare a milder dimensionwise cap (1.5 effective gain) to distinguish an excessive intervention from an unsupported mechanism.
restore('base')
for l in [7,11,15,19,23]:
 n=f'model.language_model.layers.{l}.self_attn.q_norm.weight';w=weight(n);save_edit(n,np.minimum(w,0.5))
checkpoint('query_gain_cap050');print('candidate tensors',len(list_edits()))

# Turn 040; tool success: True
# All tested variants failed to beat baseline on the diagnostic set. Use the final call for an independent 32-problem paired check of the best-performing non-base candidate (full projection), with source-stratified IDs chosen without inspecting outcomes.
base=artifact('generation-001')['base'];full=artifact('generation-007')['current'];print('full projection rescues',sum(c['correct'] and not b['correct'] for b,c in zip(base,full)),'losses',sum(b['correct'] and not c['correct'] for b,c in zip(base,full)))
idx=[int(i) for i in np.linspace(149,7323,8)]+[int(i) for i in np.linspace(7541,14681,20)]+[14823,15002,15181,15360]
validation_ids=[practice_ids[i] for i in idx];print(validation_ids)
restore('loop_projection_full');print('candidate tensors',len(list_edits()))

# Turn 042; tool success: True
# Audit the independent gains for real completed mathematical answers, not merely shorter outputs or accidental answer extraction, and aggregate paired results across both 32-problem sets.
d=artifact('generation-010');b=d['base'];c=d['current'];b0=artifact('generation-001')['base'];c0=artifact('generation-007')['current']
for x,y in zip(b,c):
 if y['correct'] and not x['correct']:
  print('RESCUE',x['id'],'expected',x['answer'],'tokens',y['tokens'],'closed',y['thinking_closed'],'cutoff',y['hit_token_limit'],'tail',y['text'][-420:].replace('\n',' '))
print('DIAGNOSTIC losses')
for x,y in zip(b0,c0):
 if x['correct'] and not y['correct']:print(x['id'],'edited tokens',y['tokens'],'closed',y['thinking_closed'],'tail',y['text'][-180:].replace('\n',' '))
for label,bb,cc in [('diagnostic',b0,c0),('validation',b,c),('combined',b0+b,c0+c)]:
 print(label,'correct',sum(x['correct'] for x in bb),sum(x['correct'] for x in cc),'rescues',sum(y['correct'] and not x['correct'] for x,y in zip(bb,cc)),'losses',sum(x['correct'] and not y['correct'] for x,y in zip(bb,cc)),'completed_correct',sum(x['correct'] and x['thinking_closed'] and not x['hit_token_limit'] for x in bb),sum(x['correct'] and x['thinking_closed'] and not x['hit_token_limit'] for x in cc))
 for src in ['gsm8k','math','aime']:
  print(src,sum(x['correct'] for x in bb if x['id'].startswith(src)),sum(x['correct'] for x in cc if x['id'].startswith(src)))
print('independent exact paired two-sided sign probability',2.0*(0.5**7))
print('edit count',len(list_edits()))
