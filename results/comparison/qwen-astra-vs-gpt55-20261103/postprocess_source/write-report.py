from pathlib import Path
import json,re
r=Path('results/comparison/qwen-astra-vs-gpt55-20261103')
a=json.loads((r/'analysis.json').read_text());u=json.loads((r/'api_usage_summary.json').read_text());g=json.loads((r/'grader_validation.json').read_text())
c=a['conditions'];f={k:json.loads((r/k/'frozen.json').read_text()) for k in ['astra','gpt55']}
metric=[('Correct boxed answers','correct'),('Finished with a correct answer','finished_and_correct'),('Thinking closed','thinking_closed'),('Hit the 32,768-token limit','hit_token_limit')]
rows='\n'.join('| '+label+' | '+' | '.join(str(c[k][key])+' / 64' for k in ['base','astra','gpt55'])+' |' for label,key in metric)
rows+='\n| Mean generated tokens | '+' | '.join(f"{c[k]['mean_tokens']:,.1f}" for k in ['base','astra','gpt55'])+' |'
comparisons=[]
for key,label in [('astra_minus_base','Astra − base'),('gpt55_minus_base','GPT-5.5 − base'),('gpt55_minus_astra','GPT-5.5 − Astra')]:
 v=a['comparisons'][key];lo,hi=v['problem_cluster_bootstrap_95_interval'];comparisons.append(f"| {label} | {100*v['accuracy_difference']:+.2f} pp | [{100*lo:+.2f}, {100*hi:+.2f}] pp | {v['paired_rescues']} / {v['paired_losses']} |")
costs='\n'.join(f"| {'GPT-6 Astra' if k=='astra' else 'GPT-5.5'} | {u['editors'][k]['api_calls']} | 10 | ${u['editors'][k]['estimated_api_cost_usd']:.6f} |" for k in ['astra','gpt55'])
main=f'''# Astra vs GPT-5.5: Qwen weight-repair pilot

On the shared unseen test, base Qwen scored **{c['base']['correct']}/64**,
Astra's selected edit **{c['astra']['correct']}/64**, and GPT-5.5's selected edit
**{c['gpt55']['correct']}/64**. Each editor independently selected its own repair
with ten successful GPU feedback calls. Neither received the published edit,
the other editor's work, or test feedback.

## Shared test

32 new MATH-500 questions × two samples per condition; the same questions,
prompts, seed and decoding settings for base and both frozen choices.

| Audited metric | Base | Astra edit | GPT-5.5 edit |
| --- | --- | --- | --- |
{rows}

GPT-5.5 independently found a repair that improved both scored and completed
correct answers in this run. Its two-answer lead over Astra is small: the
paired interval below includes either editor leading. This pilot does not
establish a general capability ranking.

The standard grader can credit a correct box in unfinished thinking. The
finished-and-correct metric additionally requires stopping before the token cap.
All **192** saved outputs were regraded after both choices were frozen; there
were **{len(g['changes'])}** score changes from the online grader.
[Analysis](analysis.json) · [Per-sample scores](graded_scores.jsonl) ·
[Grader validation](grader_validation.json).

| Paired comparison | Accuracy difference | Descriptive 95% interval | Rescues / losses |
| --- | --- | --- | --- |
{chr(10).join(comparisons)}

Intervals resample the 32 questions as clusters, using 10,000 replicates. They
measure variation across this small question set. There is one discovery run
per editor, so these intervals do not measure variation across editor attempts.

## Selected edits

| Editor | Edited tensors | Changed BF16 entries | Practice evidence for selected edit |
| --- | --- | --- | --- |
| GPT-5.5 | {len(f['gpt55']['edits'])} | {sum(e['changed_entries'] for e in f['gpt55']['edits']):,} | Four batches: base 14/128 → edit 34/128 |
| Astra | {len(f['astra']['edits'])} | {sum(e['changed_entries'] for e in f['astra']['edits']):,} | Diagnostic batch: 5/32 → 5/32; fresh paired batch: 3/32 → 10/32 |

GPT-5.5 chose a rank-one change to the final MLP output matrix, in layer 23.
It computed an analytic detector from loop/control activation means and used
it to reduce a Wait-token direction relative to several continuation/closure
token embeddings. The selected strength was 8. Available reference answers
were used for practice evaluation; the edit uses no gradients or optimizer.
See its [CPU cells](gpt55/editor_analysis_transcript.py),
[selection](gpt55/frozen.json) and [practice ledger](gpt55/practice_summary.json).

Astra computed a direction per layer from matched-token hidden-state changes
between short and repeated continuations of two questions. It selected
`W_new = W - outer(d, d @ W)` for MLP and attention/linear-attention output
matrices in layers 7–18: 24 matrices in total. This removes the component along
a normalized direction. Its behavioral interpretation is a hypothesis; the
recorded evidence establishes the measured performance, not a complete account
of the reasoning circuit. [CPU cells](astra/editor_analysis_transcript.py) ·
[Selection](astra/frozen.json) · [Practice ledger](astra/practice_summary.json).

Complete checkpoint reconstruction verified all **488** tensors for each
candidate: exactly the selected tensors changed, with every other tensor
matching the pinned base. [Astra verification](astra/replay_validation.json) ·
[GPT-5.5 verification](gpt55/replay_validation.json).

## Protocol and limits

- Editors: `gpt-6-astra` and `gpt-5.5-2026-04-23`, both `xhigh`, 16,000 maximum
  output tokens per API call. Astra's separate `max` setting was not tested.
- Target: the full `Qwen/Qwen3.5-0.8B` release at
  `2fc06364715b967f1860aea9cf38778875588b17`, BF16, used with text inputs.
- Shared practice pool: GSM8K train 7,469, MATH train 7,275 and AIME 714.
  Editors chose their own practice subsets. Tool-call budgets were matched;
  actual generated token counts and batch choices differed.
- Both initial API requests were identical apart from model ID. The initial
  numerical payload contained only generic tools and practice data.
- The test excluded all 48 questions used in earlier pilots. Its hash was
  committed before discovery, and it was uploaded only after both choices
  froze. There were zero editor API requests after either editor's freeze.
- Astra's run required an API context recovery, correction of a recovery
  launcher bug, and reconstruction after a user pause and Colab VM loss.
  Checkpoint tensor hashes matched exactly after reconstruction. Three early
  practice generations retain only their original summaries; **all final test
  outputs are retained**. These deviations affect interpretation of search
  trajectory, time and cost. [Recovery](recovery_001/) ·
  [Pause](pause_001/manifest.json) · [Resumption](resumption_002/manifest.json).

This is a bounded task comparison with smaller batches/probes and a restricted
NumPy interface. It supports conclusions about these observed repair attempts;
additional attempts and tasks would be needed to estimate general model parity.
The blinding audit concerns supplied runtime inputs, not unknown pretraining
exposure. [Full protocol and commands](../../../docs/qwen-editor-comparison.md) ·
[Machine-readable protocol](protocol.json) · [Blinding audit](blinding_audit.json).

## Cost and hardware

| Editor | Completed API responses | Successful GPU feedback calls | Estimated API cost |
| --- | --- | --- | --- |
{costs}

Recorded comparison API usage totaled **${u['estimated_total_api_cost_usd']:.6f}**,
excluding Colab. The earlier excluded tooling trial cost another $1.584578.
Astra's figure includes infrastructure detours. Costs use recorded token usage
and the official rates linked in [the usage summary](api_usage_summary.json);
they are estimates, not a billing invoice.

Qwen generation/probes used Colab **G4**, allocated as an NVIDIA RTX PRO 6000
Blackwell Server Edition with 96 GB memory. Weight analysis used CPU. Results
were downloaded and the G4 was stopped. Runtime/software details are in each
editor directory; source snapshots and raw logs are retained.

## Reproduce

```sh
python3 experiments/fetch_artifacts.py --group qwen-comparison
uv run --extra qwen --no-editable python experiments/qwen_compare/analyze.py results/comparison/qwen-astra-vs-gpt55-20261103
```

Regrading saved texts needs CPU and no API key. To reconstruct either full
candidate, use the [frozen-choice replay commands](../../../docs/qwen-editor-comparison.md#replay-a-frozen-choice).
They verify the selected file and tensor hashes against the pinned base.
[Model artifacts](https://huggingface.co/jsaurabh/weight-space-lab-artifacts)
are pinned in the repository's `model_artifacts.json`. Fresh searches and GPU
generations can vary; exact score verification uses the saved texts.
'''
(r/'README.md').write_text(main)
p=Path('README.md');text=p.read_text();row=f"| Astra vs GPT-5.5 Qwen repair | Base **{c['base']['correct']}/64**; Astra **{c['astra']['correct']}/64**; GPT-5.5 **{c['gpt55']['correct']}/64** | [Report](results/comparison/qwen-astra-vs-gpt55-20261103/README.md) |"
if row not in text:
 marker='| Blind Astra Qwen repair search |'
 lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith(marker));lines.insert(i+1,row);text='\n'.join(lines)+'\n'
text=text.replace('Tiny-task scores cover the complete stated domains. Qwen scores are small pilots\non different question sets, not a direct comparison or full benchmark replication.','Tiny-task scores cover the complete stated domains. Qwen pilots use different\nquestion sets; compare editors within the matched comparison above. These are\nsmall pilots rather than full benchmark replications.')
command='uv run --extra qwen --no-editable python experiments/qwen_compare/analyze.py results/comparison/qwen-astra-vs-gpt55-20261103'
if command not in text:text=text.replace('uv run --extra qwen --no-editable python experiments/blind/analyze.py results/qwen/blind-discovery','uv run --extra qwen --no-editable python experiments/blind/analyze.py results/qwen/blind-discovery\n'+command)
m=json.loads(Path('model_artifacts.json').read_text());size=round(sum(v['bytes'] for v in m['files'].values())/1e6)
text=re.sub(r'download is about \d+ MB','download is about '+str(size)+' MB',text);p.write_text(text)
print('Comparison report and compact root README updated.')
