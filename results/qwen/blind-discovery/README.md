# Independent Qwen3.5-0.8B repair search on Colab G4

**This trial did not discover a useful repair.** A fresh GPT-6 Astra editor
selected its own weight intervention without receiving the published edit,
earlier conversation, repository access or final test questions. Its candidate
did not reduce looping or improve held-out math accuracy.

| Audited metric | Base | Independent candidate |
| --- | --- | --- |
| Correct boxed answers | 5 / 64 (7.81%) | 3 / 64 (4.69%) |
| Finished with a correct answer | 3 / 64 (4.69%) | 3 / 64 (4.69%) |
| Thinking closed | 3 / 64 | 3 / 64 |
| Hit the 32,768-token limit | 61 / 64 | 61 / 64 |
| Mean generated tokens | 31,415.5 | 31,535.2 |

The standard-score difference was **−3.125 percentage points**: no paired
sample rescues and two regressions. A descriptive bootstrap that resamples the
32 problems as clusters gives a 95% interval of **−7.8125 to 0 percentage
points**. This is a small pilot, not a reliable estimate for all math tasks.
The standard grader can credit a correct box in unfinished thinking; the
finished-and-correct row requires generation to stop before the token limit.
One previously unfinished sample became a correct completed answer, while one
previously correct completed sample began looping. There was no net completion
gain.

All 128 saved outputs were regraded **after the edit was frozen** with the
published extraction and equivalence checker. There were no score changes:
[grader validation](grader_validation.json), [analysis](analysis.json),
[per-problem scores](per_problem.json).

## What the editor chose

The editor's hypothesis was that the final block suppresses the signal to end
thinking. It chose a Householder reflection of a single output direction in
two layer-23 matrices:

```text
d = (embedding[</think>] − embedding[<think>]) * (1 + final_norm_weight)
u = d / ||d||
W_new = W − 2 outer(u, u @ W)
```

The affected matrices are `model.language_model.layers.23.mlp.down_proj.weight`
and `model.language_model.layers.23.self_attn.o_proj.weight`. Before BF16
rounding, the reflection preserves matrix singular values and reverses only
one output direction. This is an explanatory hypothesis, not an established
repair mechanism. The embeddings, output head, normalization and forward
program remain fixed. No training or parameter fitting was used.

The edit changes **5,183,186 BF16 entries**. A complete checkpoint was rebuilt
locally and every named tensor was hashed: exactly these two of 488 tensors
changed, and all 486 others matched the pinned base. See
[the freeze manifest](frozen.json), [full replay validation](replay_validation.json)
and [the selected weights](selected.safetensors).

## Discovery protocol and limitations

- Editor: `gpt-6-astra`, maximum reasoning effort, fresh Responses API context,
  no prior research chat or published repair. The clean VM payload contained
  only generic numerical tools and practice data. The full original payload,
  checksums and all 55 API request/response pairs are saved here.
- Target: `Qwen/Qwen3.5-0.8B` revision
  `2fc06364715b967f1860aea9cf38778875588b17`, BF16. The pinned full multimodal
  release was used with text inputs only.
- Available practice: 256 GSM8K **training** problems selected with seed
  20261102. The editor evaluated eight baseline problems and candidate subsets
  of two or four. **Every practice generation used its chosen 2,048-token cap,
  hit that cap and scored zero.** This short, narrow screening was a major
  limitation. The final choice came from activation diagnostics rather than a
  measured practice accuracy improvement; the editor said so in its own
  [selection rationale](editor_selection.json).
- Search: 55 API turns, 16 successful forward-tool calls, and 36 CPU calls
  including two host preflight/maintenance calls. Limits were 60 API turns,
  20 forward calls, 100 CPU calls and $25 estimated API usage. The editor
  finished voluntarily within these limits.
- Explored hypotheses included termination-direction edits, semantic direction
  removal, linear-attention retention changes, residual row interventions and
  a single attention-head ablation. Later speculative edits were reverted.
  [Editor CPU cells](editor_analysis_transcript.py), [tool log](tool_log.jsonl)
  and [checkpoint manifests](candidate_manifests/) record the actual search.
- During discovery, two candidate weight transfers failed before generation.
  The service was repaired to use vLLM's native checkpoint reload API. Exact
  current weights, checkpoints, observations and API history were retained;
  temporary CPU variables were cleared. The same conversation resumed at turn
  25 with an infrastructure-only notice. The two failed transfers were
  refunded. GPU checksums verified candidate loading and exact base restoration.
  Original and repaired source snapshots and maintenance records are included.

The auditable blinding claim concerns what was supplied to this API run. It
cannot establish absence of relevant knowledge in the editor's pretraining.
See [blinding audit](blinding_audit.json) and the
[full protocol](../../../docs/blind-qwen-discovery.md).

## Independent evaluation

The 32 MATH-500 questions were selected and sealed before discovery, with seed
20261103, stratified across seven subjects and excluding the earlier 16-question
published-edit pilot. Dataset revision:
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`. The test file was physically absent
from the VM when the weights were frozen. It was uploaded only afterward;
there were **zero API requests after freeze**.

Each condition received two samples per problem with identical prompts,
thinking enabled, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5,
seed 0 and a 32,768-token cap. Test data and seal, raw generations, activation
arrays, source snapshots and grading results are saved in this directory.
The earlier published-edit pilot used different questions, so its headline
accuracy is not a direct comparison against this candidate.

Hardware was a requested Colab **G4**, allocated as an NVIDIA RTX PRO 6000
Blackwell Server Edition with 96 GB memory. Software: vLLM 0.33.0,
Transformers 5.20.2, Torch 2.14.0 and math-verify 0.9.0. Each test condition took
about 6.6 minutes. Results were downloaded and the G4 was stopped; no active
Colab sessions remained. Runtime details are in [runtime.json](runtime.json).

Estimated editor API usage was **$7.128361**, excluding Colab compute charges.
This is calculated from recorded token usage and the
[official model rates](https://developers.openai.com/api/docs/models/gpt-6-astra)
verified on 2026-11-03 UTC; it is not a billing invoice. See
[usage summary](api_usage_summary.json).

## Replay without further API calls

With the pinned base cached locally, reconstruct the exact full candidate:

```sh
uv run --extra qwen --no-editable python experiments/blind/replay.py \
  --base .artifacts/qwen-base --output .artifacts/qwen-blind-replay
```

The command verifies the frozen file hash and the original/edited hashes for
both affected tensors before writing a new checkpoint directory. It requires
CPU and disk space, not a GPU or an API key. Regrade the saved texts with:

```sh
uv run --extra qwen --no-editable python experiments/blind/analyze.py \
  results/qwen/blind-discovery
```

This result supports improving the discovery protocol before another trial:
longer practice generation and demonstrated gains across more practice
problems would provide a stronger basis for selecting a repair. The sealed
test results from this run must not be fed back to this editor as further
discovery feedback.
