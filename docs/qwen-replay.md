# Replaying the published Qwen3.5-0.8B weight edit

This experiment compares the original model with the published 50-pass edit.
It reconstructs that edit exactly; it does not independently rediscover a new
edit with an API programmer. No training or gradients are involved.

## Sources and fixed artifacts

- [Original experiment logs](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math)
- [Published edited model and independent evaluation](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)
- Base: `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`.
- Upstream diff revision: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
- MATH-500 dataset revision: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- Published numerical diff and attribution: `third_party/qwen_p50c/`.

The edit affects only `model.language_model.layers.15.mlp.down_proj.weight` and
`model.language_model.layers.19.mlp.down_proj.weight`. They have shape
1024 × 3584. Both base and edited tensors are verified against raw bfloat16
SHA-256 hashes; 130,188 entries actually differ after rounding. The authors
describe this as removing a looping direction from 64 columns of each matrix.

Our independent numerical audit is saved in `results/qwen/edit-analysis.json`.
On the 64 changed columns of each layer, the first singular component explains
99.99945% and 99.99914% of the squared difference, respectively. Up to bfloat16
rounding, each edit is approximately `W' = W - 6*u*(u.T @ W)` on those columns.
Thus it counteracts that component strongly: the new projection is roughly -5
times the original projection, rather than an orthogonal deletion. The two
layer-specific directions have absolute cosine similarity 0.849. This audit
describes the published numerical change; it does not independently establish
the behavioral meaning of those directions.

## Local feasibility

The initial device has 16 GB unified memory and an Apple GPU. Full bfloat16
weights fit, and the local Transformers pilot completed on MPS. However, its
unoptimized recurrent/linear-attention kernels yielded roughly 10–30 tokens/s
in the short warm-up run. These timing samples include first-use overhead and
are not a fair throughput comparison between base and edit.

The local pilot used one problem with a 64-token cap. Both conditions hit that
cap before an answer. Its zero scores are a setup check, not an estimate of
math accuracy or evidence against the edit.

A second local run used a 4,096-token cap on the same fixed problem. Both
conditions again hit the cap without a correct boxed answer, at roughly 31
tokens/s. This is recorded under `results/qwen/local-4096/`; the G4 comparison
retains the published 32,768-token cap rather than treating the short local run
as a benchmark replication.

```sh
uv sync --extra qwen --no-editable
uv run --extra qwen --no-editable weight-space-qwen prepare
uv run --extra qwen --no-editable weight-space-qwen verify
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --extra qwen --no-editable weight-space-qwen compare \
  --device mps --limit 1 --max-new-tokens 64 --output results/qwen/new-local-pilot
```

The local commands expect the pinned model under `.artifacts/qwen-base/` and
the pinned dataset under `.artifacts/math500/test.jsonl`. Large artifacts are
ignored by Git. `verify_checkpoint()` reads only the two relevant tensors.
`compare` runs the base first, then applies the verified edit in memory.
Its custom presence penalty penalizes token types generated after the prompt,
once each, rather than applying a repetition multiplier or penalizing prompt
tokens. Math grading uses the last complete boxed answer with math-verify 0.9.0.

## Colab G4 pilot

Use a Colab **G4**, per the user's preference. The scripts do not provision
hardware themselves. The allocated session must always be stopped after results
have been downloaded. The first L4 setup session was terminated before evaluation.

The G4 pilot uses a fresh RTX PRO 6000 Blackwell Server Edition (96 GB).
Evaluation dependencies match the independent published rerun: vLLM 0.33.0,
Transformers 5.20.2, and math-verify 0.9.0. The runner records actual versions.

Sixteen MATH-500 problems are selected before generation with Python seed
20261102. Both conditions use those exact problems and the same sampling:
thinking enabled, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5,
four samples per problem, seed 0, and up to 32,768 generated tokens. The user
message is the problem followed by the original boxed-answer instruction.

The base and edited models run in separate processes to release GPU resources
between conditions. Each condition has a 20-minute execution limit. Unchanged
model tensors are copied from the same pinned base snapshot; only the two
verified tensors are replaced. No quantization is applied.

Two differences from the original benchmark remain: 16 problems instead of 500,
and the full multimodal release with text-only inputs instead of the authors'
text-only export. The published independent rerun uses that same full model
substitution. The vision tower receives no images and profiling is disabled for
image/video inputs. The numerical language tensors remain unchanged.

Results report correctness, token-limit hits, generated lengths, thinking-tag
closure, repetition of word 4-grams, and paired changes. The 95% bootstrap
interval resamples problems as clusters, keeping the four samples together.
With only 16 problems, the interval will be wide. This pilot cannot reproduce
the published full MATH-500 percentage exactly or establish broad capability
improvements beyond the evaluated set.

Saved generations are regraded locally with extraction and equivalence functions
copied from the published harness (`third_party/qwen_p50c/published_grading.py`).
That grader reads the final text after the last thinking-close tag when present,
and supports the authors' alternate box formats. The initial remote grader was
slightly narrower; original outputs/scores remain saved, and `grader_validation.json`
records any score changes and agreement with the corrected local implementation.
No generations are repeated for this grading check.

## Remote payload and replay

The uploaded payload contains only the runner, the independent edit-verification
helper, public MATH-500 data, and the published diff. No `.env` file, API key,
GitHub credential, or local model cache is uploaded. The VM downloads the pinned
public base weights directly.

Scripts:

- `experiments/qwen_colab_setup.py`: installs the pinned evaluator dependencies.
- `experiments/qwen_colab_runner.py`: reconstructs checkpoints, runs both conditions,
  and writes results and an archive under `/content`.
- `experiments/qwen_colab_launch.py`: unpacks the explicit payload and starts an
  isolated background process.
- `experiments/qwen_colab_status.py`: reads progress and the final status.

After the local `prepare` command above, a complete remote run is:

```sh
python3 experiments/prepare_qwen_colab.py
colab new -s weight-space-qwen-g4 --gpu G4
colab exec -s weight-space-qwen-g4 -f experiments/qwen_colab_setup.py --timeout 900
colab upload -s weight-space-qwen-g4 .artifacts/qwen_payload.tar.gz /content/qwen_payload.tar.gz
colab exec -s weight-space-qwen-g4 -f experiments/qwen_colab_launch.py
colab exec -s weight-space-qwen-g4 -f experiments/qwen_colab_status.py
```

Wait until status is `completed`, then download the archive and release the VM:

```sh
colab download -s weight-space-qwen-g4 /content/qwen_results.tar.gz .artifacts/qwen_results.tar.gz
colab stop -s weight-space-qwen-g4
```

If setup or evaluation fails, download any available archive/logs and stop the
session as well. The `exec` timeout does not replace session teardown.

The completed archive includes the exact problems, full generated texts,
per-sample scores, hashes, settings, software versions, and GPU identity.
Benchmark problem content remains under the dataset's own license. No OpenAI
API calls are needed for this published-edit replay.
