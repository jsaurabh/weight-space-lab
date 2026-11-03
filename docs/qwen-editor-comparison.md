# Astra vs GPT-5.5: Qwen repair pilot

Both editors independently edit the same pinned Qwen3.5-0.8B checkpoint. Qwen
generation and probes run on a Colab **G4**; weight analysis runs on CPU. The
editor API key stays on the local machine. No published repair or peer-editor
outputs are supplied to either conversation.

| Setting | Both editors |
| --- | --- |
| Editor | `gpt-6-astra` or `gpt-5.5-2026-04-23` |
| Reasoning | `xhigh`, the highest shared named setting |
| Feedback budget | 10 successful generation/probe calls |
| Generation | Fixed 32,768-token cap; up to 64 total answers per call |
| Diagnostics | Up to 16 texts, 32,768 tokens/text; selected hidden states, module vectors and token probabilities |
| Analysis budget | 100 CPU calls, 120 API turns, four hours/editor |
| API output cap | 16,000 tokens/call, including reasoning |
| API cost ceiling | None; actual estimated usage is recorded |
| Practice | GSM8K train 7,469; MATH train 7,275; AIME 714, with available solutions |
| Final evaluation | 32 new MATH-500 questions × 2 samples, shared across base and both candidates |

The tool implementation and initial editor inputs are identical. Each editor
starts in an isolated workspace with fresh API context and unedited weights.
The fixed order is GPT-5.5 then Astra, selected using seed 20261103. Failed
forward requests are refunded. An editor cannot finish before using its full
feedback budget; if it reaches the API-turn or wall limit, the host freezes its
current candidate and labels that fallback.

The test is selected and sealed locally before discovery. It excludes all 48
questions used in earlier pilots and remains absent from the VM until **both**
choices are frozen. Neither editor gets test feedback. All three conditions use
the same prompts and sampling: thinking enabled, temperature 0.6, top-p 0.95,
top-k 20, presence penalty 1.5, seed 0 and 32,768 new tokens.

This is one independent discovery attempt per editor. It compares performance
under these constraints, not overall model equivalence. The
[authors evaluate 10, 25 and 50 forward passes](https://rikiyat.github.io/blog/weight-space-programming/).
Our 10-call pilot uses smaller generation/probe batches, a restricted NumPy
analysis interface, and the pinned full model with text inputs. Astra's separate
`max` reasoning setting is not tested.

The initial hardware preflight uses a separate workspace: a temporary generic
tensor change verifies GPU loading and exact base restoration, then a short
probe checks probability outputs. These observations are not given to editors
and consume none of their feedback budget.

Source: [authors' public training data and setup](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math).
The per-run `protocol.json`, payload hashes, tool logs and frozen manifests are
the authoritative record. Model artifacts belong on Hugging Face; source,
reports and API exchanges belong on GitHub.

An [interrupted GPT-5.5-only attempt](../results/comparison/qwen-astra-vs-gpt55-20261103-interrupted/)
is excluded: a validator falsely rejected a sparse column edit. The corrected
interface is tested before both fresh discovery runs.

## Run

Install the repository's `qwen` extra and configure `OPENAI_API_KEY` locally.
The GPU commands require an authenticated Colab CLI and G4 access. This run used
CLI 0.7.4 on macOS; install it with
`uv tool install google-colab-cli==0.7.4` and consult `colab readme` for setup.
The preparation step uses public data and no editor API calls. Use a new result
directory; the cache and Colab session below are intended for one active run.

```sh
uv run --extra qwen --no-editable python experiments/qwen_compare/prepare.py --output results/comparison/new-run
colab new -s weight-space-qwen-compare --gpu G4
colab exec -s weight-space-qwen-compare -f experiments/blind/setup.py --timeout 600
colab exec -s weight-space-qwen-compare -f experiments/qwen_compare/init.py --timeout 60
colab upload -s weight-space-qwen-compare .artifacts/qwen-compare/payload.tar.gz /content/qwen-comparison/payload.tar.gz
uv run --extra qwen --no-editable python experiments/qwen_compare/preflight.py --output results/comparison/new-run
uv run --extra qwen --no-editable python experiments/qwen_compare/driver.py --output results/comparison/new-run
uv run --extra qwen --no-editable python experiments/qwen_compare/analyze.py results/comparison/new-run
```

The driver exports evidence and stops the G4 in its cleanup path. If interrupted
before the driver starts, stop the provisioned session with
`colab stop -s weight-space-qwen-compare`.

## Replay a frozen choice

Fetch the completed run’s saved tensors and activation
arrays with `python3 experiments/fetch_artifacts.py --group qwen-comparison`.
Download the pinned base separately (about 1.8 GB), then reconstruct either
editor’s frozen choice:

```sh
uv run --extra qwen --no-editable hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir .artifacts/qwen-base
uv run --extra qwen --no-editable python experiments/blind/replay.py --result results/comparison/qwen-astra-vs-gpt55-20261103/gpt55 --base .artifacts/qwen-base --output .artifacts/qwen-gpt55-replay
uv run --extra qwen --no-editable python experiments/blind/replay.py --result results/comparison/qwen-astra-vs-gpt55-20261103/astra --base .artifacts/qwen-base --output .artifacts/qwen-astra-replay
```

Reconstruction verifies the frozen patch and replaced tensor hashes, needs CPU
and disk space, and makes no editor calls. Regrading the saved evaluation with
`analyze.py` also needs only CPU. Generating new answers uses the GPU.

## Recorded API interruption

In the November 3 run, Astra's tenth API request received a 400 prompt-filter
rejection after a long raw Qwen math continuation entered its history. It had
used one GPU feedback call and made no edits. GPT-5.5 had already frozen.
The controller exported the available records and stopped the G4.

One clarified continuation was accepted before allocating a replacement G4.
The original rejected request and stdout are retained. The recovered history
omits that single long stdout item and adds an infrastructure notice requesting
compact diagnostics. Astra's own artifacts, base checkpoint and counters are
restored; temporary CPU variables are cleared. The numerical backend, tools,
sampling and ten-call budget stay the same. This is a declared context change
and a limitation of the head-to-head comparison.

The run-specific recovery uses `recover_api.py` once, then `resume.py` after
provisioning and initializing another G4 with the commands above. It is not an
automatic retry mechanism. Recovery sources, before/after histories and hashes
are recorded under the run's `recovery_001` directory. Peer weights and the test
are uploaded to the replacement VM only after Astra freezes.

The first recovery launcher then exposed a missing multiprocessing entry-point
guard. Failed generation requests ran no answers and were refunded. The host
stopped API execution, preserved the exact current checkpoint, corrected the
launcher, and restored cumulative counters from the actual editor tool records
(three successful forward calls). Both completed probes were checked by
reconstructing the edited projection's outputs from its recorded inputs; their
claimed base/current states matched. Native GPU checks verified candidate
loading and exact base restoration before the editor continued. CPU variables
were cleared again. These infrastructure detours also limit interpretations of
Astra's discovery time and API cost.

The user then paused the run during Astra call 35. Colab reported the VM lost,
and that call's result was unavailable. On resumption, all named checkpoints
were reconstructed on Colab CPU from the recorded edit programs and saved
probes. Every reconstructed tensor hash matched its original manifest. The
full records of practice generations 004–006 were lost; their original returned
summaries remain explicitly marked as summary-only artifacts. Call 35 was
reissued without another editor API request, followed by the three remaining
feedback calls. The user pause was excluded from the wall budget. Subsequent
forward feedback, checkpoint backups and final-evaluation progress are saved
locally during execution. See `resumption_002` for the exact recovery record.
