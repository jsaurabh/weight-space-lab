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

After the completed run is published, fetch its saved tensors and activation
arrays with `python3 experiments/fetch_artifacts.py --group qwen-comparison`.
The shared reconstruction tool accepts either editor's result directory:

```sh
uv run --extra qwen --no-editable python experiments/blind/replay.py --result results/comparison/qwen-astra-vs-gpt55-20261103/gpt55 --base .artifacts/qwen-base --output .artifacts/qwen-gpt55-replay
uv run --extra qwen --no-editable python experiments/blind/replay.py --result results/comparison/qwen-astra-vs-gpt55-20261103/astra --base .artifacts/qwen-base --output .artifacts/qwen-astra-replay
```

Use a local copy of the pinned base revision listed in `protocol.json`.
Reconstruction verifies the frozen patch and replaced tensor hashes, needs CPU
and disk space, and makes no editor calls. Regrading the saved evaluation with
`analyze.py` also needs only CPU. Generating new answers uses the GPU.
