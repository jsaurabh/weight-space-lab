# Weight Space Lab

On-device CPU experiments in assigning tiny transformer weights directly, without
gradient descent, pretrained target weights, or an external answer function.

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
and [Weight-Space Programming from scratch on simple tasks](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html).
This repository is an independent implementation of the idea, not the authors' code.

## Current experiment

Two guided reference constructions establish a working, inspectable local testbed.
These results do **not** measure whether a separate LLM can independently discover
the weights. No OpenAI API credential is required to run or replay local models.
A separate API discovery runner now tests fresh numerical constructions using a
programmer model, without providing the reference implementations or weights.

| Task | Input | Output | Final evaluation |
| --- | --- | --- | --- |
| Parity | 1–10 bits, padded to 10 positions, then `=` | `0` for even ones, `1` for odd | All 2,046 nonempty bit strings |
| Addition | Two zero-padded two-digit numbers, e.g. `89+76=` | Exactly three digits, e.g. `165` | All 10,000 ordered pairs |

First local run (CPU, macOS arm64, float64):

| Task | Correct | Stored values | Evaluation time |
| --- | --- | --- | --- |
| [Parity](results/reference/parity/report.json) | 2,046 / 2,046 | 170 | 0.009 s |
| [Addition](results/reference/addition/report.json) | 10,000 / 10,000 | 10,429 | 0.193 s |

All nine automated checks passed. These are guided reference results, not API
programmer discovery results. Times exclude interpreter startup and model construction.

Both use an unchanged dense causal transformer forward implementation: token and
position embeddings, multihead softmax attention, residual connections, ReLU MLP,
and linear token readout. There is one block and no LayerNorm. This is deliberately
a simplified transformer, not a stock GPT model. The architecture is fixed per task;
only the weight construction differs. Arithmetic in the evaluator is used only to
score answers; the network has no access to it.

## Run locally

Install [uv](https://docs.astral.sh/uv/) if needed, then:

```sh
uv sync --locked
uv run weight-space-lab run --task all
uv run python -m unittest discover -s tests -v
```

Python 3.12 is selected by `.python-version`. The model uses CPU float64 and one
compute thread; no GPU, Colab session, API key, or network connection is required
after dependencies are installed. The lockfile records dependency versions.

Inspect individual guided reference computations:

```sh
uv run weight-space-lab inspect parity 10101
uv run weight-space-lab inspect addition '89+76='
```

Each run writes a JSON report and a loadable `weights.pt` under
`results/reference/<task>/`. Reports include accuracy, group counts, failures,
timing, tensor shapes, all parameter and buffer elements, nonzero values, and a
SHA-256 digest of the named tensor state. Checkpoints are ignored by Git and can
be recreated locally. Timing depends on hardware and package versions.

## How the reference weights work

### Parity

Three residual channels hold the bit value, the count of ones, and parity.
Zero query and key weights produce uniform causal attention. At the final `=`
position all 11 positions are visible. A value projection multiplies bit values
by 11, so their average is the count of ones. Padding and `=` contribute zero.

Ten ReLU units implement a zigzag on integer counts from zero to ten:

```text
p(n) = ReLU(n) + 2 * sum((-1)^k * ReLU(n-k), k=1..9)
```

The readout chooses token `0` or `1`. This construction is inspired by the report,
but its architecture and parameter count are not an exact reproduction of the
reported 174-parameter model.

### Addition

Four attention heads route the two tens and two ones digits into dedicated
residual channels. Queries and keys depend on constant and positional channels;
softmax routing is sharp but finite. The MLP forms `n = 10a + b + 10c + d`.

For integer inputs, a difference of shifted ReLUs gives a threshold step:
`H(n-k) = ReLU(n-k+1) - ReLU(n-k)`. Summing steps at multiples of ten gives
`floor(n/10)` on the bounded domain. One step at 100 gives `floor(n/100)`.
Position-gated hinges select the hundreds, tens, and ones calculation on
successive autoregressive output steps. A linear readout selects the nearest
digit via `argmax_d(2*d*x - d*d)`.

This is a bounded arithmetic construction, not a table of all 10,000 answers.
It assumes the documented input length, number range, and output format.

## What the checks establish

- Exhaustive exact-match correctness over both specified finite domains.
- Intermediate channels match the stated count, routed digits, and output digits.
- Removing the parity counting path reduces accuracy to 50%; removing an addition
  digit route breaks a known example.
- Causal attention prevents future tokens affecting earlier outputs.
- Saved weights reconstruct the same tensor state, including full storage accounting.

Exhaustive finite-domain testing is not evidence of generalization to longer
strings or larger numbers. The programs intentionally encode their domain bounds.
The reference designers know both algorithms and evaluation domains; this is not
a blind evaluation of an autonomous programmer. The tests are correctness checks,
not a statistical estimate of a model's discovery ability.

## API-driven discovery

The [first live run](results/discovery/astra-initial/README.md) succeeded on the
first proposal for both tasks, with no feedback revisions:

| Task | Full domain | Unseen remainder | Stored / nonzero values |
| --- | --- | --- | --- |
| Parity | 2,046 / 2,046 | 1,846 / 1,846 | 170 / 37 |
| Addition | 10,000 / 10,000 | 9,800 / 9,800 | 10,429 / 106 |

Total estimated API cost: **$0.325747**, including cache writes. These are actual API-produced numerical
proposals, saved without hand edits. They are separate from the guided results
above. Counts and explanations are limited to the defined bounded tasks.

The programmer runs through the OpenAI API; target execution and evaluation stay
on the local CPU. The configured API key is read from `OPENAI_API_KEY` or this
project's ignored `.env.local`; it is never included in experiment logs.

```sh
uv run weight-space-discover run --task all --max-attempts 3 --max-cost-usd 5
```

The initial model is GPT-6 Astra with high reasoning effort. Each task starts a
fresh conversation with its spec, vocabulary, fixed tensor shapes, the exact
forward implementation, and a generic assignment format. Neither reference
construction, reference weights, repository access, nor tools are supplied.
The architecture was chosen using the guided references, so architecture
selection itself is not an independent discovery result.

The model returns structured JSON containing an explanation and numerical
assignments. A patch names a tensor, selects a Cartesian product of indices, and
supplies constants. No generated source code or expressions are executed.
The loader enforces index bounds, shape, finite values, and size limits. Every
attempt starts from zero; the architecture and evaluator cannot be replaced.

For each task, 200 development examples are sampled without replacement using a
fixed recorded seed. Up to five development failures are sent back per attempt.
Selection uses only development accuracy, preferring the earliest tied candidate,
and stops at 100% development accuracy or the attempt/cost limit. The selected
proposal and state digest are frozen before final evaluation. Final results
include both the full finite domain and the disjoint unseen remainder: 1,846
parity inputs or 9,800 addition inputs. Final feedback is never sent to the
programmer. Zero and one seeded random model provide holdout baselines.

Each run saves requests, raw response objects, proposals, development reports,
selection metadata, final reports, baseline reports, token usage, and estimated
cost. Response IDs identify remote responses, but authorization headers and
credentials are never logged. `store=False` is requested and API retries are
disabled. No optional paid tools are enabled. New runs require a fresh directory;
rerunning a command creates a new paid experiment rather than resuming a run.

The default ceiling is **$5 estimated standard API usage**, with at most three
attempts per task and 16,000 output tokens per attempt, including reasoning.
Before each call the runner reserves a conservative token-based estimate using
the published input/output rates plus a cache-write allowance. It stops before
making a request that would exceed the estimated ceiling. This is a local
estimate, not an account billing limit or an invoice; prices can change. Rates
and their official source/date are recorded in each run manifest.

Replay a saved proposal with no API calls:

```sh
uv run weight-space-discover replay results/discovery/astra-initial/parity/selected.json --task parity
```

Known parity solutions are public, so even a context-isolated API run does not
establish novelty or absence of pretraining exposure. A single run also does not
establish reliable discovery rates across attempts or models. Next experiments
can repeat independent runs, vary model size, test additional tasks, and separately
investigate edits to a small pretrained language model. No credentials belong in
Git; `.env` files and weight checkpoints are ignored.
