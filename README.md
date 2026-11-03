# Weight Space Lab

On-device CPU experiments in assigning tiny transformer weights directly, without
gradient descent, pretrained target weights, or an external answer function.

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
and [Weight-Space Programming from scratch on simple tasks](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html).
This repository is an independent implementation of the idea, not the authors' code.

## Current experiment

Two guided reference constructions establish a working, inspectable local testbed.
These results do **not** measure whether a separate LLM can independently discover
the weights. No OpenAI API integration or credential is required for this phase.

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

Inspect individual computations:

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

## Next experiments

1. Freeze the architecture, task spec, and a restricted development evaluator.
2. Give an API programmer only the spec and tensor schema, excluding reference
   constructions, final test cases, and this repository's solutions from its context.
3. Accept numerical tensors through a constrained format rather than letting the
   programmer replace the forward function or evaluation code.
4. Record prompts, every attempt, development feedback, evaluation budget, API
   usage, and tensor digests. Freeze the submitted weights before final evaluation.
5. Evaluate independently; distinguish guided, one-shot, and feedback-assisted runs.
6. Repeat across independent attempts and compare random and guided baselines.
7. Separately investigate targeted edits to a small pretrained language model.

Known parity solutions are public, so even a context-isolated API run will not
establish novelty or absence of pretraining exposure. API access and spending
limits will be configured separately before that phase. No credentials belong
in Git; `.env` files and weight checkpoints are ignored.
