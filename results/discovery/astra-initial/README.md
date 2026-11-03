# First API discovery run

GPT-6 Astra received each task specification, fixed architecture, tensor shapes,
exact forward implementation, and a generic numerical assignment schema in a
fresh conversation. No reference weights, reference constructors, previous task
conversation, repository tools, or source-reading tools were supplied. We made
no edits to either generated proposal.

| Task | Attempts | Development | Unseen holdout | Entire domain | Stored / nonzero values |
| --- | --- | --- | --- | --- | --- |
| Parity | 1 | 200 / 200 | 1,846 / 1,846 | 2,046 / 2,046 | 170 / 37 |
| Addition | 1 | 200 / 200 | 9,800 / 9,800 | 10,000 / 10,000 | 10,429 / 106 |

Both candidates were selected and frozen before final evaluation. Because the
first proposal passed every development example, neither run received feedback
or a revision request. The API programmer's reasoning effort was `high`; all
target execution used the local CPU and float64. Total estimated standard API
cost was **$0.325747**, including cache writes, calculated from recorded token usage rather than a billing
invoice. The run's declared estimated ceiling was $5.

## Parity circuit

The generated [proposal](parity/selected.json) independently assigns a counting
circuit. Token `1` activates residual channel 0. Uniform attention and a value
scale of 11 place the number of ones in residual channel 1 at the answer position.
Ten shifted ReLUs form a zigzag, written into residual channel 2 with values -1
for even counts and +1 for odd counts. Opposite readout weights select the bit.

The mechanism resembles the known reference strategy, with different readout
scaling and constants. This is a numerical construction discovered without the
reference in its context, not evidence of a previously unknown parity algorithm.

## Addition circuit

The generated [proposal](addition/selected.json) uses a different construction
from the guided reference and leaves all but three of 81 MLP units inactive.

- Head 0 averages the two operand tens digits; head 1 averages their ones digits.
- Output projection factors 20 and 2 recover the total sum from those averages.
- Heads 2 and 3 retrieve previously generated hundreds and tens digits, when
  present, and subtract 100 and 10 times their values from the total. Before those
  outputs exist, the heads retrieve `+`, whose value is zero.
- Residual channel 8 therefore holds the remaining amount at each output step.
- Three position-gated ReLUs rescale this remainder for hundreds, tens, or ones.
- Position-dependent linear logits select the correct digit.

For `89+76=`, the remainder is 165 before the hundreds digit, 65 before the tens
digit, and 5 before the ones digit. Generated tokens are `1`, `6`, and `5`.
This is genuine autoregressive use of the network's own earlier output tokens.
It avoids the guided reference's many decimal threshold units: 106 nonzero
weights rather than 665, within the same 10,429-value architecture.

Queries and keys use large scales; attention score gaps exceed 1,400. Unwanted
softmax terms underflow to zero in the tested float64 implementation. Correctness
is established for this numerical setup and the defined domain, not arbitrary
precision arithmetic, other runtimes, longer inputs, or larger numbers.

## Evidence and replay

- [Run manifest and estimated cost](manifest.json)
- [Parity selection and final score](parity/selection.json), [full report](parity/final_all.json)
- [Addition selection and final score](addition/selection.json), [full report](addition/final_all.json)
- Each task folder contains the exact API request, raw response, parsed proposal,
  development results, unseen results, and baseline results.
- All 20 local checks pass, including replay of both saved API proposals,
  intermediate-channel checks, and an intervention that disables addition's
  output-feedback path and changes `89+76=` from `165` to `199`.
- Numerical proposal replay does not call the API:

```sh
uv run weight-space-discover replay results/discovery/astra-initial/parity/selected.json --task parity
uv run weight-space-discover replay results/discovery/astra-initial/addition/selected.json --task addition
```

The architecture and task bounds were selected using guided constructions. The
API did not discover those architectures. Known algorithmic solutions may occur
in the programmer's pretraining data. One success per task is an existence result
and does not estimate repeatability or success rates. No general-purpose language
model weights were edited in this experiment.
