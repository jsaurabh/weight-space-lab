# Tiny transformer reference constructions

These are guided numerical constructions, separate from the independent API
experiments. Both use a fixed dense causal transformer: token and position
embeddings, softmax attention, residual connections, a ReLU MLP and linear
readout. There is one block and no LayerNorm. Only the weights differ by task.

| Task | Domain | Output |
| --- | --- | --- |
| Parity | All 2,046 bit strings of length 1–10 | Even/odd bit |
| Addition | All 10,000 ordered pairs of two-digit numbers | Three-digit sum |

Per-task reports are restored under `results/reference/` by the artifact fetch
command. Independent API constructions: [report](../reports/tiny-discovery.md).

## Mechanisms

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

