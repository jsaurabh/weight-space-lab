# Qwen3.5-0.8B: published edit replay on Colab G4

The published edit improved this fixed MATH-500 pilot from **6.25% to 43.75%**
accuracy and substantially reduced answers that exhausted the token budget.
This is a reconstruction of the authors' existing edit, not a newly discovered
edit or a full 500-problem benchmark replication.

| Metric | Base | Published edit |
| --- | --- | --- |
| Correct generations | 4 / 64 | 28 / 64 |
| Accuracy | 6.25% | 43.75% |
| Hit 32,768-token limit | 60 / 64 (93.75%) | 25 / 64 (39.06%) |
| Mean generated tokens | 30,870.5 | 24,373.1 |

The increase was **37.5 percentage points**. There were 24 paired sample rescues
and no paired regressions. A bootstrap that resamples the 16 problems as
clusters gives a descriptive 95% interval of **+20.3 to +54.7 percentage points**.
Four samples of one problem are kept together in that bootstrap.

## Protocol and limits

- Hardware: requested Colab **G4**, allocated NVIDIA RTX PRO 6000 Blackwell Server
  Edition with 96 GB GPU memory. The session was stopped after downloading the
  complete results; no active Colab sessions remained.
- Data: 16 MATH-500 problems selected before generation with Python seed
  20261102; four samples per problem for each model. The subset contains eight
  Algebra, six Intermediate Algebra, one Precalculus and one Number Theory
  problem. It is small and heavily weighted toward algebra.
- Sampling: thinking enabled, temperature 0.6, top-p 0.95, top-k 20, presence
  penalty 1.5, seed 0, at most 32,768 new tokens. Same prompts and settings in
  both conditions. No quantization or training.
- Software: vLLM 0.33.0, Transformers 5.20.2, Torch 2.14.0, math-verify 0.9.0.
- The language model comes from the pinned full multimodal release and receives
  only text. The original paper used a text-only export. The independent
  published checkpoint rerun made the same full-model substitution.
- Generation took about 6.7 minutes for base and 5.0 minutes for edited. Total
  evaluator wall time, including checkpoint preparation, initialization and
  grading, was about 16 minutes. Colab environment setup was additional.

These results support the reported direction of improvement on this pilot.
They do not reproduce the paper's exact MATH-500 percentage, cover the full
benchmark, establish gains on other tasks, or show that an editor independently
rediscovered the intervention.

## Weight verification and mechanism

Both rebuilt tensors match the authors' raw bfloat16 SHA-256 hashes. Exactly
130,188 stored values differ after rounding, across 64 columns in each of the
MLP output matrices in layers 15 and 19. All other tensors are copied from the
same pinned base model. The published 131,072 count refers to touched values
before rounding.

[Verification](verification.json) records the hashes. Our separate
[numerical analysis](../edit-analysis.json) finds that each change is almost
rank one over those columns and is approximately a sixfold counter-projection
along a layer-specific direction. This describes the stored edit; identifying
the semantic role of a direction requires behavioral evidence as well.

For the first selected problem, `test/algebra/634.json`, all four base samples
hit the limit and failed, while all four edited samples answered correctly.
The short local 4,096-token comparison on that same problem had been
inconclusive, illustrating why the token budget matters.

## Grading audit

The final numbers above use the [published grader excerpts](../../../third_party/qwen_p50c/published_grading.py).
All 128 saved texts were regraded with the authors' extraction rules and
math-verify equivalence. The corrected local implementation agrees on every
case. One preliminary edited score changed from correct to incorrect: it had
the right boxed value inside reasoning, but the final answer after the thinking
close tag was incomplete. Original remote scores and raw texts are preserved.
No model generation was repeated for this audit.

- [Final analysis](analysis.json), [per-sample audited scores](graded_scores.jsonl)
- [Grader agreement and score correction](grader_validation.json)
- [Original remote summary](summary.json), [settings and versions](manifest.json)
- [Base generated texts](base_generations.jsonl), [edited generated texts](published_edit_generations.jsonl)
- [Exact frozen problems](problems.json), [source snapshots](source/), [runtime log](run.log)
- [Teardown and integrity checks](execution_record.json)

Methods and replay commands are in [the experiment guide](../../../docs/qwen-replay.md).
Dataset content remains under the MATH-500 dataset's license; the published
weight diff and grading excerpts retain their upstream attribution and licenses.
