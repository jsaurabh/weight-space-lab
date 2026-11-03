# Published Qwen p50c edit

Unmodified numerical diff and manifest from Experiment 8, run `p50c`, in
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit).

Attribution: Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab,
*Sparks of Weight-Space Programming*, 2026. The diff and manifest are licensed
CC BY 4.0, as specified in `LICENSE.upstream`. `provenance.json` records the
upstream revision, archive checksum, base-model identifier and revision.

The diff changes 130,188 raw bfloat16 entries across two MLP output matrices.
The paper's 131,072 figure counts all touched entries before bfloat16 rounding.
Our replay checks both base and edited tensor SHA-256 hashes against the
published manifest and makes no changes to the numerical diff.

Base weights are downloaded separately from `Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17`, under its Apache 2.0 license.
Base weights and full edited checkpoints are not committed to this repository.
