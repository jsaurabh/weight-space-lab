# Astra vs GPT-5.5: Qwen weight repair

Both editors independently repaired the same Qwen3.5-0.8B checkpoint, with
identical starting prompts/tools and ten successful GPU feedback calls each.
Neither received the published edit, peer work or test feedback.

| Shared unseen test | Base | Astra repair | GPT-5.5 repair |
| --- | ---: | ---: | ---: |
| Correct answers | 7/64 | 11/64 | 13/64 |
| Finished correctly before the token limit | 7/64 | 8/64 | 13/64 |
| Hit the token limit | 57/64 | 56/64 | 51/64 |

The test used 32 new MATH-500 questions, two samples each, with a 32,768-token
cap. All 192 outputs were independently regraded with no score changes.

GPT-5.5 found a useful repair in this attempt. Its two-answer lead over Astra
is too small to establish a general ranking: the question-cluster bootstrap
95% interval for the difference is −4.69 to +12.50 percentage points.

## Selected repairs

- **GPT-5.5:** rank-one change to the final MLP output matrix, using loop/control
  activations and token embeddings; one tensor, 2,037,563 changed BF16 entries.
- **Astra:** remove an activation-derived direction from MLP and attention output
  matrices in layers 7–18; 24 tensors, 56,568,423 changed BF16 entries.

Both full reconstructions verified all 488 tensors against the pinned base.
No gradients or optimizer were used. Both editors used `xhigh` reasoning;
Astra's separate `max` setting was not tested.

## Limits and cost

One discovery attempt per editor, smaller batches and a more restricted interface
than the paper. Astra required context recovery, a launcher correction and VM-loss
reconstruction. All reconstructed tensor hashes matched; three early practice
batches retain summaries rather than full generated texts. Final test records
are complete. These interruptions limit comparisons of trajectory, time and cost.

A separate GPT-5.5 tooling trial was excluded after a validator incorrectly
rejected a sparse edit. Both editors restarted with the corrected interface.

Comparison API usage: **$7.545884**; excluded trial: **$1.584578**. Colab charges
are additional. Qwen ran on Colab G4; the session was stopped after export.

## Reproduce

```sh
python3 experiments/fetch_artifacts.py --group qwen-comparison
uv run --extra qwen --no-editable python experiments/qwen_compare/analyze.py results/comparison/qwen-astra-vs-gpt55-20261103
```

Regrading needs CPU and no API key. [Protocol and fresh-run instructions](../docs/qwen-editor-comparison.md).
Checkpoints, raw conversations, outputs and recovery records are on
[Hugging Face](https://huggingface.co/jsaurabh/weight-space-lab-artifacts/tree/main/records).
The fetch command restores them under ignored `results/`; hashes and revisions
are pinned in [model_artifacts.json](../model_artifacts.json).
