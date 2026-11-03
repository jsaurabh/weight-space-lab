# Published Qwen edit replay

Replaying the authors' existing edit improved this small Qwen3.5-0.8B pilot.
This tested a known repair, rather than independent discovery by an editor.

| Fixed pilot | Base | Published edit |
| --- | ---: | ---: |
| Correct answers | 4/64 | 28/64 |
| Hit the 32,768-token limit | 60/64 | 25/64 |
| Mean generated tokens | 30,870.5 | 24,373.1 |

Sixteen MATH-500 questions, four samples each, identical decoding per condition.
There were 24 paired rescues and no losses. The question-cluster bootstrap 95%
interval for the accuracy gain was +20.3 to +54.7 percentage points.

Both reconstructed tensors matched the authors' BF16 hashes. The edit changes
130,188 stored entries across selected columns in two MLP output matrices.
Regrading all 128 outputs with the published grader corrected one preliminary
score. The table shows the audited results.

This is a small, algebra-heavy subset, not a replication of the full benchmark.
Qwen ran on Colab G4; the session was stopped after export.

## Reproduce

```sh
python3 experiments/fetch_artifacts.py --group qwen-published
uv run --extra qwen --no-editable python experiments/analyze_qwen_results.py results/qwen/g4-pilot
```

[Exact edit verification and fresh GPU runs](../docs/qwen-replay.md).
Generated answers, grading records and source snapshots are archived on
[Hugging Face](https://huggingface.co/jsaurabh/weight-space-lab-artifacts/tree/main/records)
and restored into ignored `results/` by the fetch command.
