# Independent Astra Qwen repair search

This first blind trial did not find a useful repair. Astra received generic
weight/activation tools, training examples and the base Qwen3.5-0.8B checkpoint,
without the published edit or final test questions.

| Unseen test | Base | Selected repair |
| --- | ---: | ---: |
| Correct answers | 5/64 | 3/64 |
| Finished correctly | 3/64 | 3/64 |
| Hit the 32,768-token limit | 61/64 | 61/64 |

The test used 32 MATH-500 questions with two samples each. The selected repair
reflected a thinking-termination direction in two final-layer output matrices.
Exactly two of 488 tensors changed; all others matched the pinned base.

Practice feedback was narrow: small batches capped at 2,048 tokens, all
unfinished and incorrect. The final choice therefore rested on activation
hypotheses rather than demonstrated practice gains. A native weight-reload
repair was also required during discovery. These are material limitations.

Estimated editor API usage was **$7.128361**, excluding Colab. The G4 was stopped.
This pilot used different questions from the other Qwen experiments.

## Reproduce

```sh
python3 experiments/fetch_artifacts.py --group qwen-blind
uv run --extra qwen --no-editable python experiments/blind/analyze.py results/qwen/blind-discovery
```

[Protocol and checkpoint replay](../docs/blind-qwen-discovery.md).
Raw outputs, API exchanges, source snapshots and audit records are archived on
[Hugging Face](https://huggingface.co/jsaurabh/weight-space-lab-artifacts/tree/main/records)
and restored into ignored `results/` by the fetch command.
