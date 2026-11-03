# Astra-generated tiny transformers

Astra independently supplied numerical weights for two fixed transformer
architectures. Each first proposal solved its complete finite domain on CPU.
The editor saw task specifications, shapes and the forward implementation;
it did not receive reference constructors or weights.

| Task | Development | Unseen holdout | Entire domain |
| --- | ---: | ---: | ---: |
| Parity | 200/200 | 1,846/1,846 | 2,046/2,046 |
| Two-digit addition | 200/200 | 9,798/9,798 | 10,000/10,000 |

Parity uses uniform attention to count ones and a ReLU zigzag to select parity.
Addition combines operand digits, subtracts previously emitted digits, and
reads out the remaining amount. The addition construction uses 106 nonzero
weights, compared with 665 in the guided reference.

The editor used `high` reasoning; target execution used float64. Estimated
API usage totaled **$0.325747**. Neither proposal needed revision. The unseen
split excludes development inputs and the two illustrative addition prompts.

These are existence demonstrations, not success-rate estimates. Architectures
were chosen in advance, and the tasks have known algorithmic solutions.
No general-purpose language model weights were edited.

## Reproduce

```sh
python3 experiments/fetch_artifacts.py --group tiny
uv run --extra qwen --no-editable weight-space-discover replay results/discovery/astra-initial/parity/selected.json --task parity
uv run --extra qwen --no-editable weight-space-discover replay results/discovery/astra-initial/addition/selected.json --task addition
```

Replay runs on CPU without API calls. [Architecture and guided constructions](../docs/tiny-transformers.md).
Proposals, original API exchanges and detailed scores are archived on
[Hugging Face](https://huggingface.co/jsaurabh/weight-space-lab-artifacts/tree/main/records)
and restored into ignored `results/` by the fetch command.
