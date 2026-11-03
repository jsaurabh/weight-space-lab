# Weight Space Lab

Experiments in writing and editing transformer weights directly, without training.

[Model artifacts](https://huggingface.co/jsaurabh/weight-space-lab-artifacts) ·
[Original research](https://rikiyat.github.io/blog/weight-space-programming/) ·
[Related toy experiments](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html)

## Results

| Experiment | Result | Details |
| --- | --- | --- |
| Guided tiny transformers | Parity: 2,046/2,046; addition: 10,000/10,000 | [Methods](docs/tiny-transformers.md) |
| Astra-generated tiny transformers | Both tasks solved on the first proposal | [Report](results/discovery/astra-initial/README.md) |
| Local Qwen feasibility tests | Both 64- and 4,096-token runs reached the cap | [Notes](docs/qwen-replay.md#local-feasibility) |
| Published Qwen edit replay | Base **4/64 → 28/64** | [Report](results/qwen/g4-pilot/README.md) |
| Blind Astra Qwen repair search | Base **5/64 → 3/64**; no improvement | [Report](results/qwen/blind-discovery/README.md) |

Tiny-task scores cover the complete stated domains. Qwen scores are small pilots
on different question sets, not a direct comparison or full benchmark replication.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python and dependencies are pinned.

```sh
git clone https://github.com/jsaurabh/weight-space-lab.git
cd weight-space-lab
uv sync --locked --extra qwen --no-editable
uv run --extra qwen --no-editable python experiments/fetch_artifacts.py
uv run --extra qwen --no-editable python -m unittest discover -s tests -v
```

The artifact download is about 43 MB and verifies the pinned hashes in
[model_artifacts.json](model_artifacts.json). No GPU or API key is needed for
setup, tests or saved-result replay.

## Reproduce the tiny models

Run the guided constructions or replay the saved Astra proposals on CPU:

```sh
uv run --extra qwen --no-editable weight-space-lab run --task all --output results/reproduced/reference
uv run --extra qwen --no-editable weight-space-discover replay results/discovery/astra-initial/parity/selected.json --task parity
uv run --extra qwen --no-editable weight-space-discover replay results/discovery/astra-initial/addition/selected.json --task addition
```

## Reproduce the Qwen results

Regrade the saved generations on CPU:

```sh
uv run --extra qwen --no-editable python experiments/analyze_qwen_results.py results/qwen/g4-pilot
uv run --extra qwen --no-editable python experiments/blind/analyze.py results/qwen/blind-discovery
```

For fresh model runs and checkpoint reconstruction:

- [Published-edit replay](docs/qwen-replay.md): pinned base model, exact edit verification, local and G4 commands.
- [Blind-search protocol](docs/blind-qwen-discovery.md): editor tools, budgets and test isolation.
- [Astra vs GPT-5.5 on Qwen](docs/qwen-editor-comparison.md): matched editor budgets and G4 reproduction commands.
- [Frozen blind candidate](results/qwen/blind-discovery/README.md#replay-without-further-api-calls): reconstruct the selected checkpoint without calling the editor.

The base Qwen model is downloaded separately. Generating new Qwen answers is
practical on a GPU; regrading saved answers requires only CPU.

## Run a new discovery experiment

New editor runs require `OPENAI_API_KEY` in the environment or ignored
`.env.local` and incur API usage. Use a new output directory:

```sh
uv run --extra qwen --no-editable weight-space-discover run --task all --max-attempts 3 --max-cost-usd 5 --output results/discovery/new-run
```

The cost limit is an estimated API ceiling. Prompts, responses, selections and
scores are saved with each run. Checkpoints and activation arrays are hosted on
Hugging Face; code, reports and logs are kept here.
