# Independent Qwen weight-repair discovery

This protocol gives a fresh API editor the unedited public Qwen3.5-0.8B model,
generic numerical tools and training-practice feedback. It receives no edit
from the earlier replay, no prior research conversation, and no repair recipe.
The same base-model revision is used throughout.

## Separation of roles

The local driver calls GPT-6 Astra with maximum reasoning effort. Its initial
context contains only the new task, public architecture configuration and tool
schemas. Later context contains its own outputs and tool results. Complete
requests and responses are recorded for auditing. The API key stays local.

A fresh Colab **G4** runs a clean numerical service. The uploaded payload is
limited to `backend.py`, `protocol.py`, and `practice.json`. It contains no
earlier edit, tensor diff, research report, benchmark output or credential. The
VM downloads only the pinned unedited base from Hugging Face. The editor has
no shell, internet, repository or unrestricted file-reading tool.

The orchestrator has seen the earlier replay; it implements the generic tools
and monitors transport, not the repair choices. It supplies no targeted layer,
neuron, direction, coefficient or candidate weight patch to the editor. One
preflight CPU call printed the tensor count and hidden width, without changing
weights or running behavior. That call remains in the audit log and consumes
one of the 100 CPU-call slots.

Context isolation does not establish absence of relevant knowledge in the
programmer's pretraining. The auditable claim is that this run supplies no
published repair or earlier conversation to the editor.

## Practice and sealed evaluation

The practice pool contains 256 GSM8K **training** questions, chosen with seed
20261102 from the official `openai/grade-school-math` training file at commit
`3101c7d5072418e28b9008a6636bde82a006892c`. Questions and numeric answers are
available to the editor. It can choose practice IDs for feedback and can inspect
its own generated texts and captured activations.

The final sample is sealed locally before any editor call: 32 MATH-500 problems
stratified across all seven subjects with seed 20261103, excluding the 16
problems used in the earlier replay. The seal records the exact file hash. The
test file is physically absent from the Colab service throughout discovery.

Only after candidate weights are frozen does the local driver upload the test
file. The editor receives no subsequent API calls or test scores. Final base
and candidate evaluations use the same 32 problems, two samples each, thinking
enabled, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, seed 0 and
a 32,768-token cap. This is a new pilot protocol, not the full published benchmark.

## Tools and intervention rules

- `practice`: read public training questions and answers.
- `python_analysis`: persistent CPU NumPy analysis over the base weights and
  artifacts. Array copies expose no neural-network forward or autograd object.
  Helpers allow tensor replacement and checkpoint save/restore.
- `generate`: run the base, current candidate, or a paired comparison on practice
  IDs, with up to 32 total generations per call.
- `probe`: teacher-force up to eight texts and inspect layer hidden states and
  selected module inputs/outputs at requested token positions.
- `finish`: select a checkpoint and permanently freeze it for testing.

No training, backpropagation, gradient estimation, optimizers, supervised fitting
or changes to the forward program/sampling are allowed. Numerical statistics and
closed-form algebraic changes are allowed. The analysis interface restricts
imports and rejects external access and common training operations; generated
code remains available for a separate scientific audit.

Only internal text-layer tensors can change. Embeddings, the output head and
the final normalization are fixed. Pure whole-tensor scalar rescaling is
rejected. No layer-specific, direction-specific or sparsity recipe is built
into the tool. Candidate storage is limited to 64 internal tensors and 160
million stored elements. Values must retain shape and dtype, remain finite,
and have absolute value at most one million.

The generator uses vLLM 0.33.0. Arbitrary candidate tensors are applied through
its native model weight loader, and prior changes are restored from the pinned
base when switching conditions. Prefix caching is disabled. A separate frozen
Transformers model provides teacher-forced probes under inference mode.

## Budgets and audit trail

The initial run permits 20 forward-tool calls, 100 CPU analysis calls including
preflight, and 60 API turns, with a conservative **$25 estimated API ceiling**.
No automatic API retries are enabled. Usage accounting includes cached input,
cache writes and reasoning/output tokens, including long-context pricing when
applicable. The local estimate is not an account billing limit or an invoice.

The service has a four-hour lifetime; discovery has a three-hour wall limit.
The driver exports available evidence and stops the G4 in its cleanup path,
including on errors. If a budget prevents an editor-submitted final choice,
the current candidate is frozen and that fallback is labeled explicitly.

Logs include tool code, results, generation texts, activation artifacts,
candidate manifests, final selected tensors, source snapshots and checksums.
Large intermediate checkpoint copies are excluded from the export; their
manifests and the generating code are retained. The final test is not used to
choose among edits. Any improvement must be described as a measured result,
not assumed from the fact that an edit was produced.
