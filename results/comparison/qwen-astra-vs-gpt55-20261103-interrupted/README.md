# Interrupted Qwen editor trial

**Excluded from the Astra vs GPT-5.5 comparison because of an infrastructure defect.**

The edit validator checked a strided sample when rejecting whole-matrix scalar
gains. That sample missed GPT-5.5's single edited column and wrongly classified
its legitimate local intervention as a forbidden scalar gain. GPT-5.5 then
broadened the edit to work around the rejection. The host stopped the trial
rather than continue a comparison affected by that defect.

| Recorded usage | Value |
| --- | --- |
| Editor API calls | 26, GPT-5.5 only |
| Successful GPU feedback calls | 6 |
| CPU analysis calls | 16 |
| Estimated editor API cost | $1.584578, excluding Colab |
| Astra discovery | Not started |
| Final candidate / held-out evaluation | None |

The pending sixth feedback call finished while evidence was being exported;
its result was not sent back to the editor. The GPU was then stopped. The test
remained sealed locally and was never uploaded. This is an interrupted tooling
trial, not a completed editor result.

The corrected validator compares the full array and allows sparse changes.
A regression test reproduces the missed-column case; the replacement hardware
preflight also verifies sparse-entry and column edits. Both editors restart
with fresh contexts and the corrected interface in the
[replacement comparison](../qwen-astra-vs-gpt55-20261103/).

Original inputs, source snapshots, API requests/responses and practice records
are retained here. Binary activation arrays are restored from Hugging Face with
`python3 experiments/fetch_artifacts.py --group qwen-comparison`.
