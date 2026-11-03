"""Prepare peer artifacts only after the last editor has frozen."""
import json
from pathlib import Path
shared = Path('/content/qwen-comparison')
frozen = json.loads((shared / 'astra/results/frozen.json').read_text())
assert not frozen['test_data_present_at_freeze']
(shared / 'gpt55/results').mkdir(parents=True, exist_ok=False)
print('Prepared frozen-peer artifact directory after Astra selection.')
