from pathlib import Path
import json
from huggingface_hub import HfApi, CommitOperationAdd
root=Path('results/comparison/qwen-astra-vs-gpt55-20261103');m=json.loads(Path('model_artifacts.json').read_text())
assert json.loads((root/'run.json').read_text())['g4_stopped']
a=json.loads((root/'analysis.json').read_text());c=a['conditions'];prefix=str(root)
d=Path('.artifacts/qwen-compare/hf-metadata');card=(d/'README.md').read_text()
old="- `results/comparison/qwen-astra-vs-gpt55-20261103/gpt55/selected.safetensors`: GPT-5.5's independently selected patch from the matched Qwen pilot, frozen before held-out evaluation. Contains one edited matrix; source and practice records are on [GitHub](https://github.com/jsaurabh/weight-space-lab/tree/main/results/comparison/qwen-astra-vs-gpt55-20261103/gpt55). Associated `artifacts/*.npz` files contain its two activation captures."
new=f"""- `{prefix}/{{gpt55,astra}}/selected.safetensors`: independent frozen patches from the matched ten-feedback-call Qwen pilot. GPT-5.5 changes one matrix; Astra changes 24. On 32 unseen MATH-500 questions with two samples each, base scored **{c['base']['correct']}/64**, Astra **{c['astra']['correct']}/64**, and GPT-5.5 **{c['gpt55']['correct']}/64**. This is one attempt per editor; Astra required declared infrastructure recoveries. See the [full report and limits](https://github.com/jsaurabh/weight-space-lab/tree/main/{prefix}). Associated `artifacts/*.npz` files contain two activation captures per editor.
- `{prefix}/recovery_001/launcher_repair/current_checkpoint.safetensors`: an intermediate Astra checkpoint retained as recovery evidence. This is **not** either final selected repair."""
assert old in card;card=card.replace(old,new);(d/'README.md').write_text(card)
inventory={**m,'revision':None,'source_github_repository':'https://github.com/jsaurabh/weight-space-lab','comparison_report':f'https://github.com/jsaurabh/weight-space-lab/tree/main/{prefix}','note':'GitHub model_artifacts.json pins the immutable Hugging Face revision. Inventory hashes cover all 21 binary assets.'}
(d/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
attr=d/prefix/'WEIGHT_ATTRIBUTION.md';attr.write_text('''# Qwen comparison weight attribution

The saved model tensors derive from Qwen/Qwen3.5-0.8B, revision
2fc06364715b967f1860aea9cf38778875588b17, by the Qwen team.
The original Apache 2.0 license is preserved in QWEN_MODEL_LICENSE.

GPT-5.5 and GPT-6 Astra independently generated their selected patches with
the recorded numerical interface, each using ten successful GPU feedback
calls. The GPT-5.5 patch changes one MLP output matrix; Astra's changes 24
MLP and attention output matrices. Both were frozen on 2026-11-03 before
held-out evaluation. No gradients or optimizer were used.

The launcher-repair checkpoint is an intermediate Astra candidate, preserved
for recovery verification rather than a final selection. The full base model
is not duplicated here. Probe arrays are saved inference diagnostics; their
JSON descriptors, tool/API records, source and limitations are on GitHub:
https://github.com/jsaurabh/weight-space-lab/tree/main/results/comparison/qwen-astra-vs-gpt55-20261103
''')
ops=[CommitOperationAdd(path_in_repo=p,path_or_fileobj=str(d/p)) for p in ['README.md','artifact_inventory.json',prefix+'/WEIGHT_ATTRIBUTION.md']]
for relative in ['astra/frozen.json','gpt55/frozen.json','analysis.json','protocol.json','run.json']:
 ops.append(CommitOperationAdd(path_in_repo=prefix+'/'+relative,path_or_fileobj=str(root/relative)))
api=HfApi();assert not api.model_info(m['repo_id']).private
commit=api.create_commit(repo_id=m['repo_id'],operations=ops,commit_message='Document completed Astra versus GPT-5.5 Qwen comparison',parent_commit=m['revision'])
m['revision']=commit.oid;Path('model_artifacts.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print('Public artifact metadata revision:',commit.oid)
