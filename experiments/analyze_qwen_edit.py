"""Analyze the published weight difference; do not change or train a model."""
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from weight_space_lab.qwen import read_edit, rebuild_tensor, save_json

torch.set_num_threads(4)
base = Path(".artifacts/qwen-base")
manifest, archive = read_edit(Path("third_party/qwen_p50c"))
mapping = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
results, directions = [], []
for index, spec in enumerate(manifest["tensors"]):
    with safe_open(base / mapping[spec["name"]], framework="pt", device="cpu") as source:
        before = source.get_tensor(spec["name"])
    after = rebuild_tensor(before, spec, archive, index)
    changed = (before.view(torch.uint16) != after.view(torch.uint16))
    columns = torch.where(changed.any(dim=0))[0]
    before_selected = before.float()[:, columns].double().numpy()
    after_selected = after.float()[:, columns].double().numpy()
    delta = after_selected - before_selected
    u, singular, _ = np.linalg.svd(delta, full_matrices=False)
    direction = u[:, 0]
    projected_before = direction @ before_selected
    projected_after = direction @ after_selected
    removed = direction @ delta
    scale = -float(np.dot(removed, projected_before) / np.dot(projected_before, projected_before))
    directions.append(direction)
    results.append({"name": spec["name"], "changed_columns": columns.tolist(),
                    "changed_column_count": int(columns.numel()),
                    "top_singular_values": singular[:5].tolist(),
                    "leading_singular_energy_fraction": float(singular[0] ** 2 / np.sum(singular ** 2)),
                    "estimated_direction_removal_scale": scale,
                    "after_to_before_projection_norm_ratio": float(np.linalg.norm(projected_after) / np.linalg.norm(projected_before))})
archive.close()
result = {"tensors": results, "absolute_direction_cosine_between_layers": float(abs(directions[0] @ directions[1])),
          "method": "SVD of the published numerical diff on its 64 changed columns; analysis only, no model fitting or training"}
destination = Path("results/qwen/edit-analysis.json")
destination.parent.mkdir(parents=True, exist_ok=True)
save_json(destination, result)
print(json.dumps(result, indent=2))
