"""Parse numerical assignments; never execute model-generated source code."""

import itertools
import math

import torch

from .model import TinyTransformer
from .specs import CONFIGS

TENSOR_NAMES = [
    "token.weight", "position.weight", "query.weight", "key.weight", "value.weight",
    "attention_out.weight", "mlp_in.weight", "mlp_in.bias", "mlp_out.weight",
    "mlp_out.bias", "readout.weight", "readout.bias",
]

PROPOSAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "explanation": {"type": "string"},
        "patches": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "tensor": {"type": "string", "enum": TENSOR_NAMES},
                    "axes": {"type": "array", "items": {
                        "type": "array", "items": {"type": "integer"}}},
                    "values": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["tensor", "axes", "values"],
            },
        },
    },
    "required": ["name", "explanation", "patches"],
}


@torch.no_grad()
def load_proposal(task, proposal):
    if not isinstance(proposal, dict) or set(proposal) != {"name", "explanation", "patches"}:
        raise ValueError("Proposal must contain name, explanation, and patches only")
    if not all(isinstance(proposal[k], str) for k in ("name", "explanation")):
        raise ValueError("Name and explanation must be strings")
    patches = proposal["patches"]
    if not isinstance(patches, list) or len(patches) > 10000:
        raise ValueError("Patches must be a list of at most 10000 assignments")
    model = TinyTransformer(CONFIGS[task])
    state = dict(model.named_parameters())
    assigned = 0
    for patch in patches:
        if not isinstance(patch, dict) or set(patch) != {"tensor", "axes", "values"}:
            raise ValueError("Invalid patch keys")
        if patch["tensor"] not in state:
            raise ValueError("Unknown tensor")
        tensor = state[patch["tensor"]]
        axes, values = patch["axes"], patch["values"]
        if not isinstance(axes, list) or len(axes) != tensor.ndim:
            raise ValueError(f"Expected {tensor.ndim} axes for {patch['tensor']}")
        for axis, size in zip(axes, tensor.shape, strict=True):
            if not isinstance(axis, list) or not axis or any(type(i) is not int or not 0 <= i < size for i in axis):
                raise ValueError("Axis must contain valid nonnegative integer indices")
            if len(set(axis)) != len(axis):
                raise ValueError("Duplicate axis indices are forbidden")
        count = math.prod(map(len, axes))
        assigned += count
        if assigned > 100000:
            raise ValueError("Too many assigned elements")
        if not isinstance(values, list) or len(values) not in (1, count):
            raise ValueError("Supply one broadcast value or one value per selected element")
        if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e6 for v in values):
            raise ValueError("Values must be finite numbers with absolute value at most 1e6")
        for index, coordinates in enumerate(itertools.product(*axes)):
            tensor[coordinates] = values[0] if len(values) == 1 else values[index]
    return model
