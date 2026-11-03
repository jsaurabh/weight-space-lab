"""Guided reference constructions. These are not independent LLM discoveries."""

import torch

from .model import Config, TinyTransformer
from .specs import VOCABS

PARITY_VOCAB = VOCABS["parity"]
ADDITION_VOCAB = VOCABS["addition"]


@torch.no_grad()
def parity():
    """Uniform attention counts ones; a piecewise-linear zigzag gives parity."""
    model = TinyTransformer(Config(4, 11, 3, 1, 10))
    model.token.weight[1, 0] = 1
    model.value.weight[1, 0] = 11
    model.attention_out.weight[1, 1] = 1
    for threshold in range(10):
        model.mlp_in.weight[threshold, 1] = 1
        model.mlp_in.bias[threshold] = -threshold
        model.mlp_out.weight[2, threshold] = 1 if threshold == 0 else 2 * (-1) ** threshold
    model.readout.weight[0, 2] = -1
    model.readout.bias[0] = 0.5
    model.readout.weight[1, 2] = 1
    model.readout.bias[1] = -0.5
    model.readout.bias[2:] = -100
    return model


@torch.no_grad()
def addition():
    """Route four digits, sum place values, extract digits with ReLU steps.

    Input is exactly 'ab+cd='; output is always three digits, including zeros.
    Fixed positional routing uses sharp but finite softmax, not hard indexing.
    """
    model = TinyTransformer(Config(12, 8, 32, 4, 81))
    for token in range(12):
        model.token.weight[token, 19] = 1
    for digit in range(10):
        model.token.weight[digit, digit] = 1
    for position in range(8):
        model.position.weight[position, 10 + position] = 1
    for head, source in enumerate((0, 1, 3, 4)):
        lane = head * 8
        model.query.weight[lane, 19] = 16
        model.key.weight[lane, 10 + source] = 16
        for digit in range(10):
            model.value.weight[lane, digit] = digit
        model.attention_out.weight[20 + head, lane] = 1

    unit = 0

    def hinge(position, threshold, coefficient):
        nonlocal unit
        # Sum = 10*a_tens + a_ones + 10*b_tens + b_ones.
        model.mlp_in.weight[unit, 20:24] = torch.tensor([10, 1, 10, 1])
        # Disable this hinge at every position except the selected output step.
        model.mlp_in.weight[unit, 10 + position] = 512
        model.mlp_in.bias[unit] = -512 - threshold
        model.mlp_out.weight[24, unit] = coefficient
        unit += 1

    def step(position, threshold, coefficient):
        hinge(position, threshold - 1, coefficient)
        hinge(position, threshold, -coefficient)

    # Hundreds: floor(sum / 100).
    step(5, 100, 1)
    # Tens: floor(sum / 10) - 10*floor(sum / 100).
    for threshold in range(10, 199, 10):
        step(6, threshold, 1)
    step(6, 100, -10)
    # Ones: sum - 10*floor(sum / 10).
    hinge(7, 0, 1)
    for threshold in range(10, 199, 10):
        step(7, threshold, -10)
    assert unit == model.config.hidden
    # Nearest-integer decoder: argmax_d(2*d*x - d*d).
    for digit in range(10):
        model.readout.weight[digit, 24] = 2 * digit
        model.readout.bias[digit] = -(digit ** 2)
    model.readout.bias[10:] = -1000
    return model


BUILDERS = {"parity": parity, "addition": addition}
