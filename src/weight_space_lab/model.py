"""A fixed causal transformer made only of ordinary dense tensor operations."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    vocab_size: int
    context: int
    width: int
    heads: int
    hidden: int


class TinyTransformer(nn.Module):
    """One attention block and ReLU MLP, residuals, no normalization.

    This is a simplified transformer, not an unchanged GPT architecture.
    No task-dependent operations occur in forward().
    """

    def __init__(self, config: Config):
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Width must be divisible by the number of heads")
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.query = nn.Linear(config.width, config.width, bias=False)
        self.key = nn.Linear(config.width, config.width, bias=False)
        self.value = nn.Linear(config.width, config.width, bias=False)
        self.attention_out = nn.Linear(config.width, config.width, bias=False)
        self.mlp_in = nn.Linear(config.width, config.hidden)
        self.mlp_out = nn.Linear(config.hidden, config.width)
        self.readout = nn.Linear(config.width, config.vocab_size)
        self.to(device="cpu", dtype=torch.float64)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()
        self.requires_grad_(False)

    def forward(self, tokens, *, trace=False):
        batch, length = tokens.shape
        if length > self.config.context:
            raise ValueError("Input exceeds configured context length")
        positions = torch.arange(length, device=tokens.device)
        x = self.token(tokens) + self.position(positions)
        head_width = self.config.width // self.config.heads

        def split(projection):
            return projection(x).reshape(batch, length, self.config.heads, head_width).transpose(1, 2)

        q, k, v = split(self.query), split(self.key), split(self.value)
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_width)
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        attention = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        routed = (attention @ v).transpose(1, 2).reshape(batch, length, self.config.width)
        after_attention = x + self.attention_out(routed)
        hidden = torch.relu(self.mlp_in(after_attention))
        after_mlp = after_attention + self.mlp_out(hidden)
        logits = self.readout(after_mlp)
        if trace:
            return logits, {"attention": attention, "after_attention": after_attention,
                            "hidden": hidden, "after_mlp": after_mlp}
        return logits

    @torch.inference_mode()
    def generate(self, tokens, count):
        for _ in range(count):
            next_token = self(tokens)[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    def inventory(self):
        parameters = list(self.named_parameters())
        buffers = list(self.named_buffers())
        return {
            "config": asdict(self.config),
            "parameters": sum(t.numel() for _, t in parameters),
            "buffer_elements": sum(t.numel() for _, t in buffers),
            "stored_elements": sum(t.numel() for _, t in parameters + buffers),
            "nonzero_elements": sum(int(torch.count_nonzero(t)) for _, t in parameters + buffers),
            "tensors": {name: list(t.shape) for name, t in parameters + buffers},
        }
