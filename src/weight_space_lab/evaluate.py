"""Independent input enumeration, answer oracle, and exhaustive evaluation."""

import hashlib
import itertools
import json
import platform
import time

import torch

from .programs import VOCABS


def examples(task):
    if task == "parity":
        for length in range(1, 11):
            for bits in itertools.product("01", repeat=length):
                text = "".join(bits)
                yield text.rjust(10, "_") + "=", str(text.count("1") % 2), str(length)
    elif task == "addition":
        for left in range(100):
            for right in range(100):
                total = left + right
                yield f"{left:02d}+{right:02d}=", f"{total:03d}", "carry" if left % 10 + right % 10 >= 10 else "no_ones_carry"
    else:
        raise ValueError(f"Unknown task: {task}")


def encode(texts, vocab):
    return torch.tensor([[vocab.index(c) for c in text] for text in texts], dtype=torch.long)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(bytes(tensor.detach().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(model, task, batch_size=256):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model.eval()
    vocab = VOCABS[task]
    rows = list(examples(task))
    groups, failures = {}, []
    correct = 0
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        inputs = encode([row[0] for row in batch], vocab)
        output_size = len(batch[0][1])
        generated = model.generate(inputs, output_size)[:, -output_size:]
        predictions = ["".join(vocab[index] for index in row) for row in generated.tolist()]
        for (prompt, expected, group), prediction in zip(batch, predictions, strict=True):
            passed = prediction == expected
            correct += int(passed)
            stats = groups.setdefault(group, {"correct": 0, "total": 0})
            stats["correct"] += int(passed)
            stats["total"] += 1
            if not passed and len(failures) < 10:
                failures.append({"prompt": prompt, "expected": expected, "actual": prediction})
    return {
        "task": task,
        "provenance": "guided_reference_construction",
        "evaluation": "exhaustive_finite_domain",
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "groups": groups,
        "failures_first_10": failures,
        "seconds": round(time.perf_counter() - started, 6),
        "device": "cpu",
        "dtype": str(next(model.parameters()).dtype),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "torch_version": torch.__version__,
        "state_sha256": state_digest(model),
        **model.inventory(),
    }


def write_report(report, path):
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
