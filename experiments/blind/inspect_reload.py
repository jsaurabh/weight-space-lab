"""Inspect native string-based weight update interfaces without loading a model."""
import inspect
from vllm import LLM
from vllm.v1.worker.gpu_worker import Worker
print("LLM weight methods:", [n for n in dir(LLM) if any(k in n for k in ["weight", "rpc", "sleep", "wake", "shutdown"])])
for name in dir(Worker):
    if any(k in name for k in ["weight", "reload", "update"]):
        value = getattr(Worker, name)
        if callable(value):
            try:
                print(name, inspect.signature(value))
                if name in {"reload_weights", "update_weights", "update_config"}:
                    print(inspect.getsource(value))
            except Exception:
                pass
