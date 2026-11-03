"""Read installed runtime interfaces before any repair search."""
import inspect
from vllm import LLM
print("apply_model available:", hasattr(LLM, "apply_model"))
if hasattr(LLM, "apply_model"):
    print(inspect.signature(LLM.apply_model))
