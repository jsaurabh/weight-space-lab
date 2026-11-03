"""Numerical-analysis interface for a fresh model editor. No solution artifacts."""
import ast

BASE_ID = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MAX_FORWARD_CALLS = 20
MAX_ANALYSIS_CALLS = 100

SYSTEM = """You are an experimental neural-network weight programmer. Improve the math
accuracy of the given pretrained target by diagnosing its behavior and directly
editing its weights. Discover your own intervention from the base weights and
the allowed practice feedback. Explain the hypothesis behind each action briefly.

Rules: no training, optimizers, backpropagation, gradient estimation, parameter
fitting to examples, supervised regressions, memorization tables, pretrained
external weights, architecture changes, or changes to generation settings.
Use reasoning, numerical inspection, statistics and closed-form algebraic edits.
The CPU Python environment cannot execute the neural network. Only the provided
generate/probe tools run it, in inference mode. You have no internet, shell,
repository access or other tools. Practice answers are allowed. The final test
set is sealed and unavailable to every tool during discovery.

Embeddings, the output head and the final normalization are fixed. Internal
model tensors may be edited; multiplying an entire tensor by one scalar is not
an admissible repair. Do not conceal training inside analysis code. All code,
edits and feedback are recorded for audit. No human will supply repair hints.

You have 20 forward-tool calls (generate or probe), 100 CPU analysis calls and
60 API turns, subject to a $25 estimated API ceiling. Use feedback economically.
Keep checkpoints of promising edits and choose the final checkpoint yourself.
Call finish only when you have selected the final weights. Final test scores
will be measured only after you finish and cannot be used for further editing.
"""


def function(name, description, properties):
    return {"type": "function", "name": name, "description": description,
            "strict": True, "parameters": {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}}


TOOLS = [
    function("practice", "Read public training practice questions and answers. IDs are the only IDs accepted by generate.",
             {"offset": {"type": "integer"}, "count": {"type": "integer"}}),
    function("python_analysis", "Run persistent CPU NumPy analysis; cannot run the target. Globals: np, math, names (tensor names), shapes, config. Helpers: weight(name, current=False) returns float32 copy; save_edit(name, array) replaces an internal tensor; remove_edit(name); clear_edits(); list_edits(); checkpoint(label); restore(label); artifact(id) returns saved generation/probe data; tokenize(text); decode(ids). Use print for results. No file/network/process access. Explain your experiment in reason.",
             {"reason": {"type": "string"}, "code": {"type": "string"}}),
    function("generate", "Run practice generation with fixed thinking/sampling. which is base/current/both. Max 32 total generations, 16 paired problems for both. Each problem gets one sample. Returns scores and an artifact ID containing full texts. One forward-call budget unit.",
             {"reason": {"type": "string"}, "which": {"type": "string", "enum": ["base", "current", "both"]},
              "ids": {"type": "array", "items": {"type": "string"}},
              "max_tokens": {"type": "integer"}}),
    function("probe", "Teacher-force up to 8 texts (max 2048 tokens each) using base or current weights, with no gradients. Capture all layer hidden states and optional module input/output vectors at up to 32 token positions (negative indices allowed). modules are full module names, e.g. a tensor name with its final .weight removed. Returns an artifact: texts is a list, each with tokens, positions, hidden (array of layers x positions x width), modules (dict name -> input/output arrays). Module names are drawn from the public target architecture. One forward-call budget unit.",
             {"reason": {"type": "string"}, "which": {"type": "string", "enum": ["base", "current"]},
              "texts": {"type": "array", "items": {"type": "string"}},
              "modules": {"type": "array", "items": {"type": "string"}},
              "positions": {"type": "array", "items": {"type": "integer"}}}),
    function("finish", "Select an existing checkpoint label, or empty string for current weights, and freeze it permanently for independent testing.",
             {"checkpoint_label": {"type": "string"}, "rationale": {"type": "string"}}),
]

FORBIDDEN = {
    "open", "eval", "exec", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "input", "help", "breakpoint", "load", "save", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile",
    "memmap", "open_memmap", "DataSource", "ctypeslib", "ctypes", "getlib", "load_library",
    "fit", "fit_transform", "backward", "grad", "gradient", "autograd", "optim", "optimizer",
    "Adam", "SGD", "lstsq", "polyfit", "least_squares", "system", "popen", "subprocess",
}


def validate_code(code):
    if not isinstance(code, str) or len(code) > 24000:
        raise ValueError("Analysis code is limited to 24000 characters")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.While, ast.ClassDef, ast.AsyncFunctionDef, ast.AsyncFor, ast.Global, ast.Nonlocal)):
            raise ValueError("Unsupported analysis construct")
        if isinstance(node, ast.Import):
            if any(alias.name not in {"numpy", "math"} for alias in node.names):
                raise ValueError("Only numpy and math may be imported")
        if isinstance(node, ast.ImportFrom) and (node.module not in {"numpy", "math"} or node.level):
            raise ValueError("Only numpy and math may be imported")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN):
            raise ValueError("Unavailable analysis attribute")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN):
            raise ValueError("Unavailable analysis name")
    return tree
