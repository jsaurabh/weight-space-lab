"""Matched Qwen editor interface; no previous repair or experiment outputs."""
import ast

BASE_ID = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MAX_FORWARD_CALLS = 10
MAX_ANALYSIS_CALLS = 100

SYSTEM = """Improve Qwen3.5-0.8B's math accuracy by diagnosing failures and directly
editing internal weights. Find your own intervention using only the base weights,
training-practice data and these tools. No published repair, previous editor
conversation or final test is available.

No training, gradient estimation, optimizers, supervised fitting, memorization
answer tables, pretrained external weights, architecture changes or decoding
changes. CPU analysis may inspect weights and saved observations with NumPy and
closed-form algebra. Only generate/probe execute Qwen, in inference mode on GPU.
No internet, shell, repository or arbitrary file access. Embeddings, output head,
final normalization and pure whole-tensor scalar gains are not editable.

Diagnose actual failures first. Repetition and unfinished thinking are legitimate
failure mechanisms to investigate; merely forcing a stop marker or suppressing
thinking irrespective of the question is not a demonstrated math improvement.
Use causal interventions, compare strengths, and measure accuracy on the same
practice IDs before and after editing. Prefer sufficiently large paired batches;
32 paired problems fit one generation call. Keep an edit only when practice
accuracy justifies it; fewer cut-off outputs alone are insufficient. Keep a base
checkpoint and restore it if no repair has convincing evidence.

All generation calls use thinking and the same fixed sampling, with a 32,768-token
cap. Generation artifacts contain full texts and answers for inspection. Probes
can examine long model continuations at selected positions and return hidden
states, module inputs/outputs, next-token log probabilities and top token IDs.

You have exactly 10 successful forward-tool calls (generate or probe), up to
100 CPU analysis calls and 120 API turns. Use the full forward budget before
finish; failed requests are refunded. CPU range/size/time restrictions apply.
Keep named checkpoints of candidates. Explain each experiment concisely in its
reason field. At the end select a checkpoint with finish. The held-out test will
be uploaded only after BOTH editors' choices are frozen, and no test feedback
will reach either editor. The other editor's work is never shown to you.
"""


def function(name, description, properties):
    return {"type": "function", "name": name, "description": description,
            "strict": True, "parameters": {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}}


TOOLS = [
    function("practice", "Read training questions, answers and available reference solutions. Returns up to 16 rows, using offsets in the combined pool. Metadata lists source ranges.",
        {"offset": {"type": "integer", "minimum": 0}, "count": {"type": "integer", "minimum": 1, "maximum": 16}}),
    function("python_analysis", "Persistent CPU NumPy only; cannot run the target. Globals: np, math, names, shapes, config, practice_ids (all training IDs in source order). Helpers: practice_row(id) returns a copied training question/answer/solution; weight(name,current=False) returns a float32 COPY; save_edit(name,array), remove_edit(name), clear_edits(), list_edits(), checkpoint(label), restore(label), artifact(id), tokenize(text), decode(ids). Print compact summaries. No files/network/processes or training. A checkpoint label has up to 40 letters, digits, underscores or hyphens.",
        {"reason": {"type": "string"}, "code": {"type": "string"}}),
    function("generate", "Thinking generation on distinct practice IDs: base, current or both. Fixed 32768-token cap; max 64 TOTAL generations, so max 32 paired problems for both. One sample per problem. Returns accuracy, token limits, paired rescues/losses, and an artifact ID containing complete texts. One forward-budget unit.",
        {"reason": {"type": "string"}, "which": {"type": "string", "enum": ["base", "current", "both"]},
         "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64}}),
    function("probe", "Teacher-force base/current on up to 16 raw texts, <=32768 tokens each and <=131072 tokens total. At up to 32 selected positions (negative indices allowed), capture all layer hidden states and up to 8 modules' input/output vectors, top-20 next-token IDs/logprobs and actual next-token logprobs where present. Module name is its tensor name without .weight. Artifact: texts entries have text,tokens,positions,hidden (layers x positions x width),modules (name -> input/output arrays),top_ids,top_logprobs,next_ids,next_logprobs. One forward-budget unit. Include exact chat template text for faithful diagnostic probes.",
        {"reason": {"type": "string"}, "which": {"type": "string", "enum": ["base", "current"]},
         "texts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 16},
         "modules": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
         "positions": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 32}}),
    function("finish", "After all 10 forward calls, select a checkpoint label (or empty string for current weights) and freeze permanently. The host alone performs later held-out evaluation. Restore your base checkpoint if no edit is justified.",
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
