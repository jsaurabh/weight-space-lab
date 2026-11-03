"""Grading excerpts from the published Experiment 8 harness (MIT).

Source: RikiyaT/lm_understand-logs, commit
4d09ba6d6a7e891dd8063974dfe81b3d78e3494f, exp8_math/code/mathqa.py.
Only imports and the module-loading wrapper are adapted; grading bodies are copied.
Copyright (c) 2026 the authors; see LICENSE.upstream for the MIT license.
"""
import functools
import re
import threading
import math_verify

_BOX_FORMS = re.compile(r"\\boxed|\\fbox")

def mv():
    return math_verify


def answer_text(text):
    """The text the extraction reads: a stray reasoning block cut off (up to the last '</think>')."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text

def _box_at(text, idx):
    """the content of the box that starts at text[idx] ('\\boxed' or '\\fbox'), or None."""
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):   # brace matched, as last_boxed_only_string
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None          # not closed (e.g. a response cut inside the box)
    if key == "\\boxed" and rest.startswith(" "):   # '\\boxed 5': up to the next '$' or line end (Qwen's reading)
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None

def last_boxed(text):
    """The content of the last box of a text, or None: hendrycks last_boxed_only_string + remove_boxed (the last
    '\\boxed', else the last '\\fbox'; brace matched; an unclosed last box = None), extended to '\\fbox{..}',
    '\\boxed {..}' and '\\boxed 5' (no braces: read up to the next '$' or line end, as Qwen's evaluation reads it).
    An empty box = None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)

def extract(text):
    """(answer or None, number of boxes in the text)."""
    t = answer_text(text)
    return last_boxed(t), len(_BOX_FORMS.findall(t))

def _timeouts():
    """math-verify's timeouts use signal.alarm, which works only in the main thread; elsewhere they raise and the
    library would silently return 'not equal'. Off the main thread the checks run without its timeouts."""
    return 5 if threading.current_thread() is threading.main_thread() else None

@functools.lru_cache(maxsize=1 << 16)
def _parsed(s, to):
    return mv().parse("\\boxed{" + s + "}", parsing_timeout=to)

def is_correct(pred, gold):
    """math-verify equivalence of a boxed answer and the gold answer (False for no answer / unparseable gold)."""
    if pred is None or gold is None:
        return False
    to = _timeouts()
    g, p = _parsed(gold, to), _parsed(pred, to)
    if not g or not p:
        return False
    try:
        return bool(mv().verify(g, p, timeout_seconds=to))
    except Exception:  # noqa: BLE001
        return False
