"""Frozen task interfaces, deliberately separate from reference constructions."""

from .model import Config

CONFIGS = {
    "parity": Config(4, 11, 3, 1, 10),
    "addition": Config(12, 8, 32, 4, 81),
}
VOCABS = {"parity": "01=_", "addition": "0123456789+="}
PROMPT_EXAMPLES = {"parity": (), "addition": ("00+00=", "99+99=")}
DESCRIPTIONS = {
    "parity": (
        "Given a nonempty binary string of length 1 through 10, output a single token: "
        "0 when the number of ones is even and 1 when it is odd. "
        "The input string is LEFT-padded with _ to exactly 10 characters, then = is appended. "
        "Predict the answer from the logits at the final = position. Padding is not a bit."
    ),
    "addition": (
        "Given exactly ab+cd=, where ab and cd are zero-padded decimal integers from 00 to 99, "
        "generate their sum as exactly THREE decimal digits, including leading zeros. "
        "Greedy decoding predicts the hundreds digit from the = position, appends it, "
        "predicts the tens digit, appends it, then predicts the ones digit. "
        "Examples of the format only: 00+00= produces 000; 99+99= produces 198."
    ),
}
