from __future__ import annotations
import re

CHOICE_PATTERNS = [re.compile(r"\b(?:answer is|option)?\s*\(?([A-Z])\)?\b", re.IGNORECASE)]
NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")


def parse_multiple_choice_answer(raw_output: str, valid_labels: list[str]) -> tuple[str | None, bool]:
    if not isinstance(raw_output, str):
        return None, True
    valid = {v.upper() for v in valid_labels}
    text = raw_output.strip().upper()
    for p in CHOICE_PATTERNS:
        for m in p.finditer(text):
            c = m.group(1).upper()
            if c in valid:
                return c, False
    return None, True


def parse_gsm8k_numeric_answer(raw_output: str) -> tuple[str | None, bool]:
    if not isinstance(raw_output, str):
        return None, True
    part = raw_output.split("####")[-1] if "####" in raw_output else raw_output
    matches = NUM_RE.findall(part)
    if not matches:
        return None, True
    return matches[-1].replace(",", ""), False
