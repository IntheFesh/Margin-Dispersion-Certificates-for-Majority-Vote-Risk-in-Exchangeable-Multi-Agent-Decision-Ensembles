"""Canonicalization + correctness labeling under Assumption 1.

Assumption 1 (unique-answer canonicalization): each instance has a unique
canonical gold answer and a deterministic canonicalization map. The
benchmarks in Panel A (ARC-Challenge single letter, GSM8K single integer,
MMLU single letter) satisfy this by design. Any instance lacking a unique
gold answer raises ValueError (out of scope for Panel A).

invalid-parse counts as a failure for certificate purposes (bridge lemma):
extracted_answer is None -> success indicator 0.
"""
from __future__ import annotations


_LETTER_BENCHMARKS = {"arc_challenge", "mmlu_subset"}
_NUMBER_BENCHMARKS = {"gsm8k"}


def _require_unique_gold(gold_answer: str) -> None:
    if gold_answer is None or (isinstance(gold_answer, str) and gold_answer.strip() == ""):
        raise ValueError("instance lacks a unique gold answer (Assumption 1 violated)")


def label(extracted_answer: str | None, gold_answer: str, benchmark: str) -> int:
    """Return 1 if correct, 0 if incorrect, 0 if extracted_answer is None.

    Raises ValueError if the instance lacks a unique gold answer or the
    benchmark is unknown."""
    _require_unique_gold(gold_answer)
    if benchmark in _LETTER_BENCHMARKS:
        if extracted_answer is None:
            return 0
        return int(str(extracted_answer).strip().upper() == str(gold_answer).strip().upper())
    if benchmark in _NUMBER_BENCHMARKS:
        if extracted_answer is None:
            return 0
        try:
            pred = float(str(extracted_answer).replace(",", "").replace("$", "").strip())
            gold = float(str(gold_answer).replace(",", "").replace("$", "").strip())
        except ValueError:
            return 0
        # numeric_tolerance = 0 -> exact numeric equality.
        return int(pred == gold)
    raise ValueError(f"unknown benchmark for labeling: {benchmark}")
