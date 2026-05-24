"""Per-benchmark answer extraction with an explicit failure policy.

Extraction NEVER silently returns None into the success-indicator stream.
A failed extraction is returned as (None, invalid_parse=True) and, when a
log path is supplied, a structured event is appended to
outputs/logs/extraction_failures.jsonl. The caller keeps the record with
parsed_answer=None, correct=0, invalid_parse=True and preserves raw output.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

_LETTER = re.compile(r"\b([A-D])\b")
_ANSWER_IS_LETTER = re.compile(r"answer\s*(?:is|:)\s*\(?\s*([A-D])\b", re.IGNORECASE)
_PAREN_LETTER = re.compile(r"\(\s*([A-D])\s*\)")
# Number: optional sign, optional $, digits with optional thousands commas and
# optional decimal part.
_NUMBER = re.compile(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?")
_ANSWER_IS_NUMBER = re.compile(
    r"answer\s*(?:is|:)\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE
)
_HASH = re.compile(r"####\s*([-+]?\$?\s*\d[\d,]*(?:\.\d+)?)")


def _log_failure(log_path: Path | None, payload: dict) -> None:
    if log_path is None:
        return
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def _norm_number(token: str) -> str:
    return token.replace(",", "").replace("$", "").replace(" ", "")


def extract_single_letter(raw: str, log_path: Path | None = None, instance_id: str | None = None) -> tuple[str | None, bool]:
    """Extract a single A/B/C/D. Returns (letter, invalid_parse).

    Priority: explicit 'answer is/:' phrasing, then a parenthesized letter,
    then a unique standalone letter. Zero or multiple distinct letters ->
    invalid.
    """
    if not isinstance(raw, str):
        _log_failure(log_path, {"benchmark": "single_letter", "instance_id": instance_id, "reason": "non-string output"})
        return None, True
    text = raw.strip()
    m = _ANSWER_IS_LETTER.search(text)
    if m:
        return m.group(1).upper(), False
    m = _PAREN_LETTER.search(text)
    if m:
        return m.group(1).upper(), False
    letters = {mm.group(1).upper() for mm in _LETTER.finditer(text.upper())}
    if len(letters) == 1:
        return next(iter(letters)), False
    _log_failure(
        log_path,
        {"benchmark": "single_letter", "instance_id": instance_id, "reason": f"{len(letters)} distinct letters", "raw": text[:200]},
    )
    return None, True


def extract_single_number(raw: str, log_path: Path | None = None, instance_id: str | None = None) -> tuple[str | None, bool]:
    """Extract a single number for GSM8K. Returns (normalized_number_str,
    invalid_parse). Priority: '#### N', then 'answer is/: N', then the last
    number in the output. Commas/$/spaces stripped."""
    if not isinstance(raw, str):
        _log_failure(log_path, {"benchmark": "single_number", "instance_id": instance_id, "reason": "non-string output"})
        return None, True
    text = raw.strip()
    m = _HASH.search(text)
    if m:
        return _norm_number(m.group(1)), False
    m = _ANSWER_IS_NUMBER.search(text)
    if m:
        return _norm_number(m.group(1)), False
    matches = _NUMBER.findall(text)
    if matches:
        return _norm_number(matches[-1]), False
    _log_failure(
        log_path,
        {"benchmark": "single_number", "instance_id": instance_id, "reason": "no number found", "raw": text[:200]},
    )
    return None, True


def extract_answer(raw: str, benchmark: str, log_path: Path | None = None, instance_id: str | None = None) -> tuple[str | None, bool]:
    """Dispatch to the per-benchmark extractor."""
    if benchmark in ("arc_challenge", "mmlu_subset"):
        return extract_single_letter(raw, log_path=log_path, instance_id=instance_id)
    if benchmark == "gsm8k":
        return extract_single_number(raw, log_path=log_path, instance_id=instance_id)
    raise ValueError(f"unknown benchmark for extraction: {benchmark}")
