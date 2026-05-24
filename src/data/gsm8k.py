"""GSM8K loader (HuggingFace ``datasets``).

Loads the GSM8K ``test`` split via the HuggingFace ``datasets`` library
(imported LAZILY so this module imports without the dependency),
deterministically subsamples to ``config["max_instances"]`` using a seed
derived from the global seed, and normalizes each item to the canonical
Panel A instance dict.

GSM8K is open-ended (no multiple-choice options): ``choices`` is always an
empty list and the gold answer is the final integer of the official
solution, which appears after the ``####`` marker.

No silent fallbacks: a load failure raises, and any instance whose solution
has no extractable gold number after ``####`` raises ``ValueError`` rather
than being dropped.
"""
from __future__ import annotations

import re

from src.utils.seeds import rng_for

# The official GSM8K answer is the value following the "####" marker. It may
# carry thousands separators which we strip to leave a bare integer string.
_HASH_ANSWER = re.compile(r"####\s*(-?[\d,]+)")


def _extract_gold_number(answer_field: str) -> str:
    """Return the final integer (as a string, commas stripped) after ``####``.

    Raises ``ValueError`` if the official solution does not contain a parsable
    ``#### <number>`` final answer.
    """
    if not isinstance(answer_field, str):
        raise ValueError("GSM8K answer field is not a string")
    m = _HASH_ANSWER.search(answer_field)
    if m is None:
        raise ValueError(
            f"no '#### <number>' final answer found in GSM8K solution: "
            f"{answer_field[:200]!r}"
        )
    token = m.group(1).replace(",", "").strip()
    if token in ("", "-"):
        raise ValueError(f"empty gold number after '####' in: {answer_field[:200]!r}")
    # Validate it parses as an integer; keep the canonical string form.
    try:
        value = int(token)
    except ValueError as e:
        raise ValueError(f"gold answer {token!r} is not an integer") from e
    return str(value)


def load_gsm8k(config: dict, seed: int) -> list[dict]:
    """Load and normalize GSM8K instances.

    Parameters
    ----------
    config:
        Benchmark config dict; uses ``max_instances``.
    seed:
        Global seed; the subsample RNG is ``rng_for(seed, "gsm8k")``.

    Returns
    -------
    list[dict]
        Each dict has keys: ``instance_id`` (``"gsm8k_{i}"``),
        ``benchmark`` (``"gsm8k"``), ``question``, ``choices`` (always ``[]``
        for this open-ended benchmark), ``gold_answer`` (the final integer of
        the official solution, as a string with commas stripped).

    Raises
    ------
    RuntimeError
        If the dataset fails to load.
    ValueError
        If an instance has no extractable gold number after ``####``.
    """
    try:
        import datasets  # lazy import: not installed in Phase 0
    except ImportError as e:  # pragma: no cover - depends on env
        raise RuntimeError("the 'datasets' package is required to load GSM8K") from e

    try:
        ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:  # noqa: BLE001 - structured re-raise, no fallback
        raise RuntimeError(f"failed to load GSM8K dataset: {e!r}") from e

    n_total = len(ds)
    max_instances = int(config["max_instances"])
    rng = rng_for(seed, "gsm8k")
    if max_instances < n_total:
        idx = rng.choice(n_total, size=max_instances, replace=False)
        idx = sorted(int(i) for i in idx)
    else:
        idx = list(range(n_total))

    out: list[dict] = []
    for new_i, raw_i in enumerate(idx):
        row = ds[raw_i]
        gold = _extract_gold_number(row["answer"])
        out.append(
            {
                "instance_id": f"gsm8k_{new_i}",
                "benchmark": "gsm8k",
                "question": row["question"],
                "choices": [],
                "gold_answer": gold,
            }
        )
    return out
