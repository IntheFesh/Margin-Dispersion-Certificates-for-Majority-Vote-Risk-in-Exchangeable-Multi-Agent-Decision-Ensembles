"""ARC-Challenge loader (HuggingFace ``datasets``).

Loads the ARC-Challenge ``test`` split via the HuggingFace ``datasets``
library (imported LAZILY so this module imports without the dependency),
deterministically subsamples to ``config["max_instances"]`` using a seed
derived from the global seed, and normalizes each item to the canonical
Panel A instance dict.

No silent fallbacks: a load failure raises, and any instance without a
unique single-letter (A-D) gold answer raises ``ValueError`` rather than
being dropped.
"""
from __future__ import annotations

from src.utils.seeds import rng_for

_LETTERS = ["A", "B", "C", "D", "E"]


def _normalize_answer_key(answer_key: str, labels: list[str]) -> str:
    """Map the raw ``answerKey`` to a positional letter A-D.

    ARC encodes the answer either as a letter (``"A".."E"``) or as a 1-based
    numeral (``"1".."4"``); ``labels`` are the per-choice labels in order. We
    locate the answer's position among the choice labels and map that
    position to A/B/C/D. Raises ``ValueError`` if it cannot be resolved to a
    unique letter in {A,B,C,D}.
    """
    key = str(answer_key).strip()
    if key == "":
        raise ValueError("empty answerKey")

    norm_labels = [str(lbl).strip() for lbl in labels]
    if key in norm_labels:
        pos = norm_labels.index(key)
    else:
        raise ValueError(f"answerKey {answer_key!r} not among choice labels {labels!r}")

    if pos < 0 or pos >= len(_LETTERS):
        raise ValueError(f"answer position {pos} out of letter range")
    letter = _LETTERS[pos]
    if letter not in ("A", "B", "C", "D"):
        raise ValueError(f"gold letter {letter!r} outside valid label set {{A,B,C,D}}")
    return letter


def load_arc_challenge(config: dict, seed: int) -> list[dict]:
    """Load and normalize ARC-Challenge instances.

    Parameters
    ----------
    config:
        Benchmark config dict; uses ``max_instances``.
    seed:
        Global seed; the subsample RNG is ``rng_for(seed, "arc_challenge")``.

    Returns
    -------
    list[dict]
        Each dict has keys: ``instance_id`` (``"arc_{i}"``),
        ``benchmark`` (``"arc_challenge"``), ``question``, ``choices``
        (list of ``"A. text"`` strings), ``gold_answer`` (a single letter
        A-D).

    Raises
    ------
    RuntimeError
        If the dataset fails to load.
    ValueError
        If an instance lacks a unique A-D gold letter.
    """
    try:
        import datasets  # lazy import: not installed in Phase 0
    except ImportError as e:  # pragma: no cover - depends on env
        raise RuntimeError("the 'datasets' package is required to load ARC-Challenge") from e

    try:
        ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    except Exception as e:  # noqa: BLE001 - structured re-raise, no fallback
        raise RuntimeError(f"failed to load ARC-Challenge dataset: {e!r}") from e

    n_total = len(ds)
    max_instances = int(config["max_instances"])
    rng = rng_for(seed, "arc_challenge")
    if max_instances < n_total:
        idx = rng.choice(n_total, size=max_instances, replace=False)
        idx = sorted(int(i) for i in idx)
    else:
        idx = list(range(n_total))

    out: list[dict] = []
    for new_i, raw_i in enumerate(idx):
        row = ds[raw_i]
        choices_field = row["choices"]
        texts = list(choices_field["text"])
        labels = list(choices_field["label"])
        gold = _normalize_answer_key(row["answerKey"], labels)

        labeled_choices = [f"{_LETTERS[j]}. {text}" for j, text in enumerate(texts)]
        out.append(
            {
                "instance_id": f"arc_{new_i}",
                "benchmark": "arc_challenge",
                "question": row["question"],
                "choices": labeled_choices,
                "gold_answer": gold,
            }
        )
    return out
