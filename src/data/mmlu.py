"""MMLU subset loader (HuggingFace ``datasets``).

Loads the MMLU ``test`` split (config ``"all"``) via the HuggingFace
``datasets`` library (imported LAZILY so this module imports without the
dependency), filters to the subjects listed in ``config["subjects"]``, takes
up to ``config["max_instances_per_subject"]`` deterministically per subject
using a per-subject seed derived from the global seed, and normalizes each
item to the canonical Panel A instance dict.

No silent fallbacks: a load failure raises, a missing/empty subject after
filtering raises, and any instance without a unique single-letter (A-D) gold
answer raises ``ValueError`` rather than being dropped.
"""
from __future__ import annotations

from src.utils.seeds import rng_for

_LETTERS = ["A", "B", "C", "D"]


def _letter_from_index(answer_index: int, n_choices: int) -> str:
    """Map a 0-based integer answer index to its letter A-D.

    Raises ``ValueError`` if the index is out of range for the {A,B,C,D}
    label set or is not consistent with the number of available choices.
    """
    try:
        pos = int(answer_index)
    except (TypeError, ValueError) as e:
        raise ValueError(f"non-integer MMLU answer index {answer_index!r}") from e
    if pos < 0 or pos >= len(_LETTERS):
        raise ValueError(f"MMLU answer index {pos} outside valid label set {{A,B,C,D}}")
    if pos >= n_choices:
        raise ValueError(
            f"MMLU answer index {pos} exceeds number of choices {n_choices}"
        )
    return _LETTERS[pos]


def load_mmlu_subset(config: dict, seed: int) -> list[dict]:
    """Load and normalize the MMLU subject subset.

    Parameters
    ----------
    config:
        Benchmark config dict; uses ``subjects`` (a list of subject names) and
        ``max_instances_per_subject``.
    seed:
        Global seed; each subject's subsample RNG is
        ``rng_for(seed, "mmlu_subset", subject)``.

    Returns
    -------
    list[dict]
        Each dict has keys: ``instance_id`` (``"mmlu_{subject}_{i}"``),
        ``benchmark`` (``"mmlu_subset"``), ``question``, ``choices`` (list of
        ``"A. text"`` strings), ``gold_answer`` (a single letter A-D).

    Raises
    ------
    RuntimeError
        If the dataset fails to load.
    ValueError
        If a requested subject is absent/empty, or an instance lacks a unique
        A-D gold letter.
    """
    try:
        import datasets  # lazy import: not installed in Phase 0
    except ImportError as e:  # pragma: no cover - depends on env
        raise RuntimeError("the 'datasets' package is required to load MMLU") from e

    try:
        ds = datasets.load_dataset("cais/mmlu", "all", split="test")
    except Exception as e:  # noqa: BLE001 - structured re-raise, no fallback
        raise RuntimeError(f"failed to load MMLU dataset: {e!r}") from e

    subjects = list(config["subjects"])
    max_per_subject = int(config["max_instances_per_subject"])

    # Group row indices by subject in dataset order (stable, reproducible).
    by_subject: dict[str, list[int]] = {subj: [] for subj in subjects}
    subject_set = set(subjects)
    all_subjects = ds["subject"]
    for i, subj in enumerate(all_subjects):
        if subj in subject_set:
            by_subject[subj].append(i)

    out: list[dict] = []
    for subject in subjects:
        rows = by_subject[subject]
        if not rows:
            raise ValueError(
                f"MMLU subject {subject!r} produced no instances in the test split"
            )
        rng = rng_for(seed, "mmlu_subset", subject)
        n_avail = len(rows)
        if max_per_subject < n_avail:
            picked = rng.choice(n_avail, size=max_per_subject, replace=False)
            picked = sorted(int(p) for p in picked)
            chosen_rows = [rows[p] for p in picked]
        else:
            chosen_rows = rows

        for new_i, raw_i in enumerate(chosen_rows):
            row = ds[raw_i]
            texts = list(row["choices"])
            gold = _letter_from_index(row["answer"], len(texts))
            labeled_choices = [f"{_LETTERS[j]}. {text}" for j, text in enumerate(texts)]
            out.append(
                {
                    "instance_id": f"mmlu_{subject}_{new_i}",
                    "benchmark": "mmlu_subset",
                    "question": row["question"],
                    "choices": labeled_choices,
                    "gold_answer": gold,
                }
            )
    return out
