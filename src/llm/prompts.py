"""Per-benchmark prompt templates and a deterministic formatter.

Each benchmark has a fixed instruction line appended after the question (and
the multiple-choice options, where applicable). Surface-form robustness is
exercised through exactly 16 deterministic prompt variations (persona /
phrasing prefixes). The variation used for an instance is selected purely
from the caller's ``randomization_seed`` (``seed % 16``), so the mapping is
reproducible across runs and machines.

This module is pure string manipulation: no I/O, no randomness beyond the
deterministic ``seed % 16`` selection, and no import-time side effects.
"""
from __future__ import annotations

# --- Per-benchmark instruction lines ---------------------------------------
INSTRUCTION_ARC_CHALLENGE: str = "Answer with a single letter from {A, B, C, D}."
INSTRUCTION_GSM8K: str = (
    "Show your work step by step. End with 'The answer is N.' where N is a number."
)
INSTRUCTION_MMLU: str = "Answer with a single letter."

# Map benchmark name -> instruction line.
INSTRUCTIONS: dict[str, str] = {
    "arc_challenge": INSTRUCTION_ARC_CHALLENGE,
    "gsm8k": INSTRUCTION_GSM8K,
    "mmlu_subset": INSTRUCTION_MMLU,
}

# --- 16 deterministic prompt variation prefixes ----------------------------
# Each entry is prepended (verbatim, including any trailing separator) to the
# question. Index i corresponds to prompt_variation_id == i. The list MUST
# have exactly 16 entries; this is asserted below at import time via a length
# check used only by the formatter (no side effects on import).
PROMPT_VARIATIONS: list[str] = [
    "",
    "Question:\n",
    "Please answer the following question.\n\n",
    "You are a careful expert. Answer the question below.\n\n",
    "Consider the following problem carefully.\n\n",
    "Read the problem and provide your answer.\n\n",
    "As a knowledgeable assistant, solve the following.\n\n",
    "Think step by step about the following question.\n\n",
    "Here is a question for you to answer.\n\n",
    "Solve the problem below.\n\n",
    "Carefully read and answer this question.\n\n",
    "You are taking an exam. Answer the question.\n\n",
    "Determine the correct answer to the following.\n\n",
    "Provide the best answer to the question below.\n\n",
    "Analyze the following and respond.\n\n",
    "Given the problem below, give your answer.\n\n",
]

NUM_VARIATIONS: int = 16


def _format_choices(choices: list[str]) -> str:
    """Join already-labeled choice strings (e.g. ``"A. text"``) one per line.

    The data loaders emit choices pre-labeled as ``"A. ..."``, ``"B. ..."``;
    we therefore join them verbatim rather than re-labeling, so the letter
    space stays consistent with the gold answer.
    """
    return "\n".join(choices)


def format_prompt(
    item: dict,
    benchmark: str,
    randomization_seed: int,
) -> tuple[str, int]:
    """Build the full prompt for ``item`` and return (prompt, variation_id).

    The full prompt is::

        <variation_prefix><question>[\n\n<choices>]\n\n<instruction line>

    Parameters
    ----------
    item:
        Instance dict. Must contain the question under ``"question"`` (or
        ``"prompt"``). For multiple-choice benchmarks, ``"choices"`` is a list
        of pre-labeled option strings (``"A. ..."``).
    benchmark:
        One of ``"arc_challenge"``, ``"gsm8k"``, ``"mmlu_subset"``.
    randomization_seed:
        Deterministic seed; the variation id is ``randomization_seed % 16``.

    Returns
    -------
    tuple[str, int]
        The formatted prompt text and the integer ``prompt_variation_id``.

    Raises
    ------
    ValueError
        If the benchmark is unknown or the question text is missing.
    """
    if benchmark not in INSTRUCTIONS:
        raise ValueError(f"unknown benchmark for prompt formatting: {benchmark}")
    if len(PROMPT_VARIATIONS) != NUM_VARIATIONS:
        raise ValueError(
            f"PROMPT_VARIATIONS must have exactly {NUM_VARIATIONS} entries, "
            f"found {len(PROMPT_VARIATIONS)}"
        )

    question = item.get("question")
    if question is None:
        question = item.get("prompt")
    if question is None or (isinstance(question, str) and question.strip() == ""):
        raise ValueError("item is missing a non-empty 'question'/'prompt' field")

    variation_id = int(randomization_seed) % NUM_VARIATIONS
    prefix = PROMPT_VARIATIONS[variation_id]
    instruction = INSTRUCTIONS[benchmark]

    parts: list[str] = [prefix + str(question)]

    choices = item.get("choices") or []
    if choices:
        parts.append(_format_choices(list(choices)))

    parts.append(instruction)
    prompt_text = "\n\n".join(parts)
    return prompt_text, variation_id
