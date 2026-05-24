import json
from pathlib import Path
import pytest

from src.llm.extraction import extract_answer
from src.llm.labeling import label

FIX = Path(__file__).parent / "fixtures"
BENCHMARKS = ["arc_challenge", "gsm8k", "mmlu_subset"]


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_labeling_fixtures(benchmark):
    items = json.loads((FIX / f"{benchmark}.json").read_text())
    for it in items:
        ans, _ = extract_answer(it["raw"], benchmark)
        y = label(ans, it["gold"], benchmark)
        assert y == it["expected_label"], (benchmark, it["raw"], ans, y)


def test_invalid_parse_is_failure():
    assert label(None, "A", "arc_challenge") == 0
    assert label(None, "42", "gsm8k") == 0


def test_missing_gold_raises():
    with pytest.raises(ValueError):
        label("A", "", "arc_challenge")
    with pytest.raises(ValueError):
        label("A", None, "mmlu_subset")


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError):
        label("A", "A", "unknown")


def test_gsm8k_numeric_equivalence():
    assert label("42", "42.0", "gsm8k") == 1
    assert label("1000", "1,000", "gsm8k") == 1
