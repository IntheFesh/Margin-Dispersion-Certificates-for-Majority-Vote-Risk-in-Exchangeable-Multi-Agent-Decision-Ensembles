import json
from pathlib import Path
import pytest

from src.llm.extraction import extract_answer

FIX = Path(__file__).parent / "fixtures"
BENCHMARKS = ["arc_challenge", "gsm8k", "mmlu_subset"]


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_extraction_fixtures(benchmark):
    items = json.loads((FIX / f"{benchmark}.json").read_text())
    assert len(items) == 10
    for it in items:
        ans, invalid = extract_answer(it["raw"], benchmark)
        assert invalid == it["expected_invalid"], (benchmark, it["raw"], ans, invalid)
        assert ans == it["expected_answer"], (benchmark, it["raw"], ans)


def test_extraction_logs_failure(tmp_path):
    log = tmp_path / "extraction_failures.jsonl"
    ans, invalid = extract_answer("no answer here", "arc_challenge", log_path=log, instance_id="i0")
    assert ans is None and invalid is True
    assert log.exists()
    rec = json.loads(log.read_text().strip())
    assert rec["instance_id"] == "i0"


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError):
        extract_answer("A", "unknown_bench")
