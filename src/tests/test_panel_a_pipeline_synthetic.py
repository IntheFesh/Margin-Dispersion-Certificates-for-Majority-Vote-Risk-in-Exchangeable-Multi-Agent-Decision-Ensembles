"""End-to-end exercise of the Panel A pipeline using hand-crafted small
binary correctness matrices. These rows are explicit synthetic test
fixtures and are written to a tmp path; they never appear under
results/ or data/ in the real codebase."""
from __future__ import annotations
import json
import yaml
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.panel_a.run_pilot import run as run_pilot
from src.panel_a.run_full import run as run_full


def _synthetic_correctness(seed=0, instances=40, samples=16, protocols=("A1",), benchmarks=("ARC-Challenge",), p=0.75, regime=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for proto in protocols:
        for bench in benchmarks:
            X = (rng.random((instances, samples)) < p).astype(int)
            for i in range(instances):
                for s in range(samples):
                    row = {
                        "instance_id": f"{bench}_{i}",
                        "benchmark": bench,
                        "protocol": proto,
                        "sample_id": int(s),
                        "correct": int(X[i, s]),
                    }
                    if regime is not None:
                        row["regime"] = regime
                    rows.append(row)
    return pd.DataFrame(rows)


def test_run_pilot_on_synthetic_csv(tmp_path):
    df = _synthetic_correctness(seed=1, instances=40, samples=16, protocols=("A1", "A2"))
    in_csv = tmp_path / "pilot.csv"
    df.to_csv(in_csv, index=False)
    cfg = {
        "seed": 7,
        "output_dir": str(tmp_path / "out"),
        "pilot_correctness_csv": str(in_csv),
        "K_ref": 16,
        "K_est": 8,
        "N_values": [4, 8],
        "global_delta": 0.1,
        "strict_majority": True,
    }
    summary = run_pilot(cfg, config_path=str(in_csv))
    assert summary["num_cells"] > 0
    assert Path(cfg["output_dir"], "pilot_metrics.csv").exists()
    assert Path(cfg["output_dir"], "pilot_summary.json").exists()
    assert Path(cfg["output_dir"], "pilot_report.md").exists()


def test_run_full_on_synthetic_csv(tmp_path):
    df1 = _synthetic_correctness(seed=2, instances=30, samples=16, regime="low_dispersion_strong_model", p=0.85)
    df2 = _synthetic_correctness(seed=3, instances=30, samples=16, regime="medium_dispersion_prompt_variation", p=0.65)
    df = pd.concat([df1, df2], ignore_index=True)
    in_csv = tmp_path / "full.csv"
    df.to_csv(in_csv, index=False)
    cfg = {
        "seed": 11,
        "output_dir": str(tmp_path / "out"),
        "full_correctness_csv": str(in_csv),
        "K_ref": 16,
        "K_est_values": [8],
        "N_values": [4, 8],
        "global_delta": 0.1,
        "B_boot": 5,
        "strict_majority": True,
    }
    summary = run_full(cfg, config_path=str(in_csv))
    assert summary["num_cells"] > 0
    for f in (
        "panel_a_cell_metrics.csv",
        "panel_a_bootstrap_metrics.csv",
        "panel_a_baseline_comparison.csv",
        "panel_a_nonvacuity_summary.csv",
        "panel_a_baseline_summary.csv",
        "panel_a_summary.json",
        "panel_a_report.md",
    ):
        assert Path(cfg["output_dir"], f).exists(), f


def test_run_pilot_rejects_missing_columns(tmp_path):
    bad = pd.DataFrame({"instance_id": ["i1"], "benchmark": ["b"], "protocol": ["A1"], "sample_id": [0]})
    in_csv = tmp_path / "bad.csv"
    bad.to_csv(in_csv, index=False)
    cfg = {
        "seed": 7,
        "output_dir": str(tmp_path / "out"),
        "pilot_correctness_csv": str(in_csv),
        "K_ref": 16,
        "K_est": 8,
        "N_values": [4],
        "global_delta": 0.1,
        "strict_majority": True,
    }
    with pytest.raises(ValueError):
        run_pilot(cfg)
