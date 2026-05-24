"""Phase 2 analyses: run Analyses 1-7 over the Panel A outputs.

Consumes ``outputs/panel_a/panel_a_summary.json`` (the per-cell certificate
rows produced by ``04_panel_a_full.py``) plus the per-cell estimation/oracle
JSONL matrices, and runs the seven analyses implemented under
``src.analysis``:

  1. instance-level bootstrap of certificate coverage vs R_N^MC;
  2. non-vacuity rates of the issued certificate;
  3-7. (nonvacuity by group, refusal taxonomy, budget curves, sharpness, and
        conservativeness decomposition) -- dispatched by module name so this
        orchestrator stays decoupled from each analysis's exact signature.

Each analysis writes one CSV under ``outputs/analyses/`` (e.g.
``analysis_1_bootstrap.csv``, ..., ``analysis_7_conservativeness.csv``). The
``src.analysis.*`` imports are LOCAL to ``main()`` so this module compiles and
imports in Phase 0 even before every analysis module is present; a missing
analysis module raises a clear error at run time (no silent skip).

No GPU/network: this is a pure post-processing step. It does NOT run in
Phase 0.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable

from src.utils.logging import JsonlLogger
from src.utils.provenance import capture, copy_config

# (module suffix, output csv stem). a1/a2 have known signatures and are wired
# explicitly; a3..a7 are dispatched generically by their public callable.
_ANALYSIS_MODULES: tuple[tuple[str, str], ...] = (
    ("a1_mc_bootstrap", "analysis_1_bootstrap"),
    ("a2_nonvacuity", "analysis_2_nonvacuity"),
    ("a3_refusal_taxonomy", "analysis_3_refusal_taxonomy"),
    ("a4_bidirectional", "analysis_4_bidirectional"),
    ("a5_budget_curves", "analysis_5_budget_curves"),
    ("a6_sharpness", "analysis_6_sharpness"),
    ("a7_conservativeness", "analysis_7_conservativeness"),
)


def _load_summary(panel_a_dir: Path) -> dict:
    summary_path = panel_a_dir / "panel_a_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Panel A summary not found: {summary_path}; run 04_panel_a_full.py first"
        )
    with summary_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _success_matrix(jsonl_path: Path):
    """Assemble an (M, K) {0,1} success matrix from a protocol JSONL file.

    Rows are ordered by the record ``m`` index, columns by ``k``.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"protocol records not found: {jsonl_path}")
    import numpy as np

    rows: dict[int, dict[int, int]] = {}
    max_k = -1
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m, k = int(rec["m"]), int(rec["k"])
            rows.setdefault(m, {})[k] = int(rec["success_indicator"])
            max_k = max(max_k, k)
    if not rows:
        raise RuntimeError(f"no records in {jsonl_path}")
    M = max(rows) + 1
    K = max_k + 1
    matrix = np.full((M, K), -1, dtype=int)
    for m, ks in rows.items():
        for k, v in ks.items():
            matrix[m, k] = v
    if (matrix < 0).any():
        raise RuntimeError(f"incomplete success matrix from {jsonl_path}")
    return matrix


def _find_callable(module, preferred: str) -> Callable:
    """Return the public analysis entry point of ``module``.

    Prefers ``preferred``; else the single public callable named ``analysis_*``
    or listed in ``module.__all__``. Raises if none can be resolved (no silent
    fallback to a no-op).
    """
    if hasattr(module, preferred):
        return getattr(module, preferred)
    exported = list(getattr(module, "__all__", []))
    candidates = [
        name
        for name in exported
        if callable(getattr(module, name, None)) and name.startswith("analysis")
    ]
    if len(candidates) == 1:
        return getattr(module, candidates[0])
    raise AttributeError(
        f"could not resolve analysis entry point in {module.__name__!r}; "
        f"looked for {preferred!r}, __all__={exported}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Analyses 1-7 over Panel A outputs.")
    parser.add_argument("--config", default="configs/pre_registration.yaml")
    parser.add_argument("--panel_a_dir", default="outputs/panel_a")
    parser.add_argument("--output_dir", default="outputs/analyses")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Analysis-1 bootstrap replicates")
    parser.add_argument("--seed", type=int, default=0xCAFE, help="global seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(Path("outputs/logs") / "05_run_analyses.jsonl")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    copy_config(str(config_path), str(output_dir))
    provenance = capture(args.seed, [str(config_path)])

    # --- Local imports: pandas + the analysis package (Phase 0 importability).
    import numpy as np
    import pandas as pd

    panel_a_dir = Path(args.panel_a_dir)
    summary = _load_summary(panel_a_dir)
    cells_df = pd.DataFrame(summary["cell_certificates"])
    delta_cell = float(summary["delta_cell"])
    N_values = [int(n) for n in summary["N_values"]]
    seed = int(args.seed)

    written: list[str] = []

    def _resolve(module_suffix: str, preferred: str) -> Callable:
        module = importlib.import_module(f"src.analysis.{module_suffix}")
        return _find_callable(module, preferred)

    # --- Analysis 1: instance bootstrap (needs the success matrices). --------
    a1 = _resolve("a1_mc_bootstrap", "analysis_1_bootstrap")
    a1_rows: list[dict] = []
    for cell in summary["cells"]:
        cell_dir = panel_a_dir / f"{cell['protocol']}_{cell['benchmark']}"
        est = _success_matrix(cell_dir / "estimation.jsonl")
        orc = _success_matrix(cell_dir / "oracle.jsonl")
        for N in N_values:
            res = a1(np.asarray(est), np.asarray(orc), N, delta_cell, B_boot=int(args.bootstrap), seed=seed)
            a1_rows.append(
                {
                    "protocol": cell["protocol"],
                    "benchmark": cell["benchmark"],
                    "model": cell["model"],
                    "N": N,
                    "coverage_rate": res["coverage_rate"],
                    "R_cert_ci_lo": res["R_cert_ci"][0],
                    "R_cert_ci_hi": res["R_cert_ci"][1],
                    "n_scored": res["n_scored"],
                    "n_refused": res["n_refused"],
                }
            )
    a1_path = output_dir / "analysis_1_bootstrap.csv"
    pd.DataFrame(a1_rows).to_csv(a1_path, index=False)
    written.append(str(a1_path))
    logger.event("analysis.done", analysis="a1", csv=str(a1_path), n_rows=len(a1_rows))

    # --- Analysis 2: non-vacuity rates (operates on the cells table). --------
    a2 = _resolve("a2_nonvacuity", "nonvacuity_rates")
    a2_df = a2(cells_df)
    a2_path = output_dir / "analysis_2_nonvacuity.csv"
    a2_df.to_csv(a2_path, index=False)
    written.append(str(a2_path))
    logger.event("analysis.done", analysis="a2", csv=str(a2_path), n_rows=int(len(a2_df)))

    # --- Analyses 3-7: dispatched generically over the cells table. ----------
    # Each is invoked with the per-cell certificate DataFrame; whatever frame it
    # returns is written verbatim. Signature coupling is intentionally avoided.
    for module_suffix, preferred in _ANALYSIS_MODULES[2:]:
        fn = _resolve(module_suffix, preferred)
        result = fn(cells_df)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"analysis {module_suffix} must return a DataFrame, got {type(result)!r}"
            )
        idx = module_suffix.split("_", 1)[0].lstrip("a")
        out_path = output_dir / f"analysis_{idx}_{module_suffix.split('_', 1)[1]}.csv"
        result.to_csv(out_path, index=False)
        written.append(str(out_path))
        logger.event("analysis.done", analysis=module_suffix, csv=str(out_path), n_rows=int(len(result)))

    logger.event("analyses.complete", n_csv=len(written), output_dir=str(output_dir))
    print(f"Analyses 1-7 complete; wrote {len(written)} CSVs to {output_dir}:")
    for p in written:
        print(f"  {p}")
    _ = provenance  # provenance already persisted via copy_config + capture log


if __name__ == "__main__":
    main()
