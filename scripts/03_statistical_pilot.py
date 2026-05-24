"""Phase 1.1 statistical pilot: M=500, single protocol / single benchmark.

This is the SMALLEST scale at which certificate values are reported as
findings. Unlike the engineering pilot (``02_engineering_pilot.py``, M=50,
diagnostics only), the M=500 statistical pilot exercises a full single-cell
run -- dataset load -> Protocol A1 -> extraction/labelling -> empirical
certificate over the odd N grid -> Analysis-1 instance-bootstrap of coverage --
at the reporting scale fixed in the pre-registration.

It runs a SINGLE (protocol=A1, benchmark, model) cell. The Bonferroni budget
here is the single-cell budget over the N grid; the full multi-cell budget is
established in ``04_panel_a_full.py``.

Requires a live vLLM server (GPU host); it does NOT run in Phase 0. Heavy
sibling imports are inside ``main()`` so this module imports and compiles
without those dependencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.utils.logging import JsonlLogger
from src.utils.provenance import capture, copy_config

# Reporting floor: this pilot is the smallest reportable scale.
_REPORTABLE_M_FLOOR: int = 200


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _success_matrix(jsonl_path: Path, instance_ids: list[str], K: int):
    """Assemble an (M, K) {0,1} success matrix from a Protocol A1 JSONL file."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"protocol records not found: {jsonl_path}")
    import numpy as np

    row_of = {iid: m for m, iid in enumerate(instance_ids)}
    matrix = np.full((len(instance_ids), K), -1, dtype=int)
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iid = rec["instance_id"]
            if iid not in row_of:
                continue
            matrix[row_of[iid], int(rec["k"])] = int(rec["success_indicator"])
    if (matrix < 0).any():
        raise RuntimeError("incomplete success matrix: some (instance, k) samples are missing")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical pilot (M=500, single cell; reportable).")
    parser.add_argument("--config", default="configs/panel_a_protocol_a1.yaml")
    parser.add_argument("--benchmarks_config", default="configs/benchmarks.yaml")
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--output_dir", default="outputs/pilots")
    parser.add_argument("--benchmark", default="arc_challenge")
    parser.add_argument("--model", default=None, help="single model (default: first in config)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Analysis-1 bootstrap replicates")
    parser.add_argument("--seed", type=int, default=0xCAFE, help="global seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(Path("outputs/logs") / "03_statistical_pilot.jsonl")

    config_path = Path(args.config)
    bench_cfg_path = Path(args.benchmarks_config)
    models_cfg_path = Path(args.models_config)
    for p in (config_path, bench_cfg_path, models_cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
    copy_config(str(config_path), str(output_dir))
    provenance = capture(args.seed, [str(config_path), str(bench_cfg_path), str(models_cfg_path)])

    # --- Heavy / sibling imports local to main() for Phase 0 importability. ---
    import numpy as np

    from src.analysis.a1_mc_bootstrap import analysis_1_bootstrap
    from src.certs.empirical import bonferroni_cell_budget, empirical_certificate
    from src.certs.refusal import classify_refusal
    from src.data.arc_challenge import load_arc_challenge
    from src.data.gsm8k import load_gsm8k
    from src.data.mmlu import load_mmlu_subset
    from src.protocols.pools import split_pools_by_instance
    from src.protocols.protocol_a1 import run_protocol_a1

    cfg = _load_yaml(config_path)
    bench_cfg = _load_yaml(bench_cfg_path)
    models_cfg = _load_yaml(models_cfg_path)

    benchmark = args.benchmark
    if benchmark not in ("arc_challenge", "gsm8k", "mmlu_subset"):
        raise ValueError(f"unknown benchmark: {benchmark}")
    model = args.model or cfg["models"][0]
    if model not in models_cfg:
        raise ValueError(f"model {model!r} not found in {models_cfg_path}")

    seed = int(args.seed)
    M_full = int(cfg["M_full"])
    if M_full < _REPORTABLE_M_FLOOR:
        raise ValueError(
            f"statistical pilot M_full={M_full} is below the reporting floor "
            f"{_REPORTABLE_M_FLOOR}; use the engineering pilot for smaller M"
        )
    K_full = int(cfg["K_full"])
    K_ref = int(cfg["K_ref"])
    oracle_fraction = float(cfg["oracle_fraction"])
    delta_global = float(cfg["delta_global"])
    N_values = [int(n) for n in cfg["N_values"]]
    for N in N_values:
        if N % 2 == 0:
            raise ValueError(f"N_values must be ODD (strict success-majority), got {N}")

    loaders = {
        "arc_challenge": load_arc_challenge,
        "gsm8k": load_gsm8k,
        "mmlu_subset": load_mmlu_subset,
    }
    with logger.timed("statistical_pilot.load", benchmark=benchmark, M=M_full) as ctx:
        items = loaders[benchmark](bench_cfg[benchmark], seed)
        ctx["n_loaded"] = len(items)

    instance_ids = [it["instance_id"] for it in items]
    items_by_id = {it["instance_id"]: it for it in items}
    estimation_ids, oracle_ids = split_pools_by_instance(instance_ids, oracle_fraction, seed)

    vllm_cfg = models_cfg[model].get("vllm", {})
    base_url = f"http://localhost:{int(vllm_cfg.get('port', 8000))}/v1"
    cell_config = {
        "items": items_by_id,
        "base_url": base_url,
        "served_model_name": models_cfg[model].get("hf_path", model),
        "temperature": float(cfg["temperature"]),
        "top_p": float(cfg["top_p"]),
        "max_tokens": int(cfg["max_tokens"]),
        "log_path": str(output_dir / "protocol_a1_statistical.jsonl"),
        "extraction_log_path": str(output_dir / "extraction_statistical.jsonl"),
    }

    est_path = output_dir / f"{benchmark}_{model}_estimation.jsonl"
    orc_path = output_dir / f"{benchmark}_{model}_oracle.jsonl"
    run_protocol_a1(benchmark, model, estimation_ids, K_full, "estimation", cell_config, est_path, seed)
    run_protocol_a1(benchmark, model, oracle_ids, K_ref, "oracle", cell_config, orc_path, seed)

    est_matrix = _success_matrix(est_path, estimation_ids, K_full)
    orc_matrix = _success_matrix(orc_path, oracle_ids, K_ref)

    C = len(N_values)  # single (protocol, benchmark, K_est) cell across the N grid
    delta_cell = bonferroni_cell_budget(delta_global, C)

    per_N: list[dict] = []
    for N in N_values:
        cert = empirical_certificate(np.asarray(est_matrix), N, delta_cell, use_BA=True)
        refusal = classify_refusal(cert, epsilon=0.2)
        boot = analysis_1_bootstrap(
            np.asarray(est_matrix), np.asarray(orc_matrix), N, delta_cell,
            B_boot=int(args.bootstrap), seed=seed,
        )
        per_N.append(
            {
                "N": N,
                "R_N_cert": cert["R_N_cert"],
                "Q_N_cert": cert["Q_N_cert"],
                "R_N_BA_cert": cert["R_N_BA_cert"],
                "alpha_bar_hat": cert["alpha_bar_hat"],
                "F_hat": cert["F_hat"],
                "refusal_mode": refusal["mode"],
                "refusal_sub_mode": refusal["sub_mode"],
                "coverage_rate": boot["coverage_rate"],
                "R_cert_ci": boot["R_cert_ci"],
                "n_scored": boot["n_scored"],
                "n_refused": boot["n_refused"],
            }
        )

    summary = {
        "smoke_test": False,
        "reportable": True,
        "protocol": "A1",
        "benchmark": benchmark,
        "model": model,
        "M_full": M_full,
        "M_estimation": len(estimation_ids),
        "M_oracle": len(oracle_ids),
        "delta_global": delta_global,
        "delta_cell": delta_cell,
        "C_cells": C,
        "N_values": N_values,
        "per_N": per_N,
        "provenance": provenance,
    }
    summary_path = output_dir / "statistical_pilot_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.event("statistical_pilot.done", benchmark=benchmark, model=model, M=M_full, summary_json=str(summary_path))
    print(f"Statistical pilot (M={M_full}, protocol A1) complete; wrote {summary_path}")


if __name__ == "__main__":
    main()
