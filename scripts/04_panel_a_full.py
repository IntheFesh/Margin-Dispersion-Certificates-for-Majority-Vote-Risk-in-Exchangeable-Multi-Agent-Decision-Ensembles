"""Phase 1.3 full Panel A run: Protocols A1 & A2 over benchmarks x models.

Orchestrates the complete Panel A data collection and per-cell certificate
construction:

  * Protocol A1 (prompt-randomized self-consistency): one cell per
    ``(benchmark, model)``.
  * Protocol A2 (randomized model-family ensemble): one cell per
    ``(benchmark, model_pool)`` (the model is drawn per sample; the cell is
    keyed by the pool, not an individual model).

For every cell and pool it runs the corresponding protocol, assembles the
estimation success matrix, and issues the empirical certificate over the odd N
grid. The number of certificate cells ``C`` is the number of
``(protocol, benchmark, K_est)`` cells and is computed AT RUN TIME from the two
configs; the four-event Bonferroni per-cell budget ``delta_cell = delta_global
/ C`` and ``C`` itself are written to ``outputs/panel_a/panel_a_summary.json``
together with the per-cell certificate rows and provenance.

Requires a live vLLM server (GPU host); it does NOT run in Phase 0. Heavy
sibling imports are inside ``main()`` for Phase 0 importability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.utils.logging import JsonlLogger
from src.utils.provenance import capture, copy_config


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _success_matrix(jsonl_path: Path, instance_ids: list[str], K: int):
    """Assemble an (M, K) {0,1} success matrix from a protocol JSONL file."""
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
        raise RuntimeError(f"incomplete success matrix from {jsonl_path}")
    return matrix


def _enumerate_cells(cfg_a1: dict, cfg_a2: dict) -> list[dict]:
    """Enumerate all (protocol, benchmark, K_est) certificate cells.

    Protocol A1 cells are keyed by (benchmark, model); Protocol A2 cells are
    keyed by (benchmark, model_pool). The estimation sample budget K_est is
    ``K_full`` in both protocols. Returns one descriptor dict per cell; the
    length of this list is ``C`` (the Bonferroni denominator).
    """
    cells: list[dict] = []
    for benchmark in cfg_a1["benchmarks"]:
        for model in cfg_a1["models"]:
            cells.append(
                {
                    "protocol": "A1",
                    "benchmark": benchmark,
                    "model": model,
                    "K_est": int(cfg_a1["K_full"]),
                }
            )
    pool_names = [str(e["name"]) for e in cfg_a2["model_pool"]]
    for benchmark in cfg_a2["benchmarks"]:
        cells.append(
            {
                "protocol": "A2",
                "benchmark": benchmark,
                "model": "pool:" + "+".join(pool_names),
                "K_est": int(cfg_a2["K_full"]),
            }
        )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Panel A run (Protocols A1 & A2).")
    parser.add_argument("--config", default="configs/panel_a_protocol_a1.yaml", help="Protocol A1 config")
    parser.add_argument("--config_a2", default="configs/panel_a_protocol_a2.yaml", help="Protocol A2 config")
    parser.add_argument("--benchmarks_config", default="configs/benchmarks.yaml")
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--output_dir", default="outputs/panel_a")
    parser.add_argument("--seed", type=int, default=0xCAFE, help="global seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(Path("outputs/logs") / "04_panel_a_full.jsonl")

    config_path = Path(args.config)
    config_a2_path = Path(args.config_a2)
    bench_cfg_path = Path(args.benchmarks_config)
    models_cfg_path = Path(args.models_config)
    for p in (config_path, config_a2_path, bench_cfg_path, models_cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
    copy_config(str(config_path), str(output_dir))
    provenance = capture(
        args.seed,
        [str(config_path), str(config_a2_path), str(bench_cfg_path), str(models_cfg_path)],
    )

    # --- Heavy / sibling imports local to main() for Phase 0 importability. ---
    import numpy as np

    from src.certs.empirical import bonferroni_cell_budget, empirical_certificate
    from src.certs.refusal import classify_refusal
    from src.data.arc_challenge import load_arc_challenge
    from src.data.gsm8k import load_gsm8k
    from src.data.mmlu import load_mmlu_subset
    from src.protocols.pools import split_pools_by_instance
    from src.protocols.protocol_a1 import run_protocol_a1
    from src.protocols.protocol_a2 import run_protocol_a2

    cfg_a1 = _load_yaml(config_path)
    cfg_a2 = _load_yaml(config_a2_path)
    bench_cfg = _load_yaml(bench_cfg_path)
    models_cfg = _load_yaml(models_cfg_path)

    seed = int(args.seed)
    delta_global = float(cfg_a1["delta_global"])
    N_values = [int(n) for n in cfg_a1["N_values"]]
    for N in N_values:
        if N % 2 == 0:
            raise ValueError(f"N_values must be ODD (strict success-majority), got {N}")

    # --- C is computed AT RUN TIME from the enumerated cells. -----------------
    cells = _enumerate_cells(cfg_a1, cfg_a2)
    C = len(cells)
    delta_cell = bonferroni_cell_budget(delta_global, C)
    logger.event("panel_a.cells", C=C, delta_global=delta_global, delta_cell=delta_cell)

    loaders = {
        "arc_challenge": load_arc_challenge,
        "gsm8k": load_gsm8k,
        "mmlu_subset": load_mmlu_subset,
    }

    cell_rows: list[dict] = []
    for cell in cells:
        protocol = cell["protocol"]
        benchmark = cell["benchmark"]
        cfg = cfg_a1 if protocol == "A1" else cfg_a2
        K_full = int(cfg["K_full"])
        K_ref = int(cfg["K_ref"])
        oracle_fraction = float(cfg["oracle_fraction"])

        items = loaders[benchmark](bench_cfg[benchmark], seed)
        instance_ids = [it["instance_id"] for it in items]
        items_by_id = {it["instance_id"]: it for it in items}
        estimation_ids, oracle_ids = split_pools_by_instance(instance_ids, oracle_fraction, seed)

        cell_dir = output_dir / f"{protocol}_{benchmark}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        est_path = cell_dir / "estimation.jsonl"
        orc_path = cell_dir / "oracle.jsonl"

        cell_config: dict = {
            "items": items_by_id,
            "temperature": float(cfg["temperature"]),
            "top_p": float(cfg["top_p"]),
            "max_tokens": int(cfg["max_tokens"]),
            "log_path": str(cell_dir / "protocol.jsonl"),
            "extraction_log_path": str(cell_dir / "extraction.jsonl"),
        }

        if protocol == "A1":
            model = cell["model"]
            vllm_cfg = models_cfg[model].get("vllm", {})
            cell_config["base_url"] = f"http://localhost:{int(vllm_cfg.get('port', 8000))}/v1"
            cell_config["served_model_name"] = models_cfg[model].get("hf_path", model)
            run_protocol_a1(benchmark, model, estimation_ids, K_full, "estimation", cell_config, est_path, seed)
            run_protocol_a1(benchmark, model, oracle_ids, K_ref, "oracle", cell_config, orc_path, seed)
        else:  # Protocol A2: serial model-swap over the pool.
            cell_config["model_pool"] = cfg_a2["model_pool"]
            serving: dict = {}
            for entry in cfg_a2["model_pool"]:
                name = str(entry["name"])
                vllm_cfg = models_cfg[name].get("vllm", {})
                serving[name] = {
                    "model_path": models_cfg[name].get("hf_path", name),
                    "port": int(vllm_cfg.get("port", 8000)),
                    "gpu_memory_utilization": float(vllm_cfg.get("gpu_memory_utilization", 0.90)),
                    "max_model_len": int(vllm_cfg.get("max_model_len", 4096)),
                    "enforce_eager": bool(vllm_cfg.get("enforce_eager", False)),
                }
            cell_config["serving"] = serving
            run_protocol_a2(benchmark, "", estimation_ids, K_full, "estimation", cell_config, est_path, seed)
            run_protocol_a2(benchmark, "", oracle_ids, K_ref, "oracle", cell_config, orc_path, seed)

        est_matrix = _success_matrix(est_path, estimation_ids, K_full)
        for N in N_values:
            cert = empirical_certificate(np.asarray(est_matrix), N, delta_cell, use_BA=True)
            refusal = classify_refusal(cert, epsilon=0.2)
            cell_rows.append(
                {
                    "protocol": protocol,
                    "benchmark": benchmark,
                    "model": cell["model"],
                    "K_est": cell["K_est"],
                    "M_estimation": len(estimation_ids),
                    "M_oracle": len(oracle_ids),
                    "N": N,
                    "delta_global": delta_global,
                    "delta_cell": delta_cell,
                    "C_cells": C,
                    "alpha_bar_hat": cert["alpha_bar_hat"],
                    "F_hat": cert["F_hat"],
                    "L_alpha": cert["L_alpha"],
                    "U_alpha": cert["U_alpha"],
                    "U_F": cert["U_F"],
                    "m_L": cert["m_L"],
                    "m_beta_L": cert["m_beta_L"],
                    "R_N_cert": cert["R_N_cert"],
                    "Q_N_cert": cert["Q_N_cert"],
                    "R_N_BA_cert": cert["R_N_BA_cert"],
                    "refusal_mode": refusal["mode"],
                    "refusal_sub_mode": refusal["sub_mode"],
                }
            )

    summary = {
        "C": C,
        "C_cells": C,
        "delta_global": delta_global,
        "delta_cell": delta_cell,
        "N_values": N_values,
        "cells": cells,
        "cell_certificates": cell_rows,
        "provenance": provenance,
    }
    summary_path = output_dir / "panel_a_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.event("panel_a.done", C=C, delta_cell=delta_cell, n_cell_rows=len(cell_rows), summary_json=str(summary_path))
    print(f"Full Panel A complete: C={C} cells, delta_cell={delta_cell:.3e}; wrote {summary_path}")


if __name__ == "__main__":
    main()
