"""Phase 1.1 engineering pilot: M=50 smoke test (single model, single benchmark).

This is an ENGINEERING SMOKE TEST ONLY. Its sole purpose is to exercise the
end-to-end plumbing -- dataset load -> Protocol A1 generation -> answer
extraction/labelling -> empirical certificate -> JSON summary -- on a tiny
M=50 slice so wiring bugs surface before any expensive run.

CERTIFICATE VALUES FROM M < 200 ARE NEVER REPORTED AS FINDINGS. The smallest
scale at which any certificate number is treated as a result is the statistical
pilot (``03_statistical_pilot.py``, M=500). Anything emitted here is plumbing
diagnostics, explicitly tagged ``"smoke_test": true`` in the output JSON.

Requires a live vLLM server (GPU host); it does NOT run in Phase 0. Heavy
sibling imports (data loaders, protocol, vLLM client) are performed inside
``main()`` so this module imports and compiles without those dependencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.utils.logging import JsonlLogger
from src.utils.provenance import capture, copy_config

# Hard ceiling for this smoke test. Certificate values below this M are never
# reported as findings (see module docstring).
_SMOKE_M: int = 50
_REPORTABLE_M_FLOOR: int = 200


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineering pilot (M=50 smoke test).")
    parser.add_argument("--config", default="configs/panel_a_protocol_a1.yaml")
    parser.add_argument("--benchmarks_config", default="configs/benchmarks.yaml")
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--output_dir", default="outputs/pilots")
    parser.add_argument("--benchmark", default="arc_challenge", help="single benchmark to smoke-test")
    parser.add_argument("--model", default=None, help="single model (default: first in config)")
    parser.add_argument("--seed", type=int, default=0xCAFE, help="global seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(Path("outputs/logs") / "02_engineering_pilot.jsonl")

    config_path = Path(args.config)
    bench_cfg_path = Path(args.benchmarks_config)
    models_cfg_path = Path(args.models_config)
    for p in (config_path, bench_cfg_path, models_cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
    copy_config(str(config_path), str(output_dir))
    provenance = capture(args.seed, [str(config_path), str(bench_cfg_path), str(models_cfg_path)])

    # --- Heavy / sibling imports are local so the module imports in Phase 0. ---
    import numpy as np

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
    # M=50 smoke slice: cap the benchmark config to the smoke size.
    bcfg = dict(bench_cfg[benchmark])
    if "max_instances" in bcfg:
        bcfg["max_instances"] = min(int(bcfg["max_instances"]), _SMOKE_M)

    with logger.timed("pilot.load", benchmark=benchmark, M=_SMOKE_M) as ctx:
        items = loaders[benchmark](bcfg, seed)[: _SMOKE_M]
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
        "log_path": str(output_dir / "protocol_a1_pilot.jsonl"),
        "extraction_log_path": str(output_dir / "extraction_pilot.jsonl"),
    }

    est_path = output_dir / f"{benchmark}_{model}_estimation.jsonl"
    orc_path = output_dir / f"{benchmark}_{model}_oracle.jsonl"
    run_protocol_a1(benchmark, model, estimation_ids, K_full, "estimation", cell_config, est_path, seed)
    run_protocol_a1(benchmark, model, oracle_ids, K_ref, "oracle", cell_config, orc_path, seed)

    # Build the estimation success matrix (rows = instances, cols = samples).
    est_matrix = _success_matrix(est_path, estimation_ids, K_full)

    # Bonferroni budget over this pilot's single cell across the N grid.
    C = len(N_values)
    delta_cell = bonferroni_cell_budget(delta_global, C)

    diagnostics: list[dict] = []
    for N in N_values:
        cert = empirical_certificate(np.asarray(est_matrix), N, delta_cell, use_BA=False)
        refusal = classify_refusal(cert, epsilon=0.2)
        diagnostics.append({"N": N, **{k: cert[k] for k in ("R_N_cert", "Q_N_cert", "m_L", "reason")}, **refusal})

    summary = {
        "smoke_test": True,
        "reportable": False,
        "note": (
            f"M={_SMOKE_M} < {_REPORTABLE_M_FLOOR}: certificate values are PLUMBING "
            f"DIAGNOSTICS ONLY and must never be reported as findings."
        ),
        "benchmark": benchmark,
        "model": model,
        "M_smoke": _SMOKE_M,
        "M_estimation": len(estimation_ids),
        "M_oracle": len(oracle_ids),
        "delta_global": delta_global,
        "delta_cell": delta_cell,
        "N_values": N_values,
        "diagnostics": diagnostics,
        "provenance": provenance,
    }
    summary_path = output_dir / "engineering_pilot_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.event("pilot.done", benchmark=benchmark, model=model, M=_SMOKE_M, summary_json=str(summary_path))
    print(f"Engineering pilot (SMOKE, M={_SMOKE_M}) complete; wrote {summary_path}")
    print("  NOTE: M<200 -> certificate values are diagnostics, NOT findings.")


def _success_matrix(jsonl_path: Path, instance_ids: list[str], K: int) -> list[list[int]]:
    """Assemble an (M, K) {0,1} success matrix from a Protocol A1 JSONL file.

    Raises ``FileNotFoundError`` if the records file is absent (no silent
    fabrication) and ``RuntimeError`` if any (instance, k) cell is missing.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"protocol records not found: {jsonl_path}")
    row_of = {iid: m for m, iid in enumerate(instance_ids)}
    matrix = [[None] * K for _ in instance_ids]
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iid = rec["instance_id"]
            if iid not in row_of:
                continue
            matrix[row_of[iid]][int(rec["k"])] = int(rec["success_indicator"])
    for m, row in enumerate(matrix):
        if any(v is None for v in row):
            raise RuntimeError(f"incomplete success matrix: instance row {m} has missing samples")
    return matrix


if __name__ == "__main__":
    main()
