"""Protocol A2: randomized model-family ensemble (serial model-swap policy).

Protocol A2 builds ensemble diversity by drawing the *generating model* for
each sample from a weighted pool, rather than (as in Protocol A1) varying the
surface prompt for a single fixed model. The ``model`` argument is therefore
IGNORED: the per-sample model is drawn from ``cell_config["model_pool"]``, a
list of ``{"name": str, "weight": float}`` entries.

SERIAL MODEL-SWAP POLICY (section 6.12 GPU policy)
--------------------------------------------------
The target GPU cannot host all three pool models simultaneously, so this
protocol executes in three deterministic stages:

  (1) ASSIGNMENT (no GPU). Compute the model-assignment vector for *all*
      ``(instance, k)`` pairs up front, purely from the global seed. For each
      pair we draw the assigned model name via
      ``rng_for(seed, benchmark, instance_id, k, "model_assign").choice(
      names, p=weights)``. Because the draw is keyed on the immutable
      ``(instance_id, k)`` tuple, the assignment is reproducible and does not
      depend on iteration order or on which model server happens to be up.

  (2) PER-MODEL BATCHING (one server at a time). For each model in the pool,
      in pool order: start its vLLM server via
      :mod:`src.llm.server_lifecycle`, run *every* request assigned to that
      model in step (1) against the live server (extract + label + record),
      then stop the server before moving to the next model. At most one model
      is resident on the GPU at any moment.

  (3) AGGREGATION. All per-model records carry the same unified
      :class:`PanelARecord` schema (with ``model`` set to the *drawn* model,
      not the ignored argument) and are written to a single JSONL file
      ``out_path``, ordered by ``(m, k)`` so the file layout matches Protocol
      A1's one-line-per-``(m, k)`` contract.

Determinism: the per-``(instance, k)`` generation seed is
``derive_seed(seed, benchmark, drawn_model, instance_id, k, "gen")`` and is
forwarded to the sampler. The model-assignment RNG is independent of the
generation RNG (different key suffix), so changing decoding never perturbs the
assignment and vice versa.

Pool semantics: identical to Protocol A1. This function writes records for a
SINGLE pool (``estimation`` or ``oracle``); the two pools are disjoint at the
instance level (see :mod:`src.protocols.pools`). The oracle pool only ever
feeds the R_N^MC reference and never certificate construction.

No silent skips: an invalid parse is *kept* as a record with
``invalid_parse=True``, ``success_indicator=0`` and the raw completion
preserved (invalid-parse-is-failure bridge lemma). The heavy
``openai``/``vllm`` dependencies and ``src.llm.server_lifecycle`` are imported
LAZILY inside function bodies, so this module imports cleanly in Phase 0 where
they are unavailable; the generation/serving code runs only when invoked with a
live serving environment.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.llm.extraction import extract_answer
from src.llm.labeling import label
from src.llm.prompts import NUM_VARIATIONS, format_prompt
from src.utils.logging import JsonlLogger
from src.utils.schema import PanelARecord, validate_records
from src.utils.seeds import derive_seed, rng_for


def _parse_model_pool(cell_config: dict) -> tuple[list[str], np.ndarray]:
    """Return (names, normalized_weights) from ``cell_config['model_pool']``.

    The pool is a list of ``{"name", "weight"}`` dicts. Weights must be
    non-negative and sum to a positive value; they are normalized to a
    probability vector. Raises ``ValueError`` on a malformed pool (no silent
    fallback to a uniform pool).
    """
    pool = cell_config.get("model_pool")
    if not pool:
        raise ValueError(
            "Protocol A2 requires cell_config['model_pool'] = "
            "[{'name': ..., 'weight': ...}, ...]; none supplied"
        )
    names: list[str] = []
    weights: list[float] = []
    for entry in pool:
        name = entry.get("name")
        weight = entry.get("weight")
        if name is None or weight is None:
            raise ValueError(f"model_pool entry missing 'name'/'weight': {entry!r}")
        if float(weight) < 0.0:
            raise ValueError(f"model_pool weight must be >= 0, got {weight!r}")
        names.append(str(name))
        weights.append(float(weight))
    if len(set(names)) != len(names):
        raise ValueError(f"model_pool names must be unique, got {names}")
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("model_pool weights must sum to a positive value")
    return names, w / total


def _instance_index(instance_ids: list[str]) -> dict[str, int]:
    """Map each instance id to its row index ``m`` (stable, 0-based)."""
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("instance_ids must be unique")
    return {iid: m for m, iid in enumerate(instance_ids)}


def compute_model_assignment(
    benchmark: str,
    instance_ids: list[str],
    K: int,
    names: list[str],
    weights: np.ndarray,
    seed: int,
) -> dict[tuple[str, int], str]:
    """Stage (1): deterministic model-assignment vector for all (instance, k).

    For each ``(instance_id, k)`` pair, draw the assigned model name with
    ``rng_for(seed, benchmark, instance_id, k, "model_assign").choice(
    names, p=weights)``. Keyed on the immutable pair, so the assignment is
    independent of iteration order and of which server is currently running.

    Returns
    -------
    dict[tuple[str, int], str]
        Mapping ``(instance_id, k) -> model_name``.
    """
    assignment: dict[tuple[str, int], str] = {}
    p = np.asarray(weights, dtype=float)
    for instance_id in instance_ids:
        for k in range(K):
            rng = rng_for(seed, benchmark, instance_id, k, "model_assign")
            assignment[(instance_id, k)] = str(rng.choice(names, p=p))
    return assignment


def run_protocol_a2(
    benchmark: str,
    model: str,
    instance_ids: list[str],
    K: int,
    pool_type: Literal["estimation", "oracle"],
    cell_config: dict,
    out_path: Path,
    seed: int,
) -> None:
    """Run Protocol A2 for one ``(benchmark, model-pool)`` cell and one pool.

    Same signature as :func:`src.protocols.protocol_a1.run_protocol_a1`, but
    ``model`` is IGNORED: each sample's generating model is drawn from
    ``cell_config["model_pool"]``. Executes the serial model-swap policy
    documented in the module docstring and writes one validated
    :class:`PanelARecord` JSONL line per ``(m, k)`` to ``out_path``.

    Parameters
    ----------
    benchmark:
        One of ``"arc_challenge"``, ``"gsm8k"``, ``"mmlu_subset"``.
    model:
        IGNORED (kept for signature parity with Protocol A1). The per-sample
        model is drawn from the pool and stored on each record.
    instance_ids:
        Ordered, unique instance ids for this pool; their position defines the
        row index ``m``.
    K:
        Number of ensemble samples per instance.
    pool_type:
        ``"estimation"`` or ``"oracle"``; written verbatim to each record's
        ``pool`` field. The oracle pool never feeds certificate construction.
    cell_config:
        Run configuration. Must supply ``model_pool`` (list of
        ``{"name", "weight"}``), the items dict (``cell_config["items"]``
        mapping instance_id -> instance dict), per-model serving parameters
        (``cell_config["serving"]`` mapping model name -> dict with
        ``model_path``/``port``/etc.), and decoding params ``temperature``,
        ``top_p``, ``max_tokens``.
    out_path:
        Destination JSONL file (one validated record per line, ordered by
        ``(m, k)``).
    seed:
        Global seed; per-``(instance, k)`` assignment and generation seeds are
        derived from it.

    Raises
    ------
    ValueError
        On unknown benchmark, duplicate/missing instances, or invalid config.
    RuntimeError
        On any generation/serving failure (propagated; no silent fallback).
    """
    if benchmark not in ("arc_challenge", "gsm8k", "mmlu_subset"):
        raise ValueError(f"unknown benchmark: {benchmark}")
    if pool_type not in ("estimation", "oracle"):
        raise ValueError(f"pool_type must be 'estimation' or 'oracle', got {pool_type}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    items: dict[str, dict] = cell_config.get("items") or {}
    if not items:
        raise ValueError("cell_config['items'] must map instance_id -> instance dict")

    names, weights = _parse_model_pool(cell_config)
    temperature = float(cell_config.get("temperature", 0.7))
    top_p = float(cell_config.get("top_p", 0.95))
    max_tokens = int(cell_config.get("max_tokens", 512))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(cell_config.get("log_path", out_path.parent / "protocol_a2.jsonl"))
    extract_log = cell_config.get("extraction_log_path")

    row_of = _instance_index(instance_ids)

    # --- Stage (1): deterministic assignment for all (instance, k), no GPU. --
    assignment = compute_model_assignment(
        benchmark, instance_ids, K, names, weights, seed
    )
    logger.event(
        "protocol_a2.assignment",
        benchmark=benchmark,
        pool=pool_type,
        n_instances=len(instance_ids),
        K=K,
        model_pool=names,
        weights=[float(x) for x in weights],
        n_pairs=len(assignment),
    )

    # Accumulate records keyed by (m, k) so the final file is ordered, even
    # though they are produced model-by-model out of (m, k) order.
    records_by_mk: dict[tuple[int, int], dict] = {}

    # --- Stage (2): serial per-model batching (one server resident at a time).
    serving_cfg: dict[str, dict] = cell_config.get("serving") or {}
    for model_name in names:
        # All (instance, k) pairs assigned to this model in stage (1).
        assigned_pairs = [
            (iid, k) for (iid, k), mn in assignment.items() if mn == model_name
        ]
        if not assigned_pairs:
            logger.event(
                "protocol_a2.model_skip",
                model=model_name,
                pool=pool_type,
                reason="no requests assigned by deterministic draw",
            )
            continue
        # Stable ordering of this model's batch by (m, k).
        assigned_pairs.sort(key=lambda p: (row_of[p[0]], p[1]))

        client, server_proc = _start_model_server(cell_config, serving_cfg, model_name, seed)
        logger.event(
            "protocol_a2.model_start",
            model=model_name,
            pool=pool_type,
            n_requests=len(assigned_pairs),
        )
        try:
            for instance_id, k in assigned_pairs:
                m = row_of[instance_id]
                item = items.get(instance_id)
                if item is None:
                    raise ValueError(
                        f"no item dict supplied for instance_id={instance_id!r}"
                    )
                gold = item["gold_answer"]

                # Generation seed keyed on the DRAWN model (independent of the
                # assignment RNG). Prompt variation follows the A1 convention
                # (sample-k uses variation k mod K_var) for cross-protocol
                # comparability.
                gen_seed = derive_seed(
                    seed, benchmark, model_name, instance_id, k, "gen"
                )
                prompt, variation_id = format_prompt(
                    item, benchmark, k % NUM_VARIATIONS
                )

                t0 = time.time()
                completions = client.generate(
                    prompt=prompt,
                    n_samples=1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=gen_seed,
                    top_p=top_p,
                )
                latency_ms = (time.time() - t0) * 1000.0
                raw_completion = completions[0] if completions else ""

                extracted, invalid_parse = extract_answer(
                    raw_completion,
                    benchmark,
                    log_path=extract_log,
                    instance_id=instance_id,
                )
                # invalid-parse-is-failure: keep the row, force success 0.
                success = 0 if invalid_parse else int(label(extracted, gold, benchmark))

                record = {
                    "instance_id": instance_id,
                    "pool": pool_type,
                    "m": m,
                    "k": k,
                    "prompt_variation_id": variation_id,
                    "prompt": prompt,
                    "raw_completion": raw_completion,
                    "extracted_answer": extracted,
                    "gold_answer": str(gold),
                    "success_indicator": success,
                    "invalid_parse": bool(invalid_parse),
                    "model": model_name,  # the DRAWN model, not the ignored arg
                    "latency_ms": latency_ms,
                    "seed": gen_seed,
                }
                (validated,) = validate_records([record], PanelARecord)
                records_by_mk[(m, k)] = validated
        finally:
            # Stop the server before swapping to the next model, even on error.
            _stop_model_server(server_proc)
            logger.event(
                "protocol_a2.model_stop",
                model=model_name,
                pool=pool_type,
                n_requests=len(assigned_pairs),
            )

    # --- Stage (3): aggregate into one JSONL ordered by (m, k). --------------
    expected = len(instance_ids) * K
    if len(records_by_mk) != expected:
        raise RuntimeError(
            f"Protocol A2 produced {len(records_by_mk)} records but expected "
            f"{expected} (= len(instance_ids) * K); a (m, k) pair was lost in "
            f"the serial model-swap loop"
        )
    with out_path.open("w", encoding="utf-8") as fh:
        for key in sorted(records_by_mk):
            fh.write(json.dumps(records_by_mk[key], default=str) + "\n")

    logger.event(
        "protocol_a2.done",
        benchmark=benchmark,
        pool=pool_type,
        n_records=expected,
        out_path=str(out_path),
    )


def _start_model_server(
    cell_config: dict,
    serving_cfg: dict,
    model_name: str,
    seed: int,
) -> tuple[Any, Any]:
    """Start one model's vLLM server and return (client, server_proc).

    If a live client is supplied under ``cell_config['clients'][model_name]``
    (an orchestrator already holding open connections), it is used and no
    server is spawned (``server_proc`` is ``None``). Otherwise a server is
    launched via :mod:`src.llm.server_lifecycle` from this model's serving
    config and a :class:`VLLMClient` is built against it. Both heavy imports
    are LOCAL so this module stays importable in Phase 0.
    """
    clients = cell_config.get("clients") or {}
    if model_name in clients:
        return clients[model_name], None

    cfg = serving_cfg.get(model_name)
    if cfg is None:
        raise ValueError(
            f"no live client and no cell_config['serving'][{model_name!r}] "
            f"serving config; cannot start a server (no silent fallback)"
        )
    model_path = cfg.get("model_path", model_name)
    port = cfg.get("port")
    if port is None:
        raise ValueError(f"serving config for {model_name!r} must include 'port'")

    # Lazy imports: server lifecycle pulls subprocess/urllib only; VLLMClient
    # pulls openai lazily on first generate(). Neither is needed at import.
    from src.llm.server_lifecycle import start_vllm_server
    from src.llm.vllm_client import VLLMClient

    proc = start_vllm_server(
        model_path=model_path,
        port=int(port),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.90)),
        max_model_len=int(cfg.get("max_model_len", 4096)),
        enforce_eager=bool(cfg.get("enforce_eager", False)),
        seed=int(cfg.get("seed", seed)),
    )
    client = VLLMClient(
        base_url=cfg.get("base_url", f"http://localhost:{int(port)}/v1"),
        api_key=cfg.get("api_key", "EMPTY"),
        model=cfg.get("served_model_name", model_path),
    )
    return client, proc


def _stop_model_server(server_proc: Any) -> None:
    """Stop a server started by :func:`_start_model_server`.

    A ``None`` proc means an externally-managed client was used; nothing to
    stop. The lifecycle import is local to keep Phase 0 import clean.
    """
    if server_proc is None:
        return
    from src.llm.server_lifecycle import stop_vllm_server

    stop_vllm_server(server_proc)
