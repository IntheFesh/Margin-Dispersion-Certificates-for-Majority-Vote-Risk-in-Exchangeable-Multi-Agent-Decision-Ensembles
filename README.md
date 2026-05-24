# Margin-Dispersion Certificates for Strict Success-Majority Risk in Exchangeable Multi-Agent Decision Ensembles (Panel A)

This repository implements and empirically validates **distribution-free,
two-moment certificates** that upper-bound the probability that an ensemble of
exchangeable decision agents *fails to reach a strict success-majority* on a
task instance. Given only the **mean per-agent success rate** and a
**dispersion summary** of an instance population, the certificates return a
guaranteed upper bound on the majority-vote failure risk for any odd ensemble
size *N*.

The work is organized as **Panel A only**: prompt/model-randomized ensembles of
large language models on closed-form reasoning benchmarks, validated against a
held-out Monte-Carlo majority-vote-risk reference. Inference is served by
**vLLM** through its OpenAI-compatible API.

## Scope and conventions

- **Strict success-majority**: an ensemble *succeeds* on an instance iff a
  strict majority of its members succeed. **Ties count as failures**, which is
  why every ensemble size *N* in the design grid is **odd**
  (`3, 7, 15, 31, 63, 127`): odd *N* has no ties and yields cleaner certificate
  structure.
- **Three benchmarks only**: `arc_challenge`, `gsm8k`, `mmlu_subset`.
- **Two ensemble-construction protocols**: A1 (prompt-randomized
  self-consistency for a fixed model) and A2 (randomized model-family ensemble,
  served serially under the GPU policy).
- **Estimation / oracle split is instance-level**: the estimation pool builds
  the certificate; the disjoint oracle pool feeds the Monte-Carlo reference
  `R_N^MC` only. Sample columns are never split across pools.
- **vLLM inference only**: generation goes through a live vLLM server via the
  `openai` client. We never use HuggingFace `transformers` for inference.
- **Determinism**: every random operation derives its seed from a global seed
  via `src.utils.seeds.derive_seed` / `rng_for`. There are no unseeded RNGs and
  no wall-clock seeding.
- **No-fake-data + error-handling policy**: missing input files raise
  `FileNotFoundError`; a figure is never drawn on fabricated or empty data;
  invalid answer parses are kept as recorded failures (the
  invalid-parse-is-failure bridge lemma); numerical hierarchy / primal-dual
  violations abort the pipeline. There are no silent fallbacks.

See [`docs/framework.md`](docs/framework.md) for the theory (the certificate
hierarchy, the boundedness-aware refinement, the sharp two-moment envelope, the
empirical certificates, the Bonferroni budget, and the refusal taxonomy) and
[`docs/reproduction.md`](docs/reproduction.md) for full step-by-step commands.

## Install

```bash
python -m pip install -r requirements.txt          # numpy/scipy/pandas/matplotlib/pydantic/pyyaml/openai/datasets
# vLLM is the inference engine; install it separately on the GPU host:
python -m pip install "vllm>=0.6"                   # GPU host only (Phase 1.1+)
```

Python 3.10+ is required. Plotting uses **matplotlib only** (no seaborn); every
figure is saved as both a 300-DPI PNG and a PDF.

## Reproduction (summary)

```bash
PYTHONPATH=. pytest tests                                                  # 1. proofs + estimators + plumbing unit tests
PYTHONPATH=. python scripts/00_verify_proofs.py    --output_dir outputs/logs      # 2. Phase 0: numerical proof checks (no GPU)
PYTHONPATH=. python scripts/01_design_space_precheck.py --config configs/design_grid.yaml --output_dir outputs/precheck  # 3. go/no-go pre-check (no GPU)
PYTHONPATH=. python scripts/02_engineering_pilot.py --config configs/panel_a_protocol_a1.yaml --output_dir outputs/pilots # 4. M=50 smoke test (GPU; diagnostics only, never findings)
PYTHONPATH=. python scripts/03_statistical_pilot.py --config configs/panel_a_protocol_a1.yaml --output_dir outputs/pilots # 5. M=500 single-cell pilot (GPU; smallest reportable scale)
PYTHONPATH=. python scripts/04_panel_a_full.py     --config configs/panel_a_protocol_a1.yaml --config_a2 configs/panel_a_protocol_a2.yaml --output_dir outputs/panel_a  # 6. full Panel A (GPU)
PYTHONPATH=. python scripts/05_run_analyses.py --panel_a_dir outputs/panel_a --output_dir outputs/analyses && PYTHONPATH=. python scripts/06_render_figures.py --output_dir outputs/figures && PYTHONPATH=. python scripts/07_build_tables.py --output_dir outputs/analyses/tables  # 7. analyses -> figures -> tables (no GPU)
```

Steps 1-3 and 5-7 need no GPU/network beyond dataset download in the GPU phases;
steps 4-6 (the pilots and the full run) require a live vLLM server. Certificate
values from `M < 200` (the engineering pilot) are smoke-test diagnostics and are
**never** reported as findings.

## Repository layout

- `src/certs/` — the certificates: `theorem1` (`B_N^CH`), `refinement1`
  (`B_N^CH'`), `theorem3` (`B_N^star`), `empirical` (`R_N^cert`, `Q_N^cert`,
  `R_{N,BA}^cert`), `verify` (hierarchy + primal-dual bug detectors), `refusal`.
- `src/protocols/` — the instance-level pool split and Protocols A1 / A2.
- `src/data/`, `src/llm/` — benchmark loaders and the vLLM/OpenAI generation
  stack (answer extraction, labelling, prompt variations, server lifecycle).
- `src/analysis/` — Analyses 1-7 over the Panel A outputs.
- `src/figures/` — paper-ready matplotlib figure modules.
- `src/utils/` — deterministic seeds, provenance capture, JSONL logging, schemas.
- `scripts/` — the numbered orchestration scripts `00`-`07`.
- `configs/` — the frozen design grid, protocol, model, benchmark, and
  pre-registration configs.
