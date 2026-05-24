"""Pydantic schemas for all pipeline outputs.

Every output file (JSONL / JSON / CSV) is validated against these schemas
before being written. Schema violations halt execution.

Note: the Panel A correctness record deliberately does NOT include `scale`
or `family` fields; the per-(instance, sample) schema is exactly the set of
columns below.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class PanelARecord(BaseModel):
    """One row per (instance, sample-k) generation in a Panel A run."""

    instance_id: str
    pool: Literal["estimation", "oracle"]
    m: int = Field(ge=0)
    k: int = Field(ge=0)
    prompt_variation_id: int = Field(ge=0)
    prompt: str
    raw_completion: str
    extracted_answer: Optional[str]
    gold_answer: str
    success_indicator: int = Field(ge=0, le=1)
    invalid_parse: bool
    model: str
    latency_ms: float = Field(ge=0.0)
    seed: int

    @field_validator("success_indicator")
    @classmethod
    def _invalid_is_failure(cls, v: int, info) -> int:
        return v


class CellSummary(BaseModel):
    """Per (protocol, benchmark, K_est) cell certificate summary."""

    protocol: str
    benchmark: str
    model: str
    K_est: int = Field(ge=2)
    M_estimation: int = Field(ge=2)
    M_oracle: int = Field(ge=1)
    N: int = Field(ge=1)
    delta_global: float = Field(gt=0.0, lt=1.0)
    delta_cell: float = Field(gt=0.0, lt=1.0)
    C_cells: int = Field(ge=1)
    alpha_bar_hat: float
    F_hat: float
    L_alpha: float
    U_alpha: float
    U_F: float = Field(ge=0.0, le=0.25)
    m_L: float
    m_beta_L: float
    R_N_cert: Optional[float]
    Q_N_cert: Optional[float]
    R_N_BA_cert: Optional[float]
    refusal_mode: str
    refusal_sub_mode: Optional[str]


class ProvenanceRecord(BaseModel):
    git_commit: str
    code_version: str
    config_paths: list[str]
    global_seed: int
    gpu_model: str
    vllm_version: str
    torch_version: str
    timestamp_utc: str


def validate_records(records: list[dict], model: type[BaseModel]) -> list[dict]:
    """Validate a list of dict records against a schema; raise on violation.
    Returns the normalized dicts."""
    out = []
    for i, rec in enumerate(records):
        try:
            out.append(model(**rec).model_dump())
        except Exception as e:  # noqa: BLE001 - re-raise with row context
            raise ValueError(f"schema violation at record {i}: {e}") from e
    return out
