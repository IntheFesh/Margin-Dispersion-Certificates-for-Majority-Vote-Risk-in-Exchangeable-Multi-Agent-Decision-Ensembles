"""Provenance capture for reproducibility.

Every output report records: git commit, code version, config paths, global
seed, hardware (GPU model, vLLM version, torch version), UTC timestamp. The
config YAML in use is copied verbatim to output_dir/config_used.yaml.
"""
from __future__ import annotations
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CODE_VERSION = "12.0.0"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _safe_version(import_name: str) -> str:
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"


def gpu_model() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "no-gpu-detected"


def capture(global_seed: int, config_paths: list[str]) -> dict:
    return {
        "git_commit": git_commit(),
        "code_version": CODE_VERSION,
        "config_paths": list(config_paths),
        "global_seed": int(global_seed),
        "gpu_model": gpu_model(),
        "vllm_version": _safe_version("vllm"),
        "torch_version": _safe_version("torch"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def copy_config(config_path: str, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "config_used.yaml"
    shutil.copy2(config_path, dest)
    return dest
