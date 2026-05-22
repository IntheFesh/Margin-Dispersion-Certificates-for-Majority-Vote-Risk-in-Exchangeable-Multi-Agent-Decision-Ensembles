from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_markdown_report(path: str, title: str, mode: str, config_path: str, config: dict, seed: int, num_instances: int, num_models_or_samples: int, invalid_parse_rate: float, main_table_md: str, figure_links: list[str], warnings: list[str], missing_outputs: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"- mode: {mode}", f"- config_path: {config_path}", f"- seed: {seed}", f"- datetime_utc: {datetime.now(timezone.utc).isoformat()}", f"- git_commit: {_git_hash()}", f"- num_instances: {num_instances}", f"- num_models_or_samples: {num_models_or_samples}", f"- invalid_parse_rate: {invalid_parse_rate:.6f}", "", "## Config Used", "```yaml", json.dumps(config, indent=2), "```", "", "## Main Metrics", main_table_md, "", "## Figure Links"]
    lines += [f"- {x}" for x in figure_links] if figure_links else ["- none"]
    lines += ["", "## Missing Outputs"] + ([f"- {x}" for x in missing_outputs] if missing_outputs else ["- none"])
    lines += ["", "## Warnings"] + ([f"- {x}" for x in warnings] if warnings else ["- none"])
    p.write_text("\n".join(lines) + "\n")
