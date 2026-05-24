"""Phase 2 table building: LaTeX + markdown tables from the analysis CSVs.

Reads the analysis CSVs under ``outputs/analyses/`` and emits, for each, a
paired LaTeX (``.tex``) and GitHub-flavoured markdown (``.md``) table under
``outputs/analyses/tables/``. The primary target table is Analysis 1 (the
bootstrap coverage vs the pre-registered 0.95 level); the interpretive tables
(non-vacuity, refusal taxonomy, sharpness, conservativeness) follow.

A requested-but-absent CSV raises ``FileNotFoundError`` (no fabricated table).
CSVs that are merely optional (the a3..a7 outputs, which depend on analysis
modules authored separately) are rendered when present and skipped with a
logged note when not. No GPU/network; it does NOT run in Phase 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.logging import JsonlLogger
from src.utils.provenance import capture, copy_config

# (csv stem, caption, required?). Analysis 1 and 2 are required outputs of
# 05_run_analyses.py; the rest are rendered opportunistically when present.
_TABLES: tuple[tuple[str, str, bool], ...] = (
    ("analysis_1_bootstrap", "Analysis 1: bootstrap certificate coverage of the MC majority-vote risk.", True),
    ("analysis_2_nonvacuity", "Analysis 2: non-vacuity rates of the issued certificate.", True),
    ("analysis_3_refusal_taxonomy", "Analysis 3: refusal-mode decomposition.", False),
    ("analysis_4_bidirectional", "Analysis 4: bidirectional certification rates.", False),
    ("analysis_5_budget_curves", "Analysis 5: certificate vs ensemble-size budget curves.", False),
    ("analysis_6_sharpness", "Analysis 6: certificate sharpness vs the sharp two-moment envelope.", False),
    ("analysis_7_conservativeness", "Analysis 7: four-component conservativeness decomposition.", False),
)


def _fmt(value: object) -> str:
    """Render a cell value compactly (4 dp for floats; blank for NaN/None)."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.4f}"
    return str(value)


def _to_markdown(df: pd.DataFrame, caption: str) -> str:
    """Render a DataFrame to a GitHub-flavoured markdown table (no deps)."""
    cols = list(df.columns)
    lines = [f"### {caption}", "", "| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    lines.append("")
    return "\n".join(lines)


def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Render a DataFrame to a booktabs-style LaTeX ``table`` environment."""
    cols = list(df.columns)
    col_spec = "l" * len(cols)
    header = " & ".join(_latex_escape(str(c)) for c in cols) + r" \\"
    body_lines: list[str] = []
    for _, row in df.iterrows():
        body_lines.append(" & ".join(_latex_escape(_fmt(row[c])) for c in cols) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            rf"\caption{{{_latex_escape(caption)}}}",
            rf"\label{{tab:{label}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header,
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _latex_escape(text: str) -> str:
    """Escape the LaTeX-special characters that occur in our column values."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for src_ch, dst in replacements.items():
        out = out.replace(src_ch, dst)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LaTeX/markdown tables from analysis CSVs.")
    parser.add_argument("--config", default="configs/pre_registration.yaml")
    parser.add_argument("--analyses_dir", default="outputs/analyses")
    parser.add_argument("--output_dir", default="outputs/analyses/tables")
    parser.add_argument("--seed", type=int, default=0, help="global seed (provenance only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(Path("outputs/logs") / "07_build_tables.jsonl")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    copy_config(str(config_path), str(output_dir))
    provenance = capture(args.seed, [str(config_path)])

    analyses_dir = Path(args.analyses_dir)
    written: list[str] = []
    for stem, caption, required in _TABLES:
        csv_path = analyses_dir / f"{stem}.csv"
        if not csv_path.exists():
            if required:
                raise FileNotFoundError(
                    f"required analysis CSV not found: {csv_path}; run 05_run_analyses.py first"
                )
            logger.event("table.skip", table=stem, reason="csv absent (optional analysis)")
            continue
        df = pd.read_csv(csv_path)

        md_path = output_dir / f"{stem}.md"
        tex_path = output_dir / f"{stem}.tex"
        md_path.write_text(_to_markdown(df, caption), encoding="utf-8")
        tex_path.write_text(_to_latex(df, caption, stem), encoding="utf-8")
        written.extend([str(md_path), str(tex_path)])
        logger.event("table.done", table=stem, md=str(md_path), tex=str(tex_path), n_rows=int(len(df)))

    logger.event("tables.complete", n_files=len(written), output_dir=str(output_dir))
    print(f"Built {len(written)} table files in {output_dir}:")
    for p in written:
        print(f"  {p}")
    _ = provenance


if __name__ == "__main__":
    main()
