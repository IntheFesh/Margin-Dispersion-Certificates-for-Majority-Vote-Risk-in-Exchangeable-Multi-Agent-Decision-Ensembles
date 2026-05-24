"""Integration guards for the analysis orchestration scripts.

These catch the two classes of wiring bugs between scripts/05_run_analyses.py
(and 07_build_tables.py) and the actual src/analysis modules:

  1. module/function-name drift (05 dispatching a name that does not exist),
  2. CSV-stem misalignment (07 expecting a stem 05 never writes).
"""
from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_05_dispatch_names_resolve():
    s05 = _load_script("05_run_analyses.py")
    for module_suffix, fname in s05._ANALYSIS_MODULES:
        mod = importlib.import_module(f"src.analysis.{module_suffix}")
        assert hasattr(mod, fname), f"src.analysis.{module_suffix} has no {fname!r}"
        assert callable(getattr(mod, fname)), f"{module_suffix}.{fname} is not callable"


def _expected_stems(s05) -> set[str]:
    stems = {"analysis_1_bootstrap", "analysis_2_nonvacuity"}
    for module_suffix, _ in s05._ANALYSIS_MODULES[2:]:
        idx = module_suffix.split("_", 1)[0].lstrip("a")
        name = module_suffix.split("_", 1)[1]
        stems.add(f"analysis_{idx}_{name}")
    return stems


def test_07_table_stems_align_with_05_outputs():
    s05 = _load_script("05_run_analyses.py")
    s07 = _load_script("07_build_tables.py")
    produced = _expected_stems(s05)
    table_stems = {stem for stem, _caption, _required in s07._TABLES}
    # Every table 07 declares must be a stem 05 actually writes.
    missing = table_stems - produced
    assert not missing, f"07 expects stems 05 never produces: {sorted(missing)}"
