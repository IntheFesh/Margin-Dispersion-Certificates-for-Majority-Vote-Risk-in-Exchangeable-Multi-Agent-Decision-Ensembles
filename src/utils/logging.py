"""Structured JSONL event logging.

Every significant operation appends a structured event to
outputs/logs/<phase>.jsonl with timestamp, operation, inputs, outputs,
duration, status. No silent fallbacks; logging failures raise.
"""
from __future__ import annotations
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, operation: str, status: str = "ok", **fields: Any) -> None:
        rec = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "status": status,
        }
        rec.update(fields)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    @contextmanager
    def timed(self, operation: str, **fields: Any) -> Iterator[dict]:
        start = time.time()
        ctx: dict = {}
        try:
            yield ctx
        except Exception as e:  # noqa: BLE001
            self.event(operation, status="error", error=repr(e), duration_s=time.time() - start, **fields)
            raise
        else:
            self.event(operation, status="ok", duration_s=time.time() - start, **{**fields, **ctx})
