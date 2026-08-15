"""The shape the metric routes serve, from whichever log the caller resolved.

A training run's rows come from its experiment record, an HPO trial's from the trial
directory, and both are served in one response shape, built here rather than assembled
separately at each route.
"""

from __future__ import annotations

import json
from pathlib import Path


def metrics_response(rows: list[dict], *, exists: bool) -> dict:
    """The response body both metric routes return."""
    return {"metrics": rows, "exists": exists}


def read_metrics_file(path: Path) -> dict:
    """Every parseable row of ``path``, as ``{"metrics": [...], "exists": bool}``.

    For a log this seam does not own yet: an HPO trial writes its rows beside the trial's
    own artifacts, not into an experiment record. Read line by line and skipping anything
    that will not parse, since the file is appended to while a trial is in flight, so the
    last line can be a write still in progress.
    """
    if not path.exists():
        return metrics_response([], exists=False)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return metrics_response(rows, exists=True)
