"""Reader for the append-only ``metrics.jsonl`` a training run or an HPO trial writes.

Both the Training and Tuning routes serve the same file under two different roots, so
they share this one reader rather than each parsing the format.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_metrics_file(path: Path) -> dict:
    """Every parseable row of ``path``, as ``{"metrics": [...], "exists": bool}``.

    Read line by line and skipping anything that will not parse: the file is appended to
    while a run is in flight, so the last line can be a write still in progress.
    """
    if not path.exists():
        return {"metrics": [], "exists": False}
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
    return {"metrics": rows, "exists": True}
