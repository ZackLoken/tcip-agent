"""The response shape the tuning trial-metrics route serves.

The route reads an HPO trial's own metrics log under its sweep through the storage layer and
builds the response here.
"""

from __future__ import annotations


def metrics_response(rows: list[dict], *, exists: bool) -> dict:
    """The response body the tuning trial-metrics route returns."""
    return {"metrics": rows, "exists": exists}
