"""The response shape the tuning trial-metrics route serves."""

from __future__ import annotations


def metrics_response(rows: list[dict], *, exists: bool) -> dict:
    """The response body the tuning trial-metrics route returns."""
    return {"metrics": rows, "exists": exists}
