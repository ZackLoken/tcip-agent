"""The shape the metric routes serve, from whichever log the caller resolved.

A training run's rows come from its experiment record, an HPO trial's from the trial's own
log under its sweep. Each route reads its own log through the storage layer and builds the
response here, so the two answer in one shape.
"""

from __future__ import annotations


def metrics_response(rows: list[dict], *, exists: bool) -> dict:
    """The response body both metric routes return."""
    return {"metrics": rows, "exists": exists}
