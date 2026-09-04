"""The browser and the backend meet on one declaration of a logged metric row's shape.

log_metrics accepts any JSON-encodable value per metric, wider than the frontend's own MetricRow
type, which renders only the shapes it recognizes and drops the rest silently. The backend's
docstring quotes MetricRow's own per-key value type so a reader knows what the browser actually
understands; this test holds that quote equal to the frontend's own declaration, so a widened or
narrowed MetricRow value type does not leave a stale quote behind it in the backend.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "src"
METRIC_ROW_DECLARATION = FRONTEND_SRC / "api" / "training.ts"
EXPERIMENTS_MODULE = REPO_ROOT / "packages" / "tcip-mcp" / "src" / "tcip_mcp" / "experiments.py"

_VALUE_TYPE_RE = re.compile(r"\[metric: string\]: ([^;]+);")
_QUOTED_TYPE_RE = re.compile(r"``MetricRow`` type \(``([^`]+)`` per key\)")


def test_the_backends_quoted_metric_row_shape_matches_the_frontends_own() -> None:
    """log_metrics's docstring quotes MetricRow's per-key value type; hold it equal to the type."""
    frontend_match = _VALUE_TYPE_RE.search(METRIC_ROW_DECLARATION.read_text(encoding="utf-8"))
    assert frontend_match is not None, "MetricRow's value type is no longer where this test reads it"
    docstring_match = _QUOTED_TYPE_RE.search(EXPERIMENTS_MODULE.read_text(encoding="utf-8"))
    assert docstring_match is not None, "log_metrics no longer quotes MetricRow where this test reads it"
    assert docstring_match.group(1) == frontend_match.group(1)
