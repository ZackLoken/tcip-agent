"""The test suite, the root of the repository its tests read files from, the browser's sources,
the names the person rule is tested on, and the reader of the CSV files its tests deliver."""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
"""The repository this suite belongs to."""

FRONTEND_SRC = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "src"
"""The browser's sources."""

_PERSON_NAMES = json.loads((FRONTEND_SRC / "test" / "personNames.json").read_text(
    encoding="utf-8"))
ADMITTED_NAMES: list[tuple[str, str]] = [(s, r) for s, r in _PERSON_NAMES["admitted"]]
"""Each name the person rule admits, with the name it records; the browser's tests read the
same list."""
REFUSED_NAMES: list[str] = _PERSON_NAMES["refused"]
"""Each string the person rule refuses as naming no one."""


def csv_rows(path: str | Path) -> list[dict]:
    """The rows of the UTF-8 CSV file at ``path``, each keyed by its header."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
