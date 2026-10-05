"""The test suite, the root of the repository its tests read files from, and the reader of the
CSV files its tests deliver."""

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
"""The repository this suite belongs to."""


def csv_rows(path: str | Path) -> list[dict]:
    """The rows of the UTF-8 CSV file at ``path``, each keyed by its header."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
