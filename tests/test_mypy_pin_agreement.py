"""Agreement of the mypy version pin across the files that each carry a copy.

The type gate is only one gate if CI installs the same mypy the documented environments
resolve. Nothing else guards the three copies against drifting apart, so a mismatch here
would otherwise surface only as a mypy behavior difference between CI and a local run.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_PIN_RE = re.compile(r"(?<![\w-])mypy==([\w.]+)")

_PINNED_FILES = (
    ROOT / ".github" / "workflows" / "ci.yml",
    ROOT / "environment.yml",
    ROOT / "environment.win.lock.yml",
)


def _pins(path: Path) -> list[str]:
    return _PIN_RE.findall(path.read_text(encoding="utf-8"))


def test_each_file_pins_a_mypy_version():
    for path in _PINNED_FILES:
        pins = _pins(path)
        assert pins, f"no mypy== pin found in {path}"


def test_the_three_mypy_pins_agree():
    per_file = {path: _pins(path) for path in _PINNED_FILES}
    all_versions = {version for pins in per_file.values() for version in pins}
    assert len(all_versions) == 1, (
        "mypy version pins disagree across "
        f"{[(str(path), pins) for path, pins in per_file.items()]}"
    )
