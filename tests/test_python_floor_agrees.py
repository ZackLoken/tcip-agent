"""The Python floor is one fact: every package's requires-python and ruff's target agree."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGES = ("tcip-store", "tcip-annotation", "tcip-mcp", "tcip-web")


def floor_of(package: str) -> str:
    """Return the minor version ``package``'s ``requires-python`` floor names, as ``3.N``."""
    project = tomllib.loads(
        (REPO / "packages" / package / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    requires = project["project"]["requires-python"]
    match = re.fullmatch(r">=(\d+\.\d+)", requires)
    assert match, f"{package} states requires-python {requires!r}, not >=3.N"
    return match.group(1)


def ruff_target() -> str:
    """Return the version ``ruff.toml``'s ``target-version`` names, as ``3.N``."""
    target = tomllib.loads((REPO / "ruff.toml").read_text(encoding="utf-8"))[
        "target-version"
    ]
    match = re.fullmatch(r"py(\d)(\d+)", target)
    assert match, f"ruff.toml states target-version {target!r}, not pyNM"
    return f"{match.group(1)}.{match.group(2)}"


def test_every_package_and_the_lint_target_state_one_python_floor() -> None:
    floors = {package: floor_of(package) for package in PACKAGES}
    floors["ruff.toml"] = ruff_target()
    assert len(set(floors.values())) == 1, floors
