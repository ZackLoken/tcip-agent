"""``environment.yml`` names the conda environment every process activates; nothing else may
carry a different name. Most of its copies cannot be derived at runtime (an activation key in
YAML, a bare argument in a JSON array, prose with no invocation to inspect), so this holds an
enumerated list of copy sites instead, checked against ``environment.yml``'s own name, and named
as the enumeration it is: a copy added somewhere new is not caught until it is added here too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _environment_name() -> str:
    data = yaml.safe_load((REPO_ROOT / "environment.yml").read_text(encoding="utf-8"))
    return data["name"]


def _line(path: Path, line_no: int) -> str:
    return path.read_text(encoding="utf-8").splitlines()[line_no - 1]


def test_ci_activate_environment_key_matches():
    line = _line(REPO_ROOT / ".github" / "workflows" / "ci.yml", 27)
    assert line.strip() == f"activate-environment: {_environment_name()}"


def test_mcp_json_launch_argument_matches():
    data = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    args = data["mcpServers"]["tcip"]["args"]
    assert args[args.index("-n") + 1] == _environment_name()


def test_environment_win_lock_name_matches():
    data = yaml.safe_load((REPO_ROOT / "environment.win.lock.yml").read_text(encoding="utf-8"))
    assert data["name"] == _environment_name()


# Enumerated, not derived: none of these lines carries a structured key the name could be read
# from, only prose or an invocation naming it.
_PROSE_COPY_SITES = [
    (REPO_ROOT / "README.md", 72),
    (REPO_ROOT / "README.md", 83),
    (REPO_ROOT / "CLAUDE.md", 288),
    (REPO_ROOT / "packages" / "tcip-web" / "README.md", 43),
    (REPO_ROOT / "ARCHITECTURE.md", 971),
    (REPO_ROOT / "ARCHITECTURE.md", 2057),
    (REPO_ROOT / "ARCHITECTURE.md", 2058),
    (REPO_ROOT / "scripts" / "distill_learnings.py", 8),
    (REPO_ROOT / "scripts" / "smoke_fence_e2e.py", 9),
    (REPO_ROOT / "scripts" / "smoke_phenology_e2e.py", 15),
    (REPO_ROOT / "scripts" / "smoke_terminal_e2e.py", 9),
    (REPO_ROOT / "scripts" / "watch_agent_chat.py", 13),
]


@pytest.mark.parametrize(
    "path,line_no", _PROSE_COPY_SITES,
    ids=[f"{p.name}:{n}" for p, n in _PROSE_COPY_SITES],
)
def test_enumerated_prose_copy_carries_the_environment_name(path, line_no):
    assert _environment_name() in _line(path, line_no)
