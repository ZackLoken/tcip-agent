"""The retired ``rename-subject-registry`` command's name survives nowhere in running prose.

``test_tool_renames.py`` and ``test_skill_tool_fidelity.py`` reach only a Tools table's cells (a
bare backtick-quoted name or a documented call signature); the conform command was never an MCP
tool and had no table row, so neither sweep would ever see its name. This walks every tracked file
under the four packages' own ``src`` trees, the frontend's ``src`` tree, and ``tools/`` directly,
failing on any occurrence of the command's hyphenated CLI spelling or its underscored module name,
naming the file and line, so a docstring, comment, refusal message or log line that still names
the deleted command is caught the same way a stale tool-table row is.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_SCOPE_PREFIXES = (
    "packages/tcip-annotation/src/",
    "packages/tcip-mcp/src/",
    "packages/tcip-store/src/",
    "packages/tcip-web/src/",
    "packages/tcip-web/frontend/src/",
    "tools/",
)

_NAMES = ("rename-subject-registry", "rename_subject_registry")

# Extensions a text read would corrupt or fail on; skipped rather than reported.
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".zip", ".pyc", ".db", ".sqlite", ".onnx", ".pt", ".pth",
}


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"] + list(_SCOPE_PREFIXES),
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return [line for line in out.stdout.splitlines() if line]
    except (subprocess.CalledProcessError, OSError):
        # A fail-before proof's git-archive baseline has no .git directory; the same git-free
        # fallback test_tools_readme_index.py uses.
        names: list[str] = []
        for prefix in _SCOPE_PREFIXES:
            base = REPO_ROOT / prefix
            if not base.is_dir():
                continue
            names.extend(
                str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                for p in base.rglob("*")
                if p.is_file()
            )
        return names


def _in_scope(rel: str) -> bool:
    if "node_modules" in rel.split("/"):
        return False
    if "/dist/" in f"/{rel}" or rel.startswith("dist/"):
        return False
    if not (REPO_ROOT / rel).exists():
        # git ls-files answers from the index; a working-tree deletion not yet staged still
        # lists here, and has nothing left to read or claim as a path-level hit.
        return False
    return Path(rel).suffix.lower() not in _BINARY_SUFFIXES


def _read(rel: str) -> str | None:
    try:
        return (REPO_ROOT / rel).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def test_retired_conform_command_name_absent_from_running_prose():
    files = [f for f in _tracked_files() if _in_scope(f)]
    assert files, "the scoped git ls-files walk returned nothing; the walk itself is broken"

    hits: list[str] = []
    for rel in files:
        for name in _NAMES:
            if name in rel:
                hits.append(f"{rel}: (in the path itself)")
        text = _read(rel)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name in _NAMES:
                if name in line:
                    hits.append(f"{rel}:{lineno}: {line.strip()}")

    assert not hits, "the retired command's name still appears:\n" + "\n".join(hits)
