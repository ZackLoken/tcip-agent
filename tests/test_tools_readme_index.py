"""tools/README.md names every tracked tool, so the index can't silently drift behind the
directory.

Reads the tracked file list straight from `git ls-files tools` (never a copied manifest of
what should be there) and asserts every basename other than `README.md` and anything under
`__pycache__` appears in tools/README.md's own text as a backticked filename. Coverage of the
index's completeness, not a behavior fix.

Without git (a fail-before proof's git-archive baseline has no `.git` directory), the fallback
walks `tools/` directly, the same git-free fallback `test_deleted_skills.py` uses.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO / "tools"
README = TOOLS_DIR / "README.md"


def _tracked_tool_basenames() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "tools"], cwd=REPO, capture_output=True, text=True, check=True,
        )
        lines = out.stdout.splitlines()
    except (subprocess.CalledProcessError, OSError):
        lines = [
            str(p.relative_to(REPO)).replace("\\", "/")
            for p in TOOLS_DIR.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        ]
    names = []
    for line in lines:
        if not line or "__pycache__" in line:
            continue
        basename = Path(line).name
        if basename == "README.md":
            continue
        names.append(basename)
    return names


def test_every_tracked_tool_is_named_in_the_readme_index():
    tracked = _tracked_tool_basenames()
    assert tracked, "git ls-files tools returned nothing; the walk itself is broken"
    readme_text = README.read_text(encoding="utf-8")
    missing = sorted(name for name in tracked if f"`{name}`" not in readme_text)
    assert not missing, (
        f"tools/README.md does not name {len(missing)} tracked tool(s) as a backticked "
        f"filename: {missing}"
    )
