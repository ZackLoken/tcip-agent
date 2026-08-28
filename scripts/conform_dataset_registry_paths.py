"""Conform a project's dataset registry onto the relative-path row.

``register_dataset`` now stores a dataset's ``path`` relative to the project root whenever the
dataset sits under it (the project's own tree becoming ``"."``), and absolute only for a
genuinely external dataset. A project registered before that change can still hold the old
absolute form for its own dataset, which is what this one-off operator script rewrites in
place, through the same identity-based rule (``registry_path_for``) the writer itself now uses.
Never a runtime migration: the writer already produces the new form, so this exists only to
carry an already-registered project onto it.

    python scripts/conform_dataset_registry_paths.py <project_root> [<project_root> ...]

Exit codes: 0 always; a registry with nothing to change is ordinary and is reported as such.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import tcip_store  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402

from tcip_mcp.tools.project_tools import (  # noqa: E402
    dataset_entry_path,
    dataset_registry_key,
    registry_path_for,
)


def conform_project(project_root: str) -> int:
    """Rewrite one project's registry entries onto the relative-path row.

    Returns how many entries changed. An entry already relative recomputes to the same
    spelling, so this is idempotent to run more than once.
    """
    root = Path(project_root).absolute()
    key = dataset_registry_key(root)
    versioned = tcip_store.read_versioned(key, default=[])
    entries = versioned.value if isinstance(versioned.value, list) else []
    changed = 0
    conformed = []
    for raw in entries:
        entry = dict(raw)
        path = entry.get("path")
        if isinstance(path, str) and path:
            resolved = dataset_entry_path(root, entry)
            new_path = registry_path_for(resolved, root)
            if new_path != path:
                print(f"{root}: {entry.get('id')}: {path!r} -> {new_path!r}")
                entry["path"] = new_path
                changed += 1
        conformed.append(entry)
    if changed:
        tcip_store.replace(key, conformed, expect=versioned.version)
    else:
        print(f"{root}: nothing to conform")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "project_roots", nargs="+", help="project root(s) whose dataset registry to conform"
    )
    args = ap.parse_args()

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    bind_default()

    total = 0
    for project_root in args.project_roots:
        total += conform_project(project_root)
    print(f"conformed {total} entr(ies) across {len(args.project_roots)} project(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
