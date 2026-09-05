"""Conform a project's registry index into the entries-mapping shape and respell every entry's
``checkpoint_path`` relative to the project root, per the storage convention
(``tcip_mcp.model_registry``): relative POSIX exactly when the checkpoint lives under the
registry's own scope root, absolute exactly when external.

For each project root named, in one transaction: a bare top-level array (the shape this store
carried before the family that wrapped it) wraps into ``{entries: [...]}``, then every entry whose
stored ``checkpoint_path`` resolves under the root with a matching sha256 is respelled relative;
one that does not (moved, replaced, or genuinely outside the root) is relocated among the
project's own checkpoint files by content digest, or, when no file anywhere under the root
carries that digest, left as an absolute spelling classified external-or-missing. A root already
conformed (nothing to wrap, nothing to respell) is reported unchanged. Never runs as part of any
runtime path; the import door runs the same conform on a freshly extracted archive before it
adopts or renames anything (see ``tools.project_tools.import_project``).

    python scripts/conform_model_registry_paths.py <project_root> [<project_root> ...]
    python scripts/conform_model_registry_paths.py --plan <project_root>

Exit codes: 0 once every root named was conformed, whatever its entries ended up with, or had
nothing to conform; 2 if any root's registry index will not read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

from tcip_store.binding import bind_default  # noqa: E402
from tcip_store.errors import StoreError  # noqa: E402

from tcip_mcp.model_registry import RegistryVersionRefused, conform_registry_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    bind_default()

    any_unreadable = False
    for root in args.roots:
        root = root.resolve()
        try:
            lines = conform_registry_paths(root, plan=args.plan)
        except (StoreError, RegistryVersionRefused) as exc:
            print(f"{root}: refused, {exc}")
            any_unreadable = True
            continue

        if not lines:
            print(f"{root}: nothing to conform")
            continue
        for line in lines:
            print(f"{root}: {line}")

    return 2 if any_unreadable else 0


if __name__ == "__main__":
    sys.exit(main())
