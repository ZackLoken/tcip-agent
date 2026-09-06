"""Move a root's existing record and log files into a store database.

A root that has been written by the file backend holds state no database beside it could see,
so the database backend refuses such a root until this has run. It reads every record and log
file the root's stores own, decodes all of them, and publishes a database holding exactly
those entries, stamped as already exported. Blob files (imagery, labels, predictions,
checkpoints, hand-authored documents) stay exactly where they are under every backend.

A root that already holds a database is planned too: a store whose files arrived after that
database was built is one the database has never held, and this takes exactly those in.

    tcip adopt-store --project <project_root>
    tcip adopt-store --layout <layout> <root> [<root> ...]

The layout names what kind of directory a root is, because a store's file layout is only
meaningful under the kind of root it was written for. ``--project`` supplies them for a whole
project's roots. ``--plan`` shows what would be adopted and writes nothing.

Exit codes: 0 adopted, 2 refused (nothing written).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tcip_mcp.store_catalogue import project_roots

from tcip_store.adoption import AdoptionPlan, adopt_root, plan_root, unaccounted_files
from tcip_store.errors import StoreError
from tcip_store.file_backend import FileBackend
from tcip_store.layout_claims import LAYOUTS
from tcip_store.store import bind


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog=prog)
    ap.add_argument("roots", nargs="*", help="root directories to adopt")
    ap.add_argument("--layout", default="", choices=("", *LAYOUTS), help="what kind of directory the named roots are")
    ap.add_argument("--project", default="", help="a project root: adopts its own roots and its registered datasets'")
    ap.add_argument("--plan", action="store_true", help="report what would be adopted and write nothing")
    args = ap.parse_args(argv)

    # The registry naming the project's datasets is still a file at this point, so it is read
    # through the backend that still owns it.
    bind(FileBackend())

    targets: list[tuple[str, str]] = []
    if args.roots:
        if not args.layout:
            print(f"error: name the layout of {len(args.roots)} explicit root(s) with "
                  f"--layout, one of: {', '.join(LAYOUTS)}")
            return 2
        targets += [(str(Path(root).absolute()), args.layout) for root in args.roots]
    if args.project:
        targets += list(project_roots(args.project))
    if not targets:
        print("error: name at least one root with --layout, or pass --project <project_root>")
        return 2

    try:
        plans = tuple(plan_root(root, layout) for root, layout in targets)
        left = unaccounted_files(plans)
        if left:
            listed = "\n  ".join(str(path) for path in left)
            print(
                "error: these record or log files belong to no root being adopted, and would "
                f"read as absent once a database exists:\n  {listed}\n"
                "Adopt the roots that own them in the same run, or use --project."
            )
            return 2
        _report_plans(plans)
        if args.plan:
            return 0
        for plan in plans:
            result = adopt_root(plan.root, plan.layout)
            loaded = sum(result.records.values()) + sum(result.log_entries.values())
            print(f"{plan.root}: adopted {loaded} entr(ies) into {result.database}")
    except StoreError as exc:
        print(f"error: {exc}")
        return 2
    return 0


def _report_plans(plans: tuple[AdoptionPlan, ...]) -> None:
    """Say what each root would take in, per store, before anything is written."""
    for plan in plans:
        counted: dict[str, int] = {}
        for entry in plan.entries:
            counted[entry.store] = counted.get(entry.store, 0) + 1
        summary = ", ".join(f"{store} x{count}" for store, count in sorted(counted.items()))
        print(f"{plan.root} ({plan.layout}): {summary or 'nothing to adopt'}")


if __name__ == "__main__":
    sys.exit(main())
