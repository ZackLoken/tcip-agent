"""Write a root's database-held records and logs back out as files.

Every tool that reads TCIP's state off disk rather than through the storage seam (the
data-state doctor, an archive, an auditor tailing ``.tcip/audit.jsonl``) reads what this
produces. Run it for a root, or for a whole project's roots at once:

    tcip export-store <root> [<root> ...]
    tcip export-store --project <project_root>

Exit codes: 0 everything stamped, 1 a store's counter moved while its files were being written
(rerun), 2 an export refused.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tcip_mcp.store_catalogue import project_roots

from tcip_store.errors import StoreError
from tcip_store.export import export_root
from tcip_store.file_backend import FileBackend, database_file


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog=prog)
    ap.add_argument("roots", nargs="*", help="root directories whose databases to write out")
    ap.add_argument(
        "--project",
        default="",
        help="a project root: exports its own roots and its registered datasets' roots",
    )
    args = ap.parse_args(argv)

    roots = [str(Path(root).absolute()) for root in args.roots]
    if args.project:
        # The registry naming the project's datasets is itself a record this database holds,
        # so it is read through the backend that owns it rather than off disk.
        from tcip_store.sqlite_backend import SqliteBackend
        from tcip_store.store import bind

        bind(SqliteBackend())
        roots += [root for root, _layout in project_roots(args.project)]
    if not roots:
        print("error: name at least one root, or pass --project <project_root>")
        return 2

    files = FileBackend()
    raced: list[str] = []
    for root in _unique(roots):
        if not database_file(root).is_file():
            print(f"{root}: no store database, nothing to export")
            continue
        try:
            exported = export_root(root, backend=files)
        except StoreError as exc:
            print(f"error: {root}: {exc}")
            return 2
        for store in exported.stores:
            state = "stamped" if store.stamped else "raced, rerun"
            print(
                f"{root}: {store.store}: {store.records_written} record file(s), "
                f"{store.logs_written} log file(s), {len(store.deleted)} deleted, {state}"
            )
        raced += [f"{root}:{store}" for store in exported.raced]
    if raced:
        print(f"\n{len(raced)} store(s) moved while their files were written: {', '.join(raced)}")
        return 1
    return 0


def _unique(roots: list[str]) -> list[str]:
    """The roots in the order given, without a repeat when a project names one twice."""
    seen: set[str] = set()
    ordered: list[str] = []
    for root in roots:
        marker = os.path.normcase(root)
        if marker not in seen:
            seen.add(marker)
            ordered.append(root)
    return ordered


if __name__ == "__main__":
    sys.exit(main())
