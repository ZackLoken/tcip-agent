"""Write a scope's database-held records and logs back out as files.

Every tool that reads TCIP's state off disk rather than through the storage seam (the
data-state doctor, an archive, an auditor tailing ``.tcip/audit.jsonl``) reads what this
produces. Run it for a scope, or for a whole project's scopes at once:

    python scripts/export_store.py <scope> [<scope> ...]
    python scripts/export_store.py --project <project_root>

Exit codes: 0 everything stamped, 1 a store's counter moved while its files were being written
(rerun), 2 an export refused.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _store_bootstrap import project_scopes  # noqa: E402

from tcip_store.errors import StoreError  # noqa: E402
from tcip_store.export import export_scope  # noqa: E402
from tcip_store.file_backend import FileBackend, database_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scopes", nargs="*", help="scope directories whose databases to write out")
    ap.add_argument(
        "--project",
        default="",
        help="a project root: exports its own scopes and its registered datasets' scopes",
    )
    args = ap.parse_args()

    scopes = [str(Path(scope).absolute()) for scope in args.scopes]
    if args.project:
        # The registry naming the project's datasets is itself a record this database holds,
        # so it is read through the backend that owns it rather than off disk.
        from tcip_store.sqlite_backend import SqliteBackend
        from tcip_store.store import bind

        bind(SqliteBackend())
        scopes += [scope for scope, _layout in project_scopes(args.project)]
    if not scopes:
        print("error: name at least one scope, or pass --project <project_root>")
        return 2

    files = FileBackend()
    raced: list[str] = []
    for scope in _unique(scopes):
        if not database_file(scope).is_file():
            print(f"{scope}: no store database, nothing to export")
            continue
        try:
            exported = export_scope(scope, backend=files)
        except StoreError as exc:
            print(f"error: {scope}: {exc}")
            return 2
        for store in exported.stores:
            state = "stamped" if store.stamped else "raced, rerun"
            print(
                f"{scope}: {store.store}: {store.records_written} record file(s), "
                f"{store.logs_written} log file(s), {len(store.deleted)} deleted, {state}"
            )
        raced += [f"{scope}:{store}" for store in exported.raced]
    if raced:
        print(f"\n{len(raced)} store(s) moved while their files were written: {', '.join(raced)}")
        return 1
    return 0


def _unique(scopes: list[str]) -> list[str]:
    """The scopes in the order given, without a repeat when a project names one twice."""
    seen: set[str] = set()
    ordered: list[str] = []
    for scope in scopes:
        marker = os.path.normcase(scope)
        if marker not in seen:
            seen.add(marker)
            ordered.append(scope)
    return ordered


if __name__ == "__main__":
    sys.exit(main())
