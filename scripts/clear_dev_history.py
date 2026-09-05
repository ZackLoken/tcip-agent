"""Clear a root's development-era audit lines, friction reports, retrospectives and
learning-capture lines before it reaches an alpha tester.

The production audit trail an alpha tester reads serves a different purpose than the
remediation program's own records: a tester should see the platform's own doors acting on
their data, not every experiment this repo's own development ran against the same root. This
removes exactly four things under a named root, through the storage seam and never around it:
every ``friction_reports`` record, every ``retrospectives`` record, the ``audit_log`` log's
entries, and the ``learning_capture`` log's entries. Nothing else: not annotations, images,
predictions, models, experiments, caches, or any other store.

When the root's state lives in a database (``TCIP_STORE_BACKEND=sqlite``, the default) and an
earlier ``export_store.py`` run left loose copies of any of the four beside it, those copies are
removed too, so the export stamp and the files agree: the database's own bookkeeping already
marks them behind the moment this script's mutations land, and an absent file reads the same
"nothing here" as one never exported. A file-backend root has no such copies to reconcile: its
loose files are the primary storage, already correct once the seam operations above return.

After clearing, one audit line is written into the now-empty ``audit_log``, naming the operator
(``--by``) and the reason (``--reason``) and carrying every count this run removed, so the
production log's first line states that development history was cleared and by whom. A failed
append raises ``AuditEntryNotWritten`` and this script exits nonzero naming it; every mutation
before that point has already committed, since the closing line is written last.

    python scripts/clear_dev_history.py <root> [<root> ...] [--plan]
    python scripts/clear_dev_history.py <root> [<root> ...] --apply --by <user:name> --reason <text>

``--plan`` is the default when ``--apply`` is not given: it prints exactly what ``--apply``
would remove, with counts, and writes nothing, including no audit line. ``--apply`` requires
both ``--by`` and ``--reason``, refused before any root is touched when either is missing; a
person's identity is normalized through ``tcip_mcp.identity.user_identity``. A root holding no
``.tcip`` directory is refused by name.

Exit codes: 0 when every named root was planned or cleared; 2 when a root was refused (no
``.tcip`` directory, missing ``--by``/``--reason``, or a failed closing audit append naming
``AuditEntryNotWritten``). Every root named on the command line is still processed, even after
another root's refusal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-web" / "src"))

import tcip_store as ts  # noqa: E402
from tcip_mcp.audit import (  # noqa: E402
    AUDIT_LOG_STORE,
    AuditEntryNotWritten,
    audit_log_key,
    record_event_or_raise,
)
from tcip_mcp.identity import user_identity  # noqa: E402
from tcip_mcp.tools.meta_tools import (  # noqa: E402
    FRICTION_REPORT_STORE,
    RETROSPECTIVE_STORE,
)
from tcip_store.binding import bind_default  # noqa: E402
from tcip_store.file_backend import database_file  # noqa: E402
from tcip_web.agent_learning_capture import (  # noqa: E402
    LEARNING_CAPTURE_STORE,
    learning_capture_key,
)


def _export_path(store: str, root: Path, parts: tuple[str, ...]) -> Path:
    """Where a stale exported loose copy of one entry would sit, from the store's own locator.

    The one place this script computes a filesystem path, so it never restates a locator's own
    layout: a store without a locator would already have failed ``register_store``.
    """
    descriptor = ts.get_descriptor(store)
    assert descriptor.locator is not None
    relative = descriptor.locator.relative_path(str(root), parts)
    return root.joinpath(*relative.parts)


def _remove_stale_export(store: str, root: Path, parts: tuple[str, ...]) -> bool:
    """Remove one entry's stale exported loose copy if present; True when a file was removed."""
    path = _export_path(store, root, parts)
    if path.is_file():
        path.unlink()
        return True
    return False


def plan_root(root: Path) -> list[str]:
    """Every outcome line naming what ``--apply`` would remove under ``root``, writing nothing."""
    n_reports = len(ts.keys(FRICTION_REPORT_STORE, str(root)))
    n_retros = len(ts.keys(RETROSPECTIVE_STORE, str(root)))
    n_audit = _log_length(audit_log_key(root))
    n_capture = _log_length(learning_capture_key(root))
    return [
        f"would remove {n_reports} friction_reports record(s)",
        f"would remove {n_retros} retrospectives record(s)",
        f"would remove {n_audit} audit_log line(s)",
        f"would remove {n_capture} learning_capture line(s)",
        "plan only: nothing was written, no audit line recorded",
    ]


def _log_length(key: ts.Key) -> int:
    """How many entries a log holds, decodable or not, excluding an in-flight torn tail."""
    page = ts.read_log(key)
    return len(page.records) + len(page.corrupt) + len(page.version_refused)


def clear_root(root: Path, *, by: str, reason: str) -> list[str]:
    """Remove every friction report, retrospective, audit line and learning-capture line under
    ``root``, reconcile any stale exported loose copy, and record the closing audit line.

    Raises ``AuditEntryNotWritten`` when the closing append fails; every mutation above it has
    already committed by then, since the closing line is written last.
    """
    has_database = database_file(str(root)).is_file()
    outcomes: list[str] = []
    removed_exports = 0

    report_keys = ts.keys(FRICTION_REPORT_STORE, str(root))
    for key in report_keys:
        ts.delete(key)
        if has_database and _remove_stale_export(FRICTION_REPORT_STORE, root, key.parts):
            removed_exports += 1
    outcomes.append(f"friction_reports: removed {len(report_keys)} record(s)")

    retro_keys = ts.keys(RETROSPECTIVE_STORE, str(root))
    for key in retro_keys:
        ts.delete(key, expect=ts.read_versioned(key).version)
        if has_database and _remove_stale_export(RETROSPECTIVE_STORE, root, key.parts):
            removed_exports += 1
    outcomes.append(f"retrospectives: removed {len(retro_keys)} record(s)")

    audit_key = audit_log_key(root)
    audit_removed = ts.clear_log(audit_key)
    if has_database and _remove_stale_export(AUDIT_LOG_STORE, root, audit_key.parts):
        removed_exports += 1
    outcomes.append(f"audit_log: removed {audit_removed} line(s)")

    capture_key = learning_capture_key(root)
    capture_removed = ts.clear_log(capture_key)
    if has_database and _remove_stale_export(LEARNING_CAPTURE_STORE, root, capture_key.parts):
        removed_exports += 1
    outcomes.append(f"learning_capture: removed {capture_removed} line(s)")

    if has_database:
        outcomes.append(
            f"removed {removed_exports} stale exported file(s) under .tcip so the export "
            "stamp and the files agree"
        )

    record_event_or_raise(
        "clear_dev_history",
        {
            "root": str(root),
            "by": by,
            "reason": reason,
            "friction_reports_removed": len(report_keys),
            "retrospectives_removed": len(retro_keys),
            "audit_log_lines_removed": audit_removed,
            "learning_capture_lines_removed": capture_removed,
        },
        scope=root,
    )
    outcomes.append(f"closing audit line recorded (by={by})")
    return outcomes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument(
        "--apply", action="store_true",
        help="remove development-era history; writes nothing without this flag")
    ap.add_argument(
        "--plan", action="store_true",
        help="preview only; the default when --apply is not given")
    ap.add_argument(
        "--by", default=None,
        help="the operator's identity for the closing audit line, e.g. user:zack; required "
             "with --apply")
    ap.add_argument(
        "--reason", default=None,
        help="why development history is being cleared; required with --apply")
    args = ap.parse_args()

    by = ""
    reason = ""
    if args.apply:
        if not (args.by or "").strip() or not (args.reason or "").strip():
            ap.error("--apply requires both --by and --reason, stated by the operator")
        by = user_identity(args.by)
        reason = args.reason.strip()

    bind_default()

    refused_any = False
    for root_arg in args.roots:
        root = root_arg.resolve()
        if not (root / ".tcip").is_dir():
            print(f"{root}: refused, no .tcip directory found; not a project root")
            refused_any = True
            continue
        if args.apply:
            try:
                outcomes = clear_root(root, by=by, reason=reason)
            except AuditEntryNotWritten as exc:
                print(f"{root}: refused, AuditEntryNotWritten: {exc}")
                refused_any = True
                continue
        else:
            outcomes = plan_root(root)
        for line in outcomes:
            print(f"{root}: {line}")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())
