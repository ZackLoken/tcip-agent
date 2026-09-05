"""Clear a root's development-era audit lines, friction reports, retrospectives and
learning-capture lines before it reaches an alpha tester.

The production audit trail an alpha tester reads serves a different purpose than the
remediation program's own records: a tester should see the platform's own doors acting on
their data, not every experiment this repo's own development ran against the same root. This
removes exactly four things under a named root, through the storage seam and never around it:
every ``friction_reports`` record, every ``retrospectives`` record, the ``audit_log`` log's
entries, and the ``learning_capture`` log's entries. Nothing else: not annotations, images,
predictions, models, experiments, caches, or any other store. On a file-backend root, clearing
``audit_log`` or ``learning_capture`` leaves the seam's own hidden ``.clearbase`` sidecar beside
each cleared log, the cursor watermark that keeps a cursor taken before this run comparable to
one taken after; the sidecar is the seam's bookkeeping for the log it sits beside, not a fifth
thing this script touches on its own account.

When the root's state lives in a database (``TCIP_STORE_BACKEND=sqlite``, the default) and an
earlier ``export_store.py`` run left loose copies of any of the four beside it, this script
leaves those copies exactly as they are: the seam's own record and log deletes tombstone what
they remove, so each store's ``store_counters`` row reads as behind the database from the
moment this returns, and the next ``python scripts/export_store.py <root>`` run is what deletes
the stale file, driven by that tombstone. This script does not run the export machinery itself:
doing so would rewrite every other store's exported files too, on the strength of whatever else
has changed in the database since they were last exported, which would no longer be "nothing
else" for a script that touches exactly four stores. A file-backend root has no such copies to
reconcile: its loose files are the primary storage, already correct once the seam operations
above return.

After clearing, one audit line is written into the now-empty ``audit_log``, naming the operator
(``--by``) and the reason (``--reason``) and carrying every count this run removed, so the
production log's first line states that development history was cleared and by whom. A failed
append raises ``AuditEntryNotWritten`` and this script exits nonzero naming it; every mutation
before that point has already committed, since the closing line is written last.

    python scripts/clear_dev_history.py <root> [<root> ...] [--plan]
    python scripts/clear_dev_history.py <root> [<root> ...] --apply --by <user:name> --reason <text>

``--plan`` is the default and mutually exclusive with ``--apply``: it prints exactly what
``--apply`` would remove, with counts, and writes nothing, including no audit line. ``--apply``
requires both ``--by`` and ``--reason``, refused before any root is touched when either is
missing or when ``--plan`` is given alongside it; a person's identity is normalized through
``tcip_mcp.identity.user_identity``. A root holding no ``.tcip`` directory is refused by name.

Exit codes: 0 when every named root was planned or cleared; 2 when a root was refused (no
``.tcip`` directory, ``--plan`` given with ``--apply``, missing ``--by``/``--reason``, a
``StoreError`` from the seam itself such as a busy lock or a mis-bound backend, or a failed
closing audit append naming ``AuditEntryNotWritten``). Every root named on the command line is
still processed, even after another root's refusal.
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
from tcip_web.agent_learning_capture import learning_capture_key  # noqa: E402

_DATABASE_EXPORT_NOTE = (
    "When a database backs the root, an earlier export's loose copies of what this clears "
    "still hold it until `python scripts/export_store.py <root>` runs."
)


def plan_root(root: Path) -> list[str]:
    """Every outcome line naming what ``--apply`` would remove under ``root``, writing nothing."""
    n_reports = len(ts.keys(FRICTION_REPORT_STORE, str(root)))
    n_retros = len(ts.keys(RETROSPECTIVE_STORE, str(root)))
    n_audit = _log_length(audit_log_key(root))
    n_capture = _log_length(learning_capture_key(root))
    outcomes = [
        f"would remove {n_reports} friction_reports record(s)",
        f"would remove {n_retros} retrospectives record(s)",
        f"would remove {n_audit} audit_log line(s)",
        f"would remove {n_capture} learning_capture line(s)",
        "plan only: nothing was written, no audit line recorded",
    ]
    if database_file(str(root)).is_file():
        outcomes.append(_database_export_outcome_line(root, applied=False))
    return outcomes


def _log_length(key: ts.Key) -> int:
    """How many entries a log holds, decodable or not, excluding an in-flight torn tail."""
    page = ts.read_log(key)
    return len(page.records) + len(page.corrupt) + len(page.version_refused)


def _database_export_outcome_line(root: Path, *, applied: bool) -> str:
    """What a database-backed root's exported loose copies still hold, and the command that
    reconciles them; ``applied`` picks the tense between an ``--apply`` run and a ``--plan``
    preview of the same fact."""
    holds = (
        "still hold what this run just removed" if applied
        else "would still hold what --apply removes"
    )
    return (
        f"a database backs this root; any exported .tcip/audit.jsonl, "
        f"learning_capture.jsonl, reports/*.json or retrospectives/*.md {holds} until "
        f"python scripts/export_store.py {root} runs"
    )


def clear_root(root: Path, *, by: str, reason: str) -> list[str]:
    """Remove every friction report, retrospective, audit line and learning-capture line under
    ``root``, and record the closing audit line.

    Raises ``AuditEntryNotWritten`` when the closing append fails; every mutation above it has
    already committed by then, since the closing line is written last. Under a database
    backend, this leaves a stale exported loose copy of any of the four exactly where it was;
    see the module docstring for why the next ``export_store.py`` run is what removes it.
    """
    outcomes: list[str] = []

    report_keys = ts.keys(FRICTION_REPORT_STORE, str(root))
    for key in report_keys:
        ts.delete(key)
    outcomes.append(f"friction_reports: removed {len(report_keys)} record(s)")

    retro_keys = ts.keys(RETROSPECTIVE_STORE, str(root))
    for key in retro_keys:
        ts.delete(key, expect=ts.read_versioned(key).version)
    outcomes.append(f"retrospectives: removed {len(retro_keys)} record(s)")

    audit_removed = ts.clear_log(audit_log_key(root))
    outcomes.append(f"audit_log: removed {audit_removed} line(s)")

    capture_removed = ts.clear_log(learning_capture_key(root))
    outcomes.append(f"learning_capture: removed {capture_removed} line(s)")

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
    if database_file(str(root)).is_file():
        outcomes.append(_database_export_outcome_line(root, applied=True))
    return outcomes


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__doc__.splitlines()[0]} {_DATABASE_EXPORT_NOTE}"
    )
    ap.add_argument("roots", nargs="+", type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true",
        help="remove development-era history; writes nothing without this flag")
    mode.add_argument(
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
            except (AuditEntryNotWritten, ts.StoreError) as exc:
                print(f"{root}: refused, {type(exc).__name__}: {exc}")
                refused_any = True
                continue
        else:
            outcomes = plan_root(root)
        for line in outcomes:
            print(f"{root}: {line}")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())
