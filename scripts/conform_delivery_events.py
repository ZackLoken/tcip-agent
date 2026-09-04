"""Check a project's stored ``delivery_events`` records against the current
``DeliveryEventRecord`` shape (``tcip_mcp.pipelines.delivery_events_schema``) and name, by
``event_id``, any that no longer validate.

This script conforms by checking and naming, never by rewriting: it is a one-off operator tool
per this repo's no-backward-compatibility boundary (CLAUDE.md), but unlike
``conform_view_coverage_viewing.py`` it has nothing to reshape. A ``delivery_events`` record
predating the three ``plant_mapping`` disclosure keys (``dates_delivered``, ``images_unattributed``,
``plant_attribution``) or the delivered file's own ``output_sha256`` carries no value for them
anywhere on the record: they were never computed for that delivery, so there is no old shape to
map forward, only a gap to name. A refused record is left exactly as stored; the only remedy is a
fresh delivery through the door that writes this record, or removing the record by hand.

``--plan`` is accepted for the same invocation shape as the other conform scripts in this
directory, but changes nothing here: this script never writes in either mode, so there is no
write for it to preview a difference against.

    python scripts/conform_delivery_events.py <project_root> [<project_root> ...]
    python scripts/conform_delivery_events.py --plan <project_root>

Exit codes: 0 when every stored record in every named root validates (or a root holds none); 2 if
any record in any root is refused, or a named root does not exist or holds no ``.tcip`` directory
to check. Every root named on the command line is still checked and reported, even after another
root's refusal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from tcip_mcp.pipelines.delivery_events_schema import (  # noqa: E402
    DeliveryEventRecord,
    validation_error_detail,
)
from tcip_mcp.pipelines.resolution import (  # noqa: E402
    DELIVERY_EVENTS_STORE,
    delivery_events_scope,
)
from tcip_store.binding import bind_default  # noqa: E402


def _check_record(record: object) -> tuple[bool, str]:
    """Whether ``record`` validates, and the one outcome line naming it."""
    try:
        DeliveryEventRecord.model_validate(record)
    except ValidationError as exc:
        return False, (
            f"refused, {validation_error_detail(exc)}; the disclosure this record lacks was never "
            "computed for this delivery and cannot be reconstructed from it; re-deliver, or "
            "remove the record by hand"
        )
    return True, "validates, unchanged"


def check_root(root: Path) -> tuple[list[str], bool]:
    """Every outcome line for ``root``'s stored ``delivery_events`` records, and whether any of
    them was refused.

    A ``root`` holding no ``.tcip`` directory is refused by name rather than reported as a project
    with nothing stored: a mistyped path and a real, empty project would otherwise print the same
    "nothing stored" line, and the stated root is a claim the data must positively carry.
    """
    if not (root / ".tcip").is_dir():
        return ["refused, no .tcip directory found; not a project root"], True
    scope = delivery_events_scope(root)
    outcomes: list[str] = []
    refused = False
    for key in ts.keys(DELIVERY_EVENTS_STORE, str(scope)):
        event_id = key.parts[0] if key.parts else "<unknown event>"
        ok, message = _check_record(ts.read(key))
        outcomes.append(f"{event_id}: {message}")
        if not ok:
            refused = True
    return outcomes, refused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument(
        "--plan", action="store_true",
        help="preview only; this script never rewrites a record either way")
    args = ap.parse_args()

    bind_default()

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        outcomes, refused = check_root(root)
        if refused:
            refused_any = True
        if not outcomes:
            outcomes = ["nothing stored"]
        for line in outcomes:
            print(f"{root}: {line}")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())
