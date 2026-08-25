"""Conform a project's ``annotation_stats`` record to drop its dead ``image_status`` key.

A deliberate one-off operator script, per this repo's no-backward-compatibility boundary
(CLAUDE.md): every writer of this record put an empty ``{"sessions": [...], "image_status": {}}``
on disk because the shape guard defaulted an absent document to that shape, but nothing ever reads
``image_status`` back out of it (the dataset's real image status lives in a separate store,
addressed by ``image_status_key``). This never runs as part of any runtime path.

    python scripts/drop_annotation_stats_image_status.py <project_root> [<project_root> ...]
    python scripts/drop_annotation_stats_image_status.py --plan <project_root>

For each root named: a record with no ``annotation_stats`` document is left alone (nothing to
conform); a document with no ``image_status`` key is left alone (already conformed); a document
whose ``image_status`` is present and empty has the key dropped. A document whose ``image_status``
is present and non-empty is refused rather than conformed: real data there would mean a writer this
script does not know about, and this script's job is to drop a dead key, not to decide what to do
with one that turns out to be live.

Exit codes: 0 for every root conformed (or already clean); 2 if any root was refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
for _pkg in ("tcip-store", "tcip-annotation", "tcip-mcp", "tcip-web"):
    sys.path.insert(0, str(_REPO_ROOT / "packages" / _pkg / "src"))

import tcip_store as ts  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402
from tcip_web.routes.sessions import annotation_stats_key  # noqa: E402


def conform_root(root: Path, *, plan: bool) -> str:
    """The one outcome line for ``root``: nothing to conform, conformed, or refused."""
    key = annotation_stats_key(str(root))
    stored = ts.read_versioned(key, default=None)
    data = stored.value
    if not isinstance(data, dict) or "image_status" not in data:
        return "nothing to conform"
    image_status = data["image_status"]
    if image_status != {}:
        return (f"refused: image_status is not empty ({image_status!r}); a writer this "
                "script does not know about put real data there")
    if plan:
        return "would drop the empty image_status key"
    conformed = {k: v for k, v in data.items() if k != "image_status"}
    ts.replace(key, conformed, expect=stored.version)
    return "dropped the empty image_status key"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    bind_default()

    refused = False
    for root in args.roots:
        root = root.resolve()
        try:
            outcome = conform_root(root, plan=args.plan)
        except ts.StoreError as exc:
            outcome = f"refused, {exc}"
        if outcome.startswith("refused"):
            refused = True
        print(f"{root}: {outcome}")

    return 2 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
