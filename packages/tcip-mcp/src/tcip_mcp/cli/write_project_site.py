"""Write or correct one project's authored site: the record ``initialize_project``/``ingest_images``
themselves cannot reach for a project whose name does not fit the workspace scheme, and the one
deliberate overwrite for a site typed wrong once or a record damaged by hand.

    tcip write-project-site <project_root> <site> [--replace]

Without ``--replace`` this is the same create-only write ``tcip_mcp.project_record.record_site``
always is: an absent record is written, a present record with the same site is left alone, and a
present record with a different or unreadable site refuses. With ``--replace`` the write is
unconditional, whatever was there.

Exit codes: 0 written or already recorded the same, 2 refused (nothing written).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tcip_store.binding import bind_default
from tcip_store.errors import StoreError

from tcip_mcp.project_record import record_site


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog=prog)
    ap.add_argument("project_root", help="project directory holding .tcip/")
    ap.add_argument("site", help="the orchard or station this project's plants stand in")
    ap.add_argument(
        "--replace", action="store_true",
        help="overwrite a present record unconditionally, valid or not",
    )
    args = ap.parse_args(argv)
    root = Path(args.project_root)
    if not root.is_dir():
        print(f"error: not a directory: {root}")
        return 2

    bind_default()

    try:
        result = record_site(str(root), args.site, replace=args.replace)
    except (ValueError, StoreError) as exc:
        print(f"refused: {exc}")
        return 2

    site = result["site"]
    previous = result["previous_site"]
    problem = result.get("previous_record_problem")
    if previous is None and problem is None:
        print(f"written: {root} now records site {site!r}")
    elif previous == site:
        print(f"already recorded the same: {root} already records site {site!r}")
    elif previous is not None:
        print(f"replaced: {root} recorded site {previous!r}, now records {site!r}")
    else:
        print(f"replaced: {root}'s prior record could not be read ({problem}), now records "
              f"{site!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
