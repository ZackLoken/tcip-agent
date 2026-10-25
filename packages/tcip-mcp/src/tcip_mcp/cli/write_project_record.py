"""Correct one field of a project's record, the deliberate overwrite for an answer typed wrong once.

    tcip write-project-record site <project> <site> --user <name>
    tcip write-project-record invariances <project> --[no-]horizontal-flip ... --rotation <degrees>
        --user <name>

Each correction replaces one field, keeping the record's id and every other field, records one
line naming the person in the project's log, and refuses a project whose record does not read.

Exit codes: 0 written, 2 refused (nothing written).
"""

from __future__ import annotations

import argparse
import sys

from tcip_mcp.cli import bound_project, hyphenated
from tcip_mcp.identity import actor
from tcip_mcp.project_record import (
    CaptureInvariances,
    replace_invariances,
    replace_site,
)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Replace the record field ``argv`` names and print the previous and written values.

    Returns 0 when written, or 2 after printing the refusal when the project, the person or the
    value is refused.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("project", help="the project directory")
    common.add_argument(
        "--user", required=True, help="the person making the correction"
    )
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], prog=prog
    )
    fields = ap.add_subparsers(dest="field", required=True)
    fields.add_parser("site", parents=[common]).add_argument(
        "site", help="the orchard or station this project's plants stand in"
    )
    invariances = fields.add_parser("invariances", parents=[common])
    for name, info in CaptureInvariances.model_fields.items():
        flag = f"--{hyphenated(name)}"
        if info.annotation is bool:
            invariances.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                required=True,
                help=info.description,
            )
        else:
            invariances.add_argument(
                flag, type=float, required=True, help=info.description
            )
    args = ap.parse_args(argv)
    try:
        project = bound_project(args.project)
        person = actor(args.user)
        if args.field == "site":
            site = replace_site(project, args.site, actor=person)
            previous, written = site["previous_site"], site["site"]
        else:
            block = CaptureInvariances.model_validate(
                {
                    name: getattr(args, name)
                    for name in CaptureInvariances.model_fields
                }
            )
            answer = replace_invariances(project, block, actor=person)
            previous, written = answer.previous_invariances, answer.invariances
    except (SystemExit, ValueError) as exc:
        print(f"refused: {exc}")
        return 2
    print(
        f"written: {project} recorded {args.field} {previous!r}, now records {written!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
