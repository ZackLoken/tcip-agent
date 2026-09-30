"""Correct one project's authored site, the deliberate overwrite for a site typed wrong once.

    tcip write-project-site <project> <site>

The project's record is written by ``initialize_project`` when the project is created; this command
replaces only its site, keeping its id and display name, and refuses a project whose record does
not read.

Exit codes: 0 written, 2 refused (nothing written).
"""

from __future__ import annotations

import argparse
import sys

from tcip_mcp.cli import bound_project
from tcip_mcp.project_record import replace_site


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog=prog)
    ap.add_argument("project", help="the project directory")
    ap.add_argument("site", help="the orchard or station this project's plants stand in")
    args = ap.parse_args(argv)
    try:
        project = bound_project(args.project)
        result = replace_site(project, args.site)
    except (SystemExit, ValueError) as exc:
        print(f"refused: {exc}")
        return 2
    print(f"written: {project} recorded site {result['previous_site']!r}, now records "
          f"{result['site']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
