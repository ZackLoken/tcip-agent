"""Export an annotation project as a portable bundle: a ZIP archive, or, with --output-dir, the
identical bundle written as a directory tree.

The operator/agent entry point for packaging a project's tree, its store databases copied
consistently, into one bundle an ``import-project`` run can restore from elsewhere. Wraps
``tcip_mcp.tools.project_tools.archive_project`` with no MCP tool registration.

    tcip archive-project <project_path> --output-path PATH [--include-models]
    tcip archive-project <project_path> --output-dir DIR [--include-models]

Exactly one of --output-path/--output-dir is required. This run's audit line is recorded under
``<project_path>/.tcip``, the project being archived.
"""

from __future__ import annotations

import argparse
import json
import sys

from tcip_mcp.cli import bound_project


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    from tcip_mcp.tools.project_tools import (
        TreeBundle,
        ZipBundle,
        archive_project,
    )

    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument(
        "project_path",
        help="Root directory of the project. Also where this "
        "run's audit line is recorded.",
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output-path",
        dest="destination",
        metavar="PATH",
        type=lambda path: ZipBundle(output_path=path),
        help="Destination path for the ZIP file.",
    )
    destination.add_argument(
        "--output-dir",
        dest="destination",
        metavar="DIR",
        type=lambda path: TreeBundle(output_dir=path),
        help="Destination directory to write the bundle into as a tree, "
        "instead of a ZIP. Refuses a destination inside project_path or a "
        "destination that already holds anything.",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Include registered model checkpoints (can be large).",
    )
    args = parser.parse_args(argv)

    project = bound_project(args.project_path)

    result = archive_project(
        project, args.destination, include_models=args.include_models
    )
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
