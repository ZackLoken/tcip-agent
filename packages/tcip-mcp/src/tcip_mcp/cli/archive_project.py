"""Export an annotation project as a portable bundle: a ZIP archive, or, with --output-dir, the
identical bundle written as a directory tree.

The operator/agent entry point for packaging a project's tree, its store databases copied
consistently, into one bundle an ``import-project`` run can restore from elsewhere. Wraps
``tcip_mcp.tools.project_tools.archive_project`` with no MCP tool registration.

    tcip archive-project <project_path> --output-path PATH [--include-models]
    tcip archive-project <project_path> --output-dir DIR [--include-models]

Exactly one of --output-path/--output-dir is required; a relative one resolves against the
working directory. This run's audit line is recorded under
``<project_path>/.tcip``, the project being archived.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tcip_mcp.cli import bound_project


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("project_path", help="Root directory of the project. Also where this "
                                              "run's audit line is recorded.")
    parser.add_argument("--output-path", default="",
                         help="Destination path for the ZIP file. Exactly one of "
                              "--output-path/--output-dir is required.")
    parser.add_argument("--output-dir", default="",
                         help="Destination directory to write the bundle into as a tree, "
                              "instead of a ZIP. Refuses a destination inside project_path or a "
                              "destination that already holds anything.")
    parser.add_argument("--include-models", action="store_true",
                         help="Include registered model checkpoints (can be large).")
    args = parser.parse_args(argv)

    project = bound_project(args.project_path)

    from tcip_mcp.tools.project_tools import archive_project

    result = archive_project(
        project, output_path=args.output_path and str(Path(args.output_path).resolve()),
        output_dir=args.output_dir and str(Path(args.output_dir).resolve()),
        include_models=args.include_models,
    )
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
