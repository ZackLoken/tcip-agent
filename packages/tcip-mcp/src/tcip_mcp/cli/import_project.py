"""Import an annotation project from a bundle ``tcip archive-project`` wrote: a ZIP archive, or a
directory tree written by its ``--output-dir`` mode.

The operator/agent entry point for restoring a project ``tcip archive-project`` bundled: extracts
the bundle onto ``destination``, refusing a member that is the store's own bookkeeping or that
escapes the destination. Wraps ``tcip_mcp.tools.project_tools.import_project`` with no MCP tool
registration.

    tcip import-project <bundle_path> <destination>

``bundle_path`` names either container. This run's audit line is recorded under
``<destination>/.tcip``, the project being restored.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("bundle_path", help="Path to the bundle archive-project wrote: a ZIP "
                                            "file, or a directory tree written by its "
                                            "--output-dir mode.")
    parser.add_argument("destination", help="Directory to extract into; must not already exist, "
                                             "or must be an empty directory. Also where this "
                                             "run's audit line is recorded.")
    args = parser.parse_args(argv)

    from tcip_store import bind

    from tcip_mcp.tools.project_tools import import_project

    bind()

    result = import_project(args.bundle_path, args.destination)
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
