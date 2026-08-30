"""Import an annotation project from a ZIP archive.

The operator/agent entry point for restoring a project ``scripts/archive_project.py`` bundled:
extracts into private staging, refuses on any bookkeeping, collided, undecodable or unaccounted
member, adopts what is left into a database when this process is bound to the database backend,
then moves the staged tree onto ``destination``. Wraps
``tcip_mcp.tools.project_tools.import_project`` with no MCP tool registration.

    python scripts/import_project.py <zip_path> <destination>

This run's audit line is recorded under ``<destination>/.tcip``, the project being restored,
not the process cwd.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _script_root import pin_project_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", help="Path to the .tcip.zip archive.")
    parser.add_argument("destination", help="Directory to extract into; must not already exist, "
                                             "or must be an empty directory. Also where this "
                                             "run's audit line is recorded.")
    args = parser.parse_args(argv)

    pin_project_root(args.destination)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    from tcip_mcp.tools.project_tools import import_project

    bind_default()

    result = import_project(args.zip_path, args.destination)
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
