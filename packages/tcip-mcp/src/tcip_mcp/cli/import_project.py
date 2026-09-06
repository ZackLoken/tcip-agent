"""Import an annotation project from a bundle ``tcip archive-project`` wrote: a ZIP archive, or a
directory tree written by its ``--output-dir`` mode.

The operator/agent entry point for restoring a project ``tcip archive-project`` bundled: stages
the bundle into a private directory, refuses on any bookkeeping, collided, undecodable or
unaccounted member, adopts what is left into a database when this process is bound to the
database backend, then moves the staged tree onto ``destination``. Wraps
``tcip_mcp.tools.project_tools.import_project`` with no MCP tool registration.

    tcip import-project <bundle_path> <destination>

``bundle_path`` names either container. This run's audit line is recorded under
``<destination>/.tcip``, the project being restored, not the process cwd.
"""

from __future__ import annotations

import argparse
import json
import sys

from tcip_mcp.project_paths import require_platform_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", help="Path to the bundle archive-project wrote: a ZIP "
                                            "file, or a directory tree written by its "
                                            "--output-dir mode.")
    parser.add_argument("destination", help="Directory to extract into; must not already exist, "
                                             "or must be an empty directory. Also where this "
                                             "run's audit line is recorded.")
    args = parser.parse_args(argv)

    require_platform_root(args.destination)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    from tcip_mcp.tools.project_tools import import_project

    bind_default()

    result = import_project(args.bundle_path, args.destination)
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
