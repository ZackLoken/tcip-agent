"""Export an annotation project as a portable ZIP archive.

The operator/agent entry point for packaging a project (images, ground truth, class registry,
``.tcip`` state, experiments and their claimed manifests, plus every recognized blob home) into
one ZIP an ``import_project`` run can restore from elsewhere. Wraps
``tcip_mcp.tools.project_tools.archive_project`` with no MCP tool registration.

    python scripts/archive_project.py <project_path> [--output-path PATH] [--include-models]
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_path", help="Root directory of the project.")
    parser.add_argument("--output-path", default="",
                         help="Destination path for the ZIP file. Defaults to "
                              "<project_name>.tcip.zip beside the project.")
    parser.add_argument("--include-models", action="store_true",
                         help="Include registered model checkpoints (can be large).")
    args = parser.parse_args(argv)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    # archive_project exports every database under the tree before it composes the bundle, so
    # every store must already be registered, not just the ones project_tools.py itself defines.
    import tcip_mcp.store_catalogue  # noqa: F401
    from tcip_mcp.tools.project_tools import archive_project

    bind_default()

    result = archive_project(
        args.project_path, output_path=args.output_path, include_models=args.include_models,
    )
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
