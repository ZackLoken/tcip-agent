"""Report the host's current compute headroom and the project's active training runs.

The headroom is CPU, memory and GPU free bytes, beside how many training runs the project already
has active. Reports, never caps.

Wraps ``tcip_mcp.tools.training_tools.inspect_compute_resources``.

    tcip inspect-compute-resources --project <project>

``--project`` names the project this run's active-run count resolves against.
"""

from __future__ import annotations

import argparse
import json
import sys

from tcip_mcp.cli import bound_project


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Print the compute headroom report for the project named in ``argv`` as JSON; return 0."""
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument(
        "--project",
        required=True,
        help="The project this run resolves against.",
    )
    args = parser.parse_args(argv)

    project = bound_project(args.project)

    from tcip_mcp.tools.training_tools import inspect_compute_resources

    result = inspect_compute_resources(project)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
