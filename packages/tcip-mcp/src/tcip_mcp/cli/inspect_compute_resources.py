"""Report the host's current compute headroom.

A fact to reason with before launching another concurrent training/HPO run, not an enforced
cap: the platform doesn't cap memory/CPU per run, it reports the real numbers (CPU, memory, GPU
free bytes, and how many training runs this host already has active) and leaves the judgment
call to whoever launches the next one. Wraps
``tcip_mcp.tools.training_tools.inspect_compute_resources`` with no MCP tool registration; run
it before ``launch_training``/``run_hyperparameter_search`` when compute headroom is the open question.

    tcip inspect-compute-resources --project <project_root>

``--project`` (or an already-set ``$TCIP_STATE_ROOT``) names the project this run's active-run
count and audit line resolve against; without it the answer resolves against the process cwd,
which is wrong for a run count and silent about it.
"""

from __future__ import annotations

import argparse
import json
import sys

from tcip_mcp.project_paths import require_and_pin_platform_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="",
                         help="Project root this run resolves against; falls back to "
                              "$TCIP_STATE_ROOT.")
    args = parser.parse_args(argv)

    require_and_pin_platform_root(args.project or None)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    from tcip_mcp.tools.training_tools import inspect_compute_resources

    bind_default()

    result = inspect_compute_resources()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
