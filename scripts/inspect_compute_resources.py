"""Report the host's current compute headroom.

A fact to reason with before launching another concurrent training/HPO run, not an enforced
cap: the platform doesn't cap memory/CPU per run, it reports the real numbers (CPU, memory, GPU
free bytes, and how many training runs this host already has active) and leaves the judgment
call to whoever launches the next one. Wraps
``tcip_mcp.tools.training_tools.inspect_compute_resources`` with no MCP tool registration; run
it before ``launch_training``/``run_hpo`` when compute headroom is the open question.

    python scripts/inspect_compute_resources.py
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    from tcip_mcp.tools.training_tools import inspect_compute_resources

    bind_default()

    result = inspect_compute_resources()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
