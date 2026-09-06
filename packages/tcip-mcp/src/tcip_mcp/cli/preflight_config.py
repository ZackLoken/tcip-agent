"""Validate a training configuration before launching, from the command line.

The demoted twin of ``training_tools.preflight_config``: structural checks and a builder import
always run; ``--smoke`` also builds the model and runs ``check_model_contract`` (a train+eval
forward at the resolved in_chans/num_classes/img_size), a guaranteed real-run failure otherwise;
``--overfit`` (with ``--smoke``) additionally runs the voluntary ``overfit_check`` diagnostic,
reported but never gating. This is the door for a harness with no MCP tool for it, or an
operator validating a config outside any agent session, before ``launch_training`` runs the
identical check itself.

Usage:
    tcip preflight-config --config <path.json> --project <platform_root> \
        [--smoke] [--overfit]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcip_mcp.project_paths import require_and_pin_platform_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Path to a JSON file holding the full training configuration.")
    parser.add_argument("--project", default=None,
                        help="Platform state root the config's own project-relative reads "
                             "resolve under. Required (or set $TCIP_STATE_ROOT).")
    parser.add_argument("--smoke", action="store_true",
                        help="Build the model and run check_model_contract; a contract failure "
                             "is a guaranteed real-run failure, so it blocks.")
    parser.add_argument("--overfit", action="store_true",
                        help="With --smoke, also run the voluntary overfit_check diagnostic; "
                             "never gates, a noisy-but-valid model can fail it.")
    args = parser.parse_args(argv)

    require_and_pin_platform_root(args.project)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    from tcip_store.binding import bind_default

    bind_default()

    from tcip_mcp.tools.training_tools import preflight_config

    result = preflight_config(config, smoke=args.smoke, overfit=args.overfit)
    print(json.dumps(result, indent=2))
    return 0 if not result.get("issues") else 2


if __name__ == "__main__":
    raise SystemExit(main())
