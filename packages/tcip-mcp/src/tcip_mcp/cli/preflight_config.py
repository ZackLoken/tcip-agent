r"""Validate a training configuration before launching, from the command line.

Wraps ``training_tools.preflight_config``: structural checks and a builder import always run;
``--smoke`` also builds the model and runs ``check_model_contract`` (a train+eval forward at the
run's resolved dims and img_size, every attribute head included); ``--overfit`` (with
``--smoke``) additionally runs the
voluntary ``overfit_check`` diagnostic, reported but never gating.

Usage:
    tcip preflight-config --config <path.json> --project <project> \
        [--smoke] [--overfit]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcip_mcp.cli import bound_project


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("--config", required=True,
                        help="Path to a JSON file holding the full training configuration.")
    parser.add_argument("--project", required=True,
                        help="The project the config's own project-relative reads resolve under.")
    parser.add_argument("--smoke", action="store_true",
                        help="Build the model and run check_model_contract; a contract failure "
                             "is a guaranteed real-run failure, so it blocks.")
    parser.add_argument("--overfit", action="store_true",
                        help="With --smoke, also run the voluntary overfit_check diagnostic; "
                             "never gates, a noisy-but-valid model can fail it.")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    project = bound_project(args.project)

    from tcip_mcp.tools.training_tools import preflight_config

    result = preflight_config(project, config, smoke=args.smoke, overfit=args.overfit)
    print(json.dumps(result, indent=2))
    return 0 if not result.get("issues") else 2


if __name__ == "__main__":
    raise SystemExit(main())
