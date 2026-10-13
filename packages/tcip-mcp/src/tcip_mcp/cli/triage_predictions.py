r"""Sort a checkpoint's own predictions by confidence into needs-review and unscoreable queues,
from the command line.

Wraps ``feedback_tools.triage_predictions``, which writes nothing.

Usage:
    tcip triage-predictions --checkpoint <ckpt.pt> --images-dir <dir> \
        --project <project> --low <conf> --high <conf> --rationale <text> \
        [--subject <subject>]

The checkpoint must be named by a registry entry under --project (register it with register_model
first); this command refuses one it is not, naming the digest and the project.
"""

from __future__ import annotations

import argparse
import json

from tcip_mcp.cli import bound_project


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument(
        "--checkpoint", required=True, help="Trained model checkpoint."
    )
    parser.add_argument(
        "--images-dir", required=True, help="Directory of candidate images."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="The project the checkpoint's registry entry is looked up under.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Skip the images whose label document marks this subject finished. "
        "Omitted triages every image.",
    )
    parser.add_argument(
        "--low",
        type=float,
        required=True,
        help="Lower confidence bound for the needs-review band, inclusive, and a "
        "detector's pass conf.",
    )
    parser.add_argument(
        "--high",
        type=float,
        required=True,
        help="Upper confidence bound for the needs-review band, exclusive.",
    )
    parser.add_argument(
        "--rationale", required=True, help="Why the band is stated as it is."
    )
    args = parser.parse_args(argv)

    project = bound_project(args.project)

    from tcip_mcp.tools.feedback_tools import triage_predictions

    result = triage_predictions(
        project,
        args.checkpoint,
        args.images_dir,
        low=args.low,
        high=args.high,
        rationale=args.rationale,
        subject=args.subject,
    )
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
