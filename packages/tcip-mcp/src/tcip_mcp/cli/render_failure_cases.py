"""Find and render the worst predictions for failure analysis.

Ranks by a count-mismatch + low-confidence heuristic (``get_worst_predictions``); no IoU matching,
so an image with the right box count but every box mislocated scores as good. Not a substitute for
``score_predictions(detail=True)``'s IoU-matched TP/FP/FN when mislocalization itself is the
question. Wraps ``tcip_mcp.tools.vision_tools.render_failure_cases`` and prints the grid image's
path.

    tcip render-failure-cases --dataset-root <root> --bucket <name> --project <project>
        --top-k N --shortfall-weight W --surplus-weight W --confidence-weight W
        [--task detect|segment] [--class-names NAMES]

``--project`` names where the rendered images land and what a relative ``--dataset-root`` lies
under; the bucket named ``--bucket`` under ``--dataset-root`` is what gets read, beside its images'
label documents.
"""

from __future__ import annotations

import argparse
import json
import sys

from tcip_mcp.cli import bound_project


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="The dataset root the bucket is published under.",
    )
    parser.add_argument(
        "--bucket", required=True, help="The published bucket's name."
    )
    parser.add_argument(
        "--project", required=True, help="The project the renders land under."
    )
    parser.add_argument(
        "--task", default="detect", choices=("detect", "segment")
    )
    parser.add_argument(
        "--top-k",
        type=int,
        required=True,
        help="Number of worst cases to render.",
    )
    for weight, help_text in (
        (
            "shortfall",
            "each ground-truth object the predictions fall short of",
        ),
        ("surplus", "each prediction beyond the ground-truth count"),
        ("confidence", "one minus the predictions' mean confidence"),
    ):
        parser.add_argument(
            f"--{weight}-weight",
            type=float,
            required=True,
            help=f"The ranking's weight of {help_text}.",
        )
    parser.add_argument(
        "--class-names", default="", help="Comma-separated class names."
    )
    args = parser.parse_args(argv)

    project = bound_project(args.project)

    from tcip_mcp.tools.vision_tools import render_failure_cases

    result = render_failure_cases(
        project,
        dataset_root=args.dataset_root,
        bucket=args.bucket,
        top_k=args.top_k,
        shortfall_weight=args.shortfall_weight,
        surplus_weight=args.surplus_weight,
        confidence_weight=args.confidence_weight,
        task=args.task,
        class_names=args.class_names,
    )
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
