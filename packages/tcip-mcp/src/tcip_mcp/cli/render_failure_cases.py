"""Find and render the worst predictions for failure analysis.

Ranks by a count-mismatch + low-confidence heuristic (``get_worst_predictions``); no IoU matching,
so an image with the right box count but every box mislocated scores as good. Not a substitute for
``score_predictions(detail=True)``'s IoU-matched TP/FP/FN when mislocalization itself is the
question. Wraps ``tcip_mcp.tools.vision_tools.render_failure_cases`` and prints the grid image's
path.

    tcip render-failure-cases --dataset-root <root> --bucket <name> --project <project>
        [--task detect|segment] [--top-k N] [--class-names NAMES]

``--project`` names where this run's audit line and the rendered images land; the bucket named
``--bucket`` under ``--dataset-root`` is what gets read, beside its images' label documents.
"""

from __future__ import annotations

import argparse
import json
import sys

from tcip_mcp.cli import bound_project


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("--dataset-root", required=True,
                        help="The dataset root the bucket is published under.")
    parser.add_argument("--bucket", required=True, help="The published bucket's name.")
    parser.add_argument("--project", required=True,
                         help="The project this run's audit line and renders land under.")
    parser.add_argument("--task", default="detect", choices=("detect", "segment"))
    parser.add_argument("--top-k", type=int, default=10, help="Number of worst cases to render.")
    parser.add_argument("--class-names", default="", help="Comma-separated class names.")
    args = parser.parse_args(argv)

    project = bound_project(args.project)

    from tcip_mcp.tools.vision_tools import render_failure_cases

    result = render_failure_cases(
        project,
        dataset_root=args.dataset_root,
        bucket=args.bucket,
        task=args.task,
        top_k=args.top_k,
        class_names=args.class_names,
    )
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
