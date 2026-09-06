"""Find and render the worst predictions for failure analysis.

Ranks by a count-mismatch + low-confidence heuristic (``get_worst_predictions``); no IoU
matching, so an image with the right box count but every box mislocated scores as good. Not a
substitute for ``score_predictions(detail=True)``'s IoU-matched TP/FP/FN when mislocalization
itself is the question. Wraps ``tcip_mcp.tools.vision_tools.render_failure_cases`` with no MCP
tool registration; the agent reads the grid image this prints the path to with its own
image-capable tool, then describes and recommends.

    tcip render-failure-cases <predictions_dir> <labels_dir> --project <project_root>
        [--images-dir DIR] [--task detect|segment] [--top-k N] [--class-names NAMES]

``--project`` (or an already-set ``$TCIP_STATE_ROOT``) names where this run's audit line and
the rendered grid's cache path land; ``predictions_dir``/``labels_dir`` stay what gets read.
"""

from __future__ import annotations

import argparse
import json
import sys

from tcip_mcp.project_paths import require_and_pin_platform_root


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("predictions_dir", help="Directory with prediction files.")
    parser.add_argument("labels_dir", help="Directory with ground-truth label files.")
    parser.add_argument("--project", default="",
                         help="Project root this run's audit line is recorded under; falls "
                              "back to $TCIP_STATE_ROOT.")
    parser.add_argument("--images-dir", default="",
                         help="Directory with source images. Auto-detected if omitted.")
    parser.add_argument("--task", default="detect", choices=("detect", "segment"))
    parser.add_argument("--top-k", type=int, default=10, help="Number of worst cases to render.")
    parser.add_argument("--class-names", default="", help="Comma-separated class names.")
    args = parser.parse_args(argv)

    require_and_pin_platform_root(args.project or None)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    from tcip_mcp.tools.vision_tools import render_failure_cases

    bind_default()

    result = render_failure_cases(
        predictions_dir=args.predictions_dir,
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
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
