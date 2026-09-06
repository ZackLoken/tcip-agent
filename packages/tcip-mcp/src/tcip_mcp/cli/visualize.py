"""Render annotations, predictions, a GT-vs-prediction comparison, or a sample grid, from the
command line.

The demoted twin of ``vision_tools.visualize``: one entry point for the common renders, saved to
``.tcip/artifacts/viz/``, path returned for a human to read. It writes an artifact and carries a
platform audit line, so it stays a command rather than a bare library call: --project (or
$TCIP_STATE_ROOT) is required, since both land under it.

Usage:
    tcip visualize --source annotations --path <image.jpg> \
        --project <platform_root> [--task detect] [--class-names leaf,fruit,bud] \
        [--conf-threshold <default>] [--iou-threshold 0.5] [--n 16]

--source is one of 'annotations' (path = image file), 'predictions' (path = image file),
'comparison' (path = image file), or 'dataset' (path = dataset folder containing images/ and
labels/, tiles --n random annotated samples into a grid).
"""

from __future__ import annotations

import argparse
import json

from tcip_mcp.project_paths import require_and_pin_platform_root


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("--source", required=True,
                        choices=["annotations", "predictions", "comparison", "dataset"],
                        help="What to render.")
    parser.add_argument("--path", required=True,
                        help="Image file (annotations/predictions/comparison) or dataset "
                             "folder (dataset).")
    parser.add_argument("--project", default=None,
                        help="Platform state root the artifact and the audit line land under. "
                             "Required (or set $TCIP_STATE_ROOT).")
    parser.add_argument("--task", default="detect", choices=["detect", "segment"])
    parser.add_argument("--class-names", default="",
                        help="Comma-separated class names (e.g. 'leaf,fruit,bud').")
    parser.add_argument("--conf-threshold", type=float, default=None,
                        help="Minimum confidence; filters displayed predictions (source="
                             "predictions) and the predictions matched against GT (source="
                             "comparison). Omitted uses the tool's own shared default.")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for a positive match (source=comparison only).")
    parser.add_argument("--n", type=int, default=16,
                        help="Number of samples in the grid (source=dataset only).")
    args = parser.parse_args(argv)

    require_and_pin_platform_root(args.project)

    from tcip_store.binding import bind_default

    bind_default()

    from tcip_mcp.tools.vision_tools import DEFAULT_CONF, visualize

    conf_threshold = args.conf_threshold if args.conf_threshold is not None else DEFAULT_CONF
    result = visualize(
        args.source, args.path, task=args.task, class_names=args.class_names,
        conf_threshold=conf_threshold, iou_threshold=args.iou_threshold, n=args.n,
    )
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
