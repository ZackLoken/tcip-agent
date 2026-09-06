"""Score on-disk predictions against on-disk ground truth (COCOeval), from the command line.

The demoted twin of ``annotation_tools.score_predictions``: a single image file returns
per-box matches (plus an optional per-detection breakdown with ``detail``) for a human to read;
a dataset directory returns aggregate metrics plus per-image TP/FP/FN. Both regimes share
``coco_detection_metrics``. It is a command, not an MCP tool, per CLAUDE.md: it only reads, and an
agent that already has the MCP server up calls the library function directly; this is the door
for a harness with no MCP tool for it, or an operator scoring a batch outside any agent session.

Usage:
    tcip score-predictions --path <image_or_dataset_dir> \
        [--project <platform_root>] [--iou-threshold 0.5] [--conf-threshold <default>] \
        [--detail] [--trait <trait_name>]

--project (or $TCIP_STATE_ROOT) is required only when --trait is given, since resolving a
trait's derived localization criterion reads the project's own trait registry.
"""

from __future__ import annotations

import argparse
import json

from tcip_mcp.project_paths import require_platform_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True,
                        help="Absolute path to an image file (single-image match) or a dataset "
                             "root (aggregate).")
    parser.add_argument("--project", default=None,
                        help="Platform state root the trait registry is read under. Required "
                             "(or set $TCIP_STATE_ROOT) only when --trait is given.")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for a positive match (the AP@0.5 comparability "
                             "convention).")
    parser.add_argument("--conf-threshold", type=float, default=None,
                        help="Minimum confidence to consider a prediction. Omitted uses the "
                             "tool's own default.")
    parser.add_argument("--detail", action="store_true",
                        help="Single-image only: also return the per-detection breakdown.")
    parser.add_argument("--trait", default=None,
                        help="When set, the trait's derived localization criterion governs the "
                             "reported TP/FP/FN count; map50 stays a labeled comparability "
                             "metric. Absent -> the IoU convention governs.")
    args = parser.parse_args(argv)

    if args.trait:
        require_platform_root(args.project)

    from tcip_store.binding import bind_default

    bind_default()

    from tcip_mcp.tools.annotation_tools import DEFAULT_CONF, score_predictions

    conf_threshold = args.conf_threshold if args.conf_threshold is not None else DEFAULT_CONF
    result = score_predictions(
        args.path, iou_threshold=args.iou_threshold, conf_threshold=conf_threshold,
        detail=args.detail, trait=args.trait,
    )
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
