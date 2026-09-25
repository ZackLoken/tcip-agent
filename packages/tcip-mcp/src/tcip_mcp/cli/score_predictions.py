r"""Score on-disk predictions against on-disk ground truth (COCOeval), from the command line.

Wraps ``annotation_tools.score_predictions``: a single image file returns per-box matches (plus an
optional per-detection breakdown with ``detail``); a dataset directory returns aggregate metrics
plus per-image TP/FP/FN. Both regimes share ``coco_detection_metrics``.

Usage:
    tcip score-predictions --path <image_or_dataset_dir> \
        [--project <platform_root>] [--iou-threshold 0.5] [--conf-threshold <default>] \
        [--detail] [--trait <trait_name>]

--project (or $TCIP_STATE_ROOT) is required only when --trait is given, since resolving a trait's
derived localization criterion reads the project's own trait registry.
"""

from __future__ import annotations

import argparse
import json

from tcip_mcp.project_paths import require_and_pin_platform_root


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
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
        require_and_pin_platform_root(args.project)

    from tcip_store.binding import bind_default

    bind_default()

    from tcip_mcp.tools.annotation_tools import score_predictions

    stated = {} if args.conf_threshold is None else {"conf_threshold": args.conf_threshold}
    result = score_predictions(
        args.path, iou_threshold=args.iou_threshold, detail=args.detail, trait=args.trait,
        **stated)
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
