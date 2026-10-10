r"""Score a published bucket's predictions against the images' own label documents through the
platform's one matcher, from the command line.

Wraps ``annotation_tools.score_predictions``: a single image file returns per-box matches (plus an
optional per-detection breakdown with ``detail``); an images directory returns aggregate metrics
plus per-image TP/FP/FN.

Usage:
    tcip score-predictions --path <image_or_images_dir> --bucket <name> \
        [--iou-threshold 0.5] [--conf-threshold <default>] [--detail] \
        [--trait <trait_name> --project <project>]

--project is required with --trait, since a trait's derived localization criterion is read from
the project's own confirmed revision.
"""

from __future__ import annotations

import argparse
import json

from tcip_mcp.cli import bound_project


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("--path", required=True,
                        help="An image file (single-image match) or an images directory "
                             "(aggregate); a relative one is under --project, and refused "
                             "without it.")
    parser.add_argument("--bucket", required=True,
                        help="The name of the bucket, published under --path's dataset root, "
                             "whose documents are scored.")
    parser.add_argument("--project", default=None,
                        help="The project a relative --path lies under and the trait's "
                             "confirmed revision is read from. Required with --trait.")
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

    if args.trait and args.project is None:
        parser.error("--trait requires --project, the project its confirmed revision is in")
    if args.project is not None:
        project = bound_project(args.project)
    else:
        from tcip_store import bind

        bind()
        project = None
    trait = None
    if args.trait and project is not None:
        from tcip_mcp.operationalization import latest_confirmed

        trait = latest_confirmed(args.trait, project).entry

    from tcip_mcp.tools.annotation_tools import score_predictions

    stated = {} if args.conf_threshold is None else {"conf_threshold": args.conf_threshold}
    result = score_predictions(
        args.path, args.bucket, iou_threshold=args.iou_threshold, detail=args.detail,
        trait=trait, project=project, **stated)
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
