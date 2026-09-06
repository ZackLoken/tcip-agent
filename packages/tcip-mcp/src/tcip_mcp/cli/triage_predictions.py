"""Sort a checkpoint's own predictions by confidence into auto-accept, needs-review and
unscoreable queues, from the command line.

The demoted twin of ``feedback_tools.triage_predictions``: returns predictions at or above
``--auto-threshold`` as the confident set for the caller to accept as ground truth, and writes
nothing itself. ``--auto-threshold`` keeps the tool's own refusal: omitted, it refuses to
auto-accept anything, since turning predictions into GT at a pinned threshold fabricates labels
the model was never confirmed to get right; derive it from the model's validated confidence
distribution and a breeder spot-check first. It is a command, not an MCP tool, per CLAUDE.md:
this is the door for a harness with no MCP tool for it, or an operator triaging a batch outside
any agent session.

Usage:
    tcip triage-predictions --checkpoint <ckpt.pt> --images-dir <dir> \
        --project <platform_root> [--dataset-root <dir>] [--no-skip-reviewed] \
        [--low 0.3] [--high 0.8] [--auto-threshold <conf>] [--bucket <name>] \
        [--review-state-dir <dir>]

The checkpoint must be named by a registry entry under --project (register it with
register_model first); this command refuses one it is not, naming the digest and the root.
"""

from __future__ import annotations

import argparse
import json

from tcip_mcp.project_paths import require_platform_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained model checkpoint.")
    parser.add_argument("--images-dir", required=True, help="Directory of candidate images.")
    parser.add_argument("--project", default=None,
                        help="Platform state root the checkpoint's registry entry is looked up "
                             "under. Required (or set $TCIP_STATE_ROOT).")
    parser.add_argument("--dataset-root", default="",
                        help="Root of the dataset whose review is in progress; scopes the "
                             "verdict store --skip-reviewed reads. Omitted with "
                             "--review-state-dir too, no store is read and every image triages.")
    parser.add_argument("--no-skip-reviewed", action="store_true",
                        help="Do not exclude already-completed images before triaging.")
    parser.add_argument("--low", type=float, default=0.3, help="Lower confidence bound for the "
                        "needs-review band.")
    parser.add_argument("--high", type=float, default=0.8, help="Upper confidence bound for the "
                        "needs-review band.")
    parser.add_argument("--auto-threshold", type=float, default=None,
                        help="Confidence at/above which a prediction joins the confident set. "
                             "Omitted refuses to auto-accept anything; derive it from the "
                             "model's validated confidence distribution and confirm with a "
                             "breeder spot-check first.")
    parser.add_argument("--bucket", default=None,
                        help="Which prediction bucket's completed reviews --skip-reviewed "
                             "skips. Omitted reads the store's sole bucket, refusing if several.")
    parser.add_argument("--review-state-dir", default="",
                        help="A verdict store to read instead of the dataset's own.")
    args = parser.parse_args(argv)

    root = require_platform_root(args.project)

    from tcip_store.binding import bind_default

    bind_default()

    from tcip_mcp.tools.feedback_tools import triage_predictions

    result = triage_predictions(
        args.checkpoint, args.images_dir, dataset_root=args.dataset_root,
        skip_reviewed=not args.no_skip_reviewed, low=args.low, high=args.high,
        auto_threshold=args.auto_threshold, bucket=args.bucket,
        review_state_dir=args.review_state_dir, project_path=str(root),
    )
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
