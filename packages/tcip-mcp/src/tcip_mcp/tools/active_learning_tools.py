"""Active learning MCP tools — partition predictions into review queues.

(``score_unlabeled`` was removed — ``feedback_tools.prioritize_review_queue`` is a strict
superset: same scorer + composed-detector guard, plus optional skip-already-reviewed and
per-image scores in the result.)
"""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


@mcp.tool()
@audited
def get_review_queue(
    checkpoint_path: str,
    images_dir: str,
    low: float = 0.3,
    high: float = 0.8,
    auto_threshold: float = 0.8,
) -> dict:
    """Partition predictions into auto-accept, review, and reject queues.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        images_dir: Directory of images to process.
        low: Lower confidence bound for review queue.
        high: Upper confidence bound for review queue.
        auto_threshold: Confidence threshold for auto-accepting labels.
    """
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.active_learning.selector import auto_accept, review_queue

    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    predictor = build_predictor(checkpoint_path)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    paths = sorted(str(f) for f in Path(images_dir).iterdir() if f.suffix.lower() in exts)

    predictions = predictor.predict_batch(paths)

    accepted = auto_accept(predictions, threshold=auto_threshold)
    needs_review = review_queue(predictions, low=low, high=high)

    return {
        "total_images": len(predictions),
        "auto_accepted": len(accepted),
        "needs_review": len(needs_review),
        "review_images": [r.get("image", "") for r in needs_review],
        "auto_accepted_images": [a.get("image", "") for a in accepted],
    }
