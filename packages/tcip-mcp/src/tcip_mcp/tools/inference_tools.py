"""Inference MCP tools — run models on images, export results."""

from __future__ import annotations

import logging
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing.export import export_detection_csv, result_to_yolo_lines
from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_NMS_IOU,
    DEFAULT_TILE_SIZE,
    DEFAULT_TILED,
)

logger = logging.getLogger(__name__)


@mcp.tool()
@audited
def run_inference(
    checkpoint_path: str,
    image_paths: list[str] | None = None,
    images_dir: str | None = None,
    score_threshold: float = DEFAULT_CONF,
    device: str | None = None,
    tile: bool = DEFAULT_TILED,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = 0.2,
    tile_batch_size: int = 96,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    dry_run: bool = False,
) -> dict:
    """Run a trained model on images.

    Provide either image_paths (specific images) or images_dir (all images
    in a directory). Set ``tile=True`` for SAHI-style sliding-window detection on
    high-resolution imagery with many small objects (detection heads only).

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        image_paths: List of specific image paths.
        images_dir: Directory containing images to process.
        score_threshold: Minimum confidence score.
        device: Device to use ('cuda' or 'cpu').
        tile: Enable tiled (SAHI-style) detection inference.
        tile_size: Sliding-window tile edge (px).
        overlap: Fractional tile overlap (stride = tile_size*(1-overlap)).
        tile_batch_size: Tiles per forward batch.
        global_nms_iou: Cross-tile global NMS IoU threshold.
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    if dry_run:
        # Report the effective operating point WITHOUT loading the model or running inference, so the
        # agent can see what conf/NMS/tiling will govern the object count before committing to a run.
        return {
            "dry_run": True,
            "checkpoint_path": checkpoint_path,
            "operating_point": {
                "conf": score_threshold,
                "cross_tile_nms": global_nms_iou if tile else None,
                "tiled": tile,
                "tile_size": tile_size if tile else None,
                "overlap": overlap,
            },
            "note": ("These operating-point values govern the object count (the phenotype for count "
                     "traits). For a trait with a labeled subset, resolve them per dataset "
                     "(resolve_operating_point) so the count is calibrated, not a default."),
        }

    # Lazy import to avoid torch import at module level
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        ResolvedBundle,
        ResolvedParam,
        default,
    )

    # Route the operating point through a ResolvedBundle so the count carries firewall provenance.
    # Raw inference has no per-dataset calibration, so conf is unvalidated and read with an explicit
    # acknowledgement; the delivery gate downstream enforces validation.
    op_bundle = ResolvedBundle(trait="", dataset_hash=None, params={
        "conf": ResolvedParam("conf", score_threshold, source="default",
                              derivation_class="calibration", validated_vs_gt=VALIDATED_FALSE),
        "cross_tile_nms": default("cross_tile_nms", global_nms_iou if tile else None,
                                  derivation_class="distribution"),
        "tiled": default("tiled", tile),
        "tile_size": default("tile_size", tile_size if tile else None),
    })
    conf = op_bundle.get("conf").unvalidated_value(acknowledge_unvalidated=True)

    predictor = GenericPredictor(
        checkpoint_path=checkpoint_path,
        device=device,
        score_threshold=conf,
    )

    if image_paths is None:
        if images_dir is None:
            return {"error": "Provide either image_paths or images_dir"}
        p = Path(images_dir)
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        image_paths = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in image_exts)

    # Preflight: warn (don't fail) when a slow workload will run on CPU because CUDA isn't
    # available — full tiled inference over thousands of images is hours on CPU vs minutes on
    # a GPU. Install a CUDA torch build (see environment.yml) to use the card.
    warning = None
    if device != "cpu" and (tile or len(image_paths) > 8):
        import torch

        if not torch.cuda.is_available():
            warning = (
                f"CUDA not available — running {len(image_paths)} image(s)"
                f"{' tiled' if tile else ''} on CPU, which is much slower. Install a CUDA torch "
                "build (see environment.yml) to use the GPU."
            )
            logger.warning(warning)

    results = predictor.predict_batch(
        image_paths, tile=tile, tile_size=tile_size, overlap=overlap,
        tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
    )
    total_detections = sum(r["count"] for r in results)

    out = {
        "checkpoint": checkpoint_path,
        "image_count": len(results),
        "total_detections": total_detections,
        "tiled": tile,
        "operating_point": op_bundle.to_provenance()["operating_point"],
        "results": results,
    }
    if warning:
        out["warning"] = warning
    return out


@mcp.tool()
@audited
def export_predictions_yolo(
    checkpoint_path: str,
    images_dir: str,
    output_dir: str,
    score_threshold: float = 0.5,
    device: str | None = None,
) -> dict:
    """Run inference and save predictions as YOLO-format text files.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images.
        output_dir: Directory for output .txt prediction files.
        score_threshold: Minimum confidence score.
        device: Device to use.
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    predictor = GenericPredictor(
        checkpoint_path=checkpoint_path,
        device=device,
        score_threshold=score_threshold,
    )

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    p = Path(images_dir)
    image_paths = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in image_exts)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    results = predictor.predict_batch(image_paths)
    for img_path, result in zip(image_paths, results):
        out_txt = out / f"{Path(img_path).stem}.txt"
        out_txt.write_text("\n".join(result_to_yolo_lines(result)))
        written.append(str(out_txt))

    return {"image_count": len(written), "output_dir": output_dir, "files": written}


@mcp.tool()
@audited
def export_results_csv(
    checkpoint_path: str,
    images_dir: str,
    output_path: str,
    score_threshold: float = 0.5,
    device: str | None = None,
) -> dict:
    """Run inference and export a CSV summary of detection counts per image.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images.
        output_path: Path for the output CSV file.
        score_threshold: Minimum confidence score.
        device: Device to use.
    """
    result = run_inference(
        checkpoint_path=checkpoint_path,
        images_dir=images_dir,
        score_threshold=score_threshold,
        device=device,
    )
    if "error" in result:
        return result

    csv_path = export_detection_csv(result["results"], output_path)
    return {
        "csv_path": csv_path,
        "image_count": result["image_count"],
        "total_detections": result["total_detections"],
    }
