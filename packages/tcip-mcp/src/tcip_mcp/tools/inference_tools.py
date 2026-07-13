"""Inference MCP tools — run models on images, export results."""

from __future__ import annotations

import logging
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.postprocessing.export import export_detection_csv, result_to_yolo_lines
from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_MAX_DETS,
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
    max_dets: int = DEFAULT_MAX_DETS,
    postprocess: str = "nms",
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
        max_dets: Full-frame detection cap (after any tiled merge).
        postprocess: Cross-tile merge — "nms" suppresses overlaps, "nmm" unions boxes split
            across a tile seam (better for an object straddling a boundary).
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
                "max_dets": max_dets,
                "postprocess": postprocess,
            },
            "note": ("These operating-point values govern the object count (the phenotype for count "
                     "traits). For a trait with a labeled subset, resolve them per dataset "
                     "(resolve_operating_point) so the count is calibrated, not a default."),
        }

    # Lazy import to avoid torch import at module level
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.resolution import raw_operating_point

    # Route the operating point through a ResolvedBundle so the count carries firewall provenance.
    # Raw inference has no per-dataset calibration, so conf is unvalidated and read with an explicit
    # acknowledgement; the delivery gate downstream enforces validation.
    op_bundle = raw_operating_point(
        conf=score_threshold, cross_tile_nms=global_nms_iou, tiled=tile,
        tile_size=tile_size, max_dets=max_dets,
    )
    conf = op_bundle.get("conf").unvalidated_value(acknowledge_unvalidated=True)

    # Thread NMS IoU + the full-frame detection cap into the model so they GOVERN which boxes exist
    # (torchvision in-model thresholds / ultralytics overrides), not just cross-tile merge — else
    # nms_iou has no effect on an untiled run and dense scenes truncate at the framework default.
    predictor = build_predictor(
        checkpoint_path=checkpoint_path,
        device=device,
        score_threshold=conf,
        nms_iou=global_nms_iou,
        max_dets=max_dets,
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
        tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, postprocess=postprocess,
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
    score_threshold: float = DEFAULT_CONF,
    device: str | None = None,
    tile: bool = DEFAULT_TILED,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = 0.2,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS,
) -> dict:
    """Run inference and save predictions as YOLO-format text files.

    Routes through ``run_inference`` so this delivery door resolves the SAME firewalled
    operating point (conf/NMS/tiling/max_dets) — earlier it built its own bare predictor and
    so truncated the count at the framework default and shipped labels with no provenance.
    Writes ``<stem>.txt`` per image plus an ``operating_point.json`` stamp beside them.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images.
        output_dir: Directory for output .txt prediction files.
        score_threshold: Minimum confidence score.
        device: Device to use.
        tile: Tiled (SAHI-style) inference for small dense objects.
        tile_size: Sliding-window tile edge (px).
        overlap: Fractional tile overlap.
        global_nms_iou: Cross-tile NMS IoU.
        max_dets: Full-frame detection cap.
    """
    result = run_inference(
        checkpoint_path=checkpoint_path, images_dir=images_dir, score_threshold=score_threshold,
        device=device, tile=tile, tile_size=tile_size, overlap=overlap,
        global_nms_iou=global_nms_iou, max_dets=max_dets,
    )
    if "error" in result:
        return result

    from tcip_mcp.utils.atomic_io import atomic_write_json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for r in result["results"]:
        out_txt = out / f"{Path(r['image']).stem}.txt"
        lines = result_to_yolo_lines(r)
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written.append(str(out_txt))

    # Stamp the operating point beside the delivered labels (raw inference conf is unvalidated).
    atomic_write_json(out / "operating_point.json",
                      {"operating_point": result.get("operating_point"), "validated": False})

    return {"image_count": len(written), "output_dir": output_dir, "files": written,
            "operating_point": result.get("operating_point")}


@mcp.tool()
@audited
def export_results_csv(
    checkpoint_path: str,
    images_dir: str,
    output_path: str,
    score_threshold: float = DEFAULT_CONF,
    device: str | None = None,
    tile: bool = DEFAULT_TILED,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = 0.2,
    global_nms_iou: float = DEFAULT_NMS_IOU,
    max_dets: int = DEFAULT_MAX_DETS,
) -> dict:
    """Run inference and export a CSV summary of detection counts per image.

    Routes through ``run_inference`` so the per-image counts resolve the same firewalled
    operating point (conf/NMS/tiling/max_dets) as ``run_inference``/``export_predictions_yolo`` —
    the CSV is a count-bearing deliverable (the count is the phenotype for count traits), so it
    must not be produced at a different, untiled, truncating operating point. Earlier this door
    hardcoded ``score_threshold=0.5`` and passed no tiling/max_dets, under-reporting dense
    small-object counts relative to the other two doors.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        images_dir: Directory containing input images.
        output_path: Path for the output CSV file.
        score_threshold: Minimum confidence score.
        device: Device to use.
        tile: Tiled (SAHI-style) inference for small dense objects.
        tile_size: Sliding-window tile edge (px).
        overlap: Fractional tile overlap.
        global_nms_iou: Cross-tile NMS IoU.
        max_dets: Full-frame detection cap.
    """
    result = run_inference(
        checkpoint_path=checkpoint_path,
        images_dir=images_dir,
        score_threshold=score_threshold,
        device=device,
        tile=tile,
        tile_size=tile_size,
        overlap=overlap,
        global_nms_iou=global_nms_iou,
        max_dets=max_dets,
    )
    if "error" in result:
        return result

    csv_path = export_detection_csv(result["results"], output_path)
    return {
        "csv_path": csv_path,
        "image_count": result["image_count"],
        "total_detections": result["total_detections"],
        # Carry the operating point that produced these counts — the CSV is a count-bearing
        # deliverable and the numbers are only as trustworthy as the operating point behind them.
        "operating_point": result.get("operating_point"),
    }
