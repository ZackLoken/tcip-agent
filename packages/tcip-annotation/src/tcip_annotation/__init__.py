"""Headless annotation library — canonical per-image JSON labels + the assembled dataset COCO."""

from tcip_annotation.state import (
    AnnotationState,
    BBox,
    Polygon,
    PredBBox,
    PredPolygon,
    boxes_from_polygons,
)
# Label I/O is the canonical per-image JSON (json_io); the dataset-level COCO is assembled from it.
from tcip_annotation.json_io import (
    read_detect as parse_detect_labels,
    read_segment as parse_segment_labels,
    read_detect_pred as parse_detect_predictions,
    read_segment_pred as parse_segment_predictions,
    write_detect as write_detect_labels,
    write_segment as write_segment_labels,
)
from tcip_annotation.format_io import (
    detect_format,
    load_annotations as load_annotations_any,
    save_annotations as save_annotations_any,
    parse_coco_detect,
    parse_coco_segment,
    write_coco_detect,
    write_coco_segment,
)
from tcip_annotation.matching import compute_matches, box_iou, polygon_iou, point_in_polygon
from tcip_annotation.annotation_engine import AnnotationEngine
from tcip_annotation.review_engine import ReviewEngine, ReviewDetection, ReviewContext

# SAM wrapper — lazy-import safe (requires segment-anything optional dep)
try:
    from tcip_annotation.sam_wrapper import auto_mask, grid_to_pixel
except ImportError:
    pass

__all__ = [
    "AnnotationState",
    "BBox",
    "Polygon",
    "PredBBox",
    "PredPolygon",
    "boxes_from_polygons",
    # Canonical per-image COCO/JSON — the platform's native on-disk label format (primary read/write path)
    "parse_detect_labels",
    "parse_segment_labels",
    "parse_detect_predictions",
    "parse_segment_predictions",
    "write_detect_labels",
    "write_segment_labels",
    # Multi-format import/export (auto-detect + dispatch; behind the load/save_annotations tools)
    "detect_format",
    "load_annotations_any",
    "save_annotations_any",
    # COCO-specific
    "parse_coco_detect",
    "parse_coco_segment",
    "write_coco_detect",
    "write_coco_segment",
    # Matching
    "compute_matches",
    "box_iou",
    "polygon_iou",
    "point_in_polygon",
    # SAM wrapper
    "auto_mask",
    "grid_to_pixel",
    # Engines
    "AnnotationEngine",
    "ReviewEngine",
    "ReviewDetection",
    "ReviewContext",
]
