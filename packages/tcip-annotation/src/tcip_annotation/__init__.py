"""Headless annotation library: canonical name-based per-image JSON labels + a single-file COCO."""

from tcip_annotation.state import (
    Annotation,
    AnnotationState,
    BBox,
    Point,
    Polygon,
    bbox_of,
)
# Label I/O is the canonical per-image JSON (json_io); a single-file COCO is a genuine interop format.
from tcip_annotation.json_io import (
    read_annotations,
    write_annotations,
    to_coco_dataset,
)
from tcip_annotation.format_io import (
    detect_format,
    load_annotations as load_annotations_any,
    save_annotations as save_annotations_any,
    parse_coco_annotations,
    write_coco,
)
from tcip_annotation.matching import (
    compute_classified_trait_matches,
    compute_matches,
    box_iou,
    polygon_iou,
    point_in_polygon,
)
# The one mask -> Polygon.rings extractor, shared with tcip-mcp's prediction-export path.
from tcip_annotation.mask_contours import mask_to_polygon_rings
from tcip_annotation.annotation_engine import AnnotationEngine
from tcip_annotation.review_engine import ReviewEngine, ReviewDetection, ReviewContext

# sam_wrapper's heavy engine imports all live inside function bodies, so importing from
# it is always safe; the grid-cell helpers are pure lookups over caller-supplied cells.
from tcip_annotation.sam_wrapper import auto_mask, cell_fields, grid_to_pixel

__all__ = [
    "Annotation",
    "AnnotationState",
    "BBox",
    "Point",
    "Polygon",
    "bbox_of",
    # Canonical per-image JSON: the platform's native on-disk label format (primary read/write path)
    "read_annotations",
    "write_annotations",
    "to_coco_dataset",
    # Multi-format import/export (auto-detect + dispatch; behind the load/save_annotations tools)
    "detect_format",
    "load_annotations_any",
    "save_annotations_any",
    # COCO-specific (interop)
    "parse_coco_annotations",
    "write_coco",
    # Matching
    "compute_matches",
    "compute_classified_trait_matches",
    "box_iou",
    "polygon_iou",
    "point_in_polygon",
    # Mask -> polygon rings (shared by SAM-assisted labeling and prediction export)
    "mask_to_polygon_rings",
    # Grid-cell helpers (pure lookups, no SAM dependency)
    "cell_fields",
    "grid_to_pixel",
    # SAM wrapper
    "auto_mask",
    # Engines
    "AnnotationEngine",
    "ReviewEngine",
    "ReviewDetection",
    "ReviewContext",
]
