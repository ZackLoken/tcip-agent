"""Headless annotation library: canonical name-based per-image JSON labels, and a COCO reader."""

from tcip_annotation.state import (
    Annotation,
    AnnotationState,
    BBox,
    Point,
    Polygon,
    bbox_of,
)
# Label I/O is the canonical per-image JSON (json_io); a dataset-level COCO is only ever read.
from tcip_annotation.json_io import (
    read_annotations,
    write_annotations,
)
from tcip_annotation.format_io import parse_coco_annotations
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
    # An external COCO document's records (import only)
    "parse_coco_annotations",
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
