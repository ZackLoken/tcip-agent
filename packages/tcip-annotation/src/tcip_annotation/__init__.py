"""Headless annotation library: canonical name-based per-image JSON labels, and a COCO reader."""

from tcip_annotation.state import (
    Annotation,
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
from tcip_annotation.matching import point_in_polygon
# The one mask -> Polygon.rings extractor, shared with tcip-mcp's prediction-export path.
from tcip_annotation.mask_contours import mask_to_polygon_rings
from tcip_annotation.grid import cell_fields

__all__ = [
    "Annotation",
    "BBox",
    "Point",
    "Polygon",
    "bbox_of",
    # Canonical per-image JSON: the platform's native on-disk label format (primary read/write path)
    "read_annotations",
    "write_annotations",
    # An external COCO document's records (import only)
    "parse_coco_annotations",
    "point_in_polygon",
    # Mask -> polygon rings (shared by proposers and prediction export)
    "mask_to_polygon_rings",
    "cell_fields",
]
