"""Headless annotation library."""

from tcip_annotation.state import (
    Annotation,
    BBox,
    Point,
    Polygon,
    bbox_of,
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
    # An external COCO document's records (import only)
    "parse_coco_annotations",
    "point_in_polygon",
    # Mask -> polygon rings (shared by proposers and prediction export)
    "mask_to_polygon_rings",
    "cell_fields",
]
