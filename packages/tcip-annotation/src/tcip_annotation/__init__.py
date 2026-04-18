"""Headless annotation library — YOLO, COCO, PASCAL VOC, and LabelMe format support."""

from tcip_annotation.state import AnnotationState, BBox, Polygon, PredBBox, PredPolygon
from tcip_annotation.label_io import (
    parse_detect_labels,
    parse_segment_labels,
    parse_detect_predictions,
    parse_segment_predictions,
    write_detect_labels,
    write_segment_labels,
)
from tcip_annotation.format_io import (
    detect_format,
    load_annotations as load_annotations_any,
    save_annotations as save_annotations_any,
    parse_coco_detect,
    parse_coco_segment,
    write_coco_detect,
    write_coco_segment,
    parse_voc_detect,
    write_voc_detect,
    parse_labelme_detect,
    parse_labelme_segment,
    write_labelme,
)
from tcip_annotation.matching import compute_matches, box_iou, polygon_iou, point_in_polygon

__all__ = [
    "AnnotationState",
    "BBox",
    "Polygon",
    "PredBBox",
    "PredPolygon",
    # YOLO-specific (preserved for backwards compat)
    "parse_detect_labels",
    "parse_segment_labels",
    "parse_detect_predictions",
    "parse_segment_predictions",
    "write_detect_labels",
    "write_segment_labels",
    # Format-agnostic API
    "detect_format",
    "load_annotations_any",
    "save_annotations_any",
    # COCO-specific
    "parse_coco_detect",
    "parse_coco_segment",
    "write_coco_detect",
    "write_coco_segment",
    # PASCAL VOC
    "parse_voc_detect",
    "write_voc_detect",
    # LabelMe
    "parse_labelme_detect",
    "parse_labelme_segment",
    "write_labelme",
    # Matching
    "compute_matches",
    "box_iou",
    "polygon_iou",
    "point_in_polygon",
]
