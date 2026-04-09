"""Headless annotation and review engine for YOLO-format data."""

from tcip_annotation.state import AnnotationState, BBox, Polygon, PredBBox, PredPolygon
from tcip_annotation.engine import AnnotationEngine
from tcip_annotation.review_engine import ReviewEngine
from tcip_annotation.label_io import (
    parse_detect_labels,
    parse_segment_labels,
    parse_detect_predictions,
    parse_segment_predictions,
    write_detect_labels,
    write_segment_labels,
)
from tcip_annotation.matching import compute_matches, box_iou, polygon_iou

__all__ = [
    "AnnotationState",
    "BBox",
    "Polygon",
    "PredBBox",
    "PredPolygon",
    "AnnotationEngine",
    "ReviewEngine",
    "parse_detect_labels",
    "parse_segment_labels",
    "parse_detect_predictions",
    "parse_segment_predictions",
    "write_detect_labels",
    "write_segment_labels",
    "compute_matches",
    "box_iou",
    "polygon_iou",
]
