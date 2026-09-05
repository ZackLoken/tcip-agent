"""Reader-side surfaces that hold a classified bucket to its own recorded scope: ``count_by_class``
under a coincidental detector map, ``per_plant_series``'s stamp refusals, the COCO round trip's
``attributes`` handling, and the prediction render's legend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.pipelines import resolution
from tcip_mcp.pipelines.postprocessing import phenology

SUBJECT = "bud"
ATTRIBUTE = "opening"


def test_count_by_class_a_detector_map_keyed_by_the_positive_value_name_never_counts_positive(
    tmp_path: Path,
) -> None:
    """A bare single-class detector map that happens to be spelled like the trait's own positive
    value name (a vocabulary coincidence) never counts a positive: only a bucket that classified
    along an attribute assessed a state at all."""
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="open", geometry=BBox(1, 1, 3, 3), score=0.9)], 8, 8)
    id_map = {"open": 0}
    scope = resolution.BucketScope(subject="open", attribute=None)

    total, positive, unclassified = phenology.count_by_class(p, id_map, "open", scope=scope)

    assert (total, positive, unclassified) == (1, 0, 1)


def _seed_sidecar(pred_dir: Path, sidecar: dict) -> None:
    import tcip_store

    tcip_store.replace(resolution.sidecar_key(pred_dir, "operating_point"), sidecar,
                       expect=tcip_store.Version.ABSENT)


def _damage_sidecar(pred_dir: Path) -> None:
    import os

    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
    from tcip_store.store import _backend

    key = resolution.sidecar_key(pred_dir, "operating_point")
    if (os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND) == FILE_BACKEND:
        _backend().path_for(key).write_bytes(b"{not json")
        return
    import sqlite3

    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute("update records set value = ? where store = ? and parts = ?",
                    (b"{not json", key.store, encode_parts(key.parts)))
    finally:
        conn.close()


class _Assignment:
    def __init__(self, stem, plot_name):
        self.stem = stem
        self.plot_name = plot_name
        self.accession_name = None


def test_per_plant_series_raises_for_a_neither_key_stamp(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "classifier" / "2026-05-01"
    json_io.write_annotations(
        pred_dir / "s1.json",
        [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 3, 3), attributes={ATTRIBUTE: "open"})],
        8, 8)
    _seed_sidecar(pred_dir, {"id_map": {"open": 0, "closed": 1}})
    mapping = {"2026-05-01": [_Assignment("s1", "P1")]}

    with pytest.raises(resolution.StampScopeUnstated, match="conform_classified_predictions.py"):
        phenology.per_plant_series(mapping, {"2026-05-01": str(pred_dir)}, "open")


def test_per_plant_series_raises_for_an_undecodable_stamp(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "classifier" / "2026-05-02"
    json_io.write_annotations(
        pred_dir / "s1.json",
        [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 3, 3), attributes={ATTRIBUTE: "open"})],
        8, 8)
    _seed_sidecar(pred_dir, {"id_map": {"open": 0, "closed": 1}, "subject": SUBJECT,
                            "attribute": ATTRIBUTE})
    _damage_sidecar(pred_dir)
    mapping = {"2026-05-02": [_Assignment("s1", "P1")]}

    with pytest.raises(Exception):  # the seam's own StoreError subclass
        phenology.per_plant_series(mapping, {"2026-05-02": str(pred_dir)}, "open")


def test_coco_round_trip_keeps_a_classified_predictions_value(tmp_path: Path) -> None:
    from tcip_annotation.format_io import parse_coco_annotations, write_coco

    ann = Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5), score=0.9,
                     attributes={ATTRIBUTE: "open"})
    path = tmp_path / "out.json"
    write_coco(str(path), {"img.jpg": ([ann], 10, 10)}, id_map={SUBJECT: 0})

    import json

    coco = json.loads(path.read_text(encoding="utf-8"))
    assert coco["annotations"][0]["attributes"] == {ATTRIBUTE: "open"}

    restored = parse_coco_annotations(coco, file_name="img.jpg")
    assert restored[0].subject == SUBJECT
    assert restored[0].attributes == {ATTRIBUTE: "open"}


def test_coco_parser_refuses_a_non_string_attributes_shape() -> None:
    from tcip_annotation.format_io import parse_coco_annotations
    from tcip_annotation.json_io import UnreadableLabelDocument

    coco = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 0, "name": SUBJECT}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 4, 4],
                         "attributes": {ATTRIBUTE: 7}}],
    }

    with pytest.raises(UnreadableLabelDocument, match="not a mapping of strings to strings"):
        parse_coco_annotations(coco, file_name="img.jpg")


def test_legend_name_shows_the_classified_value_under_a_classified_scope() -> None:
    from tcip_mcp.tools.vision_tools import _legend_name

    pred = Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5), score=0.9,
                      attributes={ATTRIBUTE: "open"})
    scope = resolution.BucketScope(subject=SUBJECT, attribute=ATTRIBUTE)

    assert _legend_name(pred, scope=scope) == "open"


def test_legend_name_falls_back_to_the_subject_under_no_scope() -> None:
    from tcip_mcp.tools.vision_tools import _legend_name

    pred = Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5), score=0.9)

    assert _legend_name(pred, scope=None) == SUBJECT
