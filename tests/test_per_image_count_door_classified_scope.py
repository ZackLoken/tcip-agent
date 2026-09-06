"""``per_image_counts_from_bucket``: a classified bucket delivers the object count its own scope
says its detections are of, never the value count, and refuses a neither-key stamp naming the
conform script rather than reading it as an ordinary, unscoped bucket.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcip_annotation.state import Annotation, BBox
from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
from tcip_mcp.pipelines.resolution import operating_point_stamp, sidecar_key, write_sidecar
from tests import _operationalization_fixtures as fx

SUBJECT = fx.COUNT_SUBJECT  # "stem": what the confirmed per_image_count says the counts are of
ATTRIBUTE = "condition"
ID_MAP = {"upright": 0, "lodged": 1}


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_count(tmp_path, measured_subject=SUBJECT)


def _classified_bucket(tmp_path: Path) -> Path:
    bucket = tmp_path / "predictions" / "classifier" / "2026-05-20"
    result = {"width": 100, "height": 100, "boxes": [[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 60.0, 60.0]],
             "scores": [0.9, 0.8], "labels": [1, 2]}
    write_predictions_json(bucket / "img1.json", result, created_by="test-producer",
                           subject=SUBJECT, attribute=ATTRIBUTE, id_map=ID_MAP)
    stamp = operating_point_stamp(
        {"conf": {"value": 0.5}}, validated=False, validated_by=None, tile_size_validated=None,
        shippable_issues=[], id_map=ID_MAP, subject=SUBJECT, attribute=ATTRIBUTE,
        trait=fx.COUNT_TRAIT,
        dataset_hash="H", checkpoint="m", checkpoint_sha256="f" * 64, experiment_id=None,
        images_dir=str(tmp_path / "images"), raster_path=None,
        produced_at="2026-05-20T00:00:00+00:00",
    )
    write_sidecar(bucket, stamp)
    return bucket


def test_a_classified_bucket_delivers_its_object_count_not_its_value_count(tmp_path: Path) -> None:
    from tcip_mcp.tools.inference_tools import per_image_counts_from_bucket

    bucket = _classified_bucket(tmp_path)
    out = tmp_path / "counts.csv"

    from tcip_mcp.pipelines.resolution import Acknowledgement

    result = per_image_counts_from_bucket(
        str(bucket), str(out), trait=fx.COUNT_TRAIT, project_root=tmp_path,
        acknowledgement=Acknowledgement(acknowledged_by="user:tester", reason="unvalidated fixture"))

    assert "error" not in result
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert int(rows[0]["detection_count"]) == 2  # both records counted as the object class


def test_a_neither_key_stamp_refuses_naming_the_repair_command(tmp_path: Path) -> None:
    import tcip_store

    from tcip_mcp.tools.inference_tools import per_image_counts_from_bucket

    bucket = tmp_path / "predictions" / "classifier" / "2026-05-21"
    write_predictions_json(
        bucket / "img1.json",
        {"width": 100, "height": 100, "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1]},
        subject=None, attribute=None, id_map=None,
    )
    # Overwrite the annotation's own subject/value into the classified shape by hand, then seed a
    # stamp predating the writer rail (id_map alone, no subject/attribute pair at all).
    from tcip_annotation.json_io import write_annotations

    write_annotations(
        str(bucket / "img1.json"),
        [Annotation(subject=SUBJECT, geometry=BBox(10, 10, 30, 30), score=0.9,
                   attributes={ATTRIBUTE: "upright"})],
        100, 100,
    )
    tcip_store.replace(sidecar_key(bucket, "operating_point"), {"id_map": ID_MAP},
                       expect=tcip_store.Version.ABSENT)
    out = tmp_path / "counts.csv"

    from tcip_mcp.pipelines.resolution import CountDeliveryRefused

    with pytest.raises(CountDeliveryRefused, match="repair-classified-predictions"):
        per_image_counts_from_bucket(str(bucket), str(out), trait=fx.COUNT_TRAIT,
                                     project_root=tmp_path)
