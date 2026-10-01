"""A classified bucket delivers the object count its own recorded scope says its detections are
of, never the value count."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests import _trait_fixtures as fx

SUBJECT = fx.COUNT_SUBJECT  # what the confirmed per_image_count says the counts are of
SCOPE = {"subject": SUBJECT, "attribute": "condition", "id_map": {"upright": 0, "lodged": 1}}


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_count(tmp_path, measured_subject=SUBJECT)


def test_a_classified_bucket_delivers_its_object_count_not_its_value_count(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.postprocessing.export import deliver_per_image_counts_csv
    from tests._chain_fixtures import acknowledged, predicted, published

    bucket = published(tmp_path, tmp_path / "ds" / "predictions" / "classifier" / "2026-05-20",
                       [predicted("img1", ["upright", "lodged"], SCOPE["id_map"])],
                       scope=SCOPE).path
    out = tmp_path / "counts.csv"

    acknowledged(tmp_path, lambda ack: deliver_per_image_counts_csv(
        tmp_path, bucket, str(out), trait=fx.COUNT_TRAIT, acknowledgment_id=ack,
        door="test_door"), reason="unassessed fixture")

    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert int(rows[0]["detection_count"]) == 2  # both records counted as the object class
