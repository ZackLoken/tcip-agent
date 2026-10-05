"""A bucket whose scope declares attributes delivers the object count of its own recorded subject,
never a count per attribute value."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests import _trait_fixtures as fx

SUBJECT = fx.COUNT_SUBJECT  # what the confirmed per_image_count says the counts are of


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_count(tmp_path, measured_subject=SUBJECT)


def test_an_attributed_bucket_delivers_its_object_count_not_its_value_count(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp import subject_registry as cr
    from tcip_mcp.pipelines.postprocessing.export import deliver_per_image_counts_csv
    from tests._chain_fixtures import acknowledged, predicted, published

    posture = cr.Attribute("posture", "categorical", ("upright", "lodged"))
    registry = cr.SubjectRegistry(subjects=(cr.Subject(name=SUBJECT, attributes=(posture,)),))
    image = tmp_path / "ds" / "images" / "2026-05-20" / "img1.jpg"
    bucket = published(tmp_path, "classifier/2026-05-20",
                       [predicted(image, ["upright", "lodged"], (posture,))],
                       scope={"subject": SUBJECT}, registry=registry)
    out = tmp_path / "counts.csv"

    acknowledged(tmp_path, lambda ack: deliver_per_image_counts_csv(
        tmp_path, bucket.root, bucket.name, str(out), trait=fx.COUNT_TRAIT,
        acknowledgment_id=ack, door="test_door", actor=None), reason="unassessed fixture")

    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert int(rows[0]["detection_count"]) == 2  # both records counted as the object class
