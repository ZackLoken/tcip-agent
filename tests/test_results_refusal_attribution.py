"""A Results refusal names the dimension that actually failed, and reports the others honestly.

The refusal is the breeder's hand-off into calibration: it is read to decide which operating point
to go and validate. So it carries per-dimension states, and each state belongs to the dimension it
is printed beside. A refusal that swaps them, or that reports a dimension whose evidence is on disk
as unvalidated, sends the breeder to recalibrate something that was never the problem while the real
gap stays open.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_HELD_OUT
from tcip_web.app import app
from tcip_web.state import store

from tests.test_tcip_web_results_routes import _phenology_fixture

pytestmark = pytest.mark.usefixtures("seed_bud_operationalization")

DOORS = ("phenology_measurement",)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _unvalidate_count_operating_point(body: dict) -> None:
    """Strip the count operating point of its reference, leaving the classifier evidence intact."""
    import tcip_store

    from tcip_mcp.pipelines.resolution import sidecar_key

    for bucket in body["predictions_by_date"].values():
        key = sidecar_key(bucket)
        with tcip_store.transaction(key) as txn:
            sidecar = txn.read(key)
            sidecar["validated"] = False
            sidecar["operating_point"]["conf"]["validated_against"] = VALIDATED_FALSE
            txn.write(key, sidecar)


def _unvalidate_classifier(body: dict) -> None:
    """Strip the positive-state classifier of its reference, leaving the count evidence intact."""
    import tcip_store

    from tcip_mcp.pipelines.resolution import sidecar_key

    for bucket in body["predictions_by_date"].values():
        key = sidecar_key(bucket, "classifier_operating_point")
        with tcip_store.transaction(key) as txn:
            txn.write(key, {
                "validated": False, "trait": "bud_opening", "experiment_id": "exp-1",
                "operating_point": {"classifier": {"value": "open",
                                                   "validated_against": VALIDATED_FALSE}},
            })


def _refusal_detail(client: TestClient, body: dict, route: str) -> str:
    resp = client.post(f"/api/results/{route}", json=body)
    assert resp.status_code == 400, (route, resp.text[:200])
    return resp.json()["detail"]


def test_a_refusal_over_the_count_operating_point_says_the_classifier_is_validated(
    client: TestClient, tmp_path: Path,
) -> None:
    """One dimension's evidence removed, the other's left in place: the refusal names the count
    operating point alone and still reports the classifier's real on-disk reference."""
    body = _phenology_fixture(tmp_path, validated=True, detections=4)
    assert client.post("/api/results/phenology_measurement", json=body).status_code == 200
    _unvalidate_count_operating_point(body)

    for route in DOORS:
        detail = _refusal_detail(client, body, route)
        assert "['operating_point']" in detail, route
        assert f"operating_point={VALIDATED_FALSE!r}" in detail, route
        assert f"classifier={VALIDATED_HELD_OUT!r}" in detail, route


def test_a_refusal_over_the_classifier_says_the_count_operating_point_is_validated(
    client: TestClient, tmp_path: Path,
) -> None:
    """The mirror case, so neither reading can be produced by a refusal that prints one fixed
    attribution: with only the classifier's reference gone, the count operating point is reported
    with the reference its own sidecar records."""
    body = _phenology_fixture(tmp_path, validated=True, detections=4)
    _unvalidate_classifier(body)

    for route in DOORS:
        detail = _refusal_detail(client, body, route)
        assert "['classifier']" in detail, route
        assert f"classifier={VALIDATED_FALSE!r}" in detail, route
        assert f"operating_point={VALIDATED_HELD_OUT!r}" in detail, route


def test_a_count_stamp_earned_for_a_different_trait_names_the_sidecar_and_both_traits(
    client: TestClient, tmp_path: Path,
) -> None:
    """A count stamp validated for one trait must not answer for a phenology delivery under a
    different trait: the refusal names the sidecar and both traits."""
    body = _phenology_fixture(tmp_path, validated=True, detections=4, count_trait="second_trait")

    for route in DOORS:
        detail = _refusal_detail(client, body, route)
        assert "second_trait" in detail and "bud" in detail, route
        assert any(bucket in detail for bucket in body["predictions_by_date"].values()), route


def test_the_two_refusals_do_not_read_alike(client: TestClient, tmp_path: Path) -> None:
    """The breeder is meant to act on the difference, so the two failures must not produce the same
    text. The unvalidated bucket list is the same in both; what separates them is which dimension
    is named."""
    count_broken = _phenology_fixture(tmp_path / "count", validated=True, detections=4)
    _unvalidate_count_operating_point(count_broken)
    classifier_broken = _phenology_fixture(tmp_path / "classifier", validated=True, detections=4)
    _unvalidate_classifier(classifier_broken)

    store.open_project(Path(count_broken["project_root"]).resolve())
    count_detail = _refusal_detail(client, count_broken, "phenology_measurement")
    store.open_project(Path(classifier_broken["project_root"]).resolve())
    classifier_detail = _refusal_detail(client, classifier_broken, "phenology_measurement")
    assert count_detail != classifier_detail
    assert "['classifier']" not in count_detail
    assert "['operating_point']" not in classifier_detail
