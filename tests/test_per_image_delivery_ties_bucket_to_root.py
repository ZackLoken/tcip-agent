"""A delivery ties each delivered bucket's recorded dataset to the root it is delivered under.

The per-image count door refuses a root carrying no identity naming the root and
``register_dataset``, and a bucket recording another dataset than its root's naming the bucket and
both ids, through the tool and the route, writing no CSV and no delivery event. The orthomosaic and
per-plant doors answer the same two refusals in the same words, and the gate every delivery passes
refuses both, with the tie's own sentences, when a bucket reaches it untied.

The one step made by hand is fault injection: after a bucket is published under a registered root,
the root's ``dataset.json`` is removed, which only a change to the project's files outside the
platform does. The root registered again then mints a new id.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.buckets import read_bucket
from tcip_mcp.dataset_layout import dataset_identity_path, image_dir
from tcip_mcp.delivery import (
    DeliveryRefusal,
    Unacknowledgeable,
    read_delivery_events,
)
from tests import _trait_fixtures as fx
from tests._audit_fixtures import audit_rows
from tests._mapping_fixtures import COUNT_CROP
from tests._producer_fixtures import (
    left_written,
    register_dataset_at,
    unregistered_refusal,
)
from tests._web_fixtures import BROWSER

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_mcp.buckets import Bucket

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _footprint(project: Path, root: Path) -> dict[str, list]:
    """Return what a door can write for ``project`` over ``root``, and its delivery events."""
    return {
        **left_written(project, root),
        "events": read_delivery_events(project),
    }


def _identity_removed(root: Path) -> None:
    """Remove the identity document of the registered ``root``, as a change outside the platform."""
    dataset_identity_path(root).unlink()


def _registered_again(project: Path, root: Path) -> str:
    """Register ``root`` again, which mints it a new id; return that id."""
    return register_dataset_at(project, root, crop=COUNT_CROP)


# ── the tool ─────────────────────────────────────────────────────────────


def _published(project: Path) -> tuple[Path, str]:
    """Publish a per-image count bucket under dataset ``ds``; return its root and name."""
    from tests.test_deliver_per_image_counts_bucket_regime import (
        SCOPE,
        _bucket,
    )

    fx.seed_confirmed_count(project, measured_subject=fx.COUNT_SUBJECT)
    return project / "ds", _bucket(project, SCOPE)


def _deliver(project: Path, root: Path, bucket: str) -> dict:
    """Return the answer of ``deliver_per_image_counts`` for ``bucket`` under ``root``."""
    from tests.test_deliver_per_image_counts_bucket_regime import (
        _deliver as tool_answer,
    )

    return tool_answer(project, root, bucket)


def test_a_per_image_delivery_under_a_root_with_no_identity_refuses_at_the_tool(
    tmp_path: Path,
) -> None:
    """The tool answers the sentence naming the root and ``register_dataset``; writes nothing."""
    root, bucket = _published(tmp_path)
    _identity_removed(root)
    before = _footprint(tmp_path, root)

    answered = _deliver(tmp_path, root, bucket)

    assert answered == {"error": unregistered_refusal(root)}
    assert not (tmp_path / "out" / "counts.csv").exists()
    assert _footprint(tmp_path, root) == before


def test_a_per_image_delivery_of_a_bucket_of_another_dataset_than_its_root_refuses_at_the_tool(
    tmp_path: Path,
) -> None:
    """The tool answers the sentence naming the bucket and both ids, and writes nothing."""
    root, bucket = _published(tmp_path)
    recorded = read_bucket(root, bucket).dataset_id
    _identity_removed(root)
    minted = _registered_again(tmp_path, root)
    assert minted != recorded
    before = _footprint(tmp_path, root)

    answered = _deliver(tmp_path, root, bucket)

    assert set(answered) == {"error"}, answered
    for named in (bucket, recorded, minted):
        assert named in answered["error"], answered
    assert not (tmp_path / "out" / "counts.csv").exists()
    assert _footprint(tmp_path, root) == before


def test_a_raster_bucket_under_a_root_with_no_identity_still_answers_the_raster_refusal(
    tmp_path: Path,
) -> None:
    """The raster refusal comes first, naming the orthomosaic door."""
    from tests.test_the_delivered_crop_is_the_registrys import (
        _orthomosaic_scene,
    )

    bucket = _orthomosaic_scene(tmp_path)
    fx.seed_confirmed_count(tmp_path, measured_subject=fx.COUNT_SUBJECT)
    _identity_removed(bucket.root)

    answered = _deliver(tmp_path, bucket.root, bucket.name)

    assert "deliver_orthomosaic_plant_counts" in answered["error"], answered


# ── the gate ─────────────────────────────────────────────────────────────


def _gated(project: Path, *buckets: Bucket) -> None:
    """Hand ``buckets`` to the gate of a per-image count delivery, as a door that made no tie."""
    from tcip_mcp.delivery import Result, gate
    from tcip_mcp.traits import PER_IMAGE_COUNT

    gate(
        project,
        list(buckets),
        delivery_kind=PER_IMAGE_COUNT,
        revision=fx.count_revision(project),
        result=Result((), (), population=()),
    )


def test_the_gate_over_a_root_with_no_identity_refuses_with_the_ties_sentence(
    tmp_path: Path,
) -> None:
    """A bucket handed straight to the gate under a root with no identity refuses at the gate."""
    root, name = _published(tmp_path)
    bucket = read_bucket(root, name)
    _identity_removed(root)

    with pytest.raises(ValueError) as refused:
        _gated(tmp_path, bucket)

    assert str(refused.value) == unregistered_refusal(root)


def test_the_gate_over_each_bucket_of_another_dataset_than_its_root_refuses_with_the_ties_sentence(
    tmp_path: Path,
) -> None:
    """The gate refuses the old bucket under the re-registered root, whichever comes first."""
    from tcip_mcp.buckets import require_delivered_dataset
    from tests._chain_fixtures import predicted, published
    from tests._mapping_fixtures import COUNT_CROP
    from tests.test_deliver_per_image_counts_bucket_regime import SCOPE

    root, name = _published(tmp_path)
    old = read_bucket(root, name)
    _identity_removed(root)
    minted = _registered_again(tmp_path, root)
    image = image_dir(root, "2026-01-01") / "a.png"
    new = published(
        tmp_path,
        "again/2026-01-01",
        [predicted(image, [SCOPE["subject"]] * 2)],
        scope=SCOPE,
        crop=COUNT_CROP,
    )
    assert new.dataset_id == minted != old.dataset_id
    with pytest.raises(ValueError) as tie:
        require_delivered_dataset([old])

    for buckets in ((new, old), (old, new)):
        with pytest.raises(ValueError) as refused:
            _gated(tmp_path, *buckets)
        assert str(refused.value) == str(tie.value)
    for named in (old.name, old.dataset_id, minted):
        assert named in str(tie.value), tie.value


# ── the route ────────────────────────────────────────────────────────────


def _route_bucket(project: Path) -> Bucket:
    """Publish the route fixture's bucket under the project's own root."""
    from tests.test_tcip_web_results_routes import _unassessed_count_bucket

    return _unassessed_count_bucket(project)


def _refused_by_the_route(
    client: TestClient, project: Path, bucket: Bucket, **extra: object
) -> DeliveryRefusal:
    """Post the per-image export of ``bucket``; return its 400, asserting it ships nothing.

    ``extra`` is merged into the body; a request carrying any is posted as a browser.
    """
    from tests.test_tcip_web_results_routes import _export_count, _per_image

    answered = _export_count(
        client,
        _per_image(bucket, **extra),
        headers=BROWSER if extra else None,
    )

    assert answered.status_code == 400, answered.text[:300]
    refusal = DeliveryRefusal.model_validate(answered.json()["detail"])
    assert refusal.shipping == Unacknowledgeable()
    assert not (project / "results_export" / "counts.csv").exists()
    assert read_delivery_events(project) == []
    return refusal


def test_a_per_image_export_under_a_root_with_no_identity_refuses_at_the_route(
    client: TestClient, tmp_path: Path
) -> None:
    """The route answers 400 with the sentence naming the root, and writes nothing."""
    bucket = _route_bucket(tmp_path)
    _identity_removed(tmp_path)
    before = _footprint(tmp_path, tmp_path)

    refusal = _refused_by_the_route(client, tmp_path, bucket)

    assert refusal.message == unregistered_refusal(tmp_path)
    assert _footprint(tmp_path, tmp_path) == before


def test_a_per_image_export_of_a_bucket_of_another_dataset_than_its_root_refuses_at_the_route(
    client: TestClient, tmp_path: Path
) -> None:
    """The route answers 400 with the sentence naming the bucket and both ids; writes nothing."""
    bucket = _route_bucket(tmp_path)
    recorded = read_bucket(bucket.root, bucket.name).dataset_id
    _identity_removed(tmp_path)
    minted = _registered_again(tmp_path, tmp_path)
    assert minted != recorded
    before = _footprint(tmp_path, tmp_path)

    refusal = _refused_by_the_route(client, tmp_path, bucket)

    for named in (bucket.name, recorded, minted):
        assert named in refusal.message, refusal
    assert _footprint(tmp_path, tmp_path) == before


def test_a_posted_acknowledgment_does_not_carry_a_per_image_export_past_the_tie(
    client: TestClient, tmp_path: Path
) -> None:
    """An acknowledgment of the result's real digest refuses the same way, shipping nothing.

    The digest is the one the bucket's unacknowledged export names before the identity is
    removed, so the baseline delivers under it; only the acknowledgment's own record stands.
    """
    from tests.test_tcip_web_results_routes import _digest, _per_image

    bucket = _route_bucket(tmp_path)
    digest = _digest(client, _per_image(bucket))
    _identity_removed(tmp_path)

    refusal = _refused_by_the_route(
        client,
        tmp_path,
        bucket,
        acknowledgment={"reason": "a look", "result_sha256": digest},
    )

    assert refusal.message == unregistered_refusal(tmp_path)
    assert len(audit_rows(tmp_path, "delivery_acknowledged")) == 1


# ── the three doors that read the tie ────────────────────────────────────


def test_the_per_image_the_orthomosaic_and_the_per_plant_doors_answer_one_tie(
    tmp_path: Path,
) -> None:
    """One root holding an image bucket, a raster bucket and a mapping gives each door one sentence.

    With the identity removed the three doors answer the same sentence. Registered again, each names
    its own bucket and the same two ids, the per-image and the per-plant door the same bucket, in
    words that differ from the orthomosaic door's by the bucket's name alone. The count trait's
    confirmed meaning is seeded for the door about to be called, since each door's meaning is the
    latest revision's.
    """
    from tcip_mcp.tools.orthomosaic_tools import (
        deliver_orthomosaic_plant_counts,
    )
    from tests.test_deliver_per_plant_csv import (
        _bucket,
        _deliver as deliver_per_plant,
        _mapped,
        _results,
    )
    from tests.test_orthomosaic_tools import _GRID
    from tests.test_the_delivered_crop_is_the_registrys import (
        _grid_registry,
        _orthomosaic_scene,
    )

    fx.seed_delivery_traits(tmp_path)
    root, date = _mapped(tmp_path)
    image_bucket = _bucket(tmp_path, root, date, {"P1": 1})
    raster_bucket = _orthomosaic_scene(tmp_path).name
    registry = _grid_registry(tmp_path)
    recorded = read_bucket(root, image_bucket).dataset_id
    assert read_bucket(root, raster_bucket).dataset_id == recorded

    def answers() -> tuple[str, str, str]:
        fx.seed_confirmed_count(tmp_path, measured_subject=fx.COUNT_SUBJECT)
        per_image = _deliver(tmp_path, root, image_bucket)
        fx.seed_confirmed_orthomosaic_count(tmp_path, "stem_count")
        orthomosaic = deliver_orthomosaic_plant_counts(
            tmp_path,
            str(root),
            raster_bucket,
            registry,
            str(tmp_path / "counts.csv"),
            "stem_count",
            _GRID,
            actor=None,
        )
        fx.seed_confirmed_aggregate(
            tmp_path, "stem_count", value_keys=["count"]
        )
        per_plant = deliver_per_plant(
            tmp_path,
            _results(date, ("P1", 1)),
            ["P1"],
            root,
            [image_bucket],
            plant_mapping="valley",
        )
        for door in (per_image, orthomosaic, per_plant):
            assert set(door) == {"error"}, door
        return per_image["error"], orthomosaic["error"], per_plant["error"]

    _identity_removed(root)

    assert set(answers()) == {unregistered_refusal(root)}

    minted = _registered_again(tmp_path, root)
    per_image, orthomosaic, per_plant = answers()

    assert minted != recorded
    for named in (recorded, minted, image_bucket):
        assert named in per_image, per_image
    assert per_plant == per_image
    assert orthomosaic == per_image.replace(image_bucket, raster_bucket)
    for written in ("out/counts.csv", "counts.csv", "o.csv"):
        assert not (tmp_path / written).exists(), written
    assert read_delivery_events(tmp_path) == []
