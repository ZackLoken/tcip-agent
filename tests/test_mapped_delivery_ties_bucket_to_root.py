"""A mapped delivery refuses a bucket whose recorded dataset is not the dataset of its root.

Through the doors that read a plant mapping: the per-plant tool, and the phenology tool with its
two routes. A bucket recording another dataset than the root it is delivered under, its stored
record rewritten through the store after publication as fault injection, under a mapping built over
the root, refuses naming the bucket and both ids, and no delivery event is recorded. The
orthomosaic door's own tests hold the same refusal there.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import require_dataset_identity
from tests._producer_fixtures import bucket_of_another_dataset

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _no_delivery_recorded(project: Path) -> None:
    """Assert ``project`` holds no delivery event."""
    from tcip_mcp.delivery import read_delivery_events

    assert read_delivery_events(project) == []


# ── the per-plant tool ───────────────────────────────────────────────────


@pytest.mark.usefixtures("confirmed_count_aggregate")
def test_a_per_plant_delivery_of_a_bucket_recording_another_dataset_than_its_root_refuses(
    tmp_path: Path,
) -> None:
    """The per-plant tool refuses a bucket recording another dataset than the root it is read under.

    The mapping ``valley`` over ``ds`` matches the root's identity, and the bucket records another
    dataset.
    """
    from tests.test_deliver_per_plant_csv import (
        _bucket,
        _deliver,
        _mapped,
        _results,
    )

    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})
    injected = bucket_of_another_dataset(dataset_root, bucket)

    refused = _deliver(
        tmp_path,
        _results(date, ("P1", 1)),
        ["P1"],
        dataset_root,
        [bucket],
        plant_mapping="valley",
    )

    assert "error" in refused, refused
    root_id = require_dataset_identity(dataset_root).id
    for named in (bucket, injected, root_id):
        assert named in refused["error"], refused
    assert not (tmp_path / "o.csv").exists()
    _no_delivery_recorded(tmp_path)


# ── the phenology tool and its two routes ────────────────────────────────


def test_a_phenology_delivery_of_a_bucket_recording_another_dataset_than_its_root_refuses(
    client: TestClient, tmp_path: Path
) -> None:
    """The phenology tool and both routes refuse buckets recording another dataset than their root.

    The mapping over the root matches the root's identity, and every bucket of the body records
    one other dataset.
    """
    from tests.test_second_trait_acceptance import _currant_bloom_body
    from tests.test_the_delivered_crop_is_the_registrys import (
        _phenology_refusals,
    )

    body = _currant_bloom_body(tmp_path)
    root = Path(body["dataset_root"])
    injected = bucket_of_another_dataset(root, *body["buckets"])

    refusals = _phenology_refusals(client, tmp_path, body)

    for refusal in refusals:
        for named in (injected, require_dataset_identity(root).id):
            assert named in refusal, refusal
        assert any(b in refusal for b in body["buckets"]), refusal
    _no_delivery_recorded(tmp_path)


# ── a delivery that shipped before the tie ───────────────────────────────


def test_an_acknowledged_per_plant_delivery_of_a_bucket_recording_another_dataset_refuses(
    tmp_path: Path,
) -> None:
    """A delivery shipping under a breeder's acknowledgment still refuses the bucket's dataset.

    The unassessed bucket of ``ds`` is read under ``ds``, whose mapping ``valley`` matches the
    root's identity, after its record is rewritten to record another dataset. The
    assessment-gate refusal an acknowledgment ships past is not the refusal: the delivery reaches
    the mapping and refuses there, naming the bucket and both ids, and writes no CSV and no
    delivery event.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        MappingDeliveryError,
    )
    from tcip_mcp.traits import PER_PLANT_COUNT_AGGREGATE
    from tests import _trait_fixtures as fx
    from tests._chain_fixtures import deliver_acknowledged, unassessed_bucket

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(
        tmp_path, "stem_count", value_keys=["count"], measured_subject="bud"
    )
    bucket = unassessed_bucket(
        tmp_path, experiment_id="exp-untied", images={"p1": 1}
    )
    injected = bucket_of_another_dataset(bucket.root, bucket.name)

    with pytest.raises(MappingDeliveryError) as refused:
        deliver_acknowledged(
            tmp_path,
            {"p1": [1]},
            tmp_path / "o.csv",
            "stem_count",
            value_key="count",
            delivery_kind=PER_PLANT_COUNT_AGGREGATE,
            buckets=[read_bucket(bucket.root, bucket.name)],
        )

    root_id = require_dataset_identity(bucket.root).id
    for named in (bucket.name, injected, root_id):
        assert named in str(refused.value), refused.value
    assert not (tmp_path / "o.csv").exists()
    _no_delivery_recorded(tmp_path)
