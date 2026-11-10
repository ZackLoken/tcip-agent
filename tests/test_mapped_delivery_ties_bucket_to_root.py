"""A mapped delivery refuses a bucket whose recorded dataset is not the dataset of its root.

Through the doors that read a plant mapping: the per-plant tool, and the phenology tool with its
two routes. A bucket recording another dataset than the root it is delivered under (the dataset
copied without its identity document, the copy registered and a mapping built over it) refuses
naming the bucket, and no delivery event is recorded. The orthomosaic door's own tests hold the
same refusal there.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests._mapping_fixtures import COUNT_CROP, CROP, registered_copy

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _map(project: Path, name: str, root: Path, registry: str) -> None:
    """Build the plant mapping ``name`` over ``root`` against the registered ``registry``."""
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    built = build_plant_mapping(
        project,
        name=name,
        dataset_root=str(root),
        plant_registry=registry,
        actor=None,
    )
    assert "error" not in built, built


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

    The dataset ``ds`` is copied with its store and the copy registered; a mapping built over the
    copy matches the copy's identity, and the copy's bucket still records ``ds``'s id.
    """
    from tcip_mcp.buckets import read_bucket
    from tests.test_deliver_per_plant_csv import (
        _bucket,
        _deliver,
        _mapped,
        _results,
    )

    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})
    recorded = read_bucket(dataset_root, bucket).dataset_id
    copy, minted = registered_copy(
        tmp_path, dataset_root, "ds-copy", crop=COUNT_CROP
    )
    _map(tmp_path, "copy-valley", copy, "reg")

    refused = _deliver(
        tmp_path,
        _results(date, ("P1", 1)),
        ["P1"],
        copy,
        [bucket],
        plant_mapping="copy-valley",
    )

    assert "error" in refused, refused
    for named in (bucket, recorded, minted):
        assert named in refused["error"], refused
    assert not (tmp_path / "o.csv").exists()
    _no_delivery_recorded(tmp_path)


# ── the phenology tool and its two routes ────────────────────────────────


def test_a_phenology_delivery_of_a_bucket_recording_another_dataset_than_its_root_refuses(
    client: TestClient, tmp_path: Path
) -> None:
    """The phenology tool and both routes refuse a bucket recording another dataset than its root.

    The dataset is copied with its store and the copy registered at :data:`CROP`; a mapping built
    over the copy matches the copy's identity, and the copy's buckets still record the original's.
    """
    from tcip_mcp.buckets import read_bucket
    from tests.test_second_trait_acceptance import _currant_bloom_body
    from tests.test_the_delivered_crop_is_the_registrys import (
        _phenology_refusals,
    )

    body = _currant_bloom_body(tmp_path)
    bucket = body["buckets"][0]
    recorded = read_bucket(body["dataset_root"], bucket).dataset_id
    copy, minted = registered_copy(
        tmp_path, Path(body["dataset_root"]), "ds-copy", crop=CROP
    )
    _map(tmp_path, "copy-valley", copy, "reg")

    refusals = _phenology_refusals(
        client,
        tmp_path,
        {**body, "dataset_root": str(copy), "mapping_name": "copy-valley"},
    )

    for refusal in refusals:
        for named in (recorded, minted):
            assert named in refusal, refusal
        assert any(b in refusal for b in body["buckets"]), refusal
    _no_delivery_recorded(tmp_path)


# ── a delivery that shipped before the tie ───────────────────────────────


def test_an_acknowledged_per_plant_delivery_of_a_bucket_recording_another_dataset_refuses(
    tmp_path: Path,
) -> None:
    """A delivery shipping under a breeder's acknowledgment still refuses the bucket's dataset.

    The unassessed bucket of ``ds`` is read under a registered copy of ``ds`` whose mapping
    ``valley`` matches the copy's identity. The assessment-gate refusal an acknowledgment ships
    past is not the refusal: the delivery reaches the mapping and refuses there, naming the bucket
    and both ids, and writes no CSV and no delivery event.
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
        tmp_path, experiment_id="exp-copied", images={"p1": 1}
    )
    copy, minted = registered_copy(
        tmp_path, bucket.root, "ds-copy", crop=COUNT_CROP
    )
    _map(tmp_path, "valley", copy, "valley-registry")

    with pytest.raises(MappingDeliveryError) as refused:
        deliver_acknowledged(
            tmp_path,
            {"p1": [1]},
            tmp_path / "o.csv",
            "stem_count",
            value_key="count",
            delivery_kind=PER_PLANT_COUNT_AGGREGATE,
            buckets=[read_bucket(copy, bucket.name)],
        )

    for named in (bucket.name, bucket.dataset_id, minted):
        assert named in str(refused.value), refused.value
    assert not (tmp_path / "o.csv").exists()
    _no_delivery_recorded(tmp_path)
