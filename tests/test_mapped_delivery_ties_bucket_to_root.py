"""A mapped delivery refuses a bucket whose recorded dataset is not the dataset of its root.

Through the doors that read a plant mapping: the per-plant tool, and the phenology tool with its
two routes. A bucket recording no dataset (published before its images' dataset was registered)
and a bucket recording another dataset than the root it is delivered under (the dataset copied
without its identity document, the copy registered and a mapping built over it) each refuse naming
the bucket, and no delivery event is recorded. The orthomosaic door's own tests hold the same two
refusals there.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests._mapping_fixtures import (
    COUNT_CROP,
    CROP,
    PHOTO_POSITION_ERROR_M,
    register_dataset_at,
    registered_copy,
    write_photos,
)

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
    assert recorded is not None
    for named in (bucket, recorded, minted):
        assert named in refused["error"], refused
    assert not (tmp_path / "o.csv").exists()
    _no_delivery_recorded(tmp_path)


@pytest.mark.usefixtures("confirmed_count_aggregate")
def test_a_per_plant_delivery_of_a_bucket_recording_no_dataset_refuses(
    tmp_path: Path,
) -> None:
    """The per-plant tool refuses a bucket published before its images' dataset was registered.

    A second root holds the photograph and the bucket is published under it unregistered; the
    root is then registered and a mapping built over it.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import image_dir
    from tests.test_deliver_per_plant_csv import (
        DATE,
        ONE_PLANT,
        _bucket,
        _deliver,
        _mapped,
        _results,
    )

    _mapped(tmp_path)
    bare = tmp_path / "bare"
    write_photos(
        image_dir(bare, DATE),
        {
            f"{plant}.jpg": (lat, lon, PHOTO_POSITION_ERROR_M)
            for plant, (lat, lon) in ONE_PLANT.items()
        },
        datetime.fromisoformat(DATE).replace(hour=9, minute=30),
        timedelta(minutes=5),
    )
    bucket = _bucket(tmp_path, bare, DATE, {"P1": 1})
    assert read_bucket(bare, bucket).dataset_id is None
    register_dataset_at(tmp_path, bare, crop=COUNT_CROP)
    _map(tmp_path, "bare-valley", bare, "reg")

    refused = _deliver(
        tmp_path,
        _results(DATE, ("P1", 1)),
        ["P1"],
        bare,
        [bucket],
        plant_mapping="bare-valley",
    )

    assert "error" in refused, refused
    assert bucket in refused["error"], refused
    assert "records no dataset" in refused["error"], refused
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

    assert recorded is not None
    for refusal in refusals:
        for named in (recorded, minted):
            assert named in refusal, refusal
        assert any(b in refusal for b in body["buckets"]), refusal
    _no_delivery_recorded(tmp_path)


def test_a_phenology_delivery_of_a_bucket_recording_no_dataset_refuses(
    client: TestClient, tmp_path: Path
) -> None:
    """The phenology tool and both routes refuse buckets published before their dataset existed.

    A second root holds the scene and its buckets are published under it unregistered; the root
    is then registered and a mapping built over it.
    """
    from tcip_mcp.buckets import read_bucket
    from tests._producer_fixtures import registry_over
    from tests.test_plant_mapping_binding import _write_scene
    from tests.test_second_trait_acceptance import (
        FLOWERS,
        _currant_bloom_body,
    )
    from tests.test_the_delivered_crop_is_the_registrys import (
        _phenology_refusals,
    )

    body = _currant_bloom_body(tmp_path)
    bare = tmp_path / "bare"
    bare.mkdir()
    _, buckets = _write_scene(bare)
    recorded = {read_bucket(bare, b).dataset_id for b in buckets.values()}
    assert recorded == {None}
    register_dataset_at(tmp_path, bare, crop=CROP)
    registry_over(bare, FLOWERS)
    _map(tmp_path, "bare-valley", bare, "reg")

    refusals = _phenology_refusals(
        client,
        tmp_path,
        {
            **body,
            "dataset_root": str(bare),
            "mapping_name": "bare-valley",
            "buckets": list(buckets.values()),
        },
    )

    for refusal in refusals:
        assert "records no dataset" in refusal, refusal
        assert any(b in refusal for b in buckets.values()), refusal
    _no_delivery_recorded(tmp_path)
