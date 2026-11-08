"""The crop a dataset identity and a plant registry record, through their registration doors.

Each door refuses an empty crop and one crops.yml names for no trait, stores the crop stripped,
and each reader refuses a stored record whose crop is empty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

import tcip_store
from tcip_mcp.dataset_layout import (
    dataset_identity_path,
    read_dataset_identity,
)
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.phenology_tools import register_plant_registry
from tcip_mcp.tools.project_tools import register_dataset
from tcip_mcp.traits import registered_crops
from tests._mapping_fixtures import PLANTS, write_plant_csv
from tests._record_damage_fixtures import damage_record
from tests.test_project_tools import _make_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

UNNAMED_CROP = "not-a-crop"
"""A crop name chosen to be absent from crops.yml; each test using it asserts the absence."""


def _crop(index: int = 0) -> str:
    """Return the ``index``-th crop crops.yml names, in sorted order."""
    return sorted(registered_crops())[index]


def _registered_plants(project: Path, crop: str) -> dict[str, Any]:
    """Register :data:`PLANTS` as ``valley`` with ``crop``; return the tool's answer."""
    csv_path = write_plant_csv(project / "plants.csv", PLANTS)
    return cast(
        "dict[str, Any]",
        register_plant_registry(
            project,
            name="valley",
            csv_paths=[str(csv_path)],
            crop=crop,
            site="orchard",
            actor=None,
        ),
    )


def _with_empty_crop(stored: JsonValue) -> bytes:
    """Return the record ``stored`` encoded with its ``crop`` emptied."""
    return tcip_store.encode_record(
        {**cast("dict[str, Any]", stored), "crop": ""}
    )


@pytest.mark.parametrize("crop", ["", "   ", UNNAMED_CROP])
def test_a_plant_registry_with_an_empty_or_unnamed_crop_refuses_naming_crop(
    tmp_path: Path, crop: str
) -> None:
    """``register_plant_registry`` refuses the crop by name and stores nothing."""
    assert UNNAMED_CROP not in registered_crops()

    refused = _registered_plants(tmp_path, crop)

    assert "crop" in refused.get("error", ""), refused
    assert not tcip_store.exists(
        plant_mapping.plant_registry_key(tmp_path, "valley")
    )


def test_a_dataset_with_a_crop_crops_yml_does_not_name_refuses_naming_crop(
    tmp_path: Path,
) -> None:
    """``register_dataset`` refuses an unnamed crop by name and writes no identity."""
    assert UNNAMED_CROP not in registered_crops()
    root = tmp_path / "dataset"
    _make_dataset(root)

    refused = register_dataset(tmp_path, str(root), UNNAMED_CROP, actor=None)

    assert "crop" in refused.get("error", ""), refused
    assert read_dataset_identity(root) is None


def test_each_door_stores_a_named_crop_stripped(tmp_path: Path) -> None:
    """A padded crop crops.yml names is stored stripped by both doors.

    A same-name re-registration with the bare crop, and again with the padded one, returns the
    stored record unchanged.
    """
    crop = _crop()
    root = tmp_path / "dataset"
    _make_dataset(root)

    registry = _registered_plants(tmp_path, f"  {crop} ")
    again = _registered_plants(tmp_path, crop)
    padded_again = _registered_plants(tmp_path, f"  {crop} ")
    dataset = register_dataset(tmp_path, str(root), f" {crop}  ", actor=None)

    assert "error" not in registry, registry
    stored = plant_mapping.load_registry(tmp_path, "valley")
    assert stored["crop"] == crop
    assert again == stored
    assert padded_again == stored
    assert again["registered_at"] == registry["registered_at"]
    assert "error" not in dataset, dataset
    identity = read_dataset_identity(root)
    assert identity is not None
    assert identity.crop == crop


def test_a_same_name_registration_with_another_crop_refuses_naming_both(
    tmp_path: Path,
) -> None:
    """The same plants under a taken name with a different crop refuse, the stored one kept."""
    first, other = _crop(0), _crop(1)
    stored = _registered_plants(tmp_path, first)
    assert "error" not in stored, stored

    refused = _registered_plants(tmp_path, other)

    error = refused.get("error", "")
    assert repr(first) in error and repr(other) in error, refused
    assert plant_mapping.load_registry(tmp_path, "valley") == stored


def test_a_stored_registry_whose_crop_is_empty_refuses_on_load(
    tmp_path: Path,
) -> None:
    """``load_registry`` refuses the stored record, naming the registry and ``crop``."""
    stored = _registered_plants(tmp_path, _crop())
    assert "error" not in stored, stored
    key = plant_mapping.plant_registry_key(tmp_path, "valley")
    damage_record(key, _with_empty_crop(tcip_store.read(key)))

    with pytest.raises(tcip_store.ReadRefusedError) as refused:
        plant_mapping.load_registry(tmp_path, "valley")

    assert "plant registry 'valley'" in str(refused.value)
    assert "crop" in str(refused.value)


def test_a_stored_dataset_identity_whose_crop_is_empty_refuses_on_read(
    tmp_path: Path,
) -> None:
    """``read_dataset_identity`` refuses the stored identity, naming its path and ``crop``."""
    root = tmp_path / "dataset"
    _make_dataset(root)
    assert "error" not in register_dataset(
        tmp_path, str(root), _crop(), actor=None
    )
    path = dataset_identity_path(root)
    stored = tcip_store.read_blob_versioned(path)
    tcip_store.put_blob(
        path,
        _with_empty_crop(tcip_store.decode_value(stored.value)),
        expect=stored.version,
    )

    with pytest.raises(tcip_store.ReadRefusedError) as refused:
        read_dataset_identity(root)

    assert str(path) in str(refused.value)
    assert "crop" in str(refused.value)
