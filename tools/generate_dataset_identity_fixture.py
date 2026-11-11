"""Write served dataset identity reads and registrations as a fixture file.

All are produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), the project ``valley``, created through
``tests._web_fixtures.named_project`` and opened through the picker's route, holds three dataset
roots. ``mapped`` is registered and read (``unmapped``), mapped under ``valley`` against a plant
registry of its crop (``tests._mapping_fixtures.map_captures``) and read (``mapped``), then mapped
twice more, one mapping record and one registry record damaged (:func:`_damaged_mappings`), and
read (``mapped_damaged``); ``fresh`` holds one image and no identity; ``damaged`` is registered
and its identity document then rewritten so its model refuses it (fault injection,
``tests.test_a_refused_record_reads_as_its_readers_sentence._rewritten``).
``GET /api/dataset/identity`` is answered for the other two (``unregistered``, ``refused``),
then ``POST /api/dataset/register`` registers ``fresh`` (``registered``), refuses a crop the
vocabulary does not hold (``unknown_crop``), restores ``damaged``'s id (``repaired``), refuses a
change of ``mapped``'s crop that names no stored crop (``crop_unconfirmed``) and makes it with
one (``crop_changed``), and registers a fourth root, ``unrecorded``, while every audit append
fails (``unrecorded``, ``tests._audit_fixtures.refuse_audit_appends``). Every answer, each
refusal with its status, is written normalized
(``fixture_session.normalized``), each minted id read as its name, to :data:`GENERATED_PATH`.

    python -m tools.generate_dataset_identity_fixture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import FRONTEND_SRC
from tools.fixture_session import (
    ACTOR,
    answered,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "datasetIdentityAnswers.json"

NUMBERED = frozenset({"fingerprint"})
"""The content digest a registration computes over a root's encoded images, which another
platform's encoder writes differently."""

DATE = "2026-05-02"
"""The scene's one capture date."""


def _damaged_mappings(project: Path, root: Path) -> None:
    """Build two more mappings of ``project`` over ``root``, then damage one record of each.

    ``valley-east`` is built against ``reg``, the registry ``valley`` names, and its stored record
    is damaged so it will not decode (fault injection,
    ``tests._record_damage_fixtures.damage_record``); ``valley-north`` is built against ``north``,
    registered from the same plant CSV, whose stored record is then rewritten so its model refuses
    it (fault injection, through the store's own writer).
    """
    import tcip_store
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        plant_mapping_key,
        plant_registry_key,
    )
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests._mapping_fixtures import register_plant_registry_for
    from tests._record_damage_fixtures import damage_record

    north = register_plant_registry_for(
        project, [project / "valley_plants.csv"], name="north"
    )
    for name, registry in (("valley-east", "reg"), ("valley-north", north)):
        built = build_plant_mapping(
            project,
            name=name,
            dataset_root=str(root),
            plant_registry=registry,
            actor=None,
        )
        assert "error" not in built, built
    damage_record(plant_mapping_key(project, "valley-east"), b"{not json")
    key = plant_registry_key(project, north)
    record = tcip_store.read(key)
    assert isinstance(record, dict), record
    tcip_store.replace(key, {**record, "crop": 5})


def produce() -> dict[str, JsonValue]:
    """Lay the scene down in a scratch session; return what the identity routes answered."""
    from tcip_mcp.dataset_layout import (
        dataset_identity_path,
        image_dir,
        read_dataset_identity,
    )
    from tests._audit_fixtures import refuse_audit_appends
    from tests._mapping_fixtures import COUNT_CROP, CROP, map_captures
    from tests._producer_fixtures import (
        label_image,
        register_dataset_at,
        write_image,
    )
    from tests._web_fixtures import named_project
    from tests.test_a_refused_record_reads_as_its_readers_sentence import (
        _rewritten,
    )

    with scratch_session() as (workspace, client):
        valley = named_project(workspace / "valley", "valley")
        opened(client, valley)

        def imaged(name: str) -> Path:
            root = valley.root / name
            image = write_image(image_dir(root, DATE) / "a.png")
            label_image(image, [], 100, 80)
            return root

        def identity(root: Path) -> JsonValue:
            return answered(
                client,
                "get_dataset_identity",
                200,
                params={"dataset_root": str(root)},
            )

        def register(
            root: Path,
            crop: str,
            status: int = 200,
            confirmed_crop: str | None = None,
        ) -> JsonValue:
            return answered(
                client,
                "register_dataset_route",
                status,
                {
                    "dataset_root": str(root),
                    "crop": crop,
                    "user": ACTOR,
                    "confirmed_crop": confirmed_crop,
                },
            )

        mapped = valley.root / "mapped"
        mapped.mkdir()
        register_dataset_at(valley.root, mapped, crop=CROP)
        unmapped = identity(mapped)
        map_captures(valley.root, mapped, [DATE])
        mapped_read = identity(mapped)
        _damaged_mappings(valley.root, mapped)
        mapped_damaged = identity(mapped)
        fresh = imaged("fresh")
        damaged = imaged("damaged")
        damaged_id = register_dataset_at(valley.root, damaged, crop=CROP)
        _rewritten(dataset_identity_path(damaged), {"crop": 5})

        answers = {
            "unregistered": identity(fresh),
            "unmapped": unmapped,
            "mapped": mapped_read,
            "mapped_damaged": mapped_damaged,
            "refused": identity(damaged),
            "unknown_crop": register(fresh, "no-such-crop", 400),
            "registered": register(fresh, CROP),
            "repaired": register(damaged, CROP),
            "crop_unconfirmed": register(mapped, COUNT_CROP, 409),
            "crop_changed": register(mapped, COUNT_CROP, 200, CROP),
        }
        unrecorded = imaged("unrecorded")
        with pytest.MonkeyPatch.context() as patch:
            refuse_audit_appends(patch)
            answers["unrecorded"] = register(unrecorded, CROP, 409)
        ids = {
            "mapped": read_dataset_identity(mapped),
            "fresh": read_dataset_identity(fresh),
            "unrecorded": read_dataset_identity(unrecorded),
        }
        return normalized(
            answers,
            workspace,
            numbered=NUMBERED,
            minted={
                valley.id: "<valley id>",
                damaged_id: "<damaged dataset id>",
                **{
                    record.id: f"<{name} dataset id>"
                    for name, record in ids.items()
                    if record is not None
                },
            },
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
