"""Write served dataset identity reads and registrations as a fixture file.

All are produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), the project ``valley``, created through
``tests._web_fixtures.named_project`` and opened through the picker's route, holds three dataset
roots. ``mapped`` is registered and mapped under ``valley`` against a plant registry of its crop
(``tests._mapping_fixtures.map_captures``); ``fresh`` holds one image and no identity; ``damaged``
is registered and its identity document then rewritten so its model refuses it (fault injection,
``tests.test_a_refused_record_reads_as_its_readers_sentence._rewritten``).
``GET /api/dataset/identity`` is answered for each (``unregistered``, ``mapped``, ``refused``),
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
        map_captures(valley.root, mapped, [DATE])
        fresh = imaged("fresh")
        damaged = imaged("damaged")
        damaged_id = register_dataset_at(valley.root, damaged, crop=CROP)
        _rewritten(dataset_identity_path(damaged), {"crop": 5})

        answers = {
            "unregistered": identity(fresh),
            "mapped": identity(mapped),
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
