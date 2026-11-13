"""The Setup tab's dataset identity: a root's registration read, and registered, from the web app.

``GET /api/dataset/identity`` answers whether an identity document reads at a root and at what
crop, beside the crop vocabulary and, for a registered root, the plant mappings built over it.
``POST /api/dataset/register`` registers the root through ``register_dataset`` by the person the
request names, answering each answer that function gives as it gives it; a change of a registered
crop carries the stored crop the person was shown.

The steps made by hand are fault injection: removing or damaging a root's ``dataset.json``, a
change to the project's files outside the platform; damaging a stored mapping or registry record;
and failing the audit log's append.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tcip_store
from tcip_mcp.dataset_layout import (
    dataset_identity_path,
    image_dir,
    read_dataset_identity,
)
from tcip_mcp.identity import actor
from tcip_mcp.tools.project_tools import read_datasets, register_dataset
from tests._audit_fixtures import audit_rows, refuse_audit_appends
from tests._mapping_fixtures import COUNT_CROP, CROP
from tests._producer_fixtures import (
    label_image,
    register_dataset_at,
    write_image,
)
from tests._web_fixtures import new_project
from tests.test_a_refused_record_reads_as_its_readers_sentence import (
    _rewritten,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastapi.testclient import TestClient
    from starlette.testclient import httpx

    from tcip_web.state import OpenProject

REGISTER = "/api/dataset/register"
IDENTITY = "/api/dataset/identity"
PERSON = "tester"
"""The person every request here names."""

DATE = "2026-01-01"


def _imaged(root: Path) -> Path:
    """Write one image of :data:`DATE` under ``root`` with an empty label document; return root.

    The labeled image gives the root a fingerprint.
    """
    image = write_image(image_dir(root, DATE) / "a.png")
    label_image(image, [], 100, 80)
    return root


def _register(
    client: TestClient,
    root: Path,
    crop: str = CROP,
    confirmed_crop: str | None = None,
) -> httpx.Response:
    """Post a registration of ``root`` at ``crop`` by :data:`PERSON`, with ``confirmed_crop``."""
    return client.post(
        REGISTER,
        json={
            "dataset_root": str(root),
            "crop": crop,
            "user": PERSON,
            "confirmed_crop": confirmed_crop,
        },
    )


def _identity(client: TestClient, root: Path) -> dict:
    """Return what the identity read answers for ``root``, asserting it answered 200."""
    resp = client.get(IDENTITY, params={"dataset_root": str(root)})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _written(project: Path, root: Path) -> dict[str, object]:
    """Return what a registration of ``root`` in ``project`` writes: its document and registry."""
    return {
        "identity": tcip_store.read_blob_versioned(
            dataset_identity_path(root), default=None
        ).value,
        "registry": read_datasets(project),
    }


# ── Registering ───────────────────────────────────────────────────────


def test_an_unregistered_root_is_registered_by_the_person_the_request_names(
    opened_client: TestClient, opened_project: Path
) -> None:
    """The route answers the identity written, and the root's one audit line names the person."""
    root = _imaged(opened_project / "ds")

    resp = _register(opened_client, root)

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    identity = read_dataset_identity(root)
    assert identity is not None
    assert answer == {
        "dataset_root": str(root),
        **identity.model_dump(mode="json"),
        "minted": True,
        "repaired": None,
    }
    assert identity.crop == CROP
    (line,) = audit_rows(root, "register_dataset")
    assert line["status"] == "ok"
    assert line["actor"] == actor(PERSON)


def test_the_agents_check_reads_back_what_the_route_registered(
    opened_client: TestClient, opened: OpenProject
) -> None:
    """``check_dataset_identity`` through the server answers the crop and id the route gave."""
    from tests._server_calls import call_through_handshake
    from tests._web_fixtures import bound_to

    root = _imaged(opened.root / "ds")
    registered = _register(opened_client, root).json()

    (checked,) = call_through_handshake(
        [("check_dataset_identity", {"dataset_root": str(root)})],
        bound=bound_to(opened),
    )

    assert checked["state"] == "unchanged", checked
    assert checked["identity"]["crop"] == registered["crop"]
    assert checked["identity"]["id"] == registered["id"]


def _two_registry_entries(project: Path) -> Path:
    """Return a root the project's registry holds under two ids, its document removed."""
    from tests.test_registering_a_dataset_restores_or_refuses import (
        _two_registry_entries_at_one_path,
    )

    _project, root, _cited = _two_registry_entries_at_one_path(project)
    dataset_identity_path(root).unlink()
    return root


@pytest.mark.parametrize(
    ("made", "crop"),
    [
        (lambda project: _imaged(project / "ds"), "no-such-crop"),
        (lambda project: project / "missing", CROP),
        (_two_registry_entries, CROP),
    ],
    ids=["crop_not_in_vocabulary", "no_directory", "cited_by_two_ids"],
)
def test_a_refused_registration_answers_the_functions_own_sentence_and_writes_nothing(
    opened_client: TestClient,
    opened_project: Path,
    made: Callable[[Path], Path],
    crop: str,
) -> None:
    """The route answers 400 with the error ``register_dataset`` gives, and nothing is written."""
    root = made(opened_project)
    before = _written(opened_project, root)
    refused = register_dataset(opened_project, str(root), crop, actor=None)
    assert set(refused) == {"error"}, refused

    resp = _register(opened_client, root, crop)

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == refused["error"]
    assert _written(opened_project, root) == before


def test_a_registration_over_an_unreadable_bucket_answers_the_bucket_readers_refusal(
    opened_client: TestClient, opened_project: Path
) -> None:
    """The refusal names the bucket; the identity and the registry are unchanged.

    The root's identity document is removed, so the registration reads the root's citations, and
    the root's audit log gains the call's ``exception`` line, carrying none of the facts an answer
    gives a registration's line.
    """
    from tests.test_registering_a_dataset_restores_or_refuses import (
        _damaged_bucket,
    )

    root, name = _damaged_bucket(opened_project)
    dataset_identity_path(root).unlink()
    before = _written(opened_project, root)
    lines = audit_rows(root, "register_dataset")

    resp = _register(opened_client, root)

    assert resp.status_code == 400, resp.text
    assert repr(name) in resp.json()["detail"], resp.text
    assert _written(opened_project, root) == before
    (raised,) = audit_rows(root, "register_dataset")[len(lines) :]
    assert raised["status"] == "exception"
    assert raised["actor"] == actor(PERSON)
    assert not {"dataset_id", "minted", "repaired"} & set(raised), raised


def test_a_root_whose_document_is_gone_takes_back_its_id_and_says_so(
    opened_client: TestClient, opened_project: Path
) -> None:
    """The answer is 200 with the id the registry held and the function's ``repaired`` sentence."""
    root = _imaged(opened_project / "ds")
    registered = register_dataset_at(opened_project, root, crop=CROP)
    dataset_identity_path(root).unlink()

    resp = _register(opened_client, root)

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["id"] == registered
    assert str(dataset_identity_path(root)) in answer["repaired"]
    assert repr(registered) in answer["repaired"]


def test_a_damaged_document_one_id_cites_is_restored(
    opened_client: TestClient, opened_project: Path
) -> None:
    """A document its model refuses at a registered root takes back the registry's id.

    The registration's line, by the person, carries the id, that it was not minted, and the
    answer's ``repaired`` sentence.
    """
    root = _imaged(opened_project / "ds")
    registered = register_dataset_at(opened_project, root, crop=CROP)
    _rewritten(dataset_identity_path(root), {"crop": 5})
    lines = audit_rows(root, "register_dataset")

    resp = _register(opened_client, root)

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["id"] == registered
    assert answer.get("minted") is False, answer
    assert str(dataset_identity_path(root)) in answer["repaired"]
    (line,) = audit_rows(root, "register_dataset")[len(lines) :]
    assert line["actor"] == actor(PERSON)
    assert line.get("dataset_id") == registered, line
    assert line.get("minted") is False, line
    assert line.get("repaired") == answer["repaired"]


def test_a_damaged_document_nothing_cites_refuses_and_stays(
    opened_client: TestClient, opened_project: Path
) -> None:
    """The root is another project's and holds no bucket: 400 with the function's refusal.

    The damaged document is left as it was.
    """
    other = new_project(opened_project.parent / "other").root
    root = _imaged(other.parent / "elsewhere")
    register_dataset_at(other, root, crop=CROP)
    path = dataset_identity_path(root)
    _rewritten(path, {"crop": 5})
    damaged = path.read_bytes()
    refused = register_dataset(opened_project, str(root), CROP, actor=None)

    resp = _register(opened_client, root)

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == refused["error"]
    assert "backup" in resp.json()["detail"]
    assert path.read_bytes() == damaged


def test_a_root_outside_the_allowed_roots_answers_403(
    opened_client: TestClient, opened_project: Path
) -> None:
    """A root beside the workspace is refused before any registration."""
    outside = _imaged(opened_project.parent.parent / "outside")

    resp = _register(opened_client, outside)

    assert resp.status_code == 403, resp.text
    assert read_dataset_identity(outside) is None


def test_a_registration_whose_audit_line_is_lost_answers_409_committed(
    opened_client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The identity is written; the answer says the change committed with its line unwritten."""
    from tcip_web.routes.audit_gap import AuditGapRefusal

    root = _imaged(opened_project / "ds")
    refuse_audit_appends(monkeypatch)

    resp = _register(opened_client, root)

    assert resp.status_code == 409, resp.text
    refusal = AuditGapRefusal.model_validate(resp.json()["detail"])
    assert refusal.committed is True
    assert refusal.what is None
    identity = read_dataset_identity(root)
    assert identity is not None and identity.crop == CROP


# ── Changing a registered crop ─────────────────────────────────────────


@pytest.mark.parametrize(
    "confirmed", [None, COUNT_CROP], ids=["unconfirmed", "another_crop"]
)
def test_a_crop_change_not_confirmed_with_the_stored_crop_refuses_naming_it(
    opened_client: TestClient, opened_project: Path, confirmed: str | None
) -> None:
    """409 names the stored crop, and the identity is unchanged."""
    root = _imaged(opened_project / "ds")
    register_dataset_at(opened_project, root, crop=CROP)
    before = _written(opened_project, root)

    resp = _register(opened_client, root, COUNT_CROP, confirmed)

    assert resp.status_code == 409, resp.text
    assert repr(CROP) in resp.json()["detail"], resp.text
    assert _written(opened_project, root) == before


def test_a_crop_change_confirmed_with_the_stored_crop_keeps_the_id(
    opened_client: TestClient, opened_project: Path
) -> None:
    """The answer is 200, the id kept and the crop the one given."""
    root = _imaged(opened_project / "ds")
    registered = register_dataset_at(opened_project, root, crop=CROP)

    resp = _register(opened_client, root, COUNT_CROP, CROP)

    assert resp.status_code == 200, resp.text
    identity = read_dataset_identity(root)
    assert identity is not None
    assert (identity.id, identity.crop) == (registered, COUNT_CROP)


def test_a_registration_at_the_stored_crop_needs_no_confirmation(
    opened_client: TestClient, opened_project: Path
) -> None:
    """Registering a registered root again at its own crop answers 200 with its id."""
    root = _imaged(opened_project / "ds")
    registered = register_dataset_at(opened_project, root, crop=CROP)

    resp = _register(opened_client, root)

    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == registered


def test_a_padded_crop_equal_to_the_stored_one_needs_no_confirmation(
    opened_client: TestClient, opened_project: Path
) -> None:
    """The request's crop reads as the stored crop reads, its surrounding spaces stripped."""
    root = _imaged(opened_project / "ds")
    registered = register_dataset_at(opened_project, root, crop=CROP)

    resp = _register(opened_client, root, f"  {CROP} ")

    assert resp.status_code == 200, resp.text
    assert (resp.json()["id"], resp.json()["crop"]) == (registered, CROP)


# ── Reading a root's identity ──────────────────────────────────────────


def test_the_identity_read_answers_an_unregistered_root_with_the_crop_vocabulary(
    opened_client: TestClient, opened_project: Path
) -> None:
    """A root holding no identity document reads unregistered, beside every crop crops.yml names."""
    from tcip_mcp.traits import registered_crops

    root = _imaged(opened_project / "ds")

    read = _identity(opened_client, root)

    assert read == {
        "dataset_root": str(root),
        "identity": {"kind": "unregistered"},
        "crops": sorted(registered_crops()),
    }


def test_the_identity_read_answers_a_refused_document_with_its_readers_sentence(
    opened_client: TestClient, opened_project: Path
) -> None:
    """A document its model refuses reads refused, carrying the read's own sentence."""
    root = _imaged(opened_project / "ds")
    register_dataset_at(opened_project, root, crop=CROP)
    _rewritten(dataset_identity_path(root), {"crop": 5})
    with pytest.raises(ValueError) as refusal:
        read_dataset_identity(root)

    read = _identity(opened_client, root)

    assert read["identity"] == {
        "kind": "refused",
        "reason": str(refusal.value),
    }


def _mapped(project: Path) -> tuple[Path, str]:
    """Return a root mapped under ``valley`` in ``project`` and the id it is registered under.

    A second root, mapped under ``east`` against the same plant registry, stands beside it.
    """
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests._mapping_fixtures import geolocated_captures, map_captures

    root = project / "ds"
    map_captures(project, root, [DATE])
    east = project / "east"
    east.mkdir()
    register_dataset_at(project, east, crop=CROP)
    geolocated_captures(east, [DATE])
    built = build_plant_mapping(
        project,
        name="east",
        dataset_root=str(east),
        plant_registry="reg",
        actor=None,
    )
    assert "error" not in built, built
    identity = read_dataset_identity(root)
    assert identity is not None
    return root, identity.id


def test_the_identity_read_lists_the_mappings_built_over_the_root_with_their_registrys_crop(
    opened_client: TestClient, opened_project: Path
) -> None:
    """A registered root reads its identity, and lists ``valley`` and not ``east``."""
    root, dataset_id = _mapped(opened_project)

    read = _identity(opened_client, root)

    assert read["identity"] == {
        "kind": "registered",
        "identity": {
            "crop": CROP,
            "id": dataset_id,
            "fingerprint": read["identity"]["identity"]["fingerprint"],
        },
        "mappings": {
            "placed": [
                {
                    "name": "valley",
                    "plant_registry": "reg",
                    "registry": {"kind": "crop", "crop": CROP},
                }
            ],
            "unplaced": [],
        },
    }


def test_a_mapping_or_registry_that_will_not_read_is_listed_with_its_refusal(
    opened_client: TestClient, opened_project: Path
) -> None:
    """A mapping record that will not read is unplaced; a registry's refusal stands for its crop.

    ``east``'s record is damaged, so it is listed apart by name; the registry both name is
    rewritten so its model refuses it, so ``valley`` carries that refusal; the read answers.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        load_registry,
        plant_mapping_key,
        plant_registry_key,
    )
    from tests._record_damage_fixtures import UNDECODABLE, damage_record

    root, _dataset_id = _mapped(opened_project)
    damage_record(plant_mapping_key(opened_project, "east"), UNDECODABLE)
    key = plant_registry_key(opened_project, "reg")
    record = tcip_store.read(key)
    assert isinstance(record, dict), record
    tcip_store.replace(key, {**record, "crop": 5})
    with pytest.raises(ValueError) as registry_refusal:
        load_registry(opened_project, "reg")

    mappings = _identity(opened_client, root)["identity"]["mappings"]

    assert mappings["placed"] == [
        {
            "name": "valley",
            "plant_registry": "reg",
            "registry": {
                "kind": "registry_refused",
                "refusal": str(registry_refusal.value),
            },
        }
    ]
    (unplaced,) = mappings["unplaced"]
    assert unplaced["name"] == "east"
    assert "east" in unplaced["refusal"]


def test_a_registry_whose_bytes_will_not_decode_stands_as_its_mappings_refusal(
    opened_client: TestClient, opened_project: Path
) -> None:
    """The registry's stored bytes are damaged; the read answers, ``valley`` carrying why."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        plant_registry_key,
    )
    from tests._record_damage_fixtures import UNDECODABLE, damage_record

    root, _dataset_id = _mapped(opened_project)
    key = plant_registry_key(opened_project, "reg")
    damage_record(key, UNDECODABLE)
    with pytest.raises(tcip_store.DecodeError) as refusal:
        tcip_store.read(key)

    placed = _identity(opened_client, root)["identity"]["mappings"]["placed"]

    assert placed == [
        {
            "name": "valley",
            "plant_registry": "reg",
            "registry": {
                "kind": "registry_refused",
                "refusal": str(refusal.value),
            },
        }
    ]
