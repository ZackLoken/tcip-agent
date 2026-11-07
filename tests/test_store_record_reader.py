"""The store's one record reader: a record read under the store's boundary and validated.

Each record is written by the platform's own producer, then damaged through the store's own
write (a wrong shape) or through ``damage_record`` (bytes that will not decode), and read back
through ``tcip_store.read_record`` and through the doors of readers built on it: the memory
reader and the trait reader call ``read_record``, the flags reader its raw half, ``read_raw``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tcip_store
from tcip_mcp.tools.meta_tools import (
    FrictionReport,
    MemoryKind,
    load_project_memory,
    memory_key,
    report_friction,
)
from tests._record_damage_fixtures import damage_record

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

PLACE = "the report this test reads"


def _reported(project: Path) -> tcip_store.Key:
    """Return the key of a friction report ``report_friction`` wrote under ``project``."""
    made = report_friction(
        project, category="missing_tool", detail="read back", actor=None
    )
    return memory_key(FrictionReport, str(project), made["id"])


def _decode_sentence(key: tcip_store.Key) -> str:
    """Return the sentence the store's own read raises for the record at ``key``."""
    with pytest.raises(tcip_store.DecodeError) as refused:
        tcip_store.read(key)
    return str(refused.value)


def test_a_produced_record_reads_back_as_its_model_by_a_store_or_a_transaction_read(
    project: Path,
) -> None:
    """A record its producer wrote reads back as its model, read by the store or a transaction."""
    key = _reported(project)

    by_store = tcip_store.read_record(
        tcip_store.read, key, FrictionReport, PLACE
    )
    with tcip_store.transaction(key) as txn:
        by_transaction = tcip_store.read_record(
            txn.read, key, FrictionReport, PLACE
        )

    assert by_store == FrictionReport.model_validate(tcip_store.read(key))
    assert by_transaction == by_store
    assert by_store.detail == "read back"


def test_a_record_its_model_refuses_raises_the_typed_refusal_at_the_callers_place(
    project: Path,
) -> None:
    """A record overwritten with a wrong shape refuses at the place its caller named."""
    key = _reported(project)
    tcip_store.replace(
        key,
        tcip_store.read_record(
            tcip_store.read, key, FrictionReport, PLACE
        ).model_dump(mode="json")
        | {"category": "not_a_category"},
    )

    with pytest.raises(tcip_store.ReadRefusedError) as refused:
        tcip_store.read_record(tcip_store.read, key, FrictionReport, PLACE)

    served = refused.value.served
    assert served.where == PLACE
    assert served.refusal.kind == "invalid"
    assert [issue.loc for issue in served.refusal.issues] == [["category"]]


def test_an_undecodable_record_raises_the_stores_sentence_naming_its_key(
    project: Path,
) -> None:
    """A record whose bytes do not decode refuses with the store's sentence, naming its key."""
    key = _reported(project)
    damage_record(key, b"{not valid json")

    with pytest.raises(tcip_store.ReadRefusedError) as refused:
        tcip_store.read_record(tcip_store.read, key, FrictionReport, PLACE)

    assert refused.value.served == tcip_store.RefusedRead(
        where=PLACE,
        refusal=tcip_store.Unreadable(message=_decode_sentence(key)),
    )
    assert key.parts[0] in str(refused.value)


def test_an_absent_record_raises_naming_its_key_unless_its_caller_answers_absence(
    project: Path,
) -> None:
    """An absent record refuses naming its key; one read with a default of None answers None."""
    key = memory_key(FrictionReport, str(project), "never-written")

    with pytest.raises(tcip_store.ReadRefusedError) as refused:
        tcip_store.read_record(tcip_store.read, key, FrictionReport, PLACE)
    answered = tcip_store.read_record(
        tcip_store.read, key, FrictionReport, PLACE, default=None
    )

    served = refused.value.served
    assert served.where == PLACE
    assert served.refusal.kind == "unreadable"
    assert "never-written" in str(served.refusal)
    assert answered is None


def test_an_undecodable_memory_record_and_flags_record_answer_one_refusal_at_their_doors(
    opened_project: Path, client: TestClient
) -> None:
    """An undecodable memory record and flags record each answer the store's own sentence.

    ``load_project_memory`` lists the report as unread with the ``unreadable`` refusal of the
    store's decode sentence for its key, and the editor's label load answers 400 with the flags
    reader's sentence carrying the store's decode sentence for the flags record's key.
    """
    from tcip_annotation.flags import FlagRequest, PointTarget, flag_key
    from tcip_mcp.dataset_layout import Gestures, image_dir
    from tests._producer_fixtures import (
        image_label_key,
        label_image,
        write_image,
    )

    report = _reported(opened_project)
    image = write_image(image_dir(opened_project, "2026-02-11") / "a.jpg")
    label_image(
        image,
        project=opened_project,
        labels=[{"subject": "bud", "bbox": [40.0, 30.0, 60.0, 50.0]}],
        gestures=Gestures(
            flag=(
                FlagRequest(
                    text="open?",
                    target=PointTarget(point=(50.0, 40.0), subject="bud"),
                ),
            )
        ),
    )
    flags = flag_key(image_label_key(image))
    damage_record(report, b"{not valid json")
    damage_record(flags, b"{not valid json")

    memory = load_project_memory(opened_project, MemoryKind.reports)
    loaded = client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    )

    (unread,) = [e for e in memory["entries"] if e["id"] == report.parts[0]]
    assert unread["reason"]["refusal"] == {
        "kind": "unreadable",
        "message": _decode_sentence(report),
    }
    assert loaded.status_code == 400, loaded.text
    assert loaded.json()["detail"].endswith(f": {_decode_sentence(flags)}")


def test_a_label_document_in_a_database_that_will_not_open_answers_the_editors_400(
    opened_project: Path, client: TestClient, tmp_path: Path
) -> None:
    """The editor's label load answers 400 when the dataset's database is not a database.

    The label document and its flags record live in that one database; the load answers the
    store's sentence rather than escaping.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_store.file_backend import database_file
    from tests._producer_fixtures import label_image, write_image

    dataset = tmp_path / "ds"
    image = write_image(image_dir(dataset, "2026-02-11") / "a.jpg")
    label_image(
        image,
        project=opened_project,
        labels=[{"subject": "bud", "bbox": [40.0, 30.0, 60.0, 50.0]}],
    )
    tcip_store.release_root(dataset)
    database_file(str(dataset)).write_bytes(b"not a database" * 100)

    loaded = client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    )

    assert loaded.status_code == 400, loaded.text
    assert (
        "could not complete this store's identity check"
        in loaded.json()["detail"]
    )


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_a_confirmation_over_an_undecodable_trait_record_answers_the_readers_sentence(
    client: TestClient, opened_project: Path
) -> None:
    """Confirming a trait whose record does not decode answers 400 with the reader's sentence.

    ``confirm_revision`` reads the record inside its transaction; the detail is the sentence
    ``read_trait`` raises outside one, the trait's place before the store's decode sentence.
    """
    from tcip_mcp import traits

    revision = traits.read_trait("bud_opening", opened_project).latest
    key = traits.trait_key(opened_project, "bud_opening")
    damage_record(key, b"{not valid json")

    with pytest.raises(tcip_store.ReadRefusedError) as refused:
        traits.read_trait("bud_opening", opened_project)
    resp = client.post(
        "/api/results/traits/confirm",
        json={
            "trait": "bud_opening",
            "revision": revision.number,
            "entry_sha256": revision.entry_sha256,
            "user": "tester",
            "confirmed": True,
        },
    )

    assert refused.value.served.refusal == tcip_store.Unreadable(
        message=_decode_sentence(key)
    )
    assert "'bud_opening'" in refused.value.served.where
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == str(refused.value)
