"""Digest stamping by the image-status routes, read back by the training-side quarantine reader.

A confirmation is stamped at write time with the subject's attribute-schema digest then in effect,
and ``confirmed_negative_names`` excludes a confirmation whose stamp no longer matches the subject's
current schema. Both halves run through their real implementations here, the routes doing the
writing and the dataset reader the reading, so a regression in the stamping shows up as a
confirmation that is wrongly admitted or wrongly quarantined.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import write_annotations
from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names
from tcip_web.app import app

BUD_TWO_STATES = {"opening": {"type": "categorical", "values": ["closed", "open"]}}
BUD_THREE_STATES = {
    "opening": {"type": "categorical", "values": ["closed", "elongating", "open"]}
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A dataset root whose label files are present but empty, ready to be confirmed negative."""
    labels = tmp_path / "annotations"
    labels.mkdir()
    for stem in ("img_alpha", "img_beta", "img_gamma", "img_one", "img_two"):
        write_annotations(str(labels / f"{stem}.json"), [], 800, 450, keep_empty=True)
    return tmp_path


def _save_registry(client: TestClient, root: Path, bud_attributes: dict) -> None:
    """Write the dataset registry through the real save route, carrying forward the version the
    route last reported: bud carries an attribute vocabulary, bush is detection-only, so the
    two subjects have different schemas."""
    version = client.get(
        "/api/classes/load",
        params={"project_root": str(root), "dataset_root": str(root)},
    ).json()["version"]
    resp = client.post(
        "/api/classes/save",
        json={
            "project_root": str(root),
            "dataset_root": str(root),
            "subjects": {
                "bud": {"description": "the male flower", "attributes": bud_attributes},
                "bush": {"description": "one plant crown"},
            },
            "version": version,
        },
    )
    assert resp.status_code == 200, resp.text


def _confirm_negative(client: TestClient, root: Path, image_name: str, subject: str) -> None:
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(root), "dataset_root": str(root), "image_name": image_name,
              "status": "negative", "subject": subject},
    )
    assert resp.status_code == 200, resp.text


def test_a_schema_change_quarantines_only_the_subject_it_touched(
    client: TestClient, dataset: Path
) -> None:
    """Growing bud's attribute vocabulary invalidates bud's own confirmations and leaves
    bush's alone: the stamp is per subject, so an unrelated subject keeps its human's work."""
    _save_registry(client, dataset, BUD_TWO_STATES)
    _confirm_negative(client, dataset, "img_alpha.jpg", "bud")
    _confirm_negative(client, dataset, "img_beta.jpg", "bush")

    _save_registry(client, dataset, BUD_THREE_STATES)

    labels = dataset / "annotations"
    bud_quarantined: set[str] = set()
    bud_admitted = confirmed_negative_names(
        labels, subject="bud", date=None, quarantined_out=bud_quarantined)
    bush_quarantined: set[str] = set()
    bush_admitted = confirmed_negative_names(
        labels, subject="bush", date=None, quarantined_out=bush_quarantined)

    assert bud_admitted == set()
    assert bud_quarantined == {"img_alpha.jpg"}
    assert bush_admitted == {"img_beta.jpg"}
    assert bush_quarantined == set()


def test_a_later_confirmation_in_the_same_bucket_does_not_revive_a_stale_one(
    client: TestClient, dataset: Path
) -> None:
    """One bucket holds every image touched under a subject, so a fresh confirmation written into
    it must stamp only its own image and leave the older, never-re-reviewed one quarantined."""
    _save_registry(client, dataset, BUD_TWO_STATES)
    _confirm_negative(client, dataset, "img_alpha.jpg", "bud")

    _save_registry(client, dataset, BUD_THREE_STATES)
    _confirm_negative(client, dataset, "img_gamma.jpg", "bud")

    quarantined: set[str] = set()
    admitted = confirmed_negative_names(
        dataset / "annotations", subject="bud", date=None, quarantined_out=quarantined)
    assert admitted == {"img_gamma.jpg"}
    assert quarantined == {"img_alpha.jpg"}


def test_a_bulk_write_stamps_only_the_statuses_it_applied(
    client: TestClient, dataset: Path
) -> None:
    """A name whose status the bulk route skipped was never written, so it keeps the stamp of the
    confirmation that is still standing rather than being re-dated by a write it took no part in."""
    _save_registry(client, dataset, BUD_TWO_STATES)
    first = client.post(
        "/api/classes/image_status/bulk",
        json={"project_root": str(dataset), "dataset_root": str(dataset), "subject": "bud",
              "statuses": {"img_one.jpg": "negative"}},
    )
    assert first.status_code == 200, first.text

    _save_registry(client, dataset, BUD_THREE_STATES)
    second = client.post(
        "/api/classes/image_status/bulk",
        json={"project_root": str(dataset), "dataset_root": str(dataset), "subject": "bud",
              "statuses": {"img_two.jpg": "negative", "img_one.jpg": "not_a_status"}},
    )
    assert second.status_code == 200, second.text

    quarantined: set[str] = set()
    admitted = confirmed_negative_names(
        dataset / "annotations", subject="bud", date=None, quarantined_out=quarantined)
    assert admitted == {"img_two.jpg"}
    assert quarantined == {"img_one.jpg"}


def test_a_confirmation_made_with_no_registry_present_is_still_admitted(
    client: TestClient, tmp_path: Path
) -> None:
    """A dataset with no ``classes.json`` yet has nothing to stamp against; its confirmations are
    admitted, never punished for a stamp the writer could not produce."""
    labels = tmp_path / "annotations"
    labels.mkdir()
    write_annotations(str(labels / "img_solo.json"), [], 800, 450, keep_empty=True)

    _confirm_negative(client, tmp_path, "img_solo.jpg", "bud")

    quarantined: set[str] = set()
    admitted = confirmed_negative_names(labels, subject="bud", date=None, quarantined_out=quarantined)
    assert admitted == {"img_solo.jpg"}
    assert quarantined == set()
