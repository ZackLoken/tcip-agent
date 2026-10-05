"""Subject names in the subject registry are exact and are never normalized or folded together.

A label references its subject by name, so two names that differ at all name two subjects: the
registry keeps them apart rather than merging them, a record carrying no usable name contributes
nothing to a registry derived from labels, and a new name stays addable alongside the existing ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests._producer_fixtures import box_annotation, image_label_key, label_image

DATE = "2026-03-02"


def test_subject_names_differing_only_by_case_stay_distinct(
    client: TestClient, tmp_path: Path
) -> None:
    """Nothing case-folds a subject name on the way in, so a name typed a second way is stored as
    its own subject with its own description rather than silently absorbed into the first."""
    save = client.post(
        "/api/subjects/save",
        json={
            "dataset_root": str(tmp_path),
            "subjects": {
                "bud": {"description": "the first spelling a human typed"},
                "Bud": {"description": "a second spelling a human typed"},
                "bush": {"description": "one plant crown"},
            },
            "version": None,
            "user": "breeder",
        },
    )
    assert save.status_code == 200, save.text
    assert save.json()["n_subjects"] == 3

    subjects = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()["subjects"]
    assert set(subjects) == {"bud", "Bud", "bush"}
    assert subjects["bud"]["description"] == "the first spelling a human typed"
    assert subjects["Bud"]["description"] == "a second spelling a human typed"
    assert set(json.loads((tmp_path / "subjects.json").read_text(encoding="utf-8"))) == {
        "bud", "Bud", "bush"}


def test_registry_derived_from_labels_keeps_each_name_exactly_as_labeled(
    client: TestClient, tmp_path: Path
) -> None:
    """With no saved registry, the discovered names are the ones a readable label document
    actually carries, each unchanged."""
    label_image(
        tmp_path / "images" / DATE / "IMG_A.jpg",
        [
            box_annotation(12, 30, 48, 140),
            box_annotation(300, 44, 372, 70, subject="Bud"),
            box_annotation(5, 9, 640, 480, subject="bush"),
        ],
        900, 500,
    )

    body = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()
    assert body["subjects"] is None
    assert set(body["discovered"]) == {"bud", "Bud", "bush"}
    assert body["unreadable"] == []


def test_registry_derivation_reports_a_document_it_cannot_read(
    client: TestClient, tmp_path: Path
) -> None:
    """A record whose subject name is empty makes its own document unreadable: the draft
    registry still derives from the readable documents, and names the unreadable one by stem
    rather than silently deriving nothing from it."""
    from tcip_store import encode_record

    from tests._record_damage_fixtures import damage_record

    images = tmp_path / "images" / DATE
    label_image(images / "IMG_A.jpg", [box_annotation(12, 30, 48, 140)], 900, 500)
    label_image(images / "IMG_B.jpg", [box_annotation(700, 100, 760, 220)], 900, 500)
    # No platform producer can make a record without a subject; a damaged record can carry one.
    damage_record(image_label_key(images / "IMG_B.jpg"), encode_record({
        "image": "IMG_B", "width": 900, "height": 500,
        "annotations": [{"subject": "", "bbox": [700, 100, 60, 120]}]}))

    body = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()
    assert body["discovered"] == ["bud"]
    assert body["unreadable"] == ["IMG_B"]


def test_a_new_subject_is_addable_alongside_the_saved_ones(
    client: TestClient, tmp_path: Path
) -> None:
    """Authoring a subject stays open: a later save adds the new name and updates the existing one
    in place, leaving one entry per name rather than a duplicate."""
    first = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"description": "first pass"}}, "version": None,
              "user": "breeder"},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"description": "corrected"},
                           "hazel_leaf": {"description": "one leaf blade"}},
              "version": first.json()["version"], "user": "breeder"},
    )
    assert second.status_code == 200, second.text
    assert second.json()["n_subjects"] == 2

    subjects = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()["subjects"]
    assert set(subjects) == {"bud", "hazel_leaf"}
    assert subjects["bud"]["description"] == "corrected"
    assert len(json.loads((tmp_path / "subjects.json").read_text(encoding="utf-8"))) == 2
