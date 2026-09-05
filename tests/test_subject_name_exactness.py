"""Subject names in the class registry are exact and are never normalized or folded together.

A label references its subject by name, so two names that differ at all name two subjects: the
registry keeps them apart rather than merging them, a record carrying no usable name contributes
nothing to a registry derived from labels, and a new name stays addable alongside the existing ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _box(subject: str, x1: float, y1: float, x2: float, y2: float) -> Annotation:
    return Annotation(subject=subject, geometry=BBox(x1, y1, x2, y2))


def test_subject_names_differing_only_by_case_stay_distinct(
    client: TestClient, tmp_path: Path
) -> None:
    """Nothing case-folds a subject name on the way in, so a name typed a second way is stored as
    its own subject with its own description rather than silently absorbed into the first."""
    save = client.post(
        "/api/classes/save",
        json={
            "project_root": str(tmp_path),
            "dataset_root": str(tmp_path),
            "subjects": {
                "bud": {"description": "the male flower"},
                "Bud": {"description": "a second spelling a human typed"},
                "bush": {"description": "one plant crown"},
            },
            "version": None,
        },
    )
    assert save.status_code == 200, save.text
    assert save.json()["n_subjects"] == 3

    subjects = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path)},
    ).json()["subjects"]
    assert set(subjects) == {"bud", "Bud", "bush"}
    assert subjects["bud"]["description"] == "the male flower"
    assert subjects["Bud"]["description"] == "a second spelling a human typed"
    assert set(json.loads((tmp_path / "classes.json").read_text(encoding="utf-8"))) == {
        "bud", "Bud", "bush"}


def test_registry_derived_from_labels_keeps_each_name_exactly_as_labelled(
    client: TestClient, tmp_path: Path
) -> None:
    """With no saved registry, the draft one lists the names a readable label document
    actually carries, each unchanged."""
    labels = tmp_path / "annotations" / "2026-03-02"
    labels.mkdir(parents=True)
    write_annotations(
        str(labels / "IMG_A.json"),
        [
            _box("bud", 12, 30, 48, 140),
            _box("Bud", 300, 44, 372, 70),
            _box("bush", 5, 9, 640, 480),
        ],
        900, 500,
    )

    body = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "annotations_dir": str(labels)},
    ).json()
    assert set(body["subjects"]) == {"bud", "Bud", "bush"}
    assert body["unreadable"] == []


def test_registry_derivation_reports_a_document_it_cannot_read(
    client: TestClient, tmp_path: Path
) -> None:
    """A record whose subject name is empty makes its own document unreadable: the draft
    registry still derives from the readable documents, and names the unreadable one by path
    rather than silently deriving nothing from it."""
    labels = tmp_path / "annotations" / "2026-03-02"
    labels.mkdir(parents=True)
    write_annotations(str(labels / "IMG_A.json"), [_box("bud", 12, 30, 48, 140)], 900, 500)
    write_annotations(str(labels / "IMG_B.json"), [_box("", 700, 100, 760, 220)], 900, 500)

    body = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "annotations_dir": str(labels)},
    ).json()
    assert set(body["subjects"]) == {"bud"}
    assert body["unreadable"] == [str(labels / "IMG_B.json")]


def test_a_new_subject_is_addable_alongside_the_saved_ones(
    client: TestClient, tmp_path: Path
) -> None:
    """Authoring a subject stays open: a later save adds the new name and updates the existing one
    in place, leaving one entry per name rather than a duplicate."""
    first = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"bud": {"description": "first pass"}}, "version": None},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"bud": {"description": "corrected"},
                           "hazel_leaf": {"description": "one leaf blade"}},
              "version": first.json()["version"]},
    )
    assert second.status_code == 200, second.text
    assert second.json()["n_subjects"] == 2

    subjects = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path)},
    ).json()["subjects"]
    assert set(subjects) == {"bud", "hazel_leaf"}
    assert subjects["bud"]["description"] == "corrected"
    assert len(json.loads((tmp_path / "classes.json").read_text(encoding="utf-8"))) == 2
