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
                "catkin": {"description": "the male flower"},
                "Catkin": {"description": "a second spelling a human typed"},
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
    assert set(subjects) == {"catkin", "Catkin", "bush"}
    assert subjects["catkin"]["description"] == "the male flower"
    assert subjects["Catkin"]["description"] == "a second spelling a human typed"
    assert set(json.loads((tmp_path / "classes.json").read_text(encoding="utf-8"))) == {
        "catkin", "Catkin", "bush"}


def test_registry_derived_from_labels_keeps_each_name_exactly_as_labelled(
    client: TestClient, tmp_path: Path
) -> None:
    """With no saved registry, the provisional one lists the names the labels actually carry, each
    unchanged; a record whose subject name is empty names nothing and is left out."""
    labels = tmp_path / "annotations" / "2026-03-02"
    labels.mkdir(parents=True)
    write_annotations(
        str(labels / "IMG_A.json"),
        [
            _box("catkin", 12, 30, 48, 140),
            _box("Catkin", 300, 44, 372, 70),
            _box("bush", 5, 9, 640, 480),
            _box("", 700, 100, 760, 220),
        ],
        900, 500,
    )

    subjects = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "annotations_dir": str(labels)},
    ).json()["subjects"]
    assert set(subjects) == {"catkin", "Catkin", "bush"}


def test_a_new_subject_is_addable_alongside_the_saved_ones(
    client: TestClient, tmp_path: Path
) -> None:
    """Authoring a subject stays open: a later save adds the new name and updates the existing one
    in place, leaving one entry per name rather than a duplicate."""
    first = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"catkin": {"description": "first pass"}}, "version": None},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"catkin": {"description": "corrected"},
                           "hazel_leaf": {"description": "one leaf blade"}},
              "version": first.json()["version"]},
    )
    assert second.status_code == 200, second.text
    assert second.json()["n_subjects"] == 2

    subjects = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path)},
    ).json()["subjects"]
    assert set(subjects) == {"catkin", "hazel_leaf"}
    assert subjects["catkin"]["description"] == "corrected"
    assert len(json.loads((tmp_path / "classes.json").read_text(encoding="utf-8"))) == 2
