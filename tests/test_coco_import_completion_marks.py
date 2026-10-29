"""A COCO import carries a person's completion marks into the label documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_annotation import json_io
from tcip_mcp.dataset_layout import Gestures, image_dir
from tcip_mcp.tools.ingest_tools import import_coco
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import save_labels
from tests._training_values import RATIONALE, VERTEX_SPACING
from tests.test_coco_import import (
    BOX,
    DATE,
    STEMS,
    SUBJECT,
    _dataset,
    _document,
    _label,
    _written,
)


def _reviewed(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Return a target dataset, a COCO path and the source editor's records as COCO.

    The source's first image holds both subjects, with only the first marked; its second is
    marked with no annotations. The third has no document. The marks and annotation records
    come from the label save, and only COCO's image and category identities are added.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _dataset(source_dir).root
    images, annotations = [], []
    for i, stem in enumerate(STEMS[:2], 1):
        save_labels(
            source,
            image_dir(source, DATE) / f"{stem}.png",
            [
                {"subject": subject, "bbox": BOX}
                for subject in (SUBJECT, "leaf")
            ]
            if i == 1
            else [],
            author="user:reviewer",
            gestures=Gestures(complete={SUBJECT: True}, proposals_hidden=True),
        )
        stored = ts.read(_label(source, stem))
        assert isinstance(stored, dict)
        records = stored.pop(json_io.ANNOTATIONS_KEY)
        assert isinstance(records, list)
        images.append({**stored, "id": i, "file_name": f"{stem}.png"})
        for record in records:
            assert isinstance(record, dict)
            subject = record.pop("subject")
            annotations.append(
                {
                    **record,
                    "image_id": i,
                    "category_id": (SUBJECT, "leaf").index(subject),
                }
            )
    path = _document(
        tmp_path / "reviewed.json",
        annotations=annotations,
        categories=[
            {"id": i, "name": name} for i, name in enumerate((SUBJECT, "leaf"))
        ],
    )
    coco = json.loads(path.read_text())
    coco["images"][:2] = images
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    return _dataset(target_dir).root, path, coco


def _import(root: Path, path: Path, coco: dict) -> dict:
    path.write_text(json.dumps(coco), encoding="utf-8")
    return import_coco(
        root,
        str(path),
        str(root),
        DATE,
        VERTEX_SPACING,
        RATIONALE,
        actor="user:importer",
    )


@pytest.mark.parametrize("stated_digest", [True, False])
def test_import_and_editor_agree_on_the_persons_marks(
    tmp_path: Path,
    stated_digest: bool,
) -> None:
    root, path, coco = _reviewed(tmp_path)
    source = tmp_path / "source" / "fruit_count"
    if not stated_digest:
        for record in coco["images"][:2]:
            del record[json_io.COMPLETION_KEY][SUBJECT][0]["digest"]

    result = _import(root, path, coco)

    assert "error" not in result, result
    for stem in STEMS[:2]:
        assert ts.read(_label(root, stem)) == ts.read(_label(source, stem))
    positive = json_io.read_label_document(_label(root, STEMS[0]))
    negative = json_io.read_label_document(_label(root, STEMS[1]))
    assert positive.state(SUBJECT) == "complete"
    assert positive.state("leaf") == "partial"
    assert negative.state(SUBJECT) == "negative"
    assert negative.state("leaf") == "unannotated"
    assert _written(root) == list(STEMS[:2])
    (event,) = audit_rows(root, "coco_document_imported")
    assert event["arguments"]["marked"] == {
        stem: [SUBJECT] for stem in STEMS[:2]
    }
    assert event["actor"] == "user:importer"


@pytest.mark.parametrize(
    "field,value",
    [
        ("by", "tool:reviewer"),
        ("by", "user:"),
        ("by", None),
        ("by", 1),
        ("at", ""),
        ("at", None),
        ("at", 1),
        ("proposals_hidden", None),
        ("proposals_hidden", 1),
        ("proposals_hidden", "false"),
        ("rect", None),
        ("rect", [0, 0, 64]),
        ("rect", [0, 0, "64", 64]),
        ("rect", [False, 0, 64, 64]),
        ("rect", [0, 0, 65, 64]),
        ("rect", [-1, 0, 64, 64]),
        ("rect", [0, 0, 0, 64]),
        ("rect", [0, 0, -1, 64]),
        ("rect", [0, 0, float("inf"), 64]),
        ("rect", [float("nan"), 0, 64, 64]),
        ("digest", "different"),
        ("digest", None),
        ("digest", ""),
        ("digest", 1),
    ],
)
def test_a_malformed_imported_mark_refuses_without_a_write(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root, path, coco = _reviewed(tmp_path)
    coco["images"][1][json_io.COMPLETION_KEY][SUBJECT][0][field] = value
    before = audit_rows(root)

    result = _import(root, path, coco)

    assert "error" in result and field in result["error"], result
    assert _written(root) == []
    assert audit_rows(root) == before


@pytest.mark.parametrize("field", ["rect", "by", "at", "proposals_hidden"])
def test_a_mark_missing_a_required_field_refuses(
    tmp_path: Path, field: str
) -> None:
    root, path, coco = _reviewed(tmp_path)
    del coco["images"][1][json_io.COMPLETION_KEY][SUBJECT][0][field]
    before = audit_rows(root)

    result = _import(root, path, coco)

    assert "error" in result and field in result["error"], result
    assert _written(root) == []
    assert audit_rows(root) == before


def test_mark_faults_and_annotation_faults_are_reported_together(
    tmp_path: Path,
) -> None:
    root, path, coco = _reviewed(tmp_path)
    first = coco["images"][0][json_io.COMPLETION_KEY]
    first["unregistered"] = first.pop(SUBJECT)
    first["unregistered"][0]["by"] = "tool:reviewer"
    coco["images"][1][json_io.COMPLETION_KEY][SUBJECT][0]["rect"] = [
        0,
        0,
        65,
        64,
    ]
    coco["annotations"][0]["bbox"] = [0, 0, 0, 1]
    before = audit_rows(root)

    result = _import(root, path, coco)

    for fault in (
        "unregistered",
        "recorded identity",
        "rect",
        "record 0",
    ):
        assert fault in result["error"], result
    assert _written(root) == []
    assert audit_rows(root) == before


def test_an_import_without_marks_keeps_the_annotation_document(
    tmp_path: Path,
) -> None:
    root, path, coco = _reviewed(tmp_path)
    for image in coco["images"][:2]:
        del image[json_io.COMPLETION_KEY]

    result = _import(root, path, coco)

    assert "error" not in result, result
    expected = ts.read(_label(tmp_path / "source" / "fruit_count", STEMS[0]))
    del expected[json_io.COMPLETION_KEY]
    assert ts.read(_label(root, STEMS[0])) == expected
    assert _written(root) == [STEMS[0]]
    assert result["marked"] == {}


@pytest.mark.parametrize(
    "value", ["not marks", [1], {SUBJECT: {}}, {SUBJECT: [None]}]
)
def test_a_malformed_mark_container_refuses(
    tmp_path: Path, value: object
) -> None:
    root, path, coco = _reviewed(tmp_path)
    coco["images"][1][json_io.COMPLETION_KEY] = value
    before = audit_rows(root)

    result = _import(root, path, coco)

    assert "error" in result and "mark" in result["error"], result
    assert _written(root) == []
    assert audit_rows(root) == before


def test_all_faults_of_one_mark_are_reported(tmp_path: Path) -> None:
    root, path, coco = _reviewed(tmp_path)
    mark = coco["images"][1][json_io.COMPLETION_KEY][SUBJECT][0]
    for field in mark:
        mark[field] = None
    before = audit_rows(root)

    result = _import(root, path, coco)

    for field in mark:
        assert field in result["error"], result
    assert _written(root) == []
    assert audit_rows(root) == before


def test_a_region_mark_finishes_only_the_region_a_person_marked(
    tmp_path: Path,
) -> None:
    root, path, coco = _reviewed(tmp_path)
    source = tmp_path / "source" / "fruit_count"
    image = image_dir(source, DATE) / f"{STEMS[1]}.png"
    save_labels(
        source, image, [], gestures=Gestures(complete={SUBJECT: False})
    )
    save_labels(
        source,
        image,
        [],
        gestures=Gestures(complete={SUBJECT: True}, rect=(0, 0, 32, 64)),
    )
    stored = ts.read(_label(source, STEMS[1]))
    assert isinstance(stored, dict)
    coco["images"][1][json_io.COMPLETION_KEY] = stored[json_io.COMPLETION_KEY]

    result = _import(root, path, coco)

    assert "error" not in result, result
    assert ts.read(_label(root, STEMS[1])) == stored
    assert (
        json_io.read_label_document(_label(root, STEMS[1])).state(SUBJECT)
        == "unannotated"
    )


def test_a_mark_for_an_unregistered_subject_refuses(tmp_path: Path) -> None:
    root, path, coco = _reviewed(tmp_path)
    marks = coco["images"][1][json_io.COMPLETION_KEY]
    marks["unregistered"] = marks.pop(SUBJECT)
    before = audit_rows(root)

    result = _import(root, path, coco)

    assert "completion subject 'unregistered'" in result["error"], result
    assert _written(root) == []
    assert audit_rows(root) == before


def test_a_tool_cannot_make_a_completion_mark_through_the_label_save(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path).root
    before = audit_rows(root)

    with pytest.raises(ValueError, match="person's recorded identity"):
        save_labels(
            root,
            image_dir(root, DATE) / f"{STEMS[0]}.png",
            [],
            author="save_annotations",
            gestures=Gestures(complete={SUBJECT: True}),
        )

    assert _written(root) == []
    assert audit_rows(root) == before
