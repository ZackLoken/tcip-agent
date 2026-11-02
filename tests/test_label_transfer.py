"""A dataset's labels leave it through ``export_labels`` and come back through ``import_labels``.

The labels are built in one project through the platform's own producers (the label save with
its gestures, a staged proposal accepted, flags raised and resolved, a person's negative) and
exported; a second project holding the same photographs and the same registry imports them, and
its own export equals the first record for record, each stored record byte for byte. Every fault
the import finds refuses it whole, nothing written and no line left, and the COCO door and the
label import leave one audit line of one shape, the one writer's.
"""

from __future__ import annotations

import zipfile
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_annotation.flags import (
    REVIEW_FLAGS_STORE,
    FlagRequest,
    PointTarget,
    flag_key,
    read_flags,
)
from tcip_annotation.json_io import read_document_versioned
from tcip_mcp.audit import acts_of
from tcip_mcp.dataset_layout import (
    LABEL_DOCUMENTS,
    LABELS_IMPORTED,
    Decisions,
    Gestures,
    image_dir,
    label_key,
)
from tcip_mcp.tools.ingest_tools import (
    LABELS_MEMBER,
    LabelExport,
    export_labels,
    import_labels,
)
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import label_image, staged_box

if TYPE_CHECKING:
    from pathlib import Path

FIRST, SECOND = "2025-09-14", "2025-09-21"
SIZE = 64
CAPTURES = {
    FIRST: ("boxes", "rings", "accepted"),
    SECOND: ("flagged", "negative", "untouched"),
}
"""Each capture's photographs by stem."""
SUBJECTS = ("bud", "leaf")
BREEDER = "user:breeder"
OPENING = {"type": "categorical", "values": ["closed", "open"]}
REGISTRY = {
    "bud": {"description": "one bud", "attributes": {"opening": OPENING}},
    "leaf": {"description": "one leaf"},
}
"""The subject registry both projects write."""
SHORT_OF_WHOLE = (0.0, 0.0, SIZE - 0.0001, float(SIZE))
"""A completion rect a ten-thousandth of a pixel short of the whole frame's width."""


def _photographs(where: Path, sizes: dict[str, int] | None = None) -> Path:
    """Write every photograph of :data:`CAPTURES` into ``where``, each dated its capture.

    Each is a :data:`SIZE` px square PNG, ``sizes`` naming a stem drawn at another size.
    Returns ``where``.
    """
    from PIL import Image

    from tests._image_fixtures import capture_exif

    where.mkdir()
    for i, (capture, stems) in enumerate(CAPTURES.items()):
        for j, stem in enumerate(stems):
            side = (sizes or {}).get(stem, SIZE)
            Image.new("RGB", (side, side), (40 * i, 30 * j, 90)).save(
                where / f"{stem}.png",
                exif=capture_exif(datetime.fromisoformat(capture)),
            )
    return where


def _project(path: Path, photographs: Path, registry: dict = REGISTRY) -> Path:
    """Make ``path`` a registered project holding ``photographs`` and ``registry``.

    The project is made by ``initialize_project``, the photographs brought in by
    ``ingest_images``, the registry written by ``write_subject_registry`` and the dataset
    registered by ``register_dataset``. Returns the project root.
    """
    from tcip_mcp.tools.annotation_tools import write_subject_registry
    from tcip_mcp.tools.ingest_tools import IngestManifest, ingest_images
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._web_fixtures import named_project

    root = named_project(path, path.name).root
    ingested = ingest_images(root, source=str(photographs), actor=None)
    assert isinstance(ingested, IngestManifest), ingested
    for answer in (
        write_subject_registry(root, str(root), subjects=registry, actor=None),
        register_dataset(
            root, str(root), crop=sorted(registered_crops())[0], actor=None
        ),
    ):
        assert "error" not in answer, answer
    return root


def _image(root: Path, stem: str) -> Path:
    """Return the path of the photograph ``stem`` in ``root``'s capture of it."""
    (capture,) = [c for c, stems in CAPTURES.items() if stem in stems]
    return image_dir(root, capture) / f"{stem}.png"


def _label(root: Path) -> None:
    """Label ``root``'s photographs through the label save and its gestures, as a breeder would.

    ``boxes`` holds an attributed bud and a leaf, its buds marked complete; ``rings`` a leaf
    polygon, its leaves marked complete and its buds marked complete over all but a fraction of a
    pixel's width (:data:`SHORT_OF_WHOLE`); ``accepted`` a staged proposal accepted; ``flagged``
    a bud and a crowd region with two flags raised and one resolved; ``negative`` no bud, marked
    complete; ``untouched`` nothing.
    """
    label_image(
        _image(root, "boxes"),
        [
            {
                "subject": "bud",
                "bbox": [4, 4, 20, 20],
                "attributes": {"opening": "open"},
            },
            {"subject": "leaf", "bbox": [30, 30, 50, 50]},
        ],
        project=root,
        gestures=Gestures(complete={"bud": True}),
    )
    ring = [
        {
            "subject": "leaf",
            "rings": [[[10, 10], [40, 10], [40, 40], [10, 40]]],
        }
    ]
    label_image(
        _image(root, "rings"),
        ring,
        project=root,
        gestures=Gestures(complete={"leaf": True}),
    )
    label_image(
        _image(root, "rings"),
        ring,
        project=root,
        gestures=Gestures(complete={"bud": True}, rect=SHORT_OF_WHOLE),
    )
    accepted = _image(root, "accepted")
    bucket = staged_box(root, accepted, [8.0, 8.0, 24.0, 24.0])
    label_image(
        accepted,
        [],
        project=root,
        gestures=Gestures(
            decide=Decisions(bucket=bucket, verdicts={0: "accepted"})
        ),
    )
    flagged = _image(root, "flagged")
    held = [
        {"subject": "bud", "bbox": [2, 2, 14, 14]},
        {"subject": "bud", "bbox": [30, 30, 60, 60], "iscrowd": True},
    ]
    label_image(
        flagged,
        held,
        project=root,
        gestures=Gestures(
            flag=(
                FlagRequest(
                    text="is this one bud",
                    target=PointTarget(point=(8.0, 8.0), subject="bud"),
                ),
                FlagRequest(text="the crowd's edge", target=None),
            )
        ),
    )
    first, _second = read_flags(flag_key(label_key(root, SECOND, "flagged")))
    label_image(
        flagged,
        held,
        project=root,
        gestures=Gestures(resolve={first.id: "one bud"}),
    )
    label_image(
        _image(root, "negative"),
        [],
        project=root,
        gestures=Gestures(complete={"bud": True}),
    )


def _export(root: Path, destination: Path) -> LabelExport:
    """Export ``root``'s labels to ``destination`` and return the container's record."""
    answer = export_labels(root, str(root), str(destination), actor=BREEDER)
    assert "error" not in answer, answer
    with zipfile.ZipFile(destination) as container:
        return LabelExport.model_validate_json(container.read(LABELS_MEMBER))


def _stored(root: Path) -> dict[tuple[str, ...], str]:
    """Return each label and flags record ``root`` holds, by store and parts, at its version."""
    return {
        (store, *key.parts): ts.read_versioned(key).version.token
        for store in (LABEL_DOCUMENTS, REVIEW_FLAGS_STORE)
        for key in ts.keys(store, str(root))
    }


def _states(root: Path) -> dict[tuple[str, str], str]:
    """Return each photograph's state for each subject, by stem and subject."""
    return {
        (stem, subject): read_document_versioned(
            label_key(root, capture, stem)
        )[0].state(subject)
        for capture, stems in CAPTURES.items()
        for stem in stems
        for subject in SUBJECTS
    }


@pytest.fixture
def exported(tmp_path: Path) -> tuple[Path, Path]:
    """Return a labeled project (:func:`_label`) and the container its labels were exported to."""
    root = _project(tmp_path / "first", _photographs(tmp_path / "raw"))
    _label(root)
    container = tmp_path / "labels.zip"
    _export(root, container)
    return root, container


def test_labels_imported_into_a_new_project_export_as_they_left(
    tmp_path: Path, exported: tuple[Path, Path]
) -> None:
    """A second project importing the labels exports them equal, record for record.

    Every image's state for every subject is the first project's, a negative reads negative,
    each stored label and flags record holds the first project's bytes, the accepted proposal
    keeps its producer and its acceptance, and the resolved flag keeps its resolution.
    """
    first, container = exported
    second = _project(tmp_path / "second", tmp_path / "raw")

    imported = import_labels(
        second, str(container), str(second), actor=BREEDER
    )

    assert "error" not in imported, imported
    assert imported["written"] == {
        FIRST: ["accepted", "boxes", "rings"],
        SECOND: ["flagged", "negative"],
    }
    left = _export(first, tmp_path / "first_again.zip")
    back = _export(second, tmp_path / "second.zip")
    assert back.model_dump_json(include={"images"}) == left.model_dump_json(
        include={"images"}
    )
    assert back.subjects == left.subjects
    assert _stored(second) == _stored(first)
    assert _states(second) == _states(first)
    assert _states(second)[("negative", "bud")] == "negative"
    assert _states(second)[("rings", "bud")] == "unannotated"
    (accepted,) = read_document_versioned(
        label_key(second, FIRST, "accepted")
    )[0].annotations
    assert (
        accepted.created_by is not None
        and not accepted.created_by.startswith("user:")
    )
    assert accepted.accepted_by == BREEDER
    assert read_flags(flag_key(label_key(second, SECOND, "flagged"))) == (
        read_flags(flag_key(label_key(first, SECOND, "flagged")))
    )
    (line,) = audit_rows(second, LABELS_IMPORTED)
    assert line["arguments"]["source"] == str(container.resolve())
    assert line["arguments"]["dataset"] == left.dataset.model_dump(mode="json")
    assert (
        line["source_digest"]
        == ts.read_blob_versioned(container).version.token
    )


def test_two_exports_of_one_unchanged_dataset_are_equal(
    tmp_path: Path, exported: tuple[Path, Path]
) -> None:
    """Two exports of one unchanged dataset are equal byte for byte, each leaving its line.

    The container's member states the zip format's fixed default instant, never the clock's.
    """
    first, container = exported
    again = tmp_path / "again.zip"
    _export(first, again)

    assert again.read_bytes() == container.read_bytes()
    with zipfile.ZipFile(container) as held:
        stated = held.getinfo(LABELS_MEMBER).date_time
    assert stated == zipfile.ZipInfo().date_time
    lines, _cursor = acts_of(first, ("export_labels",))
    assert [line.arguments["destination"] for line in lines] == [
        str(container),
        str(again),
    ]


def _refused(second: Path, container: Path, *named: str) -> None:
    """Import ``container`` into ``second`` and check it refuses naming ``named``, writing nothing.

    Nothing is written: the label and flags records ``second`` holds are those it held before,
    at their versions, and no import line is left.
    """
    before = _stored(second)

    answer = import_labels(second, str(container), str(second), actor=BREEDER)

    assert "error" in answer, answer
    for fault in named:
        assert fault in answer["error"], (fault, answer["error"])
    assert _stored(second) == before
    assert audit_rows(second, LABELS_IMPORTED) == []


def test_an_image_the_new_capture_does_not_hold_refuses_the_whole_import(
    tmp_path: Path, exported: tuple[Path, Path]
) -> None:
    _first, container = exported
    raw = tmp_path / "raw"
    (raw / "negative.png").unlink()

    _refused(
        _project(tmp_path / "second", raw),
        container,
        f"{SECOND}/negative.png names no image",
        "ingest_images",
    )


def test_a_frame_the_new_image_does_not_have_refuses_the_whole_import(
    tmp_path: Path, exported: tuple[Path, Path]
) -> None:
    _first, container = exported
    other = _photographs(tmp_path / "other", sizes={"rings": SIZE // 2})

    _refused(
        _project(tmp_path / "second", other),
        container,
        f"{FIRST}/rings.png is labeled at ({SIZE}, {SIZE})",
    )


def test_a_subject_the_new_registry_does_not_declare_refuses_the_whole_import(
    tmp_path: Path, exported: tuple[Path, Path]
) -> None:
    _first, container = exported

    _refused(
        _project(
            tmp_path / "second",
            tmp_path / "raw",
            registry={"bud": REGISTRY["bud"]},
        ),
        container,
        "the subject 'leaf'",
        "does not declare; declare it with write_subject_registry",
    )


def test_a_registry_declaring_the_subject_otherwise_refuses_the_whole_import(
    tmp_path: Path, exported: tuple[Path, Path]
) -> None:
    _first, container = exported
    otherwise = {
        **REGISTRY,
        "bud": {
            "description": "one bud",
            "attributes": {"opening": {**OPENING, "type": "ordinal"}},
        },
    }

    _refused(
        _project(tmp_path / "second", tmp_path / "raw", registry=otherwise),
        container,
        "the subject 'bud'",
        "declares otherwise",
    )


def test_a_label_already_present_refuses_the_whole_import_and_is_kept(
    tmp_path: Path, exported: tuple[Path, Path]
) -> None:
    _first, container = exported
    second = _project(tmp_path / "second", tmp_path / "raw")
    label_image(
        _image(second, "rings"),
        [{"subject": "leaf", "bbox": [1, 1, 9, 9]}],
        project=second,
    )

    _refused(
        second,
        container,
        f"already holds label records for ['{FIRST}/rings']",
    )


def test_an_absent_image_and_a_present_label_are_refused_together(
    tmp_path: Path, exported: tuple[Path, Path]
) -> None:
    """The one refusal names the admission's faults beside the faults found in the transaction."""
    _first, container = exported
    raw = tmp_path / "raw"
    (raw / "negative.png").unlink()
    second = _project(tmp_path / "second", raw)
    label_image(
        _image(second, "rings"),
        [{"subject": "leaf", "bbox": [1, 1, 9, 9]}],
        project=second,
    )

    _refused(
        second,
        container,
        f"{SECOND}/negative.png names no image",
        f"already holds label records for ['{FIRST}/rings']",
    )


def test_a_label_document_whose_image_is_gone_refuses_the_export(
    tmp_path: Path,
) -> None:
    """A document no image of its capture names refuses the export naming it, writing nothing."""
    first = _project(tmp_path / "first", _photographs(tmp_path / "raw"))
    _label(first)
    _image(first, "rings").unlink()
    destination = tmp_path / "labels.zip"

    answer = export_labels(first, str(first), str(destination), actor=BREEDER)

    assert f"{FIRST}/rings is a label document of no image" in answer.get(
        "error", ""
    ), answer
    assert not destination.exists()


def test_the_coco_door_decides_no_image_fact_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The COCO door hands every listed image to the one writer and refuses none of them itself.

    With the writer replaced by a recorder, an image the capture lacks, an image already holding
    a label document and an image stated at a frame it does not have are all handed over under
    their file names, the door's own faults name none of them, and nothing reaches the store.
    """
    import json

    from tcip_mcp import dataset_layout
    from tcip_mcp.tools.ingest_tools import import_coco
    from tests._training_values import RATIONALE, VERTEX_SPACING

    root = _project(tmp_path / "project", _photographs(tmp_path / "raw"))
    label_image(
        _image(root, "boxes"),
        [{"subject": "leaf", "bbox": [1, 1, 9, 9]}],
        project=root,
    )
    before = _stored(root)
    handed: list[tuple[dict, list[str]]] = []

    def recorded(
        dataset_root: Path, labels: dict, **kwargs: list[str]
    ) -> dict:
        handed.append((labels, kwargs["faults"]))
        return {"written": {}}

    monkeypatch.setattr(dataset_layout, "write_label_records", recorded)
    coco = tmp_path / "external.json"
    listed = [
        ("boxes.png", SIZE),
        ("rings.png", SIZE // 2),
        ("missing.png", SIZE),
    ]
    coco.write_text(
        json.dumps(
            {
                "images": [
                    {"id": i, "file_name": name, "width": side, "height": side}
                    for i, (name, side) in enumerate(listed, start=1)
                ],
                "annotations": [
                    {
                        "id": i,
                        "image_id": i,
                        "category_id": 1,
                        "bbox": [1, 1, 9, 9],
                    }
                    for i in range(1, len(listed) + 1)
                ],
                "categories": [{"id": 1, "name": "leaf"}],
            }
        ),
        encoding="utf-8",
    )

    answer = import_coco(
        root,
        str(coco),
        str(root),
        FIRST,
        VERTEX_SPACING,
        RATIONALE,
        actor=BREEDER,
    )

    assert answer == {"written": {}}
    ((labels, faults),) = handed
    assert sorted(labels[FIRST]) == ["boxes.png", "missing.png", "rings.png"]
    assert faults == []
    assert _stored(root) == before


def test_the_coco_door_and_the_label_import_leave_one_audit_shape(
    tmp_path: Path, exported: tuple[Path, Path]
) -> None:
    """The COCO door's line and the label import's are the one writer's, read back alike.

    Each is a ``labels_imported`` line through ``acts_of`` naming its ``source`` and what it
    ``written``, its digest its one extra fact.
    """
    import json

    from tcip_mcp.tools.ingest_tools import import_coco
    from tests._training_values import RATIONALE, VERTEX_SPACING

    _first, container = exported
    second = _project(tmp_path / "second", tmp_path / "raw")
    coco = tmp_path / "external.json"
    coco.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "untouched.png",
                        "width": SIZE,
                        "height": SIZE,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 1, 9, 9],
                    }
                ],
                "categories": [{"id": 1, "name": "leaf"}],
            }
        ),
        encoding="utf-8",
    )
    coco_answer = import_coco(
        second,
        str(coco),
        str(second),
        SECOND,
        VERTEX_SPACING,
        RATIONALE,
        actor=BREEDER,
    )
    assert "error" not in coco_answer, coco_answer
    assert "error" not in import_labels(
        second, str(container), str(second), actor=BREEDER
    )

    lines, _cursor = acts_of(second, (LABELS_IMPORTED,))

    assert [line.arguments["source"] for line in lines] == [
        str(coco.resolve()),
        str(container.resolve()),
    ]
    for line in lines:
        assert {"source", "written"} <= set(line.arguments), line
        assert set(line.model_extra or {}) == {"source_digest"}, line
