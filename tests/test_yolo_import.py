"""YOLO labels enter through the COCO conversion and its one document transaction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_annotation.json_io import canonical_digest
from tcip_annotation.state import BBox
from tests._audit_fixtures import audit_rows
from tests.test_coco_import import (
    BOX,
    DATE,
    IMG,
    RING,
    SUBJECT,
    _dataset,
    _document,
    _import as _import_coco,
    _label,
    _label_placed_before_the_writes,
    _person_label,
    _written,
)

if TYPE_CHECKING:
    from tcip_store import Versioned

BOX_LINE = "0 0.28125 0.28125 0.3125 0.25"
RING_LINE = "0 " + " ".join(str(value / IMG) for value in RING)


def _labels(folder: Path, files: dict[str, str]) -> Path:
    """Write external YOLO text files into ``folder`` and return it."""
    folder.mkdir()
    for stem, content in files.items():
        (folder / f"{stem}.txt").write_text(content, encoding="utf-8")
    return folder


def _import(folder: Path, root: Path, names: list[str] | None = None) -> dict:
    from tcip_mcp.tools.ingest_tools import import_yolo

    return import_yolo(
        root,
        str(folder),
        [SUBJECT, "leaf"] if names is None else names,
        str(root),
        DATE,
        actor="user:breeder",
    )


def test_boxes_and_a_polygon_write_the_same_documents_as_coco(
    tmp_path: Path,
) -> None:
    roots = []
    for route in ("yolo", "coco"):
        place = tmp_path / route
        place.mkdir()
        roots.append(_dataset(place).root)
    yolo, coco = roots
    folder = _labels(
        tmp_path / "labels",
        {
            "tree_01": BOX_LINE + "\n1 0.5 0.5 0.25 0.5\n",
            "tree_02": RING_LINE,
        },
    )
    result = _import(folder, yolo)
    assert result == {
        "labels_dir": str(folder),
        "names": [SUBJECT, "leaf"],
        "capture": DATE,
        "empty": [],
        "written": ["tree_01", "tree_02"],
    }
    document = _document(
        tmp_path / "external.json",
        categories=[{"id": 0, "name": SUBJECT}, {"id": 1, "name": "leaf"}],
        annotations=[
            {"image_id": 1, "category_id": 0, "bbox": BOX},
            {"image_id": 1, "category_id": 1, "bbox": [24, 16, 16, 32]},
            {"image_id": 2, "category_id": 0, "segmentation": [RING]},
        ],
    )
    assert "error" not in _import_coco(document, coco)
    for stem in result["written"]:
        assert ts.read(_label(yolo, stem)) == ts.read(_label(coco, stem))
    assert _written(yolo) == _written(coco) == ["tree_01", "tree_02"]


@pytest.mark.parametrize("content", ["", "\n \t\n"])
def test_empty_files_write_nothing_and_are_listed(
    tmp_path: Path, content: str
) -> None:
    root = _dataset(tmp_path).root
    folder = _labels(tmp_path / "labels", {"tree_01": content})
    before = audit_rows(root)

    result = _import(folder, root)

    assert result["written"] == []
    assert result["empty"] == ["tree_01"]
    assert _written(root) == []
    assert audit_rows(root) == before


@pytest.mark.parametrize(
    "line, fault",
    [
        ("0", "expected a box or a polygon"),
        ("0 0.1 0.2 0.3", "expected a box or a polygon"),
        ("0 0.1 0.2 0.3 0.4 0.5", "expected a box or a polygon"),
        ("0 0 0 1 0 1 1 0", "expected a box or a polygon"),
        ("x 0.5 0.5 0.2 0.2", "not an integer"),
        ("0.5 0.5 0.5 0.2 0.2", "not an integer"),
        ("-1 0.5 0.5 0.2 0.2", "not covered by names"),
        ("2 0.5 0.5 0.2 0.2", "not covered by names"),
        ("0 nope 0.5 0.2 0.2", "coordinates must be numbers"),
        ("0 nan 0.5 0.2 0.2", "finite and in [0, 1]"),
        ("0 0.5 inf 0.2 0.2", "finite and in [0, 1]"),
        ("0 -0.001 0.5 0.2 0.2", "finite and in [0, 1]"),
        ("0 0.5 0.5 1.001 0.2", "finite and in [0, 1]"),
        ("0 0 0 1 0 1 1.001", "finite and in [0, 1]"),
        ("0 0.5 0.5 0 0.2", "width and height must be positive"),
        ("0 0.5 0.5 0.2 0", "width and height must be positive"),
        ("0 0.5 0.5 -1e-10 0.2", "width and height must be positive"),
        ("0 0.5 0.5 0.2 -1e-10", "width and height must be positive"),
    ],
)
def test_a_bad_line_refuses_with_its_file_and_line_and_no_writes(
    tmp_path: Path, line: str, fault: str
) -> None:
    root = _dataset(tmp_path).root
    folder = _labels(
        tmp_path / "labels",
        {"tree_01": BOX_LINE, "tree_02": "\n" + BOX_LINE + "\n" + line},
    )
    before = audit_rows(root)

    result = _import(folder, root)

    assert set(result) == {"error"}
    assert "tree_02.txt, line 3:" in result["error"]
    assert fault in result["error"]
    assert _written(root) == []
    assert audit_rows(root) == before


@pytest.mark.parametrize("content", [BOX_LINE, ""])
def test_a_file_naming_no_capture_image_refuses(
    tmp_path: Path, content: str
) -> None:
    root = _dataset(tmp_path).root
    folder = _labels(
        tmp_path / "labels", {"tree_01": BOX_LINE, "tree_99": content}
    )
    before = audit_rows(root)

    result = _import(folder, root)

    assert "tree_99.txt" in result["error"]
    assert "names no image" in result["error"]
    assert _written(root) == []
    assert audit_rows(root) == before


def test_an_unregistered_used_subject_refuses(tmp_path: Path) -> None:
    root = _dataset(tmp_path).root
    folder = _labels(
        tmp_path / "labels", {"tree_01": BOX_LINE, "tree_02": RING_LINE}
    )
    before = audit_rows(root)

    result = _import(folder, root, names=["husk"])

    assert "category 'husk' is not a subject" in result["error"]
    assert _written(root) == []
    assert audit_rows(root) == before


def test_only_used_class_ids_declare_categories(tmp_path: Path) -> None:
    root = _dataset(tmp_path).root
    folder = _labels(tmp_path / "labels", {"tree_01": BOX_LINE})
    result = _import(folder, root, names=[SUBJECT, "husk"])
    assert result["written"] == ["tree_01"]


def test_floating_point_slack_is_clipped_at_the_frame_boundary(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path).root
    folder = _labels(
        tmp_path / "labels", {"tree_01": "0 -1e-10 0 1.0000000001 0 1 1"}
    )
    result = _import(folder, root)
    assert result["written"] == ["tree_01"]
    from tcip_annotation import json_io

    (annotation,) = json_io.read_label_document(
        _label(root, "tree_01")
    ).annotations
    assert annotation.geometry.rings == [[(0, 0), (IMG, 0), (IMG, IMG)]]


@pytest.mark.parametrize("interleaved", [False, True])
def test_an_existing_document_refuses_without_an_import_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interleaved: bool
) -> None:
    from tcip_annotation import json_io

    root = _dataset(tmp_path).root
    folder = _labels(
        tmp_path / "labels", {"tree_01": BOX_LINE, "tree_02": RING_LINE}
    )
    key = _label(root, "tree_02")
    if interleaved:
        _label_placed_before_the_writes(monkeypatch, key, _person_label())
    else:
        json_io.write_label_document(
            key, _person_label(), IMG, IMG, keep_empty=True
        )
    before = audit_rows(root)

    result = _import(folder, root)

    assert "already holds a label document for ['tree_02']" in result["error"]
    assert _written(root) == ["tree_02"]
    assert json_io.read_label_document(key).annotations == _person_label()
    assert audit_rows(root) == before


def test_the_audit_names_the_actor_and_versions_read_even_if_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dataset(tmp_path).root
    folder = _labels(tmp_path / "labels", {"tree_01": BOX_LINE, "tree_02": ""})
    versions = [
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(folder.glob("*.txt"))
    ]
    expected = canonical_digest(versions)
    read = ts.read_blob_versioned
    seen = []

    def replacing(path: Path) -> Versioned:
        stored = read(path)
        if path.parent == folder:
            seen.append(path.name)
            path.write_text("replacement", encoding="utf-8")
        return stored

    monkeypatch.setattr(ts, "read_blob_versioned", replacing)
    before = len(audit_rows(root))
    result = _import(folder, root)

    (event,) = audit_rows(root)[before:]
    assert event["tool"] == "yolo_labels_imported"
    assert event["status"] == "ok"
    assert event["actor"] == "user:breeder"
    assert event["arguments"] == result
    assert event["labels_digest"] == expected
    assert result["empty"] == ["tree_02"]
    assert result["written"] == ["tree_01"]
    assert seen == ["tree_01.txt", "tree_02.txt"]


def test_a_band_group_is_matched_by_stem(tmp_path: Path) -> None:
    import numpy as np
    import tifffile

    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.ingest_tools import ingest_images
    from tests._producer_fixtures import write_band_group

    root = _dataset(tmp_path).root
    images = image_dir(root, DATE)
    bands = {"Green": images / "plot_G.tif", "Red": images / "plot_R.tif"}
    raw = tmp_path / "bands"
    raw.mkdir()
    for path in bands.values():
        tifffile.imwrite(
            raw / path.name,
            np.full((32, 64), 111, dtype=np.uint16),
            datetime=DATE.replace("-", ":") + " 12:00:00",
        )
    assert "error" not in ingest_images(root, source=str(raw), actor=None)
    write_band_group(images, "plot", bands, central_wavelength_nm=None)
    folder = _labels(tmp_path / "labels", {"plot": BOX_LINE})

    result = _import(folder, root)

    assert result["written"] == ["plot"]
    from tcip_annotation import json_io

    (annotation,) = json_io.read_label_document(
        _label(root, "plot")
    ).annotations
    assert annotation.geometry == BBox(8, 5, 28, 13)


def test_a_photograph_uses_its_upright_display_frame(tmp_path: Path) -> None:
    from datetime import datetime

    from PIL import Image

    from tcip_annotation import json_io
    from tcip_mcp.tools.ingest_tools import ingest_images
    from tests._image_fixtures import capture_exif

    root = _dataset(tmp_path).root
    raw = tmp_path / "phone.jpg"
    exif = capture_exif(datetime.fromisoformat(DATE))
    exif[274] = 6
    Image.new("RGB", (64, 32)).save(raw, exif=exif)
    assert "error" not in ingest_images(root, source=str(raw), actor=None)
    folder = _labels(tmp_path / "labels", {"phone": "0 0.5 0.5 0.5 0.25"})

    result = _import(folder, root)

    assert result["written"] == ["phone"]
    (annotation,) = json_io.read_label_document(
        _label(root, "phone")
    ).annotations
    assert annotation.geometry == BBox(8, 24, 24, 40)


def test_the_tool_and_console_command_are_registered() -> None:
    from tcip_web.cli import command_names, tool_listing

    listing = tool_listing()
    tool = next(item for item in listing if item.name == "import_yolo")
    assert (
        tool.input_schema["properties"]["names"]["items"]["type"] == "string"
    )
    assert "import-yolo" in command_names(listing)


def test_a_failed_audit_append_rolls_back_the_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tcip_mcp.audit as audit

    root = _dataset(tmp_path).root
    folder = _labels(
        tmp_path / "labels", {"tree_01": BOX_LINE, "tree_02": RING_LINE}
    )
    before = audit_rows(root)
    entry = audit.audit_entry
    with monkeypatch.context() as patch:
        patch.setattr(
            audit,
            "audit_entry",
            lambda *a, **k: {**entry(*a, **k), "unencodable": float("nan")},
        )
        with pytest.raises(ts.StoreError):
            _import(folder, root)
    assert _written(root) == []
    assert audit_rows(root) == before
    assert _import(folder, root)["written"] == ["tree_01", "tree_02"]


def test_relative_paths_locate_against_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.tools.ingest_tools import import_yolo

    root = _dataset(tmp_path).root
    folder = _labels(root / "labels", {"tree_01": BOX_LINE})
    monkeypatch.chdir(tmp_path)

    result = import_yolo(root, "labels", [SUBJECT], ".", DATE, actor=None)

    assert result["written"] == ["tree_01"]
    assert result["labels_dir"] == str(folder)
    (event,) = audit_rows(root, "yolo_labels_imported")
    assert "actor" not in event


@pytest.mark.parametrize("date", ["../elsewhere", "2025-09-15"])
def test_a_date_that_names_no_capture_refuses(
    tmp_path: Path, date: str
) -> None:
    from tcip_mcp.tools.ingest_tools import import_yolo

    root = _dataset(tmp_path).root
    folder = _labels(tmp_path / "labels", {"tree_01": BOX_LINE})
    before = audit_rows(root)

    result = import_yolo(
        root, str(folder), [SUBJECT], str(root), date, actor=None
    )

    assert "names no capture" in result["error"]
    assert _written(root) == []
    assert audit_rows(root) == before


@pytest.mark.parametrize("kind", ["missing", "no_labels", "invalid_utf8"])
def test_an_unreadable_or_absent_label_source_refuses(
    tmp_path: Path, kind: str
) -> None:
    root = _dataset(tmp_path).root
    folder = tmp_path / "labels"
    if kind != "missing":
        _labels(folder, {} if kind == "no_labels" else {"tree_01": BOX_LINE})
    if kind == "invalid_utf8":
        (folder / "tree_01.txt").write_bytes(b"\xff")
    before = audit_rows(root)

    result = _import(folder, root)

    assert set(result) == {"error"}
    assert str(folder) in result["error"]
    assert _written(root) == []
    assert audit_rows(root) == before


def test_a_box_the_document_encoder_refuses_writes_nothing(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path).root
    folder = _labels(
        tmp_path / "labels",
        {"tree_01": BOX_LINE, "tree_02": "0 0.5 0.5 0.000001 0.2"},
    )
    before = audit_rows(root)

    result = _import(folder, root)

    assert "tree_02.png" in result["error"]
    assert _written(root) == []
    assert audit_rows(root) == before


def test_the_console_command_imports_labels_and_records_its_person(
    tmp_path: Path,
) -> None:
    from tests._cli_fixtures import run_tcip

    root = _dataset(tmp_path).root
    folder = _labels(root / "labels", {"tree_01": BOX_LINE})

    result = run_tcip(
        "import-yolo",
        [
            "--project",
            str(root),
            "--user",
            "breeder",
            "--labels-dir",
            "labels",
            "--names",
            json.dumps([SUBJECT]),
            "--dataset-root",
            ".",
            "--date",
            DATE,
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    answer = json.loads(result.stdout)
    assert answer["labels_dir"] == str(folder)
    assert answer["written"] == ["tree_01"]
    (event,) = audit_rows(root, "yolo_labels_imported")
    assert event["arguments"] == answer
    assert event["actor"] == "user:breeder"
