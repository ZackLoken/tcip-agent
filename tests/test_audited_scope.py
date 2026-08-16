"""An audited tool's entry lands in the one log its scope names.

A tool that mutates a record travelling with the dataset records in that dataset's own
``.tcip/audit.jsonl``, so the provenance moves with the data; every other call stays a platform
event. Exactly one log receives each entry, and a platform entry keeps the shape it always had.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import tcip_mcp.audit as audit_module

CAPTURE_DATE = "2024-05-01"


def _entries(root: Path) -> list[dict]:
    """Every audit row under ``root``, or none when that root has no log at all."""
    log = root / ".tcip" / "audit.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def _rows_for(root: Path, tool: str) -> list[dict]:
    return [row for row in _entries(root) if row["tool"] == tool]


@pytest.fixture
def platform_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The platform's own audit root, distinct from any dataset's."""
    root = tmp_path / "platform"
    root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", root)
    return root


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A dataset carrying one capture, sited away from the platform root."""
    root = tmp_path / "orchard_dataset"
    capture = root / "images" / CAPTURE_DATE
    capture.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(capture / "IMG_0001.JPG")
    return root


def _subjects() -> dict:
    return {"catkin": {"description": "a hazelnut catkin",
                       "attributes": {"elongation": {"type": "categorical",
                                                     "values": ["dormant", "elongated"]}}}}


def test_registry_write_records_in_the_dataset_named_by_its_root_argument(
    platform_root: Path, dataset_root: Path
) -> None:
    from tcip_mcp.tools.annotation_tools import write_class_map

    assert "error" not in write_class_map(str(dataset_root), subjects=_subjects())

    rows = _rows_for(dataset_root, "write_class_map")
    assert len(rows) == 1, _entries(dataset_root)
    assert rows[0]["scope"] == str(dataset_root.resolve())
    assert _rows_for(platform_root, "write_class_map") == []


def test_label_write_records_in_the_dataset_holding_the_image_it_names(
    platform_root: Path, dataset_root: Path
) -> None:
    """The scope argument carries a path inside the dataset, not the root itself."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    image = dataset_root / "images" / CAPTURE_DATE / "IMG_0001.JPG"
    result = save_annotations(
        str(image), annotations=[{"subject": "catkin", "bbox": [10, 10, 40, 40]}]
    )
    assert "error" not in result

    rows = _rows_for(dataset_root, "save_annotations")
    assert len(rows) == 1, _entries(dataset_root)
    assert rows[0]["scope"] == str(dataset_root.resolve())
    assert _rows_for(platform_root, "save_annotations") == []


def test_read_only_tool_records_in_the_platform_log_with_the_platform_entry_shape(
    platform_root: Path, dataset_root: Path
) -> None:
    """A tool that declares no scope stays a platform event, and its row grows no new field."""
    from tcip_mcp.tools.data_tools import scan_dataset

    scan_dataset(str(dataset_root))

    rows = _rows_for(platform_root, "scan_dataset")
    assert len(rows) == 1
    assert set(rows[0]) == {"timestamp", "tool", "arguments", "status", "duration_ms"}
    assert rows[0]["arguments"] == {"folder_path": str(dataset_root)}
    assert _rows_for(dataset_root, "scan_dataset") == []


def test_scope_argument_naming_no_dataset_leaves_the_call_a_platform_event(
    platform_root: Path, tmp_path: Path
) -> None:
    """A location that resolves to no dataset is never guessed into one: it stays platform-scoped."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    loose = tmp_path / "loose"
    loose.mkdir()
    image = loose / "IMG_0002.JPG"
    Image.new("RGB", (100, 80)).save(image)

    result = save_annotations(
        str(image),
        annotations=[{"subject": "catkin", "bbox": [10, 10, 40, 40]}],
        path=str(loose / "out.json"),
    )
    assert "error" not in result

    rows = _rows_for(platform_root, "save_annotations")
    assert len(rows) == 1
    assert "scope" not in rows[0]
    assert _entries(loose) == []


def test_scope_argument_left_unset_leaves_the_call_a_platform_event(platform_root: Path) -> None:
    from tcip_mcp.audit import audited

    @audited(scope_arg="dataset_root")
    def stage_something(dataset_root: str | None = None) -> dict:
        return {"ok": True}

    stage_something()

    rows = _rows_for(platform_root, "stage_something")
    assert len(rows) == 1
    assert "scope" not in rows[0]


def test_relative_output_dir_records_where_the_tool_anchors_it_not_where_the_process_runs(
    platform_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative output path anchors to the platform state root, in the entry as in the write.

    The tool resolves it through ``resolve_output_path``, which never uses the process cwd, so an
    audit entry resolved any other way would name a directory the tool did not write to. A
    same-named directory under the cwd stands in for that wrong answer here.
    """
    from tcip_mcp.tools.feedback_tools import materialize_review_dataset

    pinned_root = tmp_path / "pinned_project"
    anchored = pinned_root / "curated_dataset"
    anchored.mkdir(parents=True)
    (anchored / "classes.json").write_text("{}", encoding="utf-8")

    process_cwd = tmp_path / "process_cwd"
    decoy = process_cwd / "curated_dataset"
    decoy.mkdir(parents=True)
    (decoy / "classes.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(process_cwd)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(pinned_root))

    result = materialize_review_dataset(
        review_state_dir=str(tmp_path / "no_review_state"),
        source_images_dir=str(tmp_path / "source"),
        output_dir="curated_dataset",
    )
    assert "error" in result  # the early refusal still records the call it refused

    rows = _rows_for(anchored, "materialize_review_dataset")
    assert len(rows) == 1, _entries(anchored)
    assert rows[0]["scope"] == str(anchored.resolve())
    assert _rows_for(platform_root, "materialize_review_dataset") == []
    assert not (decoy / ".tcip").exists()


def test_a_scope_resolver_that_raises_leaves_one_platform_entry_and_the_result_intact(
    platform_root: Path,
) -> None:
    from tcip_mcp.audit import audited

    def _raises(value: object) -> Path:
        raise RuntimeError("this location cannot be resolved")

    @audited(scope_arg="output_dir", scope_via=_raises)
    def curate_something(output_dir: str) -> dict:
        return {"ok": True}

    assert curate_something("anywhere") == {"ok": True}

    rows = _rows_for(platform_root, "curate_something")
    assert len(rows) == 1
    assert "scope" not in rows[0]
    assert rows[0]["status"] == "ok"


def test_scope_argument_naming_no_parameter_is_refused_at_decoration(platform_root: Path) -> None:
    """A rail against a typo that would silently record every call in the platform log."""
    from tcip_mcp.audit import audited

    with pytest.raises(ValueError, match="names no parameter"):
        @audited(scope_arg="datset_root")
        def stage_something(dataset_root: str) -> dict:
            return {"ok": True}

    with pytest.raises(ValueError, match="no scope_arg"):
        @audited(scope_via=Path)
        def curate_something(output_dir: str) -> dict:
            return {"ok": True}
