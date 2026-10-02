"""An audited tool's entry lands in the one log its scope names.

A tool that mutates a record traveling with the dataset records in that dataset's own audit
log, so the provenance moves with the data; every other call stays in the project's log. Exactly
one log receives each entry, and a project entry keeps the shape it always had.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import tcip_mcp.audit as audit_module
import tcip_store as ts

CAPTURE_DATE = "2024-05-01"


def _entries(root: Path) -> list[dict]:
    """Every audit row scoped to ``root``, or none when that scope's log holds nothing yet."""
    return list(ts.read_log(audit_module.audit_log_key(root)).records)


def _rows_for(root: Path, tool: str) -> list[dict]:
    return [row for row in _entries(root) if row["tool"] == tool]


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A dataset carrying one capture, its own root distinct from the project's."""
    root = tmp_path / "orchard_dataset"
    capture = root / "images" / CAPTURE_DATE
    capture.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(capture / "IMG_0001.JPG")
    return root


def _subjects() -> dict:
    return {"bud": {"description": "a currant bud",
                       "attributes": {"opening": {"type": "categorical",
                                                     "values": ["closed", "open"]}}}}


def test_registry_write_records_in_the_dataset_named_by_its_root_argument(
    project: Path, dataset_root: Path
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    assert "error" not in write_subject_registry(project, str(dataset_root), subjects=_subjects())

    rows = _rows_for(dataset_root, "replace_registry")
    assert len(rows) == 1, _entries(dataset_root)
    assert _rows_for(project, "replace_registry") == []


def test_label_write_records_in_the_dataset_holding_the_image_it_names(
    project: Path, dataset_root: Path
) -> None:
    """The label path lies inside the dataset, not at its root."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    image = dataset_root / "images" / CAPTURE_DATE / "IMG_0001.JPG"
    result = save_annotations(
        project, project.parent, str(image),
        annotations=[{"subject": "bud", "bbox": [10, 10, 40, 40]}]
    )
    assert "error" not in result

    rows = _rows_for(dataset_root, "save_label_document")
    assert len(rows) == 1, _entries(dataset_root)
    assert _rows_for(project, "save_label_document") == []


def test_propose_annotations_records_in_the_dataset_named_by_the_image_it_ran_against(
    project: Path, dataset_root: Path
) -> None:
    """A proposal run is scoped like its sibling ``stage_proposals``: to the dataset the image
    belongs to, not the project's log, driven through a dotted ``module:factory`` engine so the
    test needs no torch."""
    from tcip_mcp.tools.proposal_tools import propose_annotations

    image = dataset_root / "images" / CAPTURE_DATE / "IMG_0001.JPG"
    result = propose_annotations(project, image_path=str(image),
                                 engine="tests.proposal_stub:factory")
    assert "error" not in result, result

    rows = _rows_for(dataset_root, "propose_annotations")
    assert len(rows) == 1, _entries(dataset_root)
    assert _rows_for(project, "propose_annotations") == []


def test_propose_annotations_against_a_file_under_no_images_directory_stays_in_the_project(
    project: Path, tmp_path: Path
) -> None:
    """A location that resolves to no dataset is never guessed into one, for propose_annotations
    just as for its sibling: the call still succeeds and stays in the project's log."""
    from tcip_mcp.tools.proposal_tools import propose_annotations

    loose = tmp_path / "loose"
    loose.mkdir()
    image = loose / "IMG_0002.JPG"
    Image.new("RGB", (100, 80)).save(image)

    result = propose_annotations(project, image_path=str(image),
                                 engine="tests.proposal_stub:factory")
    assert "error" not in result, result
    assert result["staged"] is False

    assert len(_rows_for(project, "propose_annotations")) == 1
    assert _entries(loose) == []


def test_unscoped_mutating_tool_records_in_the_project_log_with_the_entry_shape(
    project: Path, dataset_root: Path
) -> None:
    """A mutating tool that declares no scope records in the project's log, and its row grows no
    new field."""
    from tcip_mcp.tools.meta_tools import report_friction

    report_friction(project, "unexpected_behavior", "a project event")

    rows = _rows_for(project, "report_friction")
    assert len(rows) == 1
    assert set(rows[0]) == {
        "timestamp", "tool", "arguments", "status", "duration_ms",
    }
    assert "project" not in rows[0]["arguments"]
    assert _rows_for(dataset_root, "report_friction") == []


def test_scope_argument_naming_no_dataset_leaves_the_call_in_the_project(
    project: Path, tmp_path: Path
) -> None:
    """A location that resolves to no dataset is never guessed into one."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    loose = tmp_path / "loose"
    loose.mkdir()
    image = loose / "IMG_0002.JPG"
    Image.new("RGB", (100, 80)).save(image)

    result = save_annotations(
        project, project.parent, str(image),
        annotations=[{"subject": "bud", "bbox": [10, 10, 40, 40]}],
        path=str(loose / "out.json"),
    )
    assert "error" not in result

    assert len(_rows_for(project, "save_label_document")) == 1
    assert _entries(loose) == []


def test_scope_argument_left_unset_leaves_the_call_in_the_project(tmp_path: Path) -> None:
    from tcip_mcp.audit import audited

    @audited(scope_arg="dataset_root")
    def stage_something(project: Path, dataset_root: str | None = None) -> dict:
        return {"ok": True}

    stage_something(tmp_path)

    assert len(_rows_for(tmp_path, "stage_something")) == 1


def test_relative_scope_argument_resolves_against_the_project_not_the_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative dataset location is read against the project the call acts on; a same-named
    directory under the process cwd stands in for the wrong answer."""
    from tcip_mcp.audit import audited

    @audited(scope_arg="output_dir")
    def write_curated(project: Path, output_dir: str) -> dict:
        return {"ok": True}

    project = tmp_path / "project"
    anchored = project / "curated_dataset"
    anchored.mkdir(parents=True)
    (anchored / "subjects.json").write_text("{}", encoding="utf-8")

    process_cwd = tmp_path / "process_cwd"
    decoy = process_cwd / "curated_dataset"
    decoy.mkdir(parents=True)
    (decoy / "subjects.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(process_cwd)
    write_curated(project, "curated_dataset")

    assert len(_rows_for(anchored, "write_curated")) == 1, _entries(anchored)
    assert _rows_for(project, "write_curated") == []
    assert not (decoy / ".tcip").exists()


def test_a_scope_resolution_that_raises_refuses_rather_than_filing_the_event_in_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared dataset event whose destination cannot be worked out is not rerouted: an entry
    filed in the project's log on a resolver failure is one nobody tracing that dataset finds."""
    from tcip_mcp.audit import audited

    def _raises(value: object) -> Path:
        raise RuntimeError("this location cannot be resolved")

    monkeypatch.setattr(audit_module, "dataset_scope_of", _raises)

    @audited(scope_arg="output_dir")
    def curate_something(project: Path, output_dir: str) -> dict:
        return {"ok": True}

    with pytest.raises(RuntimeError) as caught:
        curate_something(tmp_path, "anywhere")

    assert type(caught.value) is audit_module.MutationCommittedWithoutAuditLine
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert _rows_for(tmp_path, "curate_something") == []


def test_scope_argument_naming_no_parameter_is_refused_at_decoration() -> None:
    """A rail against a typo that would silently record every call in the project's log."""
    from tcip_mcp.audit import audited

    with pytest.raises(ValueError, match="names no parameter"):
        @audited(scope_arg="datset_root")
        def stage_something(project: Path, dataset_root: str) -> dict:
            return {"ok": True}
