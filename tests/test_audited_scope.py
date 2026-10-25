"""An audited tool's entry lands in the one log its scope names.

A tool that mutates a record traveling with the dataset records in that dataset's own audit
log and not the project's; a call that declares no scope records in the project's log, and a
project entry carries its core fields and no other.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image

import tcip_mcp.audit as audit_module
import tcip_store as ts

if TYPE_CHECKING:
    from tcip_mcp.workspace import BoundProject

CAPTURE_DATE = "2024-05-01"
STUB_ENGINE = "tests.proposal_stub:factory"
"""The dotted proposal engine ``tests/proposal_stub.py`` defines."""


def _entries(root: Path) -> list[dict]:
    """Return every audit row scoped to ``root``, or none when that scope's log holds nothing."""
    return list(ts.read_log(audit_module.audit_log_key(root)).records)


def _rows_for(root: Path, tool: str) -> list[dict]:
    return [row for row in _entries(root) if row["tool"] == tool]


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Return a dataset carrying one capture, its own root distinct from the project's."""
    root = tmp_path / "orchard_dataset"
    capture = root / "images" / CAPTURE_DATE
    capture.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(capture / "IMG_0001.JPG")
    return root


def _subjects() -> dict:
    return {
        "bud": {
            "description": "a currant bud",
            "attributes": {
                "opening": {
                    "type": "categorical",
                    "values": ["closed", "open"],
                }
            },
        }
    }


def test_registry_write_records_in_the_dataset_named_by_its_root_argument(
    project: Path, dataset_root: Path
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    assert "error" not in write_subject_registry(
        project, str(dataset_root), subjects=_subjects(), actor=None
    )

    rows = _rows_for(dataset_root, "replace_registry")
    assert len(rows) == 1, _entries(dataset_root)
    assert _rows_for(project, "replace_registry") == []


def test_label_write_records_in_the_dataset_holding_the_image_it_names(
    project: Path, bound: BoundProject, dataset_root: Path
) -> None:
    """A label write records in the dataset holding its image, not in the project's log.

    The label path lies inside the dataset, not at its root.
    """
    from tcip_mcp.tools.annotation_tools import save_annotations

    image = dataset_root / "images" / CAPTURE_DATE / "IMG_0001.JPG"
    result = save_annotations(
        bound,
        str(image),
        annotations=[{"subject": "bud", "bbox": [10, 10, 40, 40]}],
        actor=None,
    )
    assert "error" not in result

    rows = _rows_for(dataset_root, "save_label_document")
    assert len(rows) == 1, _entries(dataset_root)
    assert _rows_for(project, "save_label_document") == []


def test_propose_annotations_records_in_the_dataset_named_by_the_image_it_ran_against(
    project: Path, dataset_root: Path
) -> None:
    """``propose_annotations`` records in the dataset holding the image it ran against.

    A proposal run through a dotted ``module:factory`` engine records one line in the dataset the
    image belongs to and none in the project's log.
    """
    from tcip_mcp.tools.proposal_tools import propose_annotations

    image = dataset_root / "images" / CAPTURE_DATE / "IMG_0001.JPG"
    result = propose_annotations(
        project, image_path=str(image), engine=STUB_ENGINE, actor=None
    )
    assert "error" not in result, result

    rows = _rows_for(dataset_root, "propose_annotations")
    assert len(rows) == 1, _entries(dataset_root)
    assert _rows_for(project, "propose_annotations") == []


def test_propose_annotations_staging_nothing_leaves_no_audit_line(
    project: Path, tmp_path: Path
) -> None:
    """``propose_annotations`` staging nothing leaves no audit line in any log.

    For an image under no dataset the call succeeds, stages nothing, and leaves no line in the
    project's log or beside the image.
    """
    from tcip_mcp.tools.proposal_tools import propose_annotations

    loose = tmp_path / "loose"
    loose.mkdir()
    image = loose / "IMG_0002.JPG"
    Image.new("RGB", (100, 80)).save(image)

    result = propose_annotations(
        project, image_path=str(image), engine=STUB_ENGINE, actor=None
    )
    assert "error" not in result, result
    assert result["staged"] is False

    assert _rows_for(project, "propose_annotations") == []
    assert _entries(loose) == []


def test_unscoped_mutating_tool_records_in_the_project_log_with_the_entry_shape(
    project: Path, dataset_root: Path
) -> None:
    """An unscoped mutating tool records in the project's log, its row growing no new field."""
    from tcip_mcp.tools.meta_tools import report_friction

    report_friction(
        project, "unexpected_behavior", "a project event", actor=None
    )

    rows = _rows_for(project, "report_friction")
    assert len(rows) == 1
    assert set(rows[0]) == {
        "timestamp",
        "tool",
        "arguments",
        "status",
        "duration_ms",
    }
    assert "project" not in rows[0]["arguments"]
    assert _rows_for(dataset_root, "report_friction") == []


def test_a_label_write_of_an_image_under_no_dataset_refuses_and_records_nothing(
    project: Path, bound: BoundProject, tmp_path: Path
) -> None:
    """A label write of an image under no dataset is refused and records nothing.

    An image under no dataset image tree has no label document to write: the save refuses it as
    no capture, and no log receives a line.
    """
    from tcip_mcp.tools.annotation_tools import save_annotations

    loose = tmp_path / "loose"
    loose.mkdir()
    image = loose / "IMG_0002.JPG"
    Image.new("RGB", (100, 80)).save(image)

    result = save_annotations(
        bound,
        str(image),
        annotations=[{"subject": "bud", "bbox": [10, 10, 40, 40]}],
        actor=None,
    )
    assert "is not a capture" in result["error"]

    assert _rows_for(project, "save_label_document") == []
    assert _entries(loose) == []


def test_scope_argument_left_unset_leaves_the_call_in_the_project(
    tmp_path: Path,
) -> None:
    from tcip_mcp.audit import audited

    @audited(scope_arg="dataset_root")
    def stage_something(
        project: Path, dataset_root: str | None = None
    ) -> dict:
        return {"ok": True}

    stage_something(tmp_path)

    assert len(_rows_for(tmp_path, "stage_something")) == 1


def test_relative_scope_argument_resolves_against_the_project_not_the_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative scope argument resolves against the project, never the process cwd.

    A relative dataset location is read against the project the call acts on; a same-named
    directory under the process cwd stands in for the wrong answer.
    """
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scope resolution that raises refuses the call and files nothing in the project's log.

    The call raises ``MutationCommittedWithoutAuditLineError``, caused by the resolver's failure.
    """
    from tcip_mcp.audit import audited

    def _raises(value: object) -> Path:
        raise RuntimeError("this location cannot be resolved")

    monkeypatch.setattr(audit_module, "dataset_scope_of", _raises)

    @audited(scope_arg="output_dir")
    def curate_something(project: Path, output_dir: str) -> dict:
        return {"ok": True}

    with pytest.raises(RuntimeError) as caught:
        curate_something(tmp_path, "anywhere")

    assert (
        type(caught.value)
        is audit_module.MutationCommittedWithoutAuditLineError
    )
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert _rows_for(tmp_path, "curate_something") == []


def test_two_spellings_of_a_root_reach_one_log_and_a_relative_spelling_refuses(
    dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two spellings of a root reach one audit log, and a relative spelling is refused.

    A line written under one spelling of a root (through ``images/..``) reads back under the
    root's plain spelling, and on Windows under its upper-cased spelling; a relative spelling is
    refused as not an absolute path.
    """
    import os

    from tcip_mcp.audit import audit_entries, record_event_or_raise

    record_event_or_raise(
        "curate_something",
        {"n": 1},
        actor=None,
        scope=dataset_root / "images" / "..",
    )
    spellings = [dataset_root]
    if os.name == "nt":
        spellings.append(Path(str(dataset_root).upper()))
    for spelling in spellings:
        entries, _cursor = audit_entries(spelling)
        assert [e["tool"] for e in entries] == ["curate_something"]

    monkeypatch.chdir(dataset_root.parent)
    with pytest.raises(ts.BadKeyError, match="not an absolute path"):
        audit_entries(Path(dataset_root.name))


def test_scope_argument_naming_no_parameter_is_refused_at_decoration() -> None:
    """A ``scope_arg`` naming no parameter is refused at decoration."""
    from tcip_mcp.audit import audited

    with pytest.raises(ValueError, match="names no parameter"):

        @audited(scope_arg="datset_root")
        def stage_something(project: Path, dataset_root: str) -> dict:
            return {"ok": True}
