"""scripts/doctor.py: the data-state doctor catches the field-session bug family."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry
from tcip_mcp.dataset_layout import status_bucket

PY_EXE = sys.executable
DOCTOR = str(Path(__file__).parent.parent / "scripts" / "doctor.py")


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "images" / "2026-02-11").mkdir(parents=True)
    ann = root / "annotations" / "2026-02-11"
    ann.mkdir(parents=True)
    state = root / ".tcip" / "state"
    state.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="catkin"),)))
    for name in ("IMG_A", "IMG_B", "IMG_C"):
        Image.new("RGB", (32, 32)).save(root / "images" / "2026-02-11" / f"{name}.JPG")
    # A: confirmed negative (empty + status). B: empty without confirmation (the IMG_0150 case).
    # C: has objects but status wrongly says negative (contradiction).
    json_io.write_annotations(ann / "IMG_A.json", [], 32, 32, keep_empty=True)
    json_io.write_annotations(ann / "IMG_B.json", [], 32, 32, keep_empty=True)
    json_io.write_annotations(ann / "IMG_C.json",
                              [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 32, 32)
    # Scoped by subject/date: a confirmation belongs to the subject it was made in.
    (state / "image_status.json").write_text(json.dumps(
        {status_bucket("catkin", "2026-02-11"): {"IMG_A.JPG": "negative", "IMG_B.JPG": "unannotated",
                                                 "IMG_C.JPG": "negative"}}))
    return root


def _run(root: Path):
    return subprocess.run([PY_EXE, DOCTOR, str(root)], capture_output=True, text=True)


def test_doctor_flags_the_field_session_bug_family(tmp_path):
    root = _project(tmp_path)
    # registry entry pointing at a pytest temp checkpoint (the leak the field session found)
    reg = root / ".tcip" / "models"
    reg.mkdir(parents=True)
    (reg / "registry.json").write_text(json.dumps(
        [{"name": "junk", "checkpoint_path": "C:\Temp\pytest-of-x\model.pt"}]))

    res = _run(root)
    assert res.returncode == 2  # errors present
    out = res.stdout
    assert "IMG_B" in out and "not a confirmed negative" in out   # unconfirmed empty -> warn
    assert "contradictory" in out and "IMG_C" in out              # objects + negative -> error
    assert "junk" in out and "test/temp" in out                   # registry pollution -> error
    assert "IMG_A" not in out.replace("IMG_A.JPG is negative", "")  # confirmed negative is clean


def test_doctor_flags_a_trait_spec_that_failed_to_load(tmp_path):
    """A dropped trait spec reads identically to no trait at all from the registry alone;
    doctor.py is where the agent catches the difference at session start."""
    root = _project(tmp_path)
    specs_dir = root / ".tcip" / "state" / "trait_specs"
    specs_dir.mkdir(parents=True)
    (specs_dir / "unicorn.yml").write_text(
        "name: unicorn\ndelivers: [unicorn_horn_length]\n", encoding="utf-8")

    res = _run(root)
    assert res.returncode == 2  # errors present
    assert "unicorn.yml" in res.stdout
    assert "unicorn_horn_length" in res.stdout


def test_doctor_flags_a_stale_region_completeness_attestation(tmp_path):
    """An attested cell whose annotation content has since changed is exactly the data-state
    inconsistency doctor.py exists to catch (region_completeness.json vs the label file it
    describes); confirm the ritual surfaces it, not just the route's own read path."""
    from tcip_mcp.dataset_layout import status_bucket
    from tcip_mcp.pipelines.reference_grid import reference_cells
    from tcip_mcp.pipelines.region_completeness import cell_annotation_digest

    root = tmp_path / "stale_proj"
    (root / "images" / "2026-02-11").mkdir(parents=True)
    ann_dir = root / "annotations" / "2026-02-11"
    ann_dir.mkdir(parents=True)
    state_dir = root / ".tcip" / "state"
    state_dir.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="catkin"),)))
    Image.new("RGB", (32, 32)).save(root / "images" / "2026-02-11" / "IMG_A.JPG")
    ann_path = ann_dir / "IMG_A.json"
    json_io.write_annotations(
        ann_path, [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 32, 32)

    grid = {"width": 32, "height": 32, "tile_size": 16, "overlap": 0.0, "cols": 2, "rows": 2}
    cell = next(c for c in reference_cells(32, 32, 16, clamp=True) if c.name == "A1")
    stamped = cell_annotation_digest(json_io.read_annotations(str(ann_path)), "catkin", cell)

    bucket = status_bucket("catkin", "IMG_A")
    (state_dir / "region_completeness.json").write_text(json.dumps({
        bucket: {"grid": grid, "cells_complete": ["A1"], "attested_by": "user:zack",
                "attested_at": "t", "stem": "IMG_A", "date": "2026-02-11", "subject": "catkin"},
    }))
    (state_dir / "region_completeness_digest.json").write_text(
        json.dumps({bucket: {"A1": stamped}}))

    # The label is edited after attestation: a real staleness scenario, not a fabricated one.
    json_io.write_annotations(
        ann_path, [Annotation(subject="catkin", geometry=BBox(1, 1, 20, 20))], 32, 32)

    res = _run(root)
    assert res.returncode == 2
    assert "region completeness" in res.stdout
    assert "catkin" in res.stdout and "A1" in res.stdout


def test_doctor_flags_incomplete_source_snapshot(tmp_path):
    """A bespoke run's source snapshot that failed to capture a declared file is
    self-describing (``missing``/``snapshot_errors``); doctor.py surfaces it rather than the
    manifest reading as complete."""
    root = tmp_path / "clean"
    (root / "images" / "d").mkdir(parents=True)
    ann = root / "annotations" / "d"
    ann.mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="catkin"),)))
    Image.new("RGB", (32, 32)).save(root / "images" / "d" / "IMG_A.JPG")
    json_io.write_annotations(
        ann / "IMG_A.json",
        [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9), created_by="user:zack")], 32, 32)

    manifest_dir = root / ".tcip" / "experiments" / "exp1" / "model_src"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps({
        "files": [], "missing": ["agent_helper.py"], "snapshot_errors": [],
    }))

    res = _run(root)
    assert res.returncode == 1  # warning only, no error
    assert "source snapshot" in res.stdout and "1 missing file" in res.stdout


def test_doctor_clean_project_exits_zero(tmp_path):
    root = tmp_path / "clean"
    (root / "images" / "d").mkdir(parents=True)
    ann = root / "annotations" / "d"
    ann.mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="catkin"),)))
    Image.new("RGB", (32, 32)).save(root / "images" / "d" / "IMG_A.JPG")
    json_io.write_annotations(
        ann / "IMG_A.json",
        [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9), created_by="user:zack")], 32, 32)
    res = _run(root)
    assert res.returncode == 0, res.stdout
