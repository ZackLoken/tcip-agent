"""scripts/doctor.py: the data-state doctor catches the field-session bug family."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import traits
from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry
from tcip_mcp.dataset_layout import (
    annotation_dir,
    annotation_path,
    image_dir,
    status_bucket,
    status_records,
)
from tcip_mcp.model_registry import ModelRegistry

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
        {status_bucket("catkin", "2026-02-11"): status_records(
            {"IMG_A.JPG": "negative", "IMG_B.JPG": "unannotated", "IMG_C.JPG": "negative"},
            recorded_by="user:breeder")}))
    return root


def _run(root: Path, *, file_layout: bool = False):
    """Run the doctor against ``root``.

    ``file_layout=True`` bound to the file backend on purpose: that caller's fixture wrote
    ``image_status.json``/``region_completeness.json``/``registry.json`` straight to disk, and
    the check under test reads that same file layout directly, the exact case
    ``staleness_findings`` reports as invalid once a database also holds the root, so the
    subprocess is pinned to file regardless of whatever backend the outer test run selects.
    """
    env = {**os.environ, "TCIP_STORE_BACKEND": "file"} if file_layout else None
    return subprocess.run([PY_EXE, DOCTOR, str(root)], capture_output=True, text=True, env=env)


def test_doctor_flags_the_field_session_bug_family(tmp_path):
    root = _project(tmp_path)
    # registry entry pointing at a pytest temp checkpoint (the leak the field session found)
    reg = root / ".tcip" / "models"
    reg.mkdir(parents=True)
    (reg / "registry.json").write_text(json.dumps(
        [{"name": "junk", "checkpoint_path": "C:\Temp\pytest-of-x\model.pt"}]))

    res = _run(root, file_layout=True)
    assert res.returncode == 2  # errors present
    out = res.stdout
    assert "IMG_B" in out and "not a confirmed negative" in out   # unconfirmed empty -> warn
    assert "contradictory" in out and "IMG_C" in out              # objects + negative -> error
    assert "junk" in out and "test/temp" in out                   # registry pollution -> error
    assert "IMG_A" not in out.replace("IMG_A.JPG is negative", "")  # confirmed negative is clean


def test_doctor_flags_a_trait_spec_that_failed_to_load(tmp_path):
    """A dropped trait spec reads identically to no trait at all from the registry alone;
    doctor.py is where the agent catches the difference at session start."""
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    root = _project(tmp_path)
    specs_dir = root / ".tcip" / "state" / "trait_specs"
    # This test's doctor subprocess runs with file_layout=True, so the fixture's own record has
    # to land as the same loose file the file backend reads, not the process-default backend.
    ts.bind(FileBackend())
    ts.replace(traits.trait_spec_key(specs_dir, "unicorn"),
              {"name": "unicorn", "delivers": ["unicorn_horn_length"]}, expect=ts.Version.ABSENT)

    res = _run(root, file_layout=True)
    assert res.returncode == 2  # errors present
    assert "unicorn.json" in res.stdout
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
        bucket: {"grid": grid, "cells_complete": ["A1"], "attested_by": "user:breeder",
                "attested_at": "t", "stem": "IMG_A", "date": "2026-02-11", "subject": "catkin"},
    }))
    (state_dir / "region_completeness_digest.json").write_text(
        json.dumps({bucket: {"A1": stamped}}))

    # The label is edited after attestation: a real staleness scenario, not a fabricated one.
    json_io.write_annotations(
        ann_path, [Annotation(subject="catkin", geometry=BBox(1, 1, 20, 20))], 32, 32)

    res = _run(root, file_layout=True)
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
        [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9), created_by="user:breeder")], 32, 32)

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
        [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9), created_by="user:breeder")], 32, 32)
    res = _run(root)
    assert res.returncode == 0, res.stdout


def _layout_project(tmp_path: Path, date: str | None, name: str = "resolved") -> Path:
    """A project whose image and label trees are placed by the layout resolver, so a scan root
    that drifts from the canonical layout shows up as findings the doctor never makes."""
    root = tmp_path / name
    image_dir(root, date).mkdir(parents=True)
    annotation_dir(root, date).mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    write_registry(root / "classes.json",
                   ClassRegistry(subjects=(Subject(name="catkin"), Subject(name="leaf"))))
    return root


def _lines(stdout: str, needle: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if needle in ln]


def test_labels_are_scanned_where_the_layout_resolver_places_them(tmp_path):
    """The contradiction doctor.py reports is named by the path the resolver builds, so the
    checker's scan root and the canonical annotations tree cannot drift apart unnoticed."""
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (48, 32)).save(image_dir(root, date) / "IMG_R.JPG")
    label = annotation_path(root, date, "IMG_R")
    json_io.write_annotations(
        label, [Annotation(subject="catkin", geometry=BBox(2, 3, 18, 9))], 48, 32)
    (root / ".tcip" / "state" / "image_status.json").write_text(json.dumps(
        {status_bucket("catkin", date): status_records(
            {"IMG_R.JPG": "negative"}, recorded_by="user:breeder")}))

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    contradictions = _lines(res.stdout, "contradictory")
    assert len(contradictions) == 1, res.stdout
    assert str(label.relative_to(root)) in contradictions[0]


def test_a_negative_confirmation_names_only_its_own_subject(tmp_path):
    """A confirmation is scoped to one subject: an image holding leaf annotations and confirmed
    negative for both subjects contradicts the leaf confirmation only, and the catkin
    confirmation on the same image stands."""
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (48, 32)).save(image_dir(root, date) / "IMG_S.JPG")
    json_io.write_annotations(
        annotation_path(root, date, "IMG_S"),
        [Annotation(subject="leaf", geometry=BBox(4, 2, 40, 11))], 48, 32)
    (root / ".tcip" / "state" / "image_status.json").write_text(json.dumps({
        status_bucket("catkin", date): status_records({"IMG_S.JPG": "negative"}, recorded_by="user:breeder"),
        status_bucket("leaf", date): status_records({"IMG_S.JPG": "negative"}, recorded_by="user:breeder"),
    }))

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    contradictions = _lines(res.stdout, "contradictory")
    assert len(contradictions) == 1, res.stdout
    assert "'leaf'" in contradictions[0]
    assert "'catkin'" not in contradictions[0]


def test_confirmations_are_matched_on_a_dateless_dataset(tmp_path):
    """A dataset with no capture-date buckets keys its confirmations by subject alone; the
    doctor still pairs a confirmation with the label file it contradicts."""
    root = tmp_path / "flat"
    image_dir(root, None).mkdir(parents=True)
    annotation_dir(root, None).mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="catkin"),)))
    Image.new("RGB", (40, 24)).save(image_dir(root, None) / "IMG_F.JPG")
    json_io.write_annotations(
        annotation_path(root, None, "IMG_F"),
        [Annotation(subject="catkin", geometry=BBox(3, 1, 20, 9))], 40, 24)
    (root / ".tcip" / "state" / "image_status.json").write_text(json.dumps(
        {status_bucket("catkin", None): status_records(
            {"IMG_F.JPG": "negative"}, recorded_by="user:breeder")}))

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    contradictions = _lines(res.stdout, "contradictory")
    assert len(contradictions) == 1, res.stdout
    assert "'catkin'" in contradictions[0]


def test_registry_findings_are_read_through_the_registrys_own_entry_shape(tmp_path):
    """Entries written by ModelRegistry are the shape doctor.py reports on, so an entry whose
    checkpoint is gone is named with its own name and path rather than read as nothing."""
    root = _layout_project(tmp_path, "2026-03-04")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    registry = ModelRegistry(str(root))
    paths = {}
    for name, payload in (("hazelnut_catkin_detector_v1", b"weights"),
                          ("chestnut_burr_counter_v3", b"other weights")):
        ckpt = ckpt_dir / f"{name}.pt"
        ckpt.write_bytes(payload)
        registry.register_model(name=name, checkpoint_path=str(ckpt), config={}, metrics={})
        paths[name] = ckpt
    for ckpt in paths.values():
        ckpt.unlink()

    res = _run(root)
    assert res.returncode == 2, res.stdout
    entry_lines = _lines(res.stdout, "registry entry")
    assert len(entry_lines) == 2, res.stdout
    for name, ckpt in paths.items():
        assert any(name in ln and str(ckpt) in ln for ln in entry_lines), res.stdout


def test_a_missing_checkpoint_and_a_test_checkpoint_are_distinct_registry_findings(tmp_path):
    """Pollution and absence are different data-state problems: an entry pointing at a
    throwaway test checkpoint is not reported as merely missing, and an entry whose checkpoint
    was never written is not reported as pollution."""
    root = _layout_project(tmp_path, "2026-03-04")
    models = root / ".tcip" / "models"
    models.mkdir(parents=True)
    ghost = str(Path(root.anchor) / "tcip_absent_models" / "orchard.pt")
    scratch = str(Path(root.anchor) / "scratch" / "pytest-of-someone" / "run" / "last.pt")
    (models / "registry.json").write_text(json.dumps([
        {"name": "orchard_detector_v2", "checkpoint_path": ghost},
        {"name": "scratch_detector", "checkpoint_path": scratch},
    ]))

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    entry_lines = _lines(res.stdout, "registry entry")
    assert len(entry_lines) == 2, res.stdout
    ghost_line = next(ln for ln in entry_lines if "orchard_detector_v2" in ln)
    scratch_line = next(ln for ln in entry_lines if "scratch_detector" in ln)
    assert "checkpoint missing" in ghost_line and "test/temp" not in ghost_line
    assert "test/temp" in scratch_line and "checkpoint missing" not in scratch_line


def test_trait_specs_are_read_from_the_registrys_own_directory(tmp_path):
    """The specs the doctor loads are the ones the trait registry resolves, and only the
    unloadable spec is reported: a valid spec sitting beside it stays silent."""
    import tcip_store as ts

    root = _layout_project(tmp_path, "2026-03-04")
    specs_dir = root / traits._TRAIT_SPECS_RELPATH
    ts.replace(traits.trait_spec_key(specs_dir, "bloom_length"),
              {"name": "bloom_length", "delivers": ["bloom_length"]}, expect=ts.Version.ABSENT)
    ts.replace(traits.trait_spec_key(specs_dir, "burr_size"),
              {"name": "burr_size", "delivers": ["burr_size"], "measured_with": "calipers"},
              expect=ts.Version.ABSENT)

    res = _run(root)
    assert res.returncode == 2, res.stdout
    spec_lines = _lines(res.stdout, "trait spec")
    assert len(spec_lines) == 1, res.stdout
    assert "burr_size.json" in spec_lines[0]
    assert "measured_with" in spec_lines[0]
    assert "bloom_length.json" not in res.stdout


def test_review_baselines_are_not_counted_as_label_records(tmp_path):
    """The pre-review snapshots under an annotations dir's .original are copies, not labels:
    an image whose only file there is a snapshot still has no label record and trains on
    nothing, and the snapshot itself is never reported as an empty or orphaned label."""
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    imgs = image_dir(root, date)
    for name, size in (("IMG_A", (48, 32)), ("IMG_B", (48, 32)),
                       ("IMG_C", (60, 40)), ("IMG_D", (48, 32))):
        Image.new("RGB", size).save(imgs / f"{name}.JPG")
    json_io.write_annotations(
        annotation_path(root, date, "IMG_A"),
        [Annotation(subject="catkin", geometry=BBox(2, 3, 18, 9), created_by="user:breeder")], 48, 32)
    json_io.write_annotations(
        annotation_path(root, date, "IMG_B"),
        [Annotation(subject="leaf", geometry=BBox(5, 1, 44, 12), created_by="user:breeder")], 48, 32)
    baselines = annotation_dir(root, date) / ".original"
    baselines.mkdir()
    json_io.write_annotations(baselines / "IMG_B.json", [], 48, 32, keep_empty=True)
    json_io.write_annotations(baselines / "IMG_D.json", [], 48, 32, keep_empty=True)

    res = _run(root)
    assert res.returncode == 0, res.stdout
    census = _lines(res.stdout, "have no label record")
    assert len(census) == 1, res.stdout
    assert "2 of 4 image(s)" in census[0]
    assert "IMG_C" in census[0] and "IMG_D" in census[0]
    assert ".original" not in res.stdout
