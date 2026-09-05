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
    image_status_key,
    replace_image_status_store,
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
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))
    for name in ("IMG_A", "IMG_B", "IMG_C"):
        Image.new("RGB", (32, 32)).save(root / "images" / "2026-02-11" / f"{name}.JPG")
    # A: confirmed negative (empty + status). B: empty without confirmation (the IMG_0150 case).
    # C: has objects but status wrongly says negative (contradiction).
    json_io.write_annotations(ann / "IMG_A.json", [], 32, 32, keep_empty=True)
    json_io.write_annotations(ann / "IMG_B.json", [], 32, 32, keep_empty=True)
    json_io.write_annotations(ann / "IMG_C.json",
                              [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
    # Scoped by subject/date: a confirmation belongs to the subject it was made in.
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    replace_image_status_store(root, {
        status_bucket("bud", "2026-02-11"): status_records(
            {"IMG_A.JPG": "negative", "IMG_B.JPG": "unannotated", "IMG_C.JPG": "negative"},
            recorded_by="user:breeder"),
    })
    return root


def _run(root: Path, *, file_layout: bool = False):
    """Run the doctor against ``root``.

    ``file_layout=True`` bound to the file backend on purpose: that caller's fixture wrote
    ``image_status.json``/``region_completeness.json``/``registry.json`` through the storage
    seam bound to the file backend, and the check under test reads the same file layout that
    binding serves, the exact case ``staleness_findings`` reports as invalid once a database
    also holds the root, so the subprocess is pinned to file regardless of whatever backend the
    outer test run selects.
    """
    env = {**os.environ, "TCIP_STORE_BACKEND": "file"} if file_layout else None
    return subprocess.run([PY_EXE, DOCTOR, str(root)], capture_output=True, text=True, env=env)


def test_doctor_flags_the_field_session_bug_family(tmp_path):
    root = _project(tmp_path)
    # registry entry pointing at a pytest temp checkpoint (the leak the field session found)
    reg = root / ".tcip" / "models"
    reg.mkdir(parents=True)
    (reg / "registry.json").write_text(json.dumps({"entries": [
        {"name": "junk", "checkpoint_path": r"C:\Temp\pytest-of-x\model.pt"}]}))

    res = _run(root, file_layout=True)
    assert res.returncode == 2  # errors present
    out = res.stdout
    assert "IMG_B" in out and "not a confirmed negative" in out   # unconfirmed empty -> warn
    assert "contradictory" in out and "IMG_C" in out              # objects + negative -> error
    assert "junk" in out and "test/temp" in out                   # registry pollution -> error
    assert "IMG_A" not in out.replace("IMG_A.JPG is negative", "")  # confirmed negative is clean


def test_doctor_admits_a_confirmed_negative_under_dated_labels_flat_images(tmp_path):
    """A confirmed negative resolves the same way the draw admits it when labels are dated but
    images were never split into date buckets, instead of reading as an unconfirmed empty."""
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)
    ann = root / "annotations" / "2026-02-11"
    ann.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))
    Image.new("RGB", (32, 32)).save(root / "images" / "IMG_A.JPG")
    json_io.write_annotations(ann / "IMG_A.json", [], 32, 32, keep_empty=True)

    ts.bind(FileBackend())
    replace_image_status_store(root, {
        status_bucket("bud", "2026-02-11"): status_records(
            {"IMG_A.JPG": "negative"}, recorded_by="user:breeder"),
    })

    res = _run(root, file_layout=True)

    assert "not a confirmed negative" not in res.stdout


def test_doctor_reports_a_stem_collision_and_completes(tmp_path):
    """A bucket already holding two identities for one stem key refuses at every reader; the
    doctor names it as a finding instead of crashing on the exception."""
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    root = tmp_path / "proj"
    images = root / "images" / "2026-02-11"
    images.mkdir(parents=True)
    ann = root / "annotations" / "2026-02-11"
    ann.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="bloom"),)))
    Image.new("RGB", (32, 32)).save(images / "foo.jpg")
    Image.new("RGB", (32, 32)).save(images / "foo.png")
    json_io.write_annotations(ann / "foo.json", [], 32, 32, keep_empty=True)

    ts.bind(FileBackend())

    res = _run(root, file_layout=True)

    assert res.returncode == 2
    assert "foo.jpg" in res.stdout and "foo.png" in res.stdout
    assert "Traceback" not in res.stderr


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
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))
    Image.new("RGB", (32, 32)).save(root / "images" / "2026-02-11" / "IMG_A.JPG")
    ann_path = ann_dir / "IMG_A.json"
    json_io.write_annotations(
        ann_path, [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)

    grid = {"width": 32, "height": 32, "tile_size": 16, "overlap": 0.0, "cols": 2, "rows": 2}
    cell = next(c for c in reference_cells(32, 32, 16, clamp=True) if c.name == "A1")
    stamped = cell_annotation_digest(json_io.read_annotations(str(ann_path)), "bud", cell)

    bucket = status_bucket("bud", "IMG_A")
    (state_dir / "region_completeness.json").write_text(json.dumps({
        bucket: {"grid": grid, "cells_complete": ["A1"], "attested_by": "user:breeder",
                "attested_at": "t", "stem": "IMG_A", "date": "2026-02-11", "subject": "bud"},
    }))
    (state_dir / "region_completeness_digest.json").write_text(
        json.dumps({bucket: {"A1": stamped}}))

    # The label is edited after attestation: a real staleness scenario, not a fabricated one.
    json_io.write_annotations(
        ann_path, [Annotation(subject="bud", geometry=BBox(1, 1, 20, 20))], 32, 32)

    res = _run(root, file_layout=True)
    assert res.returncode == 2
    assert "region completeness" in res.stdout
    assert "bud" in res.stdout and "A1" in res.stdout


def test_doctor_flags_an_unrecognized_region_completeness_entry(tmp_path):
    """A region-completeness store entry with no {grid, cells_complete} shape would be dropped
    by any merge, the exact state that blocks every attestation write; the doctor mirrors the
    status-store sibling and reports it by count rather than reading the store as clean."""
    from tcip_mcp.dataset_layout import region_completeness_key, status_bucket

    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    ts.replace(region_completeness_key(root),
              {status_bucket("bud", date): {"cells_complete": ["A1"]}},  # no "grid": unrecognized
              expect=ts.Version.ABSENT)

    res = _run(root, file_layout=True)
    assert "1 region-completeness entry is in a shape this reader does not recognize" in res.stdout


def test_doctor_flags_incomplete_source_snapshot(tmp_path):
    """A bespoke run's source snapshot that failed to capture a declared file is
    self-describing (``missing``/``snapshot_errors``); doctor.py surfaces it rather than the
    manifest reading as complete."""
    root = tmp_path / "clean"
    (root / "images" / "d").mkdir(parents=True)
    ann = root / "annotations" / "d"
    ann.mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))
    Image.new("RGB", (32, 32)).save(root / "images" / "d" / "IMG_A.JPG")
    json_io.write_annotations(
        ann / "IMG_A.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9), created_by="user:breeder")], 32, 32)

    manifest_dir = root / ".tcip" / "experiments" / "exp1" / "model_src"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps({
        "files": [], "missing": ["agent_helper.py"], "snapshot_errors": [],
    }))

    res = _run(root)
    assert res.returncode == 1  # warning only, no error
    assert "source snapshot" in res.stdout and "1 missing file" in res.stdout


def test_doctor_clean_project_exits_zero(tmp_path):
    from tcip_mcp.project_record import record_site

    root = tmp_path / "clean"
    (root / "images" / "d").mkdir(parents=True)
    ann = root / "annotations" / "d"
    ann.mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))
    Image.new("RGB", (32, 32)).save(root / "images" / "d" / "IMG_A.JPG")
    json_io.write_annotations(
        ann / "IMG_A.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9), created_by="user:breeder")], 32, 32)
    record_site(str(root), "north orchard")  # a clean project also carries a site record
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
                   ClassRegistry(subjects=(Subject(name="bud"), Subject(name="leaf"))))
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
        label, [Annotation(subject="bud", geometry=BBox(2, 3, 18, 9))], 48, 32)
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    replace_image_status_store(root, {
        status_bucket("bud", date): status_records(
            {"IMG_R.JPG": "negative"}, recorded_by="user:breeder"),
    })

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    contradictions = _lines(res.stdout, "contradictory")
    assert len(contradictions) == 1, res.stdout
    assert str(label.relative_to(root)) in contradictions[0]


def test_a_negative_confirmation_names_only_its_own_subject(tmp_path):
    """A confirmation is scoped to one subject: an image holding leaf annotations and confirmed
    negative for both subjects contradicts the leaf confirmation only, and the bud
    confirmation on the same image stands."""
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (48, 32)).save(image_dir(root, date) / "IMG_S.JPG")
    json_io.write_annotations(
        annotation_path(root, date, "IMG_S"),
        [Annotation(subject="leaf", geometry=BBox(4, 2, 40, 11))], 48, 32)
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    replace_image_status_store(root, {
        status_bucket("bud", date): status_records({"IMG_S.JPG": "negative"}, recorded_by="user:breeder"),
        status_bucket("leaf", date): status_records({"IMG_S.JPG": "negative"}, recorded_by="user:breeder"),
    })

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    contradictions = _lines(res.stdout, "contradictory")
    assert len(contradictions) == 1, res.stdout
    assert "'leaf'" in contradictions[0]
    assert "'bud'" not in contradictions[0]


def test_confirmations_are_matched_on_a_dateless_dataset(tmp_path):
    """A dataset with no capture-date buckets keys its confirmations by subject alone; the
    doctor still pairs a confirmation with the label file it contradicts."""
    root = tmp_path / "flat"
    image_dir(root, None).mkdir(parents=True)
    annotation_dir(root, None).mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))
    Image.new("RGB", (40, 24)).save(image_dir(root, None) / "IMG_F.JPG")
    json_io.write_annotations(
        annotation_path(root, None, "IMG_F"),
        [Annotation(subject="bud", geometry=BBox(3, 1, 20, 9))], 40, 24)
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    replace_image_status_store(root, {
        status_bucket("bud", None): status_records(
            {"IMG_F.JPG": "negative"}, recorded_by="user:breeder"),
    })

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    contradictions = _lines(res.stdout, "contradictory")
    assert len(contradictions) == 1, res.stdout
    assert "'bud'" in contradictions[0]


def test_doctor_flags_a_bare_status_token(tmp_path):
    """A status store entry with no {status, recorded_by, recorded_at} shape is unreadable and
    would be dropped by any merge; the doctor reports it by count rather than staying silent."""
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "IMG_S.JPG")
    json_io.write_annotations(annotation_path(root, date, "IMG_S"), [], 32, 32, keep_empty=True)
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    # A bare token is a shape replace_image_status_store's own writer refuses, so this fixture
    # writes it through the store primitive directly, bound to the file backend the subprocess reads.
    ts.bind(FileBackend())
    ts.replace(image_status_key(root), {status_bucket("bud", date): {"IMG_S.JPG": "unannotated"}},
              expect=ts.Version.ABSENT)

    res = _run(root, file_layout=True)
    assert "1 status entry is in a shape this reader does not recognize" in res.stdout


def test_doctor_flags_a_stale_complete_token(tmp_path):
    """A stored 'complete' whose label file holds no annotation of the confirmed subject is a
    token a human should re-confirm; the doctor reports it and does not rewrite it."""
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "IMG_S.JPG")
    json_io.write_annotations(annotation_path(root, date, "IMG_S"), [], 32, 32, keep_empty=True)
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    replace_image_status_store(root, {
        status_bucket("bud", date): status_records(
            {"IMG_S.JPG": "complete"}, recorded_by="user:breeder"),
    })

    res = _run(root, file_layout=True)
    stale = _lines(res.stdout, "re-confirm")
    assert len(stale) == 1, res.stdout
    assert "bud" in stale[0] and "IMG_S.JPG" in stale[0]


def test_registry_findings_are_read_through_the_registrys_own_entry_shape(tmp_path):
    """Entries written by ModelRegistry are the shape doctor.py reports on, so an entry whose
    checkpoint is gone is named with its own name and path rather than read as nothing."""
    root = _layout_project(tmp_path, "2026-03-04")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    registry = ModelRegistry(str(root))
    paths = {}
    for name, payload in (("currant_bud_detector_v1", b"weights"),
                          ("chestnut_burr_counter_v3", b"other weights")):
        ckpt = ckpt_dir / f"{name}.pt"
        ckpt.write_bytes(payload)
        registry.register_model(name=name, checkpoint_path=str(ckpt), config={}, metrics={},
                                metrics_source=None)
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
    (models / "registry.json").write_text(json.dumps({"entries": [
        {"name": "orchard_detector_v2", "checkpoint_path": ghost},
        {"name": "scratch_detector", "checkpoint_path": scratch},
    ]}))

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    entry_lines = _lines(res.stdout, "registry entry")
    assert len(entry_lines) == 2, res.stdout
    ghost_line = next(ln for ln in entry_lines if "orchard_detector_v2" in ln)
    scratch_line = next(ln for ln in entry_lines if "scratch_detector" in ln)
    assert "checkpoint missing" in ghost_line and "test/temp" not in ghost_line
    assert "test/temp" in scratch_line and "checkpoint missing" not in scratch_line


def test_a_checkpoint_under_a_temp_rooted_project_is_not_pollution(tmp_path):
    """``tmp_path`` itself sits under a temp tree (pytest's own fixture), so a checkpoint
    resolving inside the project is never pollution merely because the project's own location
    carries a temp-tree marker: only a checkpoint the root does not contain is scanned."""
    root = _layout_project(tmp_path, "2026-03-04")
    ckpt_dir = root / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "m.pt").write_bytes(b"weights")
    (ckpt_dir / "registry.json").write_text(json.dumps({"entries": [
        {"name": "m", "checkpoint_path": ".tcip/models/m.pt"}]}))

    res = _run(root, file_layout=True)

    assert "test/temp" not in res.stdout
    assert "checkpoint missing" not in res.stdout


def _dir_outside_any_temp_tree() -> Path:
    """A writable directory whose path carries none of the doctor's temp-tree markers.

    The doctor reads a checkpoint under pytest's tree or the OS temp directory as test pollution,
    so a test wanting the other findings needs a real file elsewhere. The interpreter's temp
    directory qualifies on Linux and not on Windows, where it sits under a Temp segment, so the
    filesystem root is tried after it; the first candidate that is creatable and unmarked wins.
    The leaf is named for this process, so two suites running at once (one per store backend)
    never write and unlink one shared file.
    """
    import os
    import tempfile

    from scripts.doctor import TEMP_TREE_MARKERS

    candidates = [Path(tempfile.gettempdir()), Path(Path.cwd().anchor)]
    for base in candidates:
        target = base / "tcip_no_metrics_source_fixture" / str(os.getpid())
        if any(marker in str(target) for marker in TEMP_TREE_MARKERS):
            continue
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return target
    raise RuntimeError(f"no writable directory outside a temp tree among {candidates}")


def test_image_census_counts_every_capture_the_loaders_admit(tmp_path):
    """The doctor's image census reads the platform's own extension set, so an .npz capture or a
    band-group manifest is matched to its label rather than reported missing."""
    from scripts.doctor import _image_stems

    images = tmp_path / "images" / "2026-03-04"
    images.mkdir(parents=True)
    (images / "plotA_0_0.npz").write_bytes(b"\x00")
    (images / "plotA_0_1.jpg").write_bytes(b"\xff\xd8")

    assert _image_stems(tmp_path) == {"plotA_0_0": "plotA_0_0.npz", "plotA_0_1": "plotA_0_1.jpg"}


def test_the_checkpoint_fixture_directory_is_this_processs_own():
    import os

    target = _dir_outside_any_temp_tree()
    try:
        assert target.name == str(os.getpid())
        assert target.parent.name == "tcip_no_metrics_source_fixture"
    finally:
        target.rmdir()


def test_registry_entry_with_no_metrics_source_is_flagged(tmp_path):
    """A registry entry that predates the metrics_source field is reported, not read as though
    the platform had verified its numbers."""
    root = _layout_project(tmp_path, "2026-03-04")
    ckpt_dir = _dir_outside_any_temp_tree()
    ckpt = ckpt_dir / "model.pt"
    ckpt.write_bytes(b"weights")
    models = root / ".tcip" / "models"
    models.mkdir(parents=True)
    (models / "registry.json").write_text(json.dumps({"entries": [
        {"name": "legacy", "checkpoint_path": str(ckpt), "metrics": {"val_map50": 0.5}}]}))

    try:
        res = _run(root, file_layout=True)
        assert res.returncode == 1, res.stdout
        matches = _lines(res.stdout, "metrics_source")
        assert len(matches) == 1 and "legacy" in matches[0], res.stdout
    finally:
        ckpt.unlink()
        ckpt_dir.rmdir()


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
    # A statement isolates bloom_length from check_trait_spec_statements's own separate finding.
    ts.replace(
        traits.trait_spec_statement_key(traits.trait_spec_statements_scope(root), "bloom_length"),
        {"trait": "bloom_length", "statement_fields": {"delivers": ["bloom_length"]},
         "rationale": "test fixture", "stated_by": "test", "stated_at": "2026-03-04T00:00:00+00:00",
         "relayed_note": "", "confirmed_by": None, "confirmed_at": None,
         "identity_from_request": None, "record_seen": None},
        expect=ts.Version.ABSENT,
    )

    res = _run(root)
    assert res.returncode == 2, res.stdout
    spec_lines = _lines(res.stdout, "trait spec")
    assert len(spec_lines) == 1, res.stdout
    assert "burr_size.json" in spec_lines[0]
    assert "measured_with" in spec_lines[0]
    assert "bloom_length.json" not in res.stdout


def test_doctor_warns_on_a_project_with_no_record(tmp_path):
    """A recordless project is the accepted standing state of one that predates the field: a
    warning, not an error, and the exit code says so."""
    root = _layout_project(tmp_path, "2026-03-04")

    res = _run(root)

    assert res.returncode == 1, res.stdout
    assert "initialize_project" in res.stdout


def test_doctor_errors_on_a_project_whose_record_does_not_decode(tmp_path):
    """A damaged record is a check that could not run, not a clean project: an error, and exit 2."""
    from tcip_mcp.project_record import project_record_key, record_site
    from tests._record_damage_fixtures import damage_record

    root = _layout_project(tmp_path, "2026-03-04")
    record_site(str(root), "north orchard")
    key = project_record_key(str(root))
    # A genuinely undecodable byte string, written under the record's own key, so the finding
    # is the store's own decode error rather than "not a site record".
    damage_record(key, b"{not valid json")

    res = _run(root)

    assert res.returncode == 2, res.stdout
    assert "does not decode" in res.stdout


def test_doctor_flags_an_unreadable_label_behind_a_confirmed_negative(tmp_path):
    """A corrupt label file behind a stored 'negative' is an error-level finding, never a pass:
    the reader raises on it, and the doctor reports it rather than letting the corruption hide
    behind the confirmed-negative status."""
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "IMG_S.JPG")
    annotation_path(root, date, "IMG_S").write_text("not json {][", encoding="utf-8")
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    replace_image_status_store(root, {
        status_bucket("bud", date): status_records(
            {"IMG_S.JPG": "negative"}, recorded_by="user:breeder"),
    })

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    unreadable = _lines(res.stdout, "will not read")
    assert len(unreadable) >= 1, res.stdout
    assert any("IMG_S" in ln for ln in unreadable), res.stdout


def test_doctor_flags_an_image_and_a_label_with_a_reserved_stem(tmp_path):
    """A stem reserved for a prediction bucket's own provenance stamp is excluded from every
    bucket walk, so it is invisible to those readers; the doctor's own ``rglob`` walk still sees
    it and reports it, since data not brought in through ingest can still carry one."""
    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "operating_point.jpg")
    json_io.write_annotations(
        annotation_path(root, date, "operating_point"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32,
    )

    res = _run(root, file_layout=True)
    assert res.returncode == 2, res.stdout
    findings = _lines(res.stdout, "reserved for a prediction bucket")
    assert any("operating_point.jpg" in ln for ln in findings), res.stdout
    assert any("operating_point.json" in ln for ln in findings), res.stdout


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
        [Annotation(subject="bud", geometry=BBox(2, 3, 18, 9), created_by="user:breeder")], 48, 32)
    json_io.write_annotations(
        annotation_path(root, date, "IMG_B"),
        [Annotation(subject="leaf", geometry=BBox(5, 1, 44, 12), created_by="user:breeder")], 48, 32)
    baselines = annotation_dir(root, date) / ".original"
    baselines.mkdir()
    json_io.write_annotations(baselines / "IMG_B.json", [], 48, 32, keep_empty=True)
    json_io.write_annotations(baselines / "IMG_D.json", [], 48, 32, keep_empty=True)
    from tcip_mcp.project_record import record_site

    record_site(str(root), "north orchard")  # a clean project also carries a site record

    res = _run(root)
    assert res.returncode == 0, res.stdout
    census = _lines(res.stdout, "have no label record")
    assert len(census) == 1, res.stdout
    assert "2 of 4 image(s)" in census[0]
    assert "IMG_C" in census[0] and "IMG_D" in census[0]
    assert ".original" not in res.stdout


def test_a_seam_written_confirmation_is_seen_by_check_negatives_and_check_data_quality_alike(
    tmp_path,
):
    """One confirmation written through replace_image_status_store is the single fact both the
    doctor's negatives check and its data-quality check read, on one root under the same
    backend."""
    from scripts import doctor

    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "IMG_S.JPG")
    json_io.write_annotations(annotation_path(root, date, "IMG_S"), [], 32, 32, keep_empty=True)
    replace_image_status_store(root, {
        status_bucket("bud", date): status_records(
            {"IMG_S.JPG": "negative"}, recorded_by="user:breeder"),
    })

    findings: list[tuple[str, str]] = []
    doctor.check_negatives(root, findings)
    assert not any("not a confirmed negative" in msg for _, msg in findings)

    quality_findings: list[tuple[str, str]] = []
    doctor.check_data_quality(root, quality_findings)
    assert quality_findings == []


def test_an_unreadable_status_store_is_its_own_error_not_a_false_negative_sweep(tmp_path, monkeypatch):
    """A status store the backend cannot read must surface as its own refusal finding, never as
    an empty negatives set: falling through with no negatives would report every empty label on
    the dataset as an unconfirmed negative, a false positive sweep hiding the real failure."""
    from scripts import doctor
    from tcip_store import StoreError

    date = "2026-03-04"
    root = _layout_project(tmp_path, date)
    Image.new("RGB", (32, 32)).save(image_dir(root, date) / "IMG_S.JPG")
    json_io.write_annotations(annotation_path(root, date, "IMG_S"), [], 32, 32, keep_empty=True)

    def _raise(*args, **kwargs):
        raise StoreError("boom")

    monkeypatch.setattr("tcip_mcp.dataset_layout.read_image_status_store", _raise)

    findings: list[tuple[str, str]] = []
    doctor.check_data_quality(root, findings)

    assert len(findings) == 1, findings
    level, message = findings[0]
    assert level == "warn"
    assert "will not read" in message and "boom" in message
    assert not any("not a confirmed negative" in msg for _, msg in findings)
