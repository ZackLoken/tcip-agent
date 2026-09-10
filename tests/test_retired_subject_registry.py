"""The retired classes.json: the three registry writers, register_dataset, the two named
absence refusals, the doctor's finding, dataset_scope_of's evidence and the two bundle doors,
all against a dataset still holding the pre-rename document. Each refusal is paired with what
still admits, per CLAUDE.md's rail rule: a bare dataset root with no registry file at all works
at every one of these sites exactly as before the subject-registry rename.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tcip_store as ts

from tcip_mcp import operationalization as op
from tcip_mcp import traits
from tcip_mcp.subject_registry import (
    Subject,
    SubjectRegistry,
    SubjectRegistryUnconformed,
    copy_registry,
    read_registry,
    registry_for_dataset_root,
    replace_registry,
    retired_document,
    write_registry,
)
from tests import _operationalization_fixtures as fx


def _dataset(root: Path, *, with_registry: bool = True) -> Path:
    """One image, one label, and (unless suppressed) the registry that decodes it."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    (root / "images" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "2026-03-04" / "a_1.jpg")
    (root / "annotations" / "2026-03-04").mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(root / "annotations" / "2026-03-04" / "a_1.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
    if with_registry:
        write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name="bud"),)))
    return root


def _retire(root: Path) -> None:
    """Stand in for the old writer: rename the current registry to the retired filename, the
    same on-disk state a dataset that predates the subject-registry rename is in."""
    (root / "subjects.json").rename(root / "classes.json")


# ── the predicate itself ────────────────────────────────────────────────────────────────────


def test_retired_document_is_none_when_nothing_is_there(tmp_path):
    assert retired_document(tmp_path) is None


def test_retired_document_answers_the_path_when_it_decodes(tmp_path):
    _dataset(tmp_path)
    _retire(tmp_path)
    assert retired_document(tmp_path) == tmp_path / "classes.json"


def test_retired_document_answers_present_even_beside_a_fresh_subjects_json(tmp_path):
    """Whether or not subjects.json exists beside it: a stray write must not be able to hide the
    retired document by placing a fresh one next to it."""
    _dataset(tmp_path)
    root = tmp_path
    (root / "classes.json").write_bytes((root / "subjects.json").read_bytes())
    assert retired_document(root) == root / "classes.json"


def test_a_classes_json_that_does_not_decode_is_not_the_retired_document(tmp_path):
    """A third-party file of that name is no claim the platform ever wrote a registry there."""
    (tmp_path / "classes.json").write_text("not a registry", encoding="utf-8")
    assert retired_document(tmp_path) is None


# ── the three writers refuse beside it ──────────────────────────────────────────────────────


def test_write_registry_refuses_beside_the_retired_document(tmp_path):
    _dataset(tmp_path)
    _retire(tmp_path)
    with pytest.raises(SubjectRegistryUnconformed) as exc:
        write_registry(tmp_path / "subjects.json", SubjectRegistry(subjects=(Subject(name="bud"),)))
    assert "classes.json" in str(exc.value)
    assert "tcip rename-subject-registry" in str(exc.value)
    assert not (tmp_path / "subjects.json").exists()


def test_replace_registry_refuses_beside_the_retired_document(tmp_path):
    _dataset(tmp_path)
    _retire(tmp_path)
    with pytest.raises(SubjectRegistryUnconformed) as exc:
        replace_registry(
            tmp_path / "subjects.json", SubjectRegistry(subjects=(Subject(name="bud"),)),
            expect=None)
    assert "tcip rename-subject-registry" in str(exc.value)
    assert not (tmp_path / "subjects.json").exists()


def test_copy_registry_refuses_at_a_destination_carrying_the_retired_document(tmp_path):
    source = _dataset(tmp_path / "source")
    dest = _dataset(tmp_path / "dest", with_registry=False)
    (dest / "classes.json").write_bytes((source / "subjects.json").read_bytes())
    with pytest.raises(SubjectRegistryUnconformed) as exc:
        copy_registry(source / "subjects.json", dest / "subjects.json")
    assert "tcip rename-subject-registry" in str(exc.value)
    assert not (dest / "subjects.json").exists()


def test_write_registry_admits_valid_work_with_no_registry_at_all(tmp_path):
    """The rail's other half: nothing retired, nothing to refuse."""
    write_registry(tmp_path / "subjects.json", SubjectRegistry(subjects=(Subject(name="bud"),)))
    assert read_registry(tmp_path / "subjects.json").subjects[0].name == "bud"


# ── write_subject_registry's tool door answers {"error": ...} ──────────────────────────────


def test_write_subject_registry_tool_answers_error_beside_the_retired_document(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    _dataset(tmp_path)
    _retire(tmp_path)
    result = write_subject_registry(str(tmp_path), {"bud": {}})
    assert "error" in result
    assert "classes.json" in result["error"]


# ── register_dataset refuses, naming the path and the command ──────────────────────────────


def test_register_dataset_refuses_beside_the_retired_document(tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    root = _dataset(tmp_path / "proj")
    _retire(root)
    result = register_dataset(str(root), crop="walnut")
    assert "error" in result
    assert str(root) in result["error"]
    assert "tcip rename-subject-registry" in result["error"]


def test_register_dataset_admits_valid_work_with_no_registry_at_all(tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    root = _dataset(tmp_path / "proj", with_registry=False)
    result = register_dataset(str(root), crop="walnut")
    assert "error" not in result


# ── the named absence refusals ──────────────────────────────────────────────────────────────


def test_resolve_registry_id_map_names_the_retired_file_for_attribute_classification(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map

    root = _dataset(tmp_path)
    _retire(root)
    with pytest.raises(ValueError) as exc:
        resolve_registry_id_map(root / "annotations" / "2026-03-04", "bud", "condition")
    assert "classes.json" in str(exc.value)
    assert "tcip rename-subject-registry" in str(exc.value)


def test_resolve_statement_registry_names_the_retired_file(tmp_path):
    from tcip_mcp.operationalization import resolve_statement_registry

    root = _dataset(tmp_path)
    _retire(root)
    with pytest.raises(ValueError) as exc:
        resolve_statement_registry(str(root), str(root))
    assert "classes.json" in str(exc.value)
    assert "tcip rename-subject-registry" in str(exc.value)


def test_unmapped_classified_run_names_the_retired_file(tmp_path):
    from tcip_mcp.tools.inference_tools import unmapped_classified_run

    root = _dataset(tmp_path)
    _retire(root)
    msg = unmapped_classified_run(
        {"subject": "bud", "attribute": "condition"}, None, images_dir=str(root / "images" / "2026-03-04"))
    assert msg is not None
    assert "classes.json" in msg
    assert "tcip rename-subject-registry" in msg


# ── the doctor reports it ───────────────────────────────────────────────────────────────────


def test_doctor_reports_the_retired_document(tmp_path):
    from tcip_mcp.cli.doctor import check_retired_subject_registry

    root = _dataset(tmp_path / "proj")
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    _retire(root)
    findings: list = []
    check_retired_subject_registry(root, findings)
    assert findings
    level, message = findings[0]
    assert level == "warn"
    assert str(root) in message
    assert "tcip rename-subject-registry" in message


def test_doctor_reports_a_stray_undecodable_classes_json_separately(tmp_path):
    from tcip_mcp.cli.doctor import check_retired_subject_registry

    root = _dataset(tmp_path / "proj", with_registry=False)
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    (root / "classes.json").write_text("not a registry", encoding="utf-8")
    findings: list = []
    check_retired_subject_registry(root, findings)
    assert findings
    level, message = findings[0]
    assert level == "warn"
    assert "stray file" in message


def test_doctor_warns_rather_than_crashes_when_roots_cannot_be_enumerated(tmp_path):
    """A project root holding an experiment's loose record files with no store.db, under the
    database backend, cannot be enumerated through the seam; the check reports it and stops
    rather than raising the StoreError out through the whole doctor run.

    Bound to the sqlite backend explicitly regardless of the ambient run: the refusal this
    proves is specific to a database backend meeting loose files with no database, the exact
    friction the file backend never has."""
    import json

    import tcip_store as ts
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_store.store import _backend

    from tcip_mcp.cli.doctor import check_retired_subject_registry

    root = _dataset(tmp_path / "proj")
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    manifest_dir = root / ".tcip" / "experiments" / "exp1" / "model_src"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"files": [], "missing": [], "snapshot_errors": []}), encoding="utf-8")

    previous = _backend()
    backend = SqliteBackend()
    ts.bind(backend)
    try:
        findings: list = []
        check_retired_subject_registry(root, findings)
    finally:
        ts.bind(previous)
        backend.close()
    assert findings
    level, message = findings[0]
    assert level == "warn"
    assert "could not enumerate" in message


def test_doctor_reports_nothing_with_no_registry_at_all(tmp_path):
    from tcip_mcp.cli.doctor import check_retired_subject_registry

    root = _dataset(tmp_path / "proj", with_registry=False)
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    findings: list = []
    check_retired_subject_registry(root, findings)
    assert findings == []


# ── draw_splits' own precondition over each materialized destination ───────────────────────


def _splittable_dataset(root: Path) -> Path:
    """Enough foreground groups (four distinct stems) for a materializing draw_splits call to
    succeed at its default minimums (one train, one val, two calibration)."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject, write_registry

    images_dir = root / "images" / "2026-03-04"
    labels_dir = root / "annotations" / "2026-03-04"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name="bud"),)))
    for i, stem in enumerate(("a", "b", "c", "d")):
        Image.new("RGB", (64, 48), (10, 20, 30)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9)) for _ in range(i + 1)], 64, 48)
    return root


def test_draw_splits_refuses_naming_a_retired_document_at_a_split_destination_nothing_written(tmp_path):
    from tcip_mcp.tools.data_tools import draw_splits

    root = _splittable_dataset(tmp_path / "ds")
    out = tmp_path / "splits"
    (out / "train").mkdir(parents=True)
    (out / "train" / "classes.json").write_bytes((root / "subjects.json").read_bytes())

    result = draw_splits(
        str(root), output_path=str(out), materialize=True, subject="bud",
        train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=3,
    )
    assert "error" in result
    assert "classes.json" in result["error"]
    assert "tcip rename-subject-registry" in result["error"]
    # Nothing written: no manifest, no materialized image/label tree.
    assert not (out / "split_manifest.json").exists()
    assert not (out / "train" / "images").exists()
    assert not (out / "val").exists()


def test_draw_splits_materializes_fine_with_no_registry_at_any_destination(tmp_path):
    """The rail's other half: nothing retired at any destination, nothing to refuse."""
    from tcip_mcp.tools.data_tools import draw_splits

    root = _splittable_dataset(tmp_path / "ds")
    out = tmp_path / "splits"
    result = draw_splits(
        str(root), output_path=str(out), materialize=True, subject="bud",
        train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=3,
    )
    assert "error" not in result, result
    assert (out / "train" / "images").is_dir() or (out / "val" / "images").is_dir()


# ── dataset_scope_of still resolves the root ────────────────────────────────────────────────


def test_dataset_scope_of_resolves_a_root_holding_only_the_retired_document(tmp_path):
    from tcip_mcp.audit import dataset_scope_of

    root = _dataset(tmp_path / "proj")
    _retire(root)
    assert dataset_scope_of(str(root)) == root.resolve()


# ── the two bundle doors ────────────────────────────────────────────────────────────────────


def test_archive_project_refuses_naming_the_retired_document(tmp_path):
    from tcip_mcp.tools.project_tools import archive_project

    root = _dataset(tmp_path / "proj")
    _retire(root)
    result = archive_project(str(root), str(tmp_path / "bundle.zip"))
    assert "error" in result
    assert "classes.json" in result["error"]
    assert "tcip rename-subject-registry" in result["error"]
    assert not (tmp_path / "bundle.zip").exists()


def test_archive_project_admits_valid_work_with_no_registry_at_all(tmp_path):
    from tcip_mcp.tools.project_tools import archive_project

    root = _dataset(tmp_path / "proj", with_registry=False)
    result = archive_project(str(root), str(tmp_path / "bundle.zip"))
    assert "error" not in result
    assert (tmp_path / "bundle.zip").is_file()


def test_import_project_refuses_naming_the_retired_document_and_the_hand_recipe(tmp_path):
    import zipfile

    from tcip_mcp.tools.project_tools import archive_project, import_project

    root = _dataset(tmp_path / "proj")
    zip_path = tmp_path / "bundle.zip"
    assert "error" not in archive_project(str(root), str(zip_path))
    # Simulate an archive made before the rename: the bundled member is still named classes.json.
    rebuilt = tmp_path / "rebuilt.zip"
    with zipfile.ZipFile(str(zip_path)) as src, zipfile.ZipFile(str(rebuilt), "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            name = "classes.json" if item.filename == "subjects.json" else item.filename
            dst.writestr(name, data)

    dest = tmp_path / "dest"
    result = import_project(str(rebuilt), str(dest))
    assert "error" in result
    assert "classes.json" in result["error"]
    assert "rename it to subjects.json" in result["error"]
    assert not dest.exists() or not any(dest.iterdir())


# ── the GUI save route answers 400 ──────────────────────────────────────────────────────────


def test_save_subjects_route_answers_400_beside_the_retired_document(tmp_path):
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    root = _dataset(tmp_path)
    _retire(root)
    client = TestClient(app, base_url="http://127.0.0.1")

    resp = client.post("/api/subjects/save", json={
        "project_root": str(root), "dataset_root": str(root),
        "subjects": {"bud": {}}, "version": None})

    assert resp.status_code == 400
    assert "classes.json" in resp.text
    assert "tcip rename-subject-registry" in resp.text
    assert not (root / "subjects.json").exists()


# ── materialize_review_dataset answers {"error": ...} naming the conform ───────────────────


def _seed_classified_verdict(state_dir: Path, *, bucket: str) -> None:
    """One accepted classified call: a classified review's own verdict shape, whose class_name
    is the confirmed value, never the object's subject."""
    from tcip_annotation.review_engine import ReviewEngine

    state = {"verdicts": {
        (bucket, "imgA.png"): {"img_status": "completed", "detections": [
            {"action": "accepted", "class_name": "healthy",
             "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
    }}
    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update(state)
    engine.save_review_state()


def test_materialize_review_dataset_answers_error_beside_the_retired_document(tmp_path):
    """A classified scope's own registry copy (materialize.py's
    _copy_source_registry_for_classified_scope, through copy_registry) refuses at a destination
    already carrying the retired document, before anything else is written, and
    materialize_review_dataset answers {"error": ...} naming the conform."""
    from PIL import Image

    from tcip_mcp.pipelines.resolution import write_sidecar
    from tcip_mcp.prediction_buckets import review_state_dir_of
    from tcip_mcp.tools.feedback_tools import materialize_review_dataset

    bucket = "predictions/classifier/2026-03-05"
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdict(review_state_dir_of(dataset_root), bucket=bucket)
    write_sidecar(dataset_root / bucket, {"id_map": {"healthy": 0, "diseased": 1},
                                          "subject": "leaf", "attribute": "condition"})

    source_root = tmp_path / "source"
    (source_root / "images").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(source_root / "images" / "imgA.png")
    write_registry(source_root / "subjects.json", SubjectRegistry(subjects=(Subject(name="leaf"),)))

    out = tmp_path / "out"
    out.mkdir()
    write_registry(out / "subjects.json", SubjectRegistry(subjects=(Subject(name="leaf"),)))
    _retire(out)

    result = materialize_review_dataset(
        str(dataset_root), str(source_root / "images"), str(out), bucket=bucket)

    assert "error" in result
    assert "classes.json" in result["error"]
    assert "tcip rename-subject-registry" in result["error"]


# ── the absence answers, with only the retired document on the root ────────────────────────


def test_absence_answers_at_every_reader_with_only_the_retired_document_present(tmp_path):
    """A root holding only the retired classes.json answers exactly as one holding no registry
    at all, at every reader that does not itself refuse: list_subjects, registry_for_dataset_root,
    the load route's draft, the fingerprint's empty registry term, and resolve_decode_id_map's
    attribute-scope precondition (the narrowest function under run_inference that carries this
    answer: an attribute-scoped run with no subjects.json returns id_map=None)."""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from tcip_mcp.dataset_layout import list_subjects
    from tcip_mcp.pipelines.data.dataset_fingerprint import _registry_term
    from tcip_mcp.tools.inference_tools import resolve_decode_id_map
    from tcip_web.app import app

    root = _dataset(tmp_path)
    _retire(root)

    assert list_subjects(root) == []
    assert registry_for_dataset_root(root) is None
    assert _registry_term(root) == ""

    predictor = SimpleNamespace(config={"data": {"subject": "bud", "attribute": "condition"}})
    assert resolve_decode_id_map(predictor, str(root / "images" / "2026-03-04")) is None

    client = TestClient(app, base_url="http://127.0.0.1")
    resp = client.get("/api/subjects/load", params={
        "project_root": str(root), "dataset_root": str(root),
        "annotations_dir": str(root / "annotations" / "2026-03-04")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] is None
    assert set(body["subjects"]) == {"bud"}  # drafted from the labels, never from classes.json


# ── the transition: a statement or confirmation still carrying the old key ─────────────────


def test_a_trait_spec_statement_rewritten_to_the_old_key_is_stale_and_the_state_door_refuses(
    tmp_path,
):
    """A trait_spec_statements record confirmed through the real producers, its snapshot then
    rewritten on disk to carry positive_class_name instead of positive_value (standing in for a
    statement confirmed before the subject-registry rename), reads stale against the live spec
    (which still carries positive_value), and state_operationalization refuses naming it."""
    root = tmp_path / "proj"
    fx.write_spec(root, fx.CROSSING_SPEC)
    fx.seed_positive_class(root, "flower", fx.CROSSING_SPEC.positive_value)
    fx.confirm_spec_statement(root, fx.CROSSING_TRAIT)

    key = traits.trait_spec_statement_key(
        traits.trait_spec_statements_scope(root), fx.CROSSING_TRAIT)
    versioned = ts.read_versioned(key)
    fields = dict(versioned.value["statement_fields"])
    fields["positive_class_name"] = fields.pop("positive_value")
    ts.replace(key, {**versioned.value, "statement_fields": fields}, expect=versioned.version)

    spec = traits.get_trait_for(fx.CROSSING_TRAIT, root)
    statement = ts.read_versioned(key).value
    assert traits.trait_spec_statement_stale(spec, statement)

    with pytest.raises(traits.TraitSpecUnconfirmed, match="no longer matches"):
        op.state_operationalization(
            root, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES,
            statement="s", mechanism="m", measured_subject="flower",
            delivered_phenotypes=list(fx.CROSSING_SPEC.delivers),
            registry=registry_for_dataset_root(root),
        )


def test_confirmed_fields_rewritten_to_the_old_key_reads_state_3_and_passes_after_reconfirmation(
    tmp_path,
):
    """A trait_operationalizations record confirmed through the real producers, its
    confirmed_fields then rewritten on disk to carry positive_class_name instead of
    positive_value, reads through _moved_fields as positive_value moved from None to the live
    value (state 3); confirm_trait_operationalization admits valid work, replacing
    confirmed_fields whole under the current key, and the check passes again."""
    root = tmp_path / "proj"
    fx.write_spec(root, fx.CROSSING_SPEC)
    fx.seed_positive_class(root, "flower", fx.CROSSING_SPEC.positive_value)
    record = fx.state_crossing(root)
    confirmed = fx.confirm(root, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)

    key = op.operationalization_key(
        op.operationalizations_scope(root), fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    versioned = ts.read_versioned(key)
    document = dict(versioned.value)
    confirmed_fields = dict(document["confirmed_fields"])
    confirmed_fields["positive_class_name"] = confirmed_fields.pop("positive_value")
    document["confirmed_fields"] = confirmed_fields
    ts.replace(key, document, expect=versioned.version)

    spec, stored, _specs_dir = op.resolve_trait_and_record(
        fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, project_root=root)
    check = op.check_operationalization(
        spec, stored, op.STATE_CROSSING_DATES, registry=registry_for_dataset_root(root))
    assert check.state == 3
    assert check.superseded[0]["field"] == "positive_value"

    fx.confirm(root, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, confirmed)

    spec, stored, _specs_dir = op.resolve_trait_and_record(
        fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, project_root=root)
    passed = op.check_operationalization(
        spec, stored, op.STATE_CROSSING_DATES, registry=registry_for_dataset_root(root))
    assert passed.state is None
    assert "positive_value" in stored.value["confirmed_fields"]
    assert "positive_class_name" not in stored.value["confirmed_fields"]
