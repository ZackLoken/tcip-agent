"""``tcip rename-subject-registry``: the conform for the subject-registry rename.

Renames a dataset root's retired ``classes.json`` back to ``subjects.json``, and stamps a
``trait_specs`` record still carrying ``positive_class_name`` to ``positive_value`` with
``schema_version: 2``. Named over a project root, it walks every root
``store_catalogue.project_roots`` answers plus each ``SPLIT_NAMES`` subdirectory under a
``SPLITS``-layout root; named over a dataset or derived root directly, it runs the registry unit
alone, in place.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tcip_store as ts
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import traits
from tcip_mcp.cli import rename_subject_registry as cli
from tcip_mcp.dataset_layout import record_image_statuses, status_bucket
from tcip_mcp.subject_registry import Attribute, Subject, SubjectRegistry, write_registry
from tcip_mcp.tools.data_tools import draw_splits
from tcip_mcp.tools.project_tools import register_dataset

DATE = "2-11-26"
SUBJECT = "bud"
CROP = "test_crop"


def _make_dataset(root: Path) -> None:
    (root / "images" / DATE).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(root / "images" / DATE / "img_000.jpg")
    (root / "annotations" / DATE).mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(root / "annotations" / DATE / "img_000.json"),
        [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 9, 9))], 32, 32)
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name=SUBJECT),)))


def _retire(root: Path) -> None:
    """Rename a written registry back to the pre-rename ``classes.json``, standing in for the
    old writer: the only way a project's on-disk state can carry the retired name today."""
    (root / "classes.json").write_bytes((root / "subjects.json").read_bytes())
    (root / "subjects.json").unlink()


# ── the registry unit, named directly over a bare dataset root ──────────────────


def test_a_bare_dataset_root_named_directly_is_renamed(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _make_dataset(root)
    _retire(root)

    outcomes, refused = cli.process_root(root, plan=False)

    assert not refused, outcomes
    assert (root / "subjects.json").is_file()
    assert not (root / "classes.json").exists()
    entries = list(ts.read_log(_audit_key(root)).records)
    assert any(e["tool"] == cli.TOOL_NAME for e in entries)


def _audit_key(root: Path):
    from tcip_mcp.audit import audit_log_key

    return audit_log_key(root)


def test_plan_writes_nothing_and_is_refusal_worthy_when_a_change_is_due(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _make_dataset(root)
    _retire(root)

    outcomes, refused = cli.process_root(root, plan=True)

    assert refused
    assert (root / "classes.json").is_file()
    assert not (root / "subjects.json").exists()


def test_both_present_and_identical_removes_the_stale_copy(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _make_dataset(root)
    (root / "classes.json").write_bytes((root / "subjects.json").read_bytes())

    outcomes, refused = cli.process_root(root, plan=False)

    assert not refused, outcomes
    assert (root / "subjects.json").is_file()
    assert not (root / "classes.json").exists()


def test_both_present_and_different_refuses_and_moves_nothing(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _make_dataset(root)
    (root / "classes.json").write_text('{"leaf": {"description": ""}}', encoding="utf-8")

    outcomes, refused = cli.process_root(root, plan=False)

    assert refused
    assert (root / "subjects.json").is_file()
    assert (root / "classes.json").is_file()


def test_a_stray_non_registry_file_named_classes_json_is_reported_and_left(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    (root / "classes.json").write_text("not a registry", encoding="utf-8")

    outcomes, refused = cli.process_root(root, plan=False)

    assert refused
    assert (root / "classes.json").is_file()
    assert not (root / "subjects.json").exists()
    assert any("stray file" in line for line in outcomes)


def test_a_bare_directory_with_no_registry_under_either_name_is_refused_when_named_directly(
    tmp_path: Path,
) -> None:
    """The direct-root form is scoped to a tree carrying a registry under either name; a
    directory with neither, named directly, is indistinguishable from any other non-project,
    non-registry directory and is refused the same way. A dataset with no registry at all is
    conformed for free through the project form instead, since the registry unit's own third
    case (neither present) is a no-op there."""
    root = tmp_path / "ds"
    _make_dataset(root)
    (root / "subjects.json").unlink()

    outcomes, refused = cli.process_root(root, plan=False)

    assert refused
    assert not (root / "subjects.json").exists() and not (root / "classes.json").exists()


def test_the_project_form_leaves_a_registered_dataset_with_no_registry_at_all_alone(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    dataset_root = project_root / "dataset"
    _make_dataset(dataset_root)
    reg = register_dataset(str(dataset_root), crop=CROP, project_root=str(project_root))
    assert "error" not in reg, reg
    (dataset_root / "subjects.json").unlink()

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert not refused, outcomes
    assert not (dataset_root / "subjects.json").exists()
    assert not (dataset_root / "classes.json").exists()


def test_a_directory_that_is_neither_a_project_nor_a_registry_root_is_refused_by_name(
    tmp_path: Path,
) -> None:
    root = tmp_path / "nothing"
    root.mkdir()

    outcomes, refused = cli.process_root(root, plan=False)

    assert refused
    assert any(str(root) in line and "refused" in line for line in outcomes)


# ── the registry unit, named directly over a split tree (out_dir/split_name) ────

NEGATIVE_STEM = "plotF_0_0"
POPULATED_STEMS = ("plotA_0_0", "plotB_0_0", "plotC_0_0", "plotD_0_0", "plotE_0_0")


def _dataset_with_one_confirmed_negative(root: Path) -> Path:
    """Six sources: five carrying annotations, one an empty label a human confirmed negative for
    ``bud``. Materializing a split over this dataset copies the registry into whichever split
    holds the negative image (:func:`data_tools._apply_negative_carry`), the only way a
    materializing ``draw_splits`` call places ``subjects.json`` into a split tree."""
    images_dir = root / "images" / DATE
    labels_dir = root / "annotations" / DATE
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name=SUBJECT),)))
    for i, stem in enumerate(POPULATED_STEMS):
        Image.new("RGB", (96, 64), (90, 120, 60)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject=SUBJECT, geometry=BBox(5 + 3 * k, 7, 25 + 3 * k, 51))
             for k in range(i + 1)],
            96, 64,
        )
    Image.new("RGB", (96, 64), (90, 120, 60)).save(images_dir / f"{NEGATIVE_STEM}.jpg")
    json_io.write_annotations(labels_dir / f"{NEGATIVE_STEM}.json", [], 96, 64, keep_empty=True)
    record_image_statuses(root, status_bucket(SUBJECT, DATE), {f"{NEGATIVE_STEM}.jpg": "negative"},
                          recorded_by="user:breeder")
    return root


def _split_holding_the_negative(out: Path) -> Path:
    holders = [out / name for name in ("train", "val", "calibration")
              if (out / name / "subjects.json").is_file()]
    assert len(holders) == 1, f"expected exactly one split to carry a registry copy: {holders}"
    return holders[0]


def test_a_split_trees_own_out_dir_named_directly_is_renamed(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _dataset_with_one_confirmed_negative(root)
    out = tmp_path / "splits"
    result = draw_splits(str(root), output_path=str(out), materialize=True, subject=SUBJECT,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25, seed=3)
    assert "error" not in result, result
    split_root = _split_holding_the_negative(out)
    _retire(split_root)

    outcomes, refused = cli.process_root(split_root, plan=False)

    assert not refused, outcomes
    assert (split_root / "subjects.json").is_file()
    assert not (split_root / "classes.json").exists()


# ── the registry unit, named directly over a curated tree with no experiment_id ─

CLASSIFIED_SUBJECT = "leaf"
CLASSIFIED_ATTRIBUTE = "condition"
CLASSIFIED_BUCKET = "predictions/classifier/2026-03-05"


def _seed_classified_verdict(state_dir: Path) -> None:
    """One accepted classified call: a classified review's own verdict shape, whose class_name
    is the confirmed value, never the object's subject."""
    from tcip_annotation.review_engine import ReviewEngine

    state = {"verdicts": {
        (CLASSIFIED_BUCKET, "imgA.png"): {"img_status": "completed", "detections": [
            {"action": "accepted", "class_name": "healthy",
             "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
    }}
    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update(state)
    engine.save_review_state()


def test_a_curated_tree_materialized_without_an_experiment_id_is_conformed_when_named_directly(
    tmp_path: Path,
) -> None:
    """``materialize_review_dataset`` called with no ``experiment_id`` records no
    ``curated_dataset`` artifact, so ``store_catalogue.project_roots`` cannot reach the tree it
    wrote; the conform's direct-root form is the only way to it. Coverage of the direct-root
    form over a real curated tree, not a guard."""
    from tcip_mcp.pipelines.resolution import write_sidecar
    from tcip_mcp.prediction_buckets import review_state_dir_of
    from tcip_mcp.tools.feedback_tools import materialize_review_dataset

    dataset_root = tmp_path / "dataset"
    _seed_classified_verdict(review_state_dir_of(dataset_root))
    write_sidecar(dataset_root / CLASSIFIED_BUCKET, {
        "id_map": {"healthy": 0, "diseased": 1},
        "subject": CLASSIFIED_SUBJECT, "attribute": CLASSIFIED_ATTRIBUTE,
    })
    source_root = tmp_path / "source_dataset"
    (source_root / "images").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(source_root / "images" / "imgA.png")
    write_registry(source_root / "subjects.json", SubjectRegistry(subjects=(
        Subject(name=CLASSIFIED_SUBJECT, attributes=(
            Attribute(name=CLASSIFIED_ATTRIBUTE, type="categorical",
                     values=("healthy", "diseased")),)),)))
    out = tmp_path / "out"

    r = materialize_review_dataset(
        str(dataset_root), str(source_root / "images"), str(out), bucket=CLASSIFIED_BUCKET)
    assert "error" not in r, r
    assert (out / "subjects.json").is_file()
    _retire(out)

    outcomes, refused = cli.process_root(out, plan=False)

    assert not refused, outcomes
    assert (out / "subjects.json").is_file()
    assert not (out / "classes.json").exists()


# ── the project form: walking a registered dataset root ─────────────────────────


def test_the_project_form_conforms_a_registered_dataset_root(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    dataset_root = project_root / "dataset"
    _make_dataset(dataset_root)
    reg = register_dataset(str(dataset_root), crop=CROP, project_root=str(project_root))
    assert "error" not in reg, reg
    _retire(dataset_root)

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert not refused, outcomes
    assert (dataset_root / "subjects.json").is_file()
    assert not (dataset_root / "classes.json").exists()


def test_an_already_conformed_project_exits_0_with_every_unit_unchanged(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    dataset_root = project_root / "dataset"
    _make_dataset(dataset_root)
    reg = register_dataset(str(dataset_root), crop=CROP, project_root=str(project_root))
    assert "error" not in reg, reg

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert not refused, outcomes
    summary = [line for line in outcomes if line.startswith(f"{project_root}:")][-1]
    assert "0 unit(s) changed" in summary


def test_the_project_form_walks_each_split_names_subdirectory_under_a_splits_root(
    tmp_path: Path,
) -> None:
    """A run bound to an existing split manifest (``persist_split_manifest``'s own
    ``manifest_binding``) makes the manifest directory a ``SPLITS``-layout root
    ``store_catalogue.project_roots`` reports; the project form then walks each of its
    ``SPLIT_NAMES`` subdirectories. Both records are written directly here, standing in for
    ``create_experiment``/``persist_split_manifest``, since only the fields ``project_roots``
    itself reads (a status record naming the experiment, ``manifest_binding.manifest_dir``)
    matter to the walk, not the run that would have produced them; both are addressed with
    ``root=project_root`` explicitly, since the autouse platform-root pin this test otherwise
    inherits is a different directory than the project root the walk is asked to read."""
    from tcip_mcp.experiments import split_key, status_key

    project_root = tmp_path / "proj"
    project_root.mkdir()
    dataset_root = project_root / "dataset"
    _dataset_with_one_confirmed_negative(dataset_root)
    out = project_root / "splits"
    result = draw_splits(str(dataset_root), output_path=str(out), materialize=True,
                         subject=SUBJECT, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25,
                         seed=3)
    assert "error" not in result, result
    split_root = _split_holding_the_negative(out)
    _retire(split_root)

    ts.replace(
        status_key("exp-a", root=project_root),
        {"state": "created", "created": None, "started": None, "ended": None},
        expect=ts.Version.ABSENT,
    )
    ts.replace(
        split_key("exp-a", root=project_root),
        {"manifest_binding": {"manifest_dir": str(out)}},
        expect=ts.Version.ABSENT,
    )

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert not refused, outcomes
    assert (split_root / "subjects.json").is_file()
    assert not (split_root / "classes.json").exists()


# ── the trait spec unit ──────────────────────────────────────────────────────────


def _seed_raw_trait_spec(project_root: Path, document: dict) -> "ts.Key":
    """A raw store write, standing in for a record written before the subject-registry rename:
    the only way a project's own state carries a document ``trait_spec_unconformed`` refuses."""
    specs_dir = traits.trait_specs_dir(str(project_root))
    key = traits.trait_spec_key(specs_dir, document["name"])
    ts.replace(key, document, expect=ts.Version.ABSENT)
    return key


def test_the_trait_spec_writes_compare_and_set_against_the_version_the_scan_itself_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The write's own ``expect`` is the version the scan already read with the document, not a
    version read fresh at write time: a concurrent write landing on the same record between the
    scan's read and the conform's own write is caught as a version conflict, rather than silently
    overwritten with content built from the document the scan read before the concurrent write."""
    project_root = tmp_path / "proj"
    (project_root / ".tcip").mkdir(parents=True)
    key = _seed_raw_trait_spec(project_root, {
        "name": "leaf", "delivers": ["leaf_length"], "positive_class_name": "open",
        "schema_version": 1,
    })

    real_read_versioned = ts.read_versioned

    def _read_then_race(k, *args, **kwargs):
        result = real_read_versioned(k, *args, **kwargs)
        if k == key:
            ts.replace(k, {**result.value, "notes": "moved by a concurrent writer"},
                      expect=result.version)
        return result

    monkeypatch.setattr(cli.ts, "read_versioned", _read_then_race)

    with pytest.raises(ts.VersionConflict):
        cli.process_root(project_root, plan=False)

    document = ts.read_versioned(key).value
    assert document["notes"] == "moved by a concurrent writer"


def test_a_version_1_trait_spec_is_rewritten_to_version_2_with_the_key_renamed(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "proj"
    (project_root / ".tcip").mkdir(parents=True)
    key = _seed_raw_trait_spec(project_root, {
        "name": "leaf", "delivers": ["leaf_length"], "positive_class_name": "open",
        "schema_version": 1,
    })
    before_version = ts.read_versioned(key).version

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert not refused, outcomes
    document = ts.read_versioned(key).value
    assert document["positive_value"] == "open"
    assert "positive_class_name" not in document
    assert document["schema_version"] == traits.TRAIT_SPEC_SCHEMA_VERSION
    assert ts.read_versioned(key).version != before_version
    entries = list(ts.read_log(_audit_key(project_root)).records)
    assert any(e["tool"] == cli.TOOL_NAME and e.get("arguments", {}).get("trait") == "leaf"
              for e in entries)


def test_a_stamped_2_trait_spec_still_carrying_the_old_key_is_rewritten(tmp_path: Path) -> None:
    """A record already stamped ``schema_version: 2`` but still carrying ``positive_class_name``
    (never written by the current encoder, but not ruled out by ``trait_spec_unconformed``, whose
    reason is about the stamp alone) is rewritten to ``positive_value`` all the same: the key
    check runs independently of the version reason. Coverage of a case the conform already
    holds, not a guard."""
    project_root = tmp_path / "proj"
    (project_root / ".tcip").mkdir(parents=True)
    key = _seed_raw_trait_spec(project_root, {
        "name": "leaf", "delivers": ["leaf_length"], "positive_class_name": "open",
        "schema_version": traits.TRAIT_SPEC_SCHEMA_VERSION,
    })
    before_version = ts.read_versioned(key).version

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert not refused, outcomes
    document = ts.read_versioned(key).value
    assert document["positive_value"] == "open"
    assert "positive_class_name" not in document
    assert document["schema_version"] == traits.TRAIT_SPEC_SCHEMA_VERSION
    assert ts.read_versioned(key).version != before_version


def test_an_unstamped_trait_spec_with_no_positive_class_name_is_still_conformed(
    tmp_path: Path,
) -> None:
    """A record with no ``schema_version`` and no ``positive_class_name`` (nothing to rename, but
    still unconformed) is stamped, not left: :func:`traits.trait_spec_unconformed` answers a
    reason for it regardless of whether the renamed field is present."""
    project_root = tmp_path / "proj"
    (project_root / ".tcip").mkdir(parents=True)
    key = _seed_raw_trait_spec(
        project_root, {"name": "leaf", "delivers": ["leaf_length"]})

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert not refused, outcomes
    document = ts.read_versioned(key).value
    assert document["schema_version"] == traits.TRAIT_SPEC_SCHEMA_VERSION


def test_a_trait_spec_carrying_both_keys_is_refused_by_name(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    (project_root / ".tcip").mkdir(parents=True)
    key = _seed_raw_trait_spec(project_root, {
        "name": "leaf", "delivers": ["leaf_length"], "positive_class_name": "open",
        "positive_value": "open",
    })

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert refused
    document = ts.read_versioned(key).value
    assert "positive_class_name" in document and "positive_value" in document


def test_an_undecodable_trait_spec_record_is_reported_and_left(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    (project_root / ".tcip").mkdir(parents=True)
    specs_dir = traits.trait_specs_dir(str(project_root))
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    path = FileBackend().path_for(traits.trait_spec_key(specs_dir, "leaf"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {", encoding="utf-8")

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert refused
    assert path.read_text(encoding="utf-8") == "not valid json {"
    assert any("will not decode" in line for line in outcomes)


def test_a_stamped_2_trait_spec_is_left_alone(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    (project_root / ".tcip").mkdir(parents=True)
    key = _seed_raw_trait_spec(project_root, {
        "name": "leaf", "delivers": ["leaf_length"], "positive_value": "open",
        "schema_version": traits.TRAIT_SPEC_SCHEMA_VERSION,
    })
    before = ts.read_versioned(key)

    outcomes, refused = cli.process_root(project_root, plan=False)

    assert not refused, outcomes
    after = ts.read_versioned(key)
    assert after.version == before.version
    assert after.value == before.value


def test_plan_over_a_trait_spec_writes_nothing(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    (project_root / ".tcip").mkdir(parents=True)
    key = _seed_raw_trait_spec(project_root, {
        "name": "leaf", "delivers": ["leaf_length"], "positive_class_name": "open",
    })
    before = ts.read_versioned(key)

    outcomes, refused = cli.process_root(project_root, plan=True)

    assert refused
    after = ts.read_versioned(key)
    assert after.version == before.version
    assert after.value == before.value


# ── main() / the console command ─────────────────────────────────────────────────


def test_main_exits_2_on_a_refused_root(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "nothing"
    root.mkdir()

    code = cli.main([str(root)], prog="tcip rename-subject-registry")

    assert code == 2
    assert "refused" in capsys.readouterr().out


def test_main_exits_0_when_every_named_root_is_already_conformed(
    tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    root = tmp_path / "ds"
    _make_dataset(root)

    code = cli.main([str(root)], prog="tcip rename-subject-registry")

    assert code == 0
