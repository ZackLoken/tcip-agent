"""``tcip repair-classified-predictions``: stamping a scope onto a classified prediction
bucket that predates the writer rail, and rewriting a bucket's own documents from the old
value-in-subject shape into the shape ``write_predictions_json`` now writes.

Every bucket a live producer could no longer mint (a pre-rail stamp, a value-in-subject document)
is seeded by writing straight through the store or the annotation writer, the same posture
``test_review_negative_coverage.py`` takes for a neither-key stamp: the rail refuses these shapes
outright, so there is no producer left to build them through.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts
from tcip_store.binding import bind_default

from tcip_annotation.json_io import write_annotations
from tcip_annotation.review_engine import ReviewEngine
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.experiments import config_key
from tcip_mcp.pipelines.resolution import bucket_scope, sidecar_key
from tcip_mcp.prediction_buckets import review_state_dir_of
from tcip_mcp.tools.project_tools import upsert_dataset

SUBJECT = "leaf"
ATTRIBUTE = "condition"
VALUE_ID_MAP = {"healthy": 0, "diseased": 1}
DETECTOR_ID_MAP = {SUBJECT: 0}


def _load_script():
    from tcip_mcp.cli import repair_classified_predictions

    return repair_classified_predictions


def _base_stamp(*, id_map: dict, experiment_id: str = "exp-1", **overrides) -> dict:
    stamp = {
        "trait": "condition", "dataset_hash": "h",
        "operating_point": {"conf": {"value": 0.5}}, "id_map": id_map,
        "validated": False, "validated_by": None, "tile_size_validated": None,
        "shippable_issues": [], "checkpoint": "m", "checkpoint_sha256": "f" * 64,
        "experiment_id": experiment_id, "images_dir": None, "raster_path": None,
        "produced_at": "2026-01-01T00:00:00+00:00",
    }
    stamp.update(overrides)
    return stamp


def _write_stamp(bucket: Path, stamp: dict) -> None:
    bucket.mkdir(parents=True, exist_ok=True)
    ts.replace(sidecar_key(bucket, "operating_point"), stamp, expect=ts.Version.ABSENT)


def _damage_record(key: ts.Key, data: bytes) -> None:
    """Put ``data`` behind a record, wherever the bound backend keeps it (mirrors
    test_project_tools.py's own helper: the record must already exist at the key)."""
    import os

    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
    from tcip_store.store import _backend

    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        _backend().path_for(key).write_bytes(data)
        return
    import sqlite3

    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()


def _write_undecodable_stamp(bucket: Path) -> None:
    bucket.mkdir(parents=True, exist_ok=True)
    key = sidecar_key(bucket, "operating_point")
    ts.replace(key, _base_stamp(id_map=VALUE_ID_MAP), expect=ts.Version.ABSENT)
    _damage_record(key, b"{not json")


def _write_doc(bucket: Path, stem: str, annotations: list[Annotation], w=100, h=80) -> Path:
    bucket.mkdir(parents=True, exist_ok=True)
    path = bucket / f"{stem}.json"
    write_annotations(str(path), annotations, w, h, keep_empty=True)
    return path


def _write_experiment_config(experiment_id: str, root: Path, data: dict) -> None:
    ts.replace(config_key(experiment_id, root=root), {"data": data}, expect=ts.Version.ABSENT)


def _pre_conform_classified_bucket(dataset_root: Path, *, date: str = "2026-01-01") -> Path:
    """A classified bucket in the old shape: the value in ``subject``, a stamp recording the
    value-keyed map but no ``(subject, attribute)`` pair, written before the writer rail existed.
    """
    bucket = dataset_root / "predictions" / "classifier" / date
    _write_doc(bucket, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_doc(bucket, "img2", [Annotation(subject="diseased", geometry=BBox(5, 5, 15, 15), score=0.8)])
    _write_stamp(bucket, _base_stamp(id_map=VALUE_ID_MAP, experiment_id="exp-classified"))
    return bucket


def _scoped_like_bucket(dataset_root: Path, *, id_map: dict, date: str = "like") -> Path:
    """A fully scoped bucket (a real ``--like`` source): its own stamp already carries the
    ``(subject, attribute)`` pair and ``id_map``, a shape a live classifier run mints, seeded
    through the same producer, ``operating_point_stamp`` and ``write_sidecar``."""
    from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar

    bucket = dataset_root / "predictions" / "classifier" / date
    value = next(iter(id_map))
    _write_doc(bucket, "imgA",
              [Annotation(subject=SUBJECT, geometry=BBox(0, 0, 10, 10), attributes={ATTRIBUTE: value})])
    stamp = operating_point_stamp(
        {"conf": {"value": 0.5}}, validated=False, validated_by=None, tile_size_validated=None,
        shippable_issues=[], id_map=id_map, trait=ATTRIBUTE, dataset_hash="h",
        checkpoint="m", checkpoint_sha256="f" * 64, experiment_id="exp-like",
        images_dir=None, raster_path=None, produced_at="2026-01-01T00:00:00+00:00",
        subject=SUBJECT, attribute=ATTRIBUTE,
    )
    write_sidecar(bucket, stamp)
    return bucket


def _scoped_detector_like_bucket(dataset_root: Path, *, date: str = "like") -> Path:
    """A fully scoped detector bucket (a real ``--like`` source with no attribute), the shape a
    live detector run mints, seeded through the same producer."""
    from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar

    bucket = dataset_root / "predictions" / "detector" / date
    _write_doc(bucket, "imgA", [Annotation(subject=SUBJECT, geometry=BBox(0, 0, 10, 10))])
    stamp = operating_point_stamp(
        {"conf": {"value": 0.5}}, validated=False, validated_by=None, tile_size_validated=None,
        shippable_issues=[], id_map=DETECTOR_ID_MAP, trait=SUBJECT, dataset_hash="h",
        checkpoint="m", checkpoint_sha256="f" * 64, experiment_id="exp-detector-like",
        images_dir=None, raster_path=None, produced_at="2026-01-01T00:00:00+00:00",
        subject=SUBJECT, attribute=None,
    )
    write_sidecar(bucket, stamp)
    return bucket


def _register(project_root: Path, dataset_root: Path, *, dataset_id: str = "ds-1") -> None:
    (project_root / ".tcip").mkdir(parents=True, exist_ok=True)
    upsert_dataset(project_root, {"id": dataset_id, "path": str(dataset_root), "crop": "currant",
                                  "fingerprint": None})


def _classes_json(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "classes.json").write_text(
        '{"leaf": {"attributes": {"condition": {"type": "categorical", '
        '"values": ["healthy", "diseased"]}}}}',
        encoding="utf-8",
    )


def test_a_classifier_bucket_is_rewritten_and_stamped(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is False
    assert any("rewrote 2 document(s)" in o for o in outcomes), outcomes
    scope = bucket_scope(bucket)
    assert scope.subject == SUBJECT and scope.attribute == ATTRIBUTE
    from tcip_annotation.json_io import read_annotations

    doc1 = read_annotations(str(bucket / "img1.json"))
    assert doc1[0].subject == SUBJECT
    assert doc1[0].attributes == {ATTRIBUTE: "healthy"}

    from tcip_mcp.pipelines.postprocessing.phenology import count_by_class

    total, positive, unclassified = count_by_class(
        bucket / "img1.json", VALUE_ID_MAP, "healthy", scope=scope)
    assert (total, positive, unclassified) == (1, 1, 0)


def test_a_sourced_attribute_with_no_subject_refuses_before_any_write(tmp_path):
    """Coverage of the named-attribute message, not a guard: the experiment source reads a run's
    config verbatim, with no validation of its own, and a hand-corrupted record naming an
    attribute under no subject refuses before any write through the same subject-is-None gate the
    (None, None) case below guards; removing only this shape's own attribute-naming branch still
    refuses through that gate's shared fallback message."""
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project,
                              {"subject": None, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})

    from tcip_mcp.prediction_buckets import bucket_content_digest

    before = bucket_content_digest(bucket)
    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is True
    assert any("no subject" in o for o in outcomes), outcomes
    assert bucket_content_digest(bucket) == before
    assert module.read_stamp_state(bucket).kind == "unstated"


def test_an_experiment_source_answering_neither_subject_nor_attribute_refuses_before_any_write(
    tmp_path,
):
    """The guard: a bespoke run's experiment record can honestly answer no scope at all (both keys
    present, both None), and this is the shape the subject-is-None gate exists to catch before any
    write, rather than admitting it as a detector pair naming no object class."""
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project, {"subject": None, "attribute": None})

    from tcip_mcp.prediction_buckets import bucket_content_digest

    before = bucket_content_digest(bucket)
    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is True
    assert any("names no subject" in o for o in outcomes), outcomes
    assert bucket_content_digest(bucket) == before
    assert module.read_stamp_state(bucket).kind == "unstated"


def test_a_second_run_rewrites_nothing_and_reports_the_same_left_behind_set(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})

    first, _ = module.process_project_root(project, plan=False, operator_subject=None,
                                           operator_attribute=None)
    second, refused = module.process_project_root(project, plan=False, operator_subject=None,
                                                   operator_attribute=None)

    assert refused is False
    assert any("conformed" in o and "rewrote" not in o for o in second), second


def test_a_detector_bucket_is_stamped(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = dataset_root / "predictions" / "detector" / "2026-01-02"
    _write_doc(bucket, "img1", [Annotation(subject=SUBJECT, geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_stamp(bucket, _base_stamp(id_map=DETECTOR_ID_MAP, experiment_id="exp-detector"))
    _write_experiment_config("exp-detector", project, {"subject": SUBJECT, "attribute": None,
                                                       "id_map": DETECTOR_ID_MAP})

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is False
    assert any("stamped the detector pair" in o for o in outcomes), outcomes
    scope = bucket_scope(bucket)
    assert scope.subject == SUBJECT and scope.attribute is None


def test_a_reviewed_detector_bucket_is_stamped_with_digests_intact(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = dataset_root / "predictions" / "detector" / "2026-01-03"
    _write_doc(bucket, "img1", [Annotation(subject=SUBJECT, geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_stamp(bucket, _base_stamp(id_map=DETECTOR_ID_MAP, experiment_id="exp-detector-2"))
    _write_experiment_config("exp-detector-2", project, {"subject": SUBJECT, "attribute": None,
                                                         "id_map": DETECTOR_ID_MAP})
    key = ("predictions/detector/2026-01-03", "img1.json")
    state_dir = review_state_dir_of(dataset_root)
    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update({"verdicts": {key: {"img_status": "completed", "detections": [
        {"action": "accepted", "class_name": SUBJECT,
         "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]}}})
    engine.save_review_state()

    from tcip_mcp.prediction_buckets import bucket_content_digest

    before = bucket_content_digest(bucket)
    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is False
    assert any("stamp completion" in o or "stamped the detector pair" in o for o in outcomes), outcomes
    assert bucket_content_digest(bucket) == before


def test_a_stated_detector_scope_refused_over_a_bucket_whose_map_is_keyed_by_values(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = dataset_root / "predictions" / "classifier" / "2026-01-04"
    _write_doc(bucket, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_stamp(bucket, _base_stamp(id_map=VALUE_ID_MAP, experiment_id="exp-no-record"))

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=SUBJECT, operator_attribute=None)

    assert refused is True
    assert any("refused" in o and "detector pair" in o for o in outcomes), outcomes


def test_a_no_map_bucket_a_decimal_key_bucket_and_an_undecodable_stamp(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)

    no_map = dataset_root / "predictions" / "classifier" / "no-map"
    _write_doc(no_map, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_stamp(no_map, _base_stamp(id_map=None, experiment_id="exp-a"))
    _write_experiment_config("exp-a", project, {"subject": SUBJECT, "attribute": ATTRIBUTE})

    decimal_key = dataset_root / "predictions" / "classifier" / "decimal-key"
    _write_doc(decimal_key, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_stamp(decimal_key, _base_stamp(id_map={"0": 0, "1": 1}, experiment_id="exp-b"))
    _write_experiment_config("exp-b", project, {"subject": SUBJECT, "attribute": ATTRIBUTE})

    undecodable = dataset_root / "predictions" / "classifier" / "bad-stamp"
    _write_doc(undecodable, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_undecodable_stamp(undecodable)

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is True
    joined = "\n".join(outcomes)
    assert "no-map" in joined and "records no id_map" in joined
    assert "decimal-key" in joined and "decimal key" in joined
    assert "bad-stamp" in joined and "refused" in joined


def test_a_stamped_bucket_holding_a_value_keyed_document_is_reported_with_re_inference(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = dataset_root / "predictions" / "classifier" / "already-scoped"
    _write_doc(bucket, "img1", [Annotation(subject=SUBJECT, geometry=BBox(0, 0, 10, 10), score=0.9,
                                          attributes={ATTRIBUTE: "healthy"})])
    _write_doc(bucket, "img2", [Annotation(subject="diseased", geometry=BBox(5, 5, 15, 15), score=0.8)])
    _write_stamp(bucket, _base_stamp(id_map=VALUE_ID_MAP, subject=SUBJECT, attribute=ATTRIBUTE))

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is False
    assert any("conformed" in o and "value-keyed record" in o for o in outcomes), outcomes


def test_a_reviewed_classified_bucket_is_reported_with_its_verdict_count_and_untouched(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = _pre_conform_classified_bucket(dataset_root, date="reviewed")
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})
    key = ("predictions/classifier/reviewed", "img1.json")
    state_dir = review_state_dir_of(dataset_root)
    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update({"verdicts": {key: {"img_status": "completed", "detections": [
        {"action": "accepted", "class_name": "healthy",
         "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]}}})
    engine.save_review_state()

    from tcip_mcp.prediction_buckets import bucket_content_digest

    before = bucket_content_digest(bucket)
    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is True
    assert any("review verdict" in o for o in outcomes), outcomes
    assert bucket_content_digest(bucket) == before


def test_plan_mode_writes_nothing_and_reports_what_would_change(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})

    from tcip_mcp.prediction_buckets import bucket_content_digest

    before = bucket_content_digest(bucket)
    outcomes, refused = module.process_project_root(
        project, plan=True, operator_subject=None, operator_attribute=None)

    assert refused is True
    assert any("would conform" in o for o in outcomes), outcomes
    assert bucket_content_digest(bucket) == before


def test_a_prediction_date_directory_with_no_images_counterpart_is_walked_and_conformed(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    assert not (dataset_root / "images").exists()
    bucket = _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is False
    assert any(str(bucket) in o and "rewrote" in o for o in outcomes), outcomes


def test_the_experiment_source_is_read_under_the_walked_root_when_the_platform_root_is_pinned_elsewhere(
    tmp_path, monkeypatch,
):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(elsewhere))

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is False
    assert any("rewrote 2 document(s)" in o and "experiment" in o and str(project) in o
              for o in outcomes), outcomes
    assert not any(str(elsewhere) in o for o in outcomes), outcomes


def test_a_bespoke_run_answering_no_scope_falls_through_to_the_operator(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project, {"dataset_source": "bespoke"})

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=SUBJECT, operator_attribute=ATTRIBUTE)

    assert refused is False
    assert any("rewrote 2 document(s)" in o and "operator statement" in o for o in outcomes), outcomes
    assert bucket_scope(bucket).subject == SUBJECT


def test_a_scopeless_bucket_with_no_source_is_refused_and_left_alone(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project, {"dataset_source": "bespoke"})

    from tcip_mcp.prediction_buckets import bucket_content_digest

    before = bucket_content_digest(bucket)
    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is True
    assert any("no source" in o for o in outcomes), outcomes
    assert bucket_content_digest(bucket) == before


def test_the_like_source_conforms_a_bare_copy(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    scoped = _pre_conform_classified_bucket(dataset_root, date="source")
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})
    module.process_project_root(project, plan=False, operator_subject=None, operator_attribute=None)
    assert bucket_scope(scoped).subject == SUBJECT

    bare_copy = tmp_path / "hand_split" / "calibration"
    _write_doc(bare_copy, "imgA", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10))])

    outcome, refused, id_map, changed = module.conform_bucket(
        bare_copy, root=None, plan=False, like_dir=scoped,
        operator_subject=None, operator_attribute=None, is_bare_named=True,
    )

    assert refused is False
    assert "rewrote" in outcome and "no stamp written" in outcome
    from tcip_annotation.json_io import read_annotations

    doc = read_annotations(str(bare_copy / "imgA.json"))
    assert doc[0].subject == SUBJECT
    assert doc[0].attributes == {ATTRIBUTE: "healthy"}
    assert module.read_stamp_state(bare_copy).kind == "absent"


def test_a_detector_like_source_refuses_to_conform_a_bare_copy(tmp_path):
    """A bare directory this command repairs holds a classified bucket's value-in-subject
    records: a --like naming a detector bucket (its scope names no attribute) has nothing to move
    the value under, and refuses by name rather than attempting a rewrite with no attribute."""
    bind_default()
    module = _load_script()
    dataset_root = tmp_path / "dataset"
    detector_like = _scoped_detector_like_bucket(dataset_root)

    bare_copy = tmp_path / "hand_split" / "calibration"
    _write_doc(bare_copy, "imgA", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10))])

    outcome, refused, id_map, changed = module.conform_bucket(
        bare_copy, root=None, plan=False, like_dir=detector_like,
        operator_subject=None, operator_attribute=None, is_bare_named=True,
    )

    assert refused is True
    assert "detector bucket" in outcome
    from tcip_annotation.json_io import read_annotations

    doc = read_annotations(str(bare_copy / "imgA.json"))
    assert doc[0].subject == "healthy"


def test_a_disagreeing_like_id_map_refuses_against_the_bucket_own_recorded_map(tmp_path):
    """``--like`` supplies a vocabulary only for a bare directory or a stamp recording none: a
    bucket whose own stamp already records an ``id_map`` is rewritten against that map, and a
    ``--like`` whose map differs from it refuses by name rather than silently overriding it.
    """
    bind_default()
    module = _load_script()
    dataset_root = tmp_path / "dataset"
    bucket = dataset_root / "predictions" / "classifier" / "unstated"
    _write_doc(bucket, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10))])
    _write_stamp(bucket, _base_stamp(id_map=VALUE_ID_MAP))  # no subject/attribute recorded yet

    other_map = {"unripe": 0, "ripe": 1}
    like_dir = _scoped_like_bucket(dataset_root, id_map=other_map, date="like")

    from tcip_mcp.prediction_buckets import bucket_content_digest

    before = bucket_content_digest(bucket)
    outcome, refused, id_map, changed_root = module.conform_bucket(
        bucket, root=None, plan=False, like_dir=like_dir,
        operator_subject=None, operator_attribute=None, is_bare_named=False,
    )

    assert refused is True
    assert "disagrees" in outcome
    assert bucket_content_digest(bucket) == before
    assert module.read_stamp_state(bucket).kind == "unstated"


def test_a_like_id_map_agreeing_with_the_bucket_own_recorded_map_conforms_it(tmp_path):
    """A matching ``--like`` map is admitted exactly as the bucket's own recorded map."""
    bind_default()
    module = _load_script()
    dataset_root = tmp_path / "dataset"
    bucket = dataset_root / "predictions" / "classifier" / "unstated"
    _write_doc(bucket, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10))])
    _write_doc(bucket, "img2", [Annotation(subject="diseased", geometry=BBox(5, 5, 15, 15))])
    _write_stamp(bucket, _base_stamp(id_map=VALUE_ID_MAP))

    like_dir = _scoped_like_bucket(dataset_root, id_map=VALUE_ID_MAP, date="like")

    outcome, refused, id_map, changed_root = module.conform_bucket(
        bucket, root=None, plan=False, like_dir=like_dir,
        operator_subject=None, operator_attribute=None, is_bare_named=False,
    )

    assert refused is False
    assert "rewrote" in outcome
    assert id_map == VALUE_ID_MAP
    scope = bucket_scope(bucket)
    assert (scope.subject, scope.attribute) == (SUBJECT, ATTRIBUTE)

    from tcip_annotation.json_io import read_annotations

    doc = read_annotations(str(bucket / "img1.json"))
    assert doc[0].subject == SUBJECT
    assert doc[0].attributes == {ATTRIBUTE: "healthy"}


def test_a_bucket_holding_a_raw_index_or_foreign_value_record_is_reported_and_untouched(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = dataset_root / "predictions" / "classifier" / "foreign"
    _write_doc(bucket, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_doc(bucket, "img2", [Annotation(subject="0", geometry=BBox(5, 5, 15, 15), score=0.7)])
    _write_stamp(bucket, _base_stamp(id_map=VALUE_ID_MAP, experiment_id="exp-foreign"))
    _write_experiment_config("exp-foreign", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})

    from tcip_mcp.prediction_buckets import bucket_content_digest

    before = bucket_content_digest(bucket)
    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is True
    assert any("unconformable" in o for o in outcomes), outcomes
    assert bucket_content_digest(bucket) == before


def test_a_stale_detector_document_is_unconformable_even_when_a_value_names_the_subject(tmp_path):
    """The one-key blind spot: a classified map can declare a value spelled like the object class
    itself. A stale detector document carrying the object class and no value must still be
    reported unconformable, never mistaken for a rewrite because its bare subject happens to be a
    key of the map too.
    """
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = dataset_root / "predictions" / "classifier" / "stale"
    coincidental_map = {SUBJECT: 0, "diseased": 1}
    _write_doc(bucket, "img1", [Annotation(subject=SUBJECT, geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_stamp(bucket, _base_stamp(id_map=coincidental_map, experiment_id="exp-coincidental"))
    _write_experiment_config("exp-coincidental", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE,
                               "id_map": coincidental_map})

    from tcip_mcp.prediction_buckets import bucket_content_digest

    before = bucket_content_digest(bucket)
    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is True
    assert any("unconformable" in o for o in outcomes), outcomes
    assert bucket_content_digest(bucket) == before


def test_a_value_keyed_ground_truth_record_is_reported_as_a_candidate(tmp_path):
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    _pre_conform_classified_bucket(dataset_root)
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})
    gt_dir = dataset_root / "annotations" / "2026-01-01"
    _write_doc(gt_dir, "gt_img", [Annotation(subject="healthy", geometry=BBox(1, 1, 9, 9))])

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert any("ground-truth candidate" in o and "healthy" in o for o in outcomes), outcomes


def _platform_audit_entries(tool: str = "repair_classified_predictions") -> list[dict]:
    from tcip_mcp.tools.meta_tools import read_audit_log

    return read_audit_log(scope=None, tool=tool)["entries"]


def test_a_bare_like_rewrite_writes_one_audit_entry_with_its_structured_fields(
    tmp_path, monkeypatch,
):
    """A bare directory conformed through --like sits under no dataset root: its one audit entry
    is filed to the platform log and carries the structured fields rule 8 names, not only the
    free-text outcome."""
    import tcip_mcp.audit as audit_module

    platform_root = tmp_path / "platform"
    platform_root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", platform_root)
    bind_default()
    module = _load_script()
    dataset_root = tmp_path / "dataset"
    scoped = _scoped_like_bucket(dataset_root, id_map=VALUE_ID_MAP, date="source")

    bare_copy = tmp_path / "hand_split" / "calibration"
    _write_doc(bare_copy, "imgA", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10))])

    outcome, refused, _id_map, changed = module.conform_bucket(
        bare_copy, root=None, plan=False, like_dir=scoped,
        operator_subject=None, operator_attribute=None, is_bare_named=True,
    )

    assert refused is False
    assert changed is True
    entries = _platform_audit_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["arguments"]["bucket"] == str(bare_copy)
    assert entry["documents_rewritten"] == 1
    assert entry["stamp_written"] is False
    assert (entry["subject"], entry["attribute"]) == (SUBJECT, ATTRIBUTE)
    assert entry["source"] == f"--like {scoped}"
    assert "digest_before" not in entry
    assert entry["outcome"] == outcome


def test_a_classified_rewrite_under_no_dataset_root_writes_one_entry_consistent_with_dataset_scope_of(
    tmp_path, monkeypatch,
):
    """Coverage of the scope-consistency invariant: a bucket outside any dataset root's canonical
    layout (no predictions/images/annotations segment in its path) files its one audit entry
    wherever ``dataset_scope_of`` resolves for it, computed once before any write into the bucket
    and reused for both the outcome note and the audit entry's own scope, never re-derived
    afterward from a state this run's own write changed.

    On the file backend that resolves to ``None`` (a genuinely bare directory carries no ``.tcip``
    of its own): the entry lands in the platform log. On the database backend, a bucket that
    already carries a stamp of its own answers to itself (``dataset_scope_of``'s own ``.tcip``
    fallback cannot tell a bucket's own prior stamp from a genuine dataset marker, the seam's
    known behaviour documented on ``_emit_conform_audit``, not this command's to change): the
    entry lands under that resolved root's own log instead. Either way the outcome text and the
    entry's filed scope never disagree, which is the scope-consistency bug this pin guards: the
    two used to be read from ``dataset_scope_of`` at two different times (once before the rewrite,
    once after), and the stamp write in between could make the second read answer differently
    from the first.

    Fails at the baseline on ``entry["outcome"]``/``outcome`` itself, on the sqlite leg only: this
    bucket resolves under no real dataset root (``dataset_layout.dataset_root_of`` finds no
    canonical segment in its path) on both backends, so the no-verdict-store note belongs in the
    outcome regardless of backend; the unfixed code omitted it on the database backend, where
    ``dataset_scope_of`` answers the bucket itself and the old code read that alone as a real
    dataset root. The scope-consistency assertions below (where the entry is filed) pass against
    the unfixed source too; they are coverage of a fact this fix leaves untouched, not this pin's
    own claim.
    """
    import tcip_mcp.audit as audit_module
    from tcip_mcp.audit import dataset_scope_of

    platform_root = tmp_path / "platform"
    platform_root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", platform_root)
    bind_default()
    module = _load_script()
    bucket = tmp_path / "standalone_run"
    _write_doc(bucket, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10))])
    _write_doc(bucket, "img2", [Annotation(subject="diseased", geometry=BBox(5, 5, 15, 15))])
    _write_stamp(bucket, _base_stamp(id_map=VALUE_ID_MAP))  # no subject/attribute recorded yet
    resolved_scope = dataset_scope_of(bucket)

    outcome, refused, id_map, changed = module.conform_bucket(
        bucket, root=None, plan=False, like_dir=None,
        operator_subject=SUBJECT, operator_attribute=ATTRIBUTE, is_bare_named=False,
    )

    assert refused is False
    assert changed is True
    assert id_map == VALUE_ID_MAP
    assert "this bucket sits under no dataset root" in outcome
    if resolved_scope is None:
        filed_entries = _platform_audit_entries()
    else:
        filed_entries = _dataset_audit_entries(resolved_scope)
        assert _platform_audit_entries() == []
    assert len(filed_entries) == 1
    entry = filed_entries[0]
    assert entry["documents_rewritten"] == 2
    assert entry["stamp_written"] is True
    assert (entry["subject"], entry["attribute"]) == (SUBJECT, ATTRIBUTE)
    assert entry["source"] == "operator statement (--subject/--attribute)"
    assert entry["digest_before"] != entry["digest_after"]
    assert entry["outcome"] == outcome


def test_a_refused_bucket_writes_no_audit_entry(tmp_path, monkeypatch):
    """A bucket this script refuses to conform (here, a --like naming a detector bucket for a
    bare copy) writes nothing, changed or audited."""
    import tcip_mcp.audit as audit_module

    platform_root = tmp_path / "platform"
    platform_root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", platform_root)
    bind_default()
    module = _load_script()
    dataset_root = tmp_path / "dataset"
    detector_like = _scoped_detector_like_bucket(dataset_root)
    bare_copy = tmp_path / "hand_split" / "calibration"
    _write_doc(bare_copy, "imgA", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10))])

    outcome, refused, _id_map, changed = module.conform_bucket(
        bare_copy, root=None, plan=False, like_dir=detector_like,
        operator_subject=None, operator_attribute=None, is_bare_named=True,
    )

    assert refused is True
    assert changed is False
    assert _platform_audit_entries() == []


def _dataset_audit_entries(
    dataset_root: Path, *, tool: str = "repair_classified_predictions",
) -> list[dict]:
    from tcip_mcp.tools.meta_tools import read_audit_log

    return read_audit_log(scope=str(dataset_root), tool=tool)["entries"]


def test_a_reviewed_classified_bucket_walked_through_process_project_root_writes_no_audit_entry(
    tmp_path, monkeypatch,
):
    """The direct-call refusal pin above (test_a_refused_bucket_writes_no_audit_entry) never
    reaches process_project_root's own caller loop: driven through it here, a bucket carrying
    review verdicts, seeded through ReviewEngine.record_detection_action the way
    tests/test_tcip_web_routes.py's _verdicted_launch_dataset does, writes nothing to either the
    platform log or the dataset's own."""
    import tcip_mcp.audit as audit_module
    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine
    from tcip_mcp.prediction_buckets import bucket_content_digest, bucket_key_of

    platform_root = tmp_path / "platform"
    platform_root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", platform_root)
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = _pre_conform_classified_bucket(dataset_root, date="reviewed-walk")
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})

    engine = ReviewEngine(dataset_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="img1.png", img_width=100, img_height=80,
                        preds=[Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10),
                                          score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="healthy", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(0.0, 0.0, 10.0, 10.0))
    engine.record_detection_action(bucket_key_of(bucket), det, ctx, action="accepted")

    before = bucket_content_digest(bucket)
    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is True
    assert any("review verdict" in o for o in outcomes), outcomes
    assert bucket_content_digest(bucket) == before
    assert _platform_audit_entries() == []
    assert _dataset_audit_entries(dataset_root) == []


def test_a_second_bare_like_run_over_an_already_conformed_copy_writes_no_entry(
    tmp_path, monkeypatch,
):
    """A no-op --like run (the second over an already-conformed bare copy) rewrites nothing, so
    it emits no audit entry and reports changed false, unlike the first run pinned above
    (test_a_bare_like_rewrite_writes_one_audit_entry_with_its_structured_fields)."""
    import tcip_mcp.audit as audit_module

    platform_root = tmp_path / "platform"
    platform_root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", platform_root)
    bind_default()
    module = _load_script()
    dataset_root = tmp_path / "dataset"
    scoped = _scoped_like_bucket(dataset_root, id_map=VALUE_ID_MAP, date="source")
    bare_copy = tmp_path / "hand_split" / "calibration"
    _write_doc(bare_copy, "imgA", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10))])

    first = module.conform_bucket(
        bare_copy, root=None, plan=False, like_dir=scoped,
        operator_subject=None, operator_attribute=None, is_bare_named=True,
    )
    assert first[1] is False and first[3] is True
    assert len(_platform_audit_entries()) == 1

    outcome, refused, id_map, changed = module.conform_bucket(
        bare_copy, root=None, plan=False, like_dir=scoped,
        operator_subject=None, operator_attribute=None, is_bare_named=True,
    )

    assert refused is False
    assert changed is False
    assert len(_platform_audit_entries()) == 1


def test_a_classified_rewrite_with_a_validated_stamp_writes_an_entry_whose_outcome_carries_the_floor_note(
    tmp_path, monkeypatch,
):
    """A rewrite of a bucket whose stamp already claimed validated floors the claim (rule 7),
    since the rewrite changes the bucket's own content digest away from the one the validation
    record was earned over. The floor note used to reach only stdout in the returned outcome
    string; it now rides the audit entry's own outcome field too."""
    import tcip_mcp.audit as audit_module
    from tests._binding_fixtures import file_validation_record
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT

    platform_root = tmp_path / "platform"
    platform_root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", platform_root)
    bind_default()
    module = _load_script()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    bucket = dataset_root / "predictions" / "classifier" / "validated"
    _write_doc(bucket, "img1", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10), score=0.9)])
    _write_doc(bucket, "img2", [Annotation(subject="diseased", geometry=BBox(5, 5, 15, 15), score=0.8)])
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})
    stamp = _base_stamp(
        id_map=VALUE_ID_MAP, experiment_id="exp-classified",
        operating_point={"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}})
    # file_validation_record hashes pred_dirs as they are on disk now, so it is called before the
    # rewrite: the record covers the pre-conform, value-in-subject content the rewrite replaces.
    bound = file_validation_record(
        stamp, dataset_root=dataset_root, pred_dirs=[bucket],
        experiment_id="exp-validated-ref", trait=ATTRIBUTE)
    bound["validated"] = True
    _write_stamp(bucket, bound)

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is False
    assert any("floors" in o for o in outcomes), outcomes
    entries = _dataset_audit_entries(dataset_root)
    assert len(entries) == 1
    assert "floors" in entries[0]["outcome"]


def test_process_project_root_summary_counts_changed_and_unchanged_buckets(tmp_path):
    """process_project_root's own returned outcomes name how many buckets its walk changed and
    how many it left as they were, so the count field conform_bucket returns finally has a
    reader."""
    module = _load_script()
    bind_default()
    project = tmp_path / "project"
    dataset_root = tmp_path / "dataset"
    _register(project, dataset_root)
    _pre_conform_classified_bucket(dataset_root, date="to-change")
    _write_experiment_config("exp-classified", project,
                              {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": VALUE_ID_MAP})
    unchanged_bucket = dataset_root / "predictions" / "classifier" / "already-scoped"
    _write_doc(unchanged_bucket, "img1",
              [Annotation(subject=SUBJECT, geometry=BBox(0, 0, 10, 10), score=0.9,
                         attributes={ATTRIBUTE: "healthy"})])
    _write_stamp(unchanged_bucket, _base_stamp(id_map=VALUE_ID_MAP, subject=SUBJECT,
                                              attribute=ATTRIBUTE))

    outcomes, refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)

    assert refused is False
    assert any("1 bucket(s) changed, 1 left as they were" in o for o in outcomes), outcomes


def test_bucket_dirs_under_agrees_with_the_shared_walk_over_a_dot_prefixed_date_directory(tmp_path):
    """``bucket_dirs_under`` calls the shared ``prediction_bucket_dirs`` walk rather than holding
    a second implementation of it: the shared walk admits every date directory regardless of its
    own name, and a dot-prefixed one is filtered back out here, the one remaining difference
    between the two.
    """
    from tcip_mcp.dataset_layout import prediction_bucket_dirs

    module = _load_script()
    dataset_root = tmp_path / "dataset"
    model_dir = dataset_root / "predictions" / "classifier"
    real_date = model_dir / "2026-01-01"
    hidden_date = model_dir / ".hidden"
    _write_doc(real_date, "img1", [Annotation(subject="leaf", geometry=BBox(0, 0, 10, 10))])
    _write_doc(hidden_date, "img1", [Annotation(subject="leaf", geometry=BBox(0, 0, 10, 10))])

    shared = prediction_bucket_dirs(dataset_root)
    assert hidden_date in shared, "the shared walk admits a directory without regard to its name"

    found = module.bucket_dirs_under(dataset_root)
    assert real_date in found
    assert hidden_date not in found
    assert set(found) == {
        d for d in shared if module.is_bucket_name(d.name) and module._looks_like_bucket(d)
    }


def test_process_project_root_early_returns_say_no_summary_applies(tmp_path):
    """``process_project_root``'s two early returns (no ``.tcip``, no registered datasets) return
    before the changed/unchanged summary line every other path appends; each says so rather than
    leaving the summary's absence unstated."""
    module = _load_script()
    bind_default()

    no_tcip_outcomes, no_tcip_refused = module.process_project_root(
        tmp_path / "not_a_project", plan=False, operator_subject=None, operator_attribute=None)
    assert no_tcip_refused is True
    assert any("no summary applies" in o for o in no_tcip_outcomes), no_tcip_outcomes

    project = tmp_path / "project"
    (project / ".tcip").mkdir(parents=True)
    no_datasets_outcomes, no_datasets_refused = module.process_project_root(
        project, plan=False, operator_subject=None, operator_attribute=None)
    assert no_datasets_refused is False
    assert any("no summary applies" in o for o in no_datasets_outcomes), no_datasets_outcomes


def test_main_prints_a_bucket_changed_unchanged_summary_for_the_bucket_path(tmp_path, capsys):
    """``main``'s ``--bucket`` loop has the same ``changed`` value ``process_project_root``'s own
    walk already summarizes; it must print the same kind of summary rather than discarding it."""
    bind_default()
    module = _load_script()
    dataset_root = tmp_path / "dataset"
    scoped = _scoped_like_bucket(dataset_root, id_map=VALUE_ID_MAP, date="source")

    bare_copy = tmp_path / "hand_split" / "calibration"
    _write_doc(bare_copy, "imgA", [Annotation(subject="healthy", geometry=BBox(0, 0, 10, 10))])

    already_scoped = tmp_path / "hand_split" / "already_scoped"
    _write_doc(already_scoped, "imgB",
              [Annotation(subject=SUBJECT, geometry=BBox(0, 0, 10, 10),
                         attributes={ATTRIBUTE: "healthy"})])
    _write_stamp(already_scoped, _base_stamp(id_map=VALUE_ID_MAP, subject=SUBJECT,
                                             attribute=ATTRIBUTE))

    exit_code = module.main([
        "--bucket", str(bare_copy), "--like", str(scoped),
        "--bucket", str(already_scoped), "--like", str(scoped),
    ])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "--bucket: 1 bucket(s) changed, 1 left as they were" in out
