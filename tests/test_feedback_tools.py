"""Review->retrain MCP tools (materialize + queue + lineage + registration)."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.tools.feedback_tools import materialize_review_dataset, prioritize_review_queue

# The prediction bucket these verdicts were recorded against, as bucket_key_of spells one.
BUCKET = "predictions/detector/2026-03-04"


def _seed_verdicts(state_dir: Path, *, bucket: str = BUCKET) -> Path:
    """Record one accepted and one rejected image's verdicts in the store at ``state_dir``."""
    state = {"verdicts": {
        (bucket, "imgA.png"): {"img_status": "completed", "detections": [
            {"action": "accepted", "class_name": "bud", "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
        (bucket, "imgB.png"): {"img_status": "completed", "detections": [
            {"action": "rejected", "class_name": "bud", "gt_bbox_norm": None, "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]}]},
    }}
    # Seed through the engine so the fixture cannot drift from the real shard format.
    from tcip_annotation.review_engine import ReviewEngine

    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update(state)
    engine.save_review_state()
    return state_dir


def _source_images(src: Path) -> Path:
    from PIL import Image
    src.mkdir(parents=True, exist_ok=True)
    for name in ("imgA.png", "imgB.png"):
        Image.new("RGB", (64, 64), (120, 120, 120)).save(src / name)
    return src


def _own_store(dataset_root: Path) -> Path:
    from tcip_mcp.prediction_buckets import review_state_dir_of

    return review_state_dir_of(dataset_root)


def _setup(tmp_path: Path):
    """A dataset whose own verdict store holds the review, plus its reviewed source images."""
    dataset_root = tmp_path / "dataset"
    _seed_verdicts(_own_store(dataset_root))
    return dataset_root, _source_images(tmp_path / "src")


def test_materialize_review_dataset_end_to_end(tmp_path):
    dataset_root, src = _setup(tmp_path)
    out = tmp_path / "out"
    r = materialize_review_dataset(str(dataset_root), str(src), str(out))
    assert "error" not in r
    assert r["positive"] == 1 and r["hard_negative"] == 1
    assert (out / "images" / "imgA.png").is_file()
    assert (out / "annotations" / "imgA.json").is_file()


def test_materialize_reads_the_dataset_s_own_store_when_none_is_stated(tmp_path):
    """The dataset root alone names the store: no second argument, no second location."""
    dataset_root, src = _setup(tmp_path)
    r = materialize_review_dataset(str(dataset_root), str(src), str(tmp_path / "out"))
    assert "error" not in r
    assert r["dataset_root"] == str(dataset_root)
    assert r["review_state_stated"] is False
    assert r["review_state"] == str(_own_store(dataset_root) / "review")
    assert str(_own_store(dataset_root)) in r["review_state_origin"]


def test_materialize_consumes_a_stated_store_outside_the_dataset(tmp_path):
    """A review recorded outside the dataset is still curated, and the response says from where.

    The dataset here has no store of its own, so the shards can only have come from the stated
    location, and the caller is told which one it was. The bucket key is stated as an absolute
    path (a directory outside any dataset root, the shape bucket_key_of gives one): a relative
    key is only ever meaningful against the root whose own store recorded it, never against a
    different dataset_root the caller states alongside an external review_state_dir.
    """
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    bucket_dir = str((tmp_path / "predictions" / "detector" / "2026-03-04").resolve())
    external = _seed_verdicts(tmp_path / "elsewhere" / "state", bucket=bucket_dir)
    src = _source_images(tmp_path / "src")

    r = materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"), review_state_dir=str(external))
    assert "error" not in r
    assert r["positive"] == 1 and r["hard_negative"] == 1
    assert r["dataset_root"] == str(dataset_root)
    assert r["review_state_stated"] is True
    assert r["review_state"] == str(external / "review")
    assert str(external) in r["review_state_origin"]
    assert str(_own_store(dataset_root)) in r["review_state_origin"]


def test_materialize_refuses_an_empty_stated_store_rather_than_the_dataset_s_own(tmp_path):
    """A stated store holding no shards is refused, never answered from the dataset's own.

    The dataset's own store holds a full review here, so a fallback would succeed and quietly
    curate a review the caller did not name.
    """
    dataset_root, src = _setup(tmp_path)
    stated = tmp_path / "elsewhere" / "state"
    stated.mkdir(parents=True)

    r = materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"), review_state_dir=str(stated))
    assert str(stated) in r["error"]
    assert "positive" not in r


def test_materialize_review_dataset_records_lineage(tmp_path, monkeypatch):
    import tcip_store as ts
    import tcip_mcp.experiments as experiments
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "exp")
    experiments.create_experiment("exp1", {"x": 1})

    dataset_root, src = _setup(tmp_path)
    out = tmp_path / "out"
    r = materialize_review_dataset(str(dataset_root), str(src), str(out), experiment_id="exp1")
    assert r["experiment_id"] == "exp1"

    lineage = ts.read(experiments.lineage_key("exp1"))
    assert lineage["data_source"] == str(dataset_root)
    assert lineage["review_session"]["dataset_root"] == str(dataset_root)
    assert lineage["review_session"]["review_state_dir"] == str(_own_store(dataset_root))
    artifacts = ts.read(experiments.artifacts_key("exp1"))
    assert artifacts["curated_dataset"]["path"] == str(out)


def test_lineage_records_a_stated_store_beside_the_dataset_it_curates(tmp_path, monkeypatch):
    """Both facts are recorded: which dataset the review was of, and where its shards were read."""
    import tcip_store as ts
    import tcip_mcp.experiments as experiments
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "exp")

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    bucket_dir = str((tmp_path / "predictions" / "detector" / "2026-03-04").resolve())
    external = _seed_verdicts(tmp_path / "elsewhere" / "state", bucket=bucket_dir)
    src = _source_images(tmp_path / "src")

    materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"),
        experiment_id="ext1", review_state_dir=str(external))

    lineage = ts.read(experiments.lineage_key("ext1"))
    assert lineage["data_source"] == str(dataset_root)
    assert lineage["review_session"]["dataset_root"] == str(dataset_root)
    assert lineage["review_session"]["review_state_dir"] == str(external)


def test_materialize_creates_experiment_when_absent(tmp_path, monkeypatch):
    import tcip_store as ts
    import tcip_mcp.experiments as experiments
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "exp")

    dataset_root, src = _setup(tmp_path)
    r = materialize_review_dataset(str(dataset_root), str(src), str(tmp_path / "out"), experiment_id="new1")
    assert r["experiment_id"] == "new1"
    lineage = ts.read(experiments.lineage_key("new1"))
    assert "review_session" in lineage


def test_a_second_curation_against_a_completed_experiment_refuses_before_writing(tmp_path, monkeypatch):
    """A pointer is checked before its write, not after: a second curation against an experiment
    whose curated_dataset pointer is already populated and terminal refuses by name, with no
    directory written for it to orphan."""
    import tcip_mcp.experiments as experiments
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "exp")
    experiments.create_experiment("exp2", {"x": 1})
    experiments.update_status("exp2", "running")

    dataset_root, src = _setup(tmp_path)
    out1 = tmp_path / "out1"
    r1 = materialize_review_dataset(str(dataset_root), str(src), str(out1), experiment_id="exp2")
    assert "error" not in r1
    experiments.update_status("exp2", "completed")

    out2 = tmp_path / "out2"
    r2 = materialize_review_dataset(str(dataset_root), str(src), str(out2), experiment_id="exp2")
    assert "error" in r2
    assert not out2.exists()


def test_materialize_invalid_inputs_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()  # a dataset root whose own store holds no shards
    r = materialize_review_dataset(str(empty), str(tmp_path), str(tmp_path / "o1"))
    assert str(_own_store(empty)) in r["error"]

    dataset_root, _src = _setup(tmp_path)
    assert "error" in materialize_review_dataset(
        str(dataset_root), str(tmp_path / "nope"), str(tmp_path / "o2"))

    assert "dataset_root" in materialize_review_dataset("", str(tmp_path), str(tmp_path / "o3"))["error"]


def test_prioritize_review_queue_checkpoint_missing(tmp_path):
    r = prioritize_review_queue(str(tmp_path / "nope.pt"), str(tmp_path))
    assert "error" in r  # early guard, no torch import needed


def test_prioritize_review_queue_skips_what_the_dataset_s_own_store_holds(tmp_path):
    """Coverage of the ranking door's own dataset_root/skip_reviewed/bucket forwarding through
    _prepare_queue_sources: both reviewed images drop out before any scorer runs, the same
    plumbing triage_predictions shares."""
    dataset_root, images = _setup(tmp_path)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    r = prioritize_review_queue(
        checkpoint_path=str(ckpt), images_dir=str(images), dataset_root=str(dataset_root),
        skip_reviewed=True, bucket=BUCKET)
    assert r["reviewed_skipped"] == 2
    assert r["total_candidates"] == 0
    assert r["queue"] == []


def test_triage_predictions_skips_what_the_dataset_s_own_store_holds(tmp_path):
    """The dataset root is enough to find the verdicts: both reviewed images drop out of the queue.

    A store the tool could not find would rank every image again and send the breeder back through
    a review they already finished.
    """
    from tcip_mcp.tools.feedback_tools import triage_predictions

    dataset_root, images = _setup(tmp_path)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    r = triage_predictions(
        checkpoint_path=str(ckpt), images_dir=str(images), dataset_root=str(dataset_root))
    assert r["reviewed_skipped"] == 2
    assert r["total_images"] == 0


def test_triage_predictions_skips_what_a_stated_store_holds(tmp_path):
    """A review recorded outside the dataset still filters the queue when its store is stated."""
    from tcip_mcp.tools.feedback_tools import triage_predictions

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    external = _seed_verdicts(tmp_path / "elsewhere" / "state")
    images = _source_images(tmp_path / "src")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    r = triage_predictions(
        checkpoint_path=str(ckpt), images_dir=str(images), dataset_root=str(dataset_root),
        review_state_dir=str(external))
    assert r["reviewed_skipped"] == 2


def test_prioritize_review_queue_rejects_non_composed_kind(tmp_path, monkeypatch):
    """Active-learning uncertainty scoring reads model logits, which a non-composed predictor
    kind (a bespoke tcip model was never built for) doesn't expose, so it must fail with a
    clear error, not crash on ``.model``.
    """
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.predictor as predmod
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    # prioritize_review_queue imports build_predictor from the predictor module at call time.
    monkeypatch.setattr(predmod, "build_predictor",
                        lambda *a, **k: SimpleNamespace(kind="foreign_kind"))

    r = prioritize_review_queue(checkpoint_path=str(ckpt), images_dir=str(images),
                                project_path=str(tmp_path))
    assert "error" in r and "foreign_kind" in r["error"]


def test_triage_predictions_surfaces_unscoreable(tmp_path, monkeypatch):
    """A regression checkpoint's predictions carry no confidence signal at all (RegressionHead's
    point estimate, deliberately no distributional output). triage_predictions must route them
    into review rather than silently drop them from every output, and tag them distinctly via
    unscoreable_images so a caller can tell this apart from a genuinely medium-confidence item."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.predictor as predmod
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")

    predictions = [{"image": "a.jpg", "width": 4, "height": 4, "head0_values": [0.42]}]
    monkeypatch.setattr(
        predmod, "build_predictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources: predictions))

    r = triage_predictions(
        checkpoint_path=str(ckpt), images_dir=str(images), project_path=str(tmp_path))
    assert r["needs_review"] == 1
    assert r["review_images"] == ["a.jpg"]
    assert r["unscoreable_images"] == ["a.jpg"]
    assert r["auto_accepted_images"] == []


def _stubbed_triage_predictions(tmp_path, monkeypatch, predictions: list[dict], **kwargs):
    """triage_predictions against a real registered checkpoint with predict_batch stubbed to a
    fixed set of confidence-bearing and unscoreable predictions, one call site both door-level
    triage tests share."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.predictor as predmod
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    for pred in predictions:
        (images / pred["image"]).write_bytes(b"x")
    monkeypatch.setattr(
        predmod, "build_predictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources: predictions))

    return triage_predictions(
        checkpoint_path=str(ckpt), images_dir=str(images), project_path=str(tmp_path), **kwargs)


def test_triage_predictions_auto_threshold_none_refuses_with_zero_auto_accepts(tmp_path, monkeypatch):
    """Coverage of the door's own auto-accept refusal (absent since the re-point moved this
    door's tests onto the ranking door): no threshold means zero auto-accepts and the refusal
    named, while the confident, mid-confidence and unscoreable predictions are still routed
    honestly rather than silently dropped."""
    predictions = [
        {"image": "high.jpg", "scores": [0.9]},
        {"image": "mid.jpg", "scores": [0.5]},
        {"image": "unscoreable.jpg", "head0_values": [0.42]},
    ]
    r = _stubbed_triage_predictions(tmp_path, monkeypatch, predictions)
    assert r["total_images"] == 3
    assert r["auto_accepted"] == 0
    assert r["auto_accepted_images"] == []
    assert "auto_accept_refused" in r
    assert r["needs_review"] == 2
    assert r["review_images"] == ["mid.jpg", "unscoreable.jpg"]
    assert r["unscoreable_images"] == ["unscoreable.jpg"]


def test_triage_predictions_explicit_auto_threshold_stamps_breeder_confirmation(tmp_path, monkeypatch):
    """Coverage of the door's own breeder-confirmation stamp (absent since the re-point): an
    explicit auto_threshold accepts exactly the predictions that clear it and carries the
    confirmation-required stamp, with the review/unscoreable routing unchanged by its presence."""
    predictions = [
        {"image": "high.jpg", "scores": [0.9]},
        {"image": "mid.jpg", "scores": [0.5]},
        {"image": "unscoreable.jpg", "head0_values": [0.42]},
    ]
    r = _stubbed_triage_predictions(tmp_path, monkeypatch, predictions, auto_threshold=0.85)
    assert r["total_images"] == 3
    assert r["auto_accepted"] == 1
    assert r["auto_accepted_images"] == ["high.jpg"]
    assert "auto_accept_requires_breeder_confirmation" in r
    assert r["needs_review"] == 2
    assert r["review_images"] == ["mid.jpg", "unscoreable.jpg"]
    assert r["unscoreable_images"] == ["unscoreable.jpg"]


def test_unresolvable_scorer_raises_valueerror_not_an_import_error():
    """The refusal is a ValueError whatever the name looks like.

    ``build_scorer``'s callers catch ``ValueError`` to turn a refusal into an error dict. A dotted
    name that fails to import used to raise ``ModuleNotFoundError`` straight out of the audited
    MCP tool instead.
    """
    import pytest

    from tcip_mcp.pipelines.active_learning.helpers import build_scorer

    with pytest.raises(ValueError):
        build_scorer("no_such_scorer", "detection")
    with pytest.raises(ValueError, match="Could not import scorer"):
        build_scorer("not_a_module.at_all:make", "detection")


def test_unresolvable_proposal_engine_raises_valueerror():
    import pytest

    from tcip_mcp.pipelines.proposal import resolve_proposer

    with pytest.raises(ValueError):
        resolve_proposer("no_such_engine")
    with pytest.raises(ValueError, match="Could not import proposal engine"):
        resolve_proposer("not_a_module.at_all:make")


# -- rail: prioritize_review_queue marks a bound run's calibration-side candidates ------------


def _bespoke_checkpoint_payload() -> dict:
    from tcip_mcp.pipelines.model_build import build_model

    src = {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
        "task": "detection",
    }
    model = build_model({"model_source": src})
    return {"model_source": src, "model_state_dict": model.state_dict()}


def _registered_checkpoint_from_experiment(tmp_path: Path, experiment_id: str) -> str:
    """A real checkpoint completed and registered against ``experiment_id`` (``complete_run`` then
    ``register_model_from_experiment``, experiment-mode registration), so
    ``checkpoint.producer`` resolves to it the way a real trained run's own registration would;
    never a hand-built registry entry."""
    import torch

    from tcip_mcp.experiments import complete_run, register_model_from_experiment

    ckpt_path = tmp_path / f"{experiment_id}.pt"
    torch.save(_bespoke_checkpoint_payload(), str(ckpt_path))
    completed = complete_run(experiment_id, str(ckpt_path))
    assert "error" not in completed, completed
    registered = register_model_from_experiment(
        experiment_id, str(ckpt_path), project_path=str(tmp_path))
    assert "error" not in registered, registered
    return str(ckpt_path)


def _stub_scorer(monkeypatch) -> None:
    """Scores every candidate 1.0 in order: the real scorer reads model logits, which this rail
    has no need to exercise, since the calibration mark is computed from the candidate's own
    path and the manifest, never from a score."""
    import tcip_mcp.pipelines.active_learning.helpers as al_helpers

    class _Scorer:
        def score(self, sources, model, device):
            return [(s, 1.0) for s in sources]

    monkeypatch.setattr(al_helpers, "build_scorer", lambda method, task: _Scorer())


def test_prioritize_review_queue_marks_a_bound_runs_calibration_side(tmp_path, monkeypatch):
    """A checkpoint whose run was bound to a split manifest marks each ranked candidate against
    that manifest's own calibration side; no mark exists at all before this family."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tests.test_selection_disjointness_label_movement import DATES, _bind_run, _dataset, _draw

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    manifest = _draw(root, manifest_dir)
    date = DATES[0]
    calibration_stems = {
        i.split("/", 1)[1] for i in manifest["splits"]["calibration"] if i.startswith(f"{date}/")
    }
    assert calibration_stems  # the fixture's own three-way ratio gives this date some

    experiment_id = "exp-pq-marks"
    _bind_run(root, manifest_dir, experiment_id, date=date)
    ckpt_path = _registered_checkpoint_from_experiment(tmp_path, experiment_id)
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        checkpoint_path=ckpt_path, images_dir=str(root / "images" / date),
        project_path=str(tmp_path))
    assert "error" not in r, r
    assert r["queue"], r
    for entry in r["queue"]:
        stem = Path(entry["image"]).stem
        assert entry["calibration_member"] == (stem in calibration_stems), entry
    assert "marks_unresolved" not in r


def test_prioritize_review_queue_scope_check_reaches_the_review_queue(tmp_path, monkeypatch):
    """Marker proof that _resolve_calibration_ids reaches manifest_scope_issues, the one
    accumulator every manifest-scope consumer shares: a site that stopped calling it would pass
    this test's own scenario silently instead of surfacing the marker below in
    marks_unresolved."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_mcp.pipelines.data.splits as splits_mod
    from tests.test_selection_disjointness_label_movement import DATES, _bind_run, _dataset, _draw

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    _draw(root, manifest_dir)
    date = DATES[0]

    experiment_id = "exp-pq-marker"
    _bind_run(root, manifest_dir, experiment_id, date=date)
    ckpt_path = _registered_checkpoint_from_experiment(tmp_path, experiment_id)
    _stub_scorer(monkeypatch)
    monkeypatch.setattr(
        splits_mod, "manifest_scope_issues",
        lambda *a, **k: (["MARKER-REVIEW-QUEUE-SCOPE-ISSUE"], None),
    )

    r = prioritize_review_queue(
        checkpoint_path=ckpt_path, images_dir=str(root / "images" / date),
        project_path=str(tmp_path))

    assert "error" not in r, r
    assert "marks_unresolved" in r
    assert "MARKER-REVIEW-QUEUE-SCOPE-ISSUE" in r["marks_unresolved"]


def test_prioritize_review_queue_unbound_run_carries_no_marks_or_reason(tmp_path, monkeypatch):
    """A checkpoint with no registry-recorded producer has nothing bound to check against: no
    ``calibration_member`` on any entry, and no ``marks_unresolved`` guessing at a reason."""
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        checkpoint_path=ckpt, images_dir=str(images), project_path=str(tmp_path))
    assert "error" not in r, r
    assert r["queue"], r
    assert all("calibration_member" not in entry for entry in r["queue"])
    assert "marks_unresolved" not in r


def test_prioritize_review_queue_marks_unresolved_when_the_manifest_cannot_be_read(
    tmp_path, monkeypatch,
):
    """A bound run whose named manifest can no longer be read serves the queue with no marks and
    a stated ``marks_unresolved`` reason, never a guess at membership."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_store as ts
    from tests.test_selection_disjointness_label_movement import DATES, _bind_run, _dataset, _draw
    from tcip_mcp.tools.data_tools import split_manifest_key

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    manifest = _draw(root, manifest_dir)
    date = DATES[0]

    experiment_id = "exp-pq-unresolved"
    _bind_run(root, manifest_dir, experiment_id, date=date)
    ckpt_path = _registered_checkpoint_from_experiment(tmp_path, experiment_id)

    # Corrupted the same way read_split_manifest_dir's own refusal rail is tested: a required
    # members-block key stripped, the record rewritten through the store, never a file deleted.
    manifest["members"][date] = {
        k: v for k, v in manifest["members"][date].items() if k != "label_digests"
    }
    ts.replace(split_manifest_key(manifest_dir), manifest)
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        checkpoint_path=ckpt_path, images_dir=str(root / "images" / date),
        project_path=str(tmp_path))
    assert "error" not in r, r
    assert r["queue"], r
    assert all("calibration_member" not in entry for entry in r["queue"])
    assert "marks_unresolved" in r
    assert str(manifest_dir) in r["marks_unresolved"]


# -- rail: calibration marks are decided by the manifest's own record, never images_dir's shape -

_FLAT_SUBJECT = "leaf"
_FLAT_IMG = 32
_FLAT_STEMS = ("a", "b", "c", "d", "e", "f")


def _save_flat_png(path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (_FLAT_IMG, _FLAT_IMG), color=(128, 128, 128)).save(path)


def _write_flat_label(root: Path, date: str, stem: str) -> None:
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    json_io.write_annotations(
        str(root / "annotations" / date / f"{stem}.json"),
        [Annotation(subject=_FLAT_SUBJECT, geometry=BBox(2, 2, 10, 10))], _FLAT_IMG, _FLAT_IMG,
    )


def _bucketed_labels_flat_images_dataset(root: Path, date: str, stems=_FLAT_STEMS) -> Path:
    """Labels bucketed under one date; images in the flat ``images/`` root (no ``images/<date>/``
    bucket), the layout whose split manifest records the flat root as that date's own
    ``images_root``."""
    for stem in stems:
        _save_flat_png(root / "images" / f"{stem}.jpg")
        _write_flat_label(root, date, stem)
    return root


def _mixed_two_date_dataset(
    root: Path, canonical_date: str, flat_date: str, stems=_FLAT_STEMS,
) -> Path:
    """One date's images bucketed under ``images/<date>/`` (canonical); the other's labels are
    bucketed the same way but its images sit in the flat ``images/`` root, no bucket of its own
    (the same mismatch :func:`_bucketed_labels_flat_images_dataset` builds for one date, beside a
    date that has no such mismatch)."""
    for stem in stems:
        _save_flat_png(root / "images" / canonical_date / f"{stem}.jpg")
        _write_flat_label(root, canonical_date, stem)
    for stem in stems:
        _save_flat_png(root / "images" / f"{stem}.jpg")
        _write_flat_label(root, flat_date, stem)
    return root


def _draw_flat(root: Path, out: Path, *, seed: int = 2) -> dict:
    import tcip_store as ts
    from tcip_mcp.tools.data_tools import draw_splits, split_manifest_key

    result = draw_splits(str(root), output_path=str(out), subject=_FLAT_SUBJECT, seed=seed,
                         train_ratio=0.4, val_ratio=0.3, calibration_ratio=0.3)
    assert "error" not in result, result
    return ts.read(split_manifest_key(out))


def _bind_dataset_run(
    root: Path, manifest_dir: Path, experiment_id: str, *, date: str, images_dir: Path,
) -> None:
    """A run bound to ``manifest_dir`` for ``date``, its own admission drawn against
    ``images_dir``: the same sequence :func:`_registered_checkpoint_from_experiment`'s callers
    use for the canonical dated layout, parameterized over which directory this date's images
    actually live in."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(root / "annotations" / date),
        "subject": _FLAT_SUBJECT, "attribute": None,
        "split": {"manifest_dir": str(manifest_dir)},
    }
    train_ds, val_ds, label_digests = auto_train_val("detection", data_cfg, None)
    create_experiment(experiment_id, {})
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg, label_digests=label_digests)


def test_prioritize_review_queue_marks_a_flat_images_tree_dataset_correctly(tmp_path, monkeypatch):
    """A dataset whose labels are bucketed by date but whose images live in the flat images/ root
    (no images/<date>/ bucket) still marks its calibration side correctly: the manifest's own
    recorded images_root for that date is the flat root, never a date guessed from images_dir's
    path shape (which cannot tell a flat root apart from a dateless one)."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2026-03-01"
    root = _bucketed_labels_flat_images_dataset(tmp_path / "data", date)
    manifest_dir = tmp_path / "manifest"
    manifest = _draw_flat(root, manifest_dir)
    calibration_stems = {
        i.split("/", 1)[1] for i in manifest["splits"]["calibration"] if i.startswith(f"{date}/")
    }
    assert calibration_stems  # the fixture's own three-way ratio gives this date some

    experiment_id = "exp-pq-flat"
    _bind_dataset_run(root, manifest_dir, experiment_id, date=date, images_dir=root / "images")
    ckpt_path = _registered_checkpoint_from_experiment(tmp_path, experiment_id)
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        checkpoint_path=ckpt_path, images_dir=str(root / "images"),
        project_path=str(tmp_path))
    assert "error" not in r, r
    assert r["queue"], r
    assert "marks_unresolved" not in r
    marked_true = {Path(e["image"]).stem for e in r["queue"] if e["calibration_member"]}
    assert marked_true == calibration_stems


def test_prioritize_review_queue_a_bound_run_never_marks_another_dates_calibration_side(
    tmp_path, monkeypatch,
):
    """A manifest spanning two dates (one canonical, the other bucketed-labels-flat-images
    like the single-date rail above), bound to the flat one, marks only that date's own
    calibration side: a stem that is a calibration member under the other, canonical, unbound
    date must never read as a member here, even though the same stem name recurs under both."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    canonical_date, flat_date = "2026-03-01", "2026-03-15"
    root = _mixed_two_date_dataset(tmp_path / "data", canonical_date, flat_date)
    manifest_dir = tmp_path / "manifest"
    manifest = _draw_flat(root, manifest_dir)
    bound_stems = {
        i.split("/", 1)[1] for i in manifest["splits"]["calibration"]
        if i.startswith(f"{flat_date}/")
    }
    other_stems = {
        i.split("/", 1)[1] for i in manifest["splits"]["calibration"]
        if i.startswith(f"{canonical_date}/")
    }
    assert bound_stems and other_stems
    leaked = other_stems - bound_stems
    assert leaked  # a stem calibration-only under the unbound date; the case that must not leak

    experiment_id = "exp-pq-two-dates"
    _bind_dataset_run(
        root, manifest_dir, experiment_id, date=flat_date, images_dir=root / "images")
    ckpt_path = _registered_checkpoint_from_experiment(tmp_path, experiment_id)
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        checkpoint_path=ckpt_path, images_dir=str(root / "images"),
        project_path=str(tmp_path))
    assert "error" not in r, r
    assert r["queue"], r
    assert "marks_unresolved" not in r
    marked_true = {Path(e["image"]).stem for e in r["queue"] if e["calibration_member"]}
    assert marked_true == bound_stems
    assert not (marked_true & leaked)


def test_prioritize_review_queue_a_root_mismatch_yields_marks_unresolved_not_false(
    tmp_path, monkeypatch,
):
    """images_dir that is not the bound date's recorded images_root cannot be that date's
    member by path shape; the response says so under marks_unresolved rather than mark every
    candidate a confident non-member."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tests.test_selection_disjointness_label_movement import DATES, _bind_run, _dataset, _draw

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    _draw(root, manifest_dir)
    bound_date, other_date = DATES

    experiment_id = "exp-pq-root-mismatch"
    _bind_run(root, manifest_dir, experiment_id, date=bound_date)
    ckpt_path = _registered_checkpoint_from_experiment(tmp_path, experiment_id)
    _stub_scorer(monkeypatch)

    # A real directory, just not the one the manifest recorded for the bound date.
    r = prioritize_review_queue(
        checkpoint_path=ckpt_path, images_dir=str(root / "images" / other_date),
        project_path=str(tmp_path))
    assert "error" not in r, r
    assert r["queue"], r
    assert all("calibration_member" not in entry for entry in r["queue"])
    assert "marks_unresolved" in r
    assert bound_date in r["marks_unresolved"]


def test_prioritize_review_queue_a_corrupted_split_record_yields_marks_unresolved_naming_it(
    tmp_path, monkeypatch,
):
    """A bound run whose own split.json will not decode must not read as an unbound run: the
    corruption is named under marks_unresolved rather than folded onto silence."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tests._record_damage_fixtures import damage_record
    from tests.test_selection_disjointness_label_movement import DATES, _bind_run, _dataset, _draw
    from tcip_mcp.experiments import split_key

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    _draw(root, manifest_dir)
    date = DATES[0]

    experiment_id = "exp-pq-corrupt-split"
    _bind_run(root, manifest_dir, experiment_id, date=date)
    ckpt_path = _registered_checkpoint_from_experiment(tmp_path, experiment_id)
    damage_record(split_key(experiment_id, root=str(tmp_path)), b"{not json at all")
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        checkpoint_path=ckpt_path, images_dir=str(root / "images" / date),
        project_path=str(tmp_path))
    assert "error" not in r, r
    assert r["queue"], r
    assert all("calibration_member" not in entry for entry in r["queue"])
    assert "marks_unresolved" in r
    assert "could not be read" in r["marks_unresolved"]


def test_prioritize_review_queue_signature_drops_the_triage_only_parameters():
    """The confidence-triage capability split off with its own parameters: prioritize_review_queue
    carries no strategy flag and none of triage_predictions's own knobs."""
    import inspect

    params = inspect.signature(prioritize_review_queue).parameters
    assert "strategy" not in params
    assert "low" not in params
    assert "high" not in params
    assert "auto_threshold" not in params


def test_triage_predictions_signature_carries_the_triage_only_parameters():
    """The door that gained the confidence-triage capability carries its own parameters."""
    import inspect

    from tcip_mcp.tools.feedback_tools import triage_predictions

    params = inspect.signature(triage_predictions).parameters
    assert "low" in params
    assert "high" in params
    assert "auto_threshold" in params


def test_feedback_tools_register_in_manifest():
    from tcip_mcp.server import list_registered_tools
    names = list_registered_tools()
    assert "materialize_review_dataset" in names
    assert "prioritize_review_queue" in names


# --- classified-scope materialization ---------------------------------------------------------

CLASSIFIED_SUBJECT = "leaf"
CLASSIFIED_ATTRIBUTE = "condition"
CLASSIFIED_BUCKET = "predictions/classifier/2026-03-05"


def _seed_classified_verdicts(state_dir: Path, *, bucket: str = CLASSIFIED_BUCKET) -> Path:
    """One accepted 'healthy' call and one rejected 'diseased' call: a classified review's own
    verdicts, whose class_name is the confirmed/predicted value, never the object's subject."""
    state = {"verdicts": {
        (bucket, "imgA.png"): {"img_status": "completed", "detections": [
            {"action": "accepted", "class_name": "healthy",
             "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
        (bucket, "imgB.png"): {"img_status": "completed", "detections": [
            {"action": "rejected", "class_name": "diseased",
             "gt_bbox_norm": None, "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]}]},
    }}
    from tcip_annotation.review_engine import ReviewEngine

    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update(state)
    engine.save_review_state()
    return state_dir


def _stamp_classified_bucket(dataset_root: Path, bucket_rel: str = CLASSIFIED_BUCKET) -> Path:
    from tcip_mcp.pipelines.resolution import write_sidecar

    bucket_dir = dataset_root / bucket_rel
    write_sidecar(bucket_dir, {"id_map": {"healthy": 0, "diseased": 1},
                              "subject": CLASSIFIED_SUBJECT, "attribute": CLASSIFIED_ATTRIBUTE})
    return bucket_dir


def _source_dataset_with_registry(root: Path) -> Path:
    """A dataset root carrying a real class registry, with the two reviewed images under its
    own images/ (the segment dataset_root_of needs to locate the root back from it)."""
    dataset_root = root / "source_dataset"
    images = dataset_root / "images"
    _source_images(images)
    (dataset_root / "classes.json").write_text(
        '{"leaf": {"attributes": {"condition": {"type": "categorical", '
        '"values": ["healthy", "diseased"]}}}}',
        encoding="utf-8",
    )
    return dataset_root


def test_materialize_writes_positives_under_a_classified_scope_in_the_ground_truth_shape(tmp_path):
    """The object class lands in subject, the confirmed value under the scope's own attribute,
    never the verdict-name-derived subject a detector review would write."""
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(_own_store(dataset_root))
    _stamp_classified_bucket(dataset_root)
    source = _source_dataset_with_registry(tmp_path)

    r = materialize_review_dataset(
        str(dataset_root), str(source / "images"), str(tmp_path / "out"),
        bucket=CLASSIFIED_BUCKET)

    assert "error" not in r
    assert r["positive"] == 1
    assert r["subject"] == CLASSIFIED_SUBJECT
    assert r["attribute"] == CLASSIFIED_ATTRIBUTE
    from tcip_annotation.json_io import read_annotations

    anns = read_annotations(str(tmp_path / "out" / "annotations" / "imgA.json"))
    assert anns[0].subject == CLASSIFIED_SUBJECT
    assert anns[0].attributes == {CLASSIFIED_ATTRIBUTE: "healthy"}


def test_materialize_refuses_a_classified_scope_bucket_recording_no_id_map(tmp_path):
    """The tool's own pre-check refuses by name before any write when the reviewed bucket's
    stamp records no id_map at all: a classified scope's confirmed values have nothing to check
    against, and passing an empty vocabulary through unchecked would let the write rail refuse
    every value with a bare ``[]`` instead of naming the bucket that lacks a map. The admitting
    case (a bucket whose stamp records a real map) is
    test_materialize_writes_positives_under_a_classified_scope_in_the_ground_truth_shape."""
    from tcip_mcp.pipelines.resolution import write_sidecar

    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(_own_store(dataset_root))
    bucket_dir = dataset_root / CLASSIFIED_BUCKET
    write_sidecar(bucket_dir, {"id_map": None,
                              "subject": CLASSIFIED_SUBJECT, "attribute": CLASSIFIED_ATTRIBUTE})
    source = _source_dataset_with_registry(tmp_path)
    out = tmp_path / "out"

    r = materialize_review_dataset(
        str(dataset_root), str(source / "images"), str(out), bucket=CLASSIFIED_BUCKET)

    assert "error" in r
    assert "records no id_map" in r["error"]
    assert not out.exists()


def test_materialize_refuses_a_positive_under_a_classified_scope_with_an_empty_id_map(tmp_path):
    """An empty recorded id_map names no vocabulary either: ``write_sidecar``'s own
    ``scope_consistent_with_map`` rail refuses an empty id_map outright, so no live producer can
    stamp this shape, and the bucket here is seeded through a raw store write instead. The tool's
    own pre-check only refuses an absent map (``bucket_id_map`` answering ``None``); an empty one
    reaches ``materialize_dataset``'s own write rail (``_write_positive_label``), which refuses
    each accepted verdict's positive the same way an absent vocabulary refuses. The admitting
    case is test_materialize_writes_positives_under_a_classified_scope_in_the_ground_truth_shape."""
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(_own_store(dataset_root))
    bucket_dir = dataset_root / CLASSIFIED_BUCKET
    tcip_store.replace(sidecar_key(bucket_dir, "operating_point"),
                       {"id_map": {}, "subject": CLASSIFIED_SUBJECT,
                        "attribute": CLASSIFIED_ATTRIBUTE},
                       expect=tcip_store.Version.ABSENT)
    source = _source_dataset_with_registry(tmp_path)
    out = tmp_path / "out"

    r = materialize_review_dataset(
        str(dataset_root), str(source / "images"), str(out), bucket=CLASSIFIED_BUCKET)

    assert "error" not in r
    assert r["positive"] == 0
    assert [e["image"] for e in r["boundary_refused"]] == ["imgA.png"]
    assert "requires the bucket's own recorded vocabulary" in r["boundary_refused"][0]["reason"]


def test_materialize_never_confirms_a_negative_under_a_classified_scope(tmp_path):
    """A rejected value call names the model's wrong-state guess, never the object's absence, so
    the rejected-only image is named in unconfirmed_negatives and no confirmed-negative status
    is ever recorded for it, even though its label file is still an empty background."""
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(_own_store(dataset_root))
    _stamp_classified_bucket(dataset_root)
    source = _source_dataset_with_registry(tmp_path)
    out = tmp_path / "out"

    r = materialize_review_dataset(
        str(dataset_root), str(source / "images"), str(out), bucket=CLASSIFIED_BUCKET)

    assert "error" not in r
    assert len(r["unconfirmed_negatives"]) == 1
    assert r["unconfirmed_negatives"][0]["image"] == "imgB.png"
    from tcip_mcp.dataset_layout import read_image_status_store

    assert read_image_status_store(out) == {}


def test_materialize_copies_the_source_registry_under_a_classified_scope(tmp_path):
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(_own_store(dataset_root))
    _stamp_classified_bucket(dataset_root)
    source = _source_dataset_with_registry(tmp_path)
    out = tmp_path / "out"

    r = materialize_review_dataset(
        str(dataset_root), str(source / "images"), str(out), bucket=CLASSIFIED_BUCKET)

    assert "error" not in r
    assert (out / "classes.json").is_file()
    assert (out / "classes.json").read_text(encoding="utf-8") == (
        (source / "classes.json").read_text(encoding="utf-8"))


def test_materialize_refuses_a_classified_scope_with_no_source_registry(tmp_path):
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(_own_store(dataset_root))
    _stamp_classified_bucket(dataset_root)
    src = _source_images(tmp_path / "src")  # a bare directory, no dataset root to derive from

    r = materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"), bucket=CLASSIFIED_BUCKET)

    assert "error" in r
    assert "register_dataset" in r["error"]


def test_materialize_refuses_a_classified_scope_into_a_populated_output(tmp_path):
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(_own_store(dataset_root))
    _stamp_classified_bucket(dataset_root)
    source = _source_dataset_with_registry(tmp_path)
    out = tmp_path / "out"
    out.mkdir(parents=True)
    (out / "classes.json").write_text('{"other": {}}', encoding="utf-8")

    r = materialize_review_dataset(
        str(dataset_root), str(source / "images"), str(out), bucket=CLASSIFIED_BUCKET)

    assert "error" in r
    assert "already holds a class registry" in r["error"]
    assert (out / "classes.json").read_text(encoding="utf-8") == '{"other": {}}'


def test_materialize_refuses_a_neither_key_stamp(tmp_path):
    from tcip_mcp.pipelines.resolution import sidecar_key
    import tcip_store

    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(_own_store(dataset_root))
    bucket_dir = dataset_root / CLASSIFIED_BUCKET
    tcip_store.replace(sidecar_key(bucket_dir, "operating_point"),
                       {"id_map": {"healthy": 0, "diseased": 1}}, expect=tcip_store.Version.ABSENT)
    src = _source_images(tmp_path / "src")

    r = materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"), bucket=CLASSIFIED_BUCKET)

    assert "error" in r
    assert "repair-classified-predictions" in r["error"]


def test_materialize_refuses_an_undecodable_stamp(tmp_path):
    import os

    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
    from tcip_store.store import _backend

    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(_own_store(dataset_root))
    bucket_dir = _stamp_classified_bucket(dataset_root)
    key = sidecar_key(bucket_dir, "operating_point")
    if (os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND) == FILE_BACKEND:
        _backend().path_for(key).write_bytes(b"{not json")
    else:
        import sqlite3

        from tcip_store.sqlite_backend import database_path, encode_parts

        conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
        try:
            conn.execute("update records set value = ? where store = ? and parts = ?",
                        (b"{not json", key.store, encode_parts(key.parts)))
        finally:
            conn.close()
    src = _source_images(tmp_path / "src")

    r = materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"), bucket=CLASSIFIED_BUCKET)

    assert "error" in r


def test_materialize_refuses_a_relative_bucket_key_against_a_stated_foreign_store(tmp_path):
    """A relative bucket key is only ever meaningful against the root whose own store recorded
    it; a caller stating a different review_state_dir must state an absolute path instead."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    external = _seed_verdicts(tmp_path / "elsewhere" / "state", bucket=BUCKET)  # relative key
    src = _source_images(tmp_path / "src")

    r = materialize_review_dataset(
        str(dataset_root), str(src), str(tmp_path / "out"), review_state_dir=str(external))

    assert "error" in r
    assert "relative bucket key" in r["error"]
