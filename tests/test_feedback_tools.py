"""Review->retrain MCP tools: materialize, the review queue and triage."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES
from tcip_mcp.project_paths import project_state_dir
from tcip_mcp.tools.feedback_tools import materialize_review_dataset, prioritize_review_queue

# The prediction bucket these verdicts were recorded against, as bucket_key_of spells one.
BUCKET = "predictions/detector/2026-03-04"


def _seed_verdicts(state_dir: Path, *, bucket: str = BUCKET) -> Path:
    """Record one accepted and one rejected image's verdicts in the store at ``state_dir``."""
    state = {"verdicts": {
        (bucket, "imgA.png"): {"img_status": "completed", "detections": [
            {"action": "accepted", "class_name": "bud", "iscrowd": False, "reviewed_by": "", "conf": None, "class_id": None, "producer_identity": None, "conf_threshold": None, "missed_object_attested": False, "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
        (bucket, "imgB.png"): {"img_status": "completed", "detections": [
            {"action": "rejected", "class_name": "bud", "iscrowd": False, "reviewed_by": "", "conf": None, "class_id": None, "producer_identity": None, "conf_threshold": None, "missed_object_attested": False, "gt_bbox_norm": None, "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]}]},
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


def _setup(tmp_path: Path):
    """A dataset whose own verdict store holds the review, plus its reviewed source images."""
    dataset_root = tmp_path / "dataset"
    _seed_verdicts(project_state_dir(dataset_root))
    return dataset_root, _source_images(tmp_path / "src")


def _reviewed(dataset_root: Path) -> str:
    """Publish the bucket the seeded verdicts were recorded against, a bucket of no documents;
    its directory."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    return str(published(dataset_root.parent, dataset_root / BUCKET, [],
                         scope={"subject": "bud", "attribute": None, "id_map": {"bud": 0}}).path)


def test_materialize_review_dataset_end_to_end(tmp_path):
    dataset_root, src = _setup(tmp_path)
    out = tmp_path / "out"
    r = materialize_review_dataset(tmp_path, str(dataset_root), str(src), str(out),
                                   predictions_dir=_reviewed(dataset_root))
    assert "error" not in r
    assert r["positive"] == 1 and r["hard_negative"] == 1
    assert (out / "images" / "imgA.png").is_file()
    assert (out / "annotations" / "imgA.json").is_file()


def test_materialize_reads_the_dataset_s_own_store_when_none_is_stated(tmp_path):
    """The dataset root alone names the store: no second argument, no second location."""
    dataset_root, src = _setup(tmp_path)
    r = materialize_review_dataset(tmp_path, str(dataset_root), str(src), str(tmp_path / "out"),
                                   predictions_dir=_reviewed(dataset_root))
    assert "error" not in r
    assert r["dataset_root"] == str(dataset_root)
    assert r["review_state_stated"] is False
    assert r["review_state"] == str(project_state_dir(dataset_root) / "review")
    assert str(project_state_dir(dataset_root)) in r["review_state_origin"]


def test_materialize_consumes_a_stated_store_outside_the_dataset(tmp_path):
    """A review recorded outside the dataset is still curated, and the response says from where.

    The dataset here has no store of its own, so the shards can only have come from the stated
    location, and the caller is told which one it was.
    """
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    external = _seed_verdicts(tmp_path / "elsewhere" / "state")
    src = _source_images(tmp_path / "src")

    r = materialize_review_dataset(
        tmp_path, str(dataset_root), str(src), str(tmp_path / "out"),
        predictions_dir=_reviewed(dataset_root), review_state_dir=str(external))
    assert "error" not in r
    assert r["positive"] == 1 and r["hard_negative"] == 1
    assert r["dataset_root"] == str(dataset_root)
    assert r["review_state_stated"] is True
    assert r["review_state"] == str(external / "review")
    assert str(external) in r["review_state_origin"]
    assert str(project_state_dir(dataset_root)) in r["review_state_origin"]


def test_materialize_refuses_an_empty_stated_store_rather_than_the_dataset_s_own(tmp_path):
    """A stated store holding no shards is refused, never answered from the dataset's own.

    The dataset's own store holds a full review here, so a fallback would succeed and quietly
    curate a review the caller did not name.
    """
    dataset_root, src = _setup(tmp_path)
    stated = tmp_path / "elsewhere" / "state"
    stated.mkdir(parents=True)

    r = materialize_review_dataset(
        tmp_path, str(dataset_root), str(src), str(tmp_path / "out"), review_state_dir=str(stated))
    assert str(stated) in r["error"]
    assert "positive" not in r


def test_materialize_review_dataset_writes_no_run(tmp_path, monkeypatch):
    """A curated dataset is data, not a run: materializing one writes its own output and opens
    no run directory."""
    from tcip_mcp.experiments import run_dirs

    dataset_root, src = _setup(tmp_path)
    r = materialize_review_dataset(tmp_path, str(dataset_root), str(src), str(tmp_path / "out"),
                                   predictions_dir=_reviewed(dataset_root))

    assert "error" not in r, r
    assert "experiment_id" not in r
    assert run_dirs(tmp_path) == []


def test_materialize_invalid_inputs_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()  # a dataset root whose own store holds no shards
    r = materialize_review_dataset(tmp_path, str(empty), str(tmp_path), str(tmp_path / "o1"))
    assert str(project_state_dir(empty)) in r["error"]

    dataset_root, _src = _setup(tmp_path)
    assert "error" in materialize_review_dataset(
        tmp_path, str(dataset_root), str(tmp_path / "nope"), str(tmp_path / "o2"))

    assert "dataset_root" in materialize_review_dataset(tmp_path, "", str(tmp_path), str(tmp_path / "o3"))["error"]


def test_prioritize_review_queue_checkpoint_missing(tmp_path):
    r = prioritize_review_queue(tmp_path, str(tmp_path / "nope.pt"), str(tmp_path))
    assert "error" in r  # early guard, no torch import needed


def test_prioritize_review_queue_skips_what_the_dataset_s_own_store_holds(tmp_path):
    """Coverage of the ranking door's own dataset_root/skip_reviewed/predictions_dir forwarding
    through _prepare_queue_sources: both reviewed images drop out before any scorer runs, the same
    plumbing triage_predictions shares."""
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset_root, images = _setup(tmp_path)
    ckpt = registered_checkpoint(tmp_path)

    r = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt, images_dir=str(images), dataset_root=str(dataset_root),
        skip_reviewed=True, predictions_dir=_reviewed(dataset_root))
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
        tmp_path, checkpoint_path=str(ckpt), images_dir=str(images), dataset_root=str(dataset_root),
        predictions_dir=_reviewed(dataset_root))
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
        tmp_path, checkpoint_path=str(ckpt), images_dir=str(images), dataset_root=str(dataset_root),
        predictions_dir=_reviewed(dataset_root), review_state_dir=str(external))
    assert r["reviewed_skipped"] == 2


def test_triage_predictions_surfaces_unscoreable(tmp_path, monkeypatch):
    """A regression checkpoint's predictions carry no confidence signal at all (RegressionHead's
    point estimate, deliberately no distributional output). triage_predictions must route them
    into review rather than silently drop them from every output, and tag them distinctly via
    unscoreable_images so a caller can tell this apart from a genuinely medium-confidence item."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")

    predictions = [{"image": "a.jpg", "width": 4, "height": 4, "head0_values": [0.42]}]
    monkeypatch.setattr(
        predmod, "GenericPredictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources, **kw: predictions))

    r = triage_predictions(
        tmp_path, checkpoint_path=str(ckpt), images_dir=str(images))
    assert r["needs_review"] == 1
    assert r["review_images"] == ["a.jpg"]
    assert r["unscoreable_images"] == ["a.jpg"]
    assert r["auto_accepted_images"] == []


def _stubbed_triage_predictions(tmp_path, monkeypatch, predictions: list[dict], **kwargs):
    """triage_predictions against a real registered checkpoint with predict_batch stubbed to a
    fixed set of confidence-bearing and unscoreable predictions, one call site both door-level
    triage tests share."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    for pred in predictions:
        (images / pred["image"]).write_bytes(b"x")
    monkeypatch.setattr(
        predmod, "GenericPredictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources, **kw: predictions))

    return triage_predictions(
        tmp_path, checkpoint_path=str(ckpt), images_dir=str(images), **kwargs)


def test_triage_predictions_auto_threshold_none_refuses_with_zero_auto_accepts(tmp_path, monkeypatch):
    """Coverage of the door's own auto-accept refusal: no threshold means zero auto-accepts and
    the refusal named, while the confident, mid-confidence and unscoreable predictions are still
    routed honestly rather than silently dropped."""
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
    """Coverage of the door's own breeder-confirmation stamp: an explicit auto_threshold
    accepts exactly the predictions that clear it and carries the confirmation-required stamp,
    with the review/unscoreable routing unchanged by its presence."""
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

    ``build_scorer``'s callers catch ``ValueError`` to turn a refusal into an error dict: a
    dotted name that fails to import must raise ``ValueError`` too, never
    ``ModuleNotFoundError`` straight out of the audited MCP tool.
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
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims

    src = {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"min_size": 64, "max_size": 128},
        "task": "detection",
    }
    config = {"model_source": src,
              "data": {"num_channels": 3, "scope": {"subject": "bud", "id_map": {"bud": 0}}}}
    model = build_model(config, recorded_model_dims(config))
    return {"config": config, "model_state_dict": model.state_dict()}


def _bound_checkpoint(project: Path, manifest_dir: Path, experiment_id: str) -> tuple[Path, str]:
    """A run of ``project`` bound to the selection at ``manifest_dir``, run through the child's
    own entry (its data resolved and recorded, its body saving the model its config builds), so
    its completed checkpoint's producer is that bound run. Returns ``(run directory, checkpoint
    path)``."""
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, worker_run

    run_dir = worker_run(project, {
        "model_source": dict(BUILT_DETECTOR),
        "training_source": "tests.bespoke_models:save_built_weights",
        "data": {"split": {"selection_dir": str(manifest_dir)}},
    }, experiment_id=experiment_id)
    checkpoint = observe(run_dir).checkpoint
    assert checkpoint is not None
    return run_dir, checkpoint["path"]


def _stub_scorer(monkeypatch) -> None:
    """Scores every candidate 1.0 in order: the real scorer reads model logits, which this rail
    has no need to exercise, since the calibration mark is computed from the candidate's own
    path and the selection, never from a score."""
    import tcip_mcp.pipelines.active_learning.helpers as al_helpers

    class _Scorer:
        def score(self, sources, predictor):
            return [(s, 1.0) for s in sources]

    monkeypatch.setattr(al_helpers, "build_scorer", lambda method, task: _Scorer())


def _reference_samples(drawn) -> list:
    """The samples a drawn selection holds on its reference sides, calibration and holdout."""
    return [s for side in REFERENCE_SIDES for s in drawn.on(side)]


def test_prioritize_review_queue_marks_a_bound_runs_reference_sides(tmp_path, monkeypatch):
    """A checkpoint whose run was bound to a selection marks each ranked candidate against that
    selection's own calibration and holdout samples under the queue's images directory."""
    from tests.test_selection_ground_truth_digests import DATES, _dataset, _draw

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    drawn = _draw(tmp_path, root, manifest_dir)
    date = DATES[0]
    reference_stems = {
        Path(s.source).stem for s in _reference_samples(drawn)
        if Path(s.source).parent.name == date
    }
    assert reference_stems  # the fixture's own three-way ratio gives this date some

    _run_dir, ckpt_path = _bound_checkpoint(tmp_path, manifest_dir, "exp-pq-marks")
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt_path, images_dir=str(root / "images" / date))
    assert "error" not in r, r
    assert r["queue"], r
    for entry in r["queue"]:
        stem = Path(entry["image"]).stem
        assert entry["reference_member"] == (stem in reference_stems), entry


def test_the_review_queue_marks_the_runs_frozen_calibration_side_after_the_selection_changed(
    tmp_path, monkeypatch,
):
    """The marks are the calibration samples the producing run's own partition froze: a selection
    rewritten after the run, its calibration side moved to train, changes none of them."""
    import tcip_store as ts
    from tcip_mcp.pipelines.data.selection import selection_document, selection_key
    from tests.test_selection_ground_truth_digests import DATES, _dataset, _draw

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    drawn = _draw(tmp_path, root, manifest_dir)
    date = DATES[0]
    reference_stems = {Path(s.source).stem for s in _reference_samples(drawn)
                         if Path(s.source).parent.name == date}
    assert reference_stems

    _run_dir, ckpt_path = _bound_checkpoint(tmp_path, manifest_dir, "exp-pq-frozen")

    document = selection_document(drawn)
    for sample in document["samples"]:
        if sample["side"] in REFERENCE_SIDES:
            sample["side"] = "train"
    ts.replace(selection_key(manifest_dir), document)
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt_path, images_dir=str(root / "images" / date))
    assert "error" not in r, r
    marked = {Path(e["image"]).stem for e in r["queue"] if e["reference_member"]}
    assert marked == reference_stems


def test_the_review_queue_scores_candidates_at_the_checkpoints_own_read_width(tmp_path):
    """Both scorers read a candidate at the width the checkpoint reads at: a one-channel model
    over three-band candidates scores them as one band, rather than handing its first convolution
    an image three times as wide as the one it was trained on."""
    from PIL import Image

    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(
        tmp_path,
        model_source={"builder": "tests.bespoke_models:build_bespoke_detection",
                      "builder_kwargs": {"min_size": 64, "max_size": 128, "image_mean": [0.4],
                                         "image_std": [0.2]},
                      "task": "detection"},
        data={"num_channels": 1, "scope": {"subject": "bud", "id_map": {"bud": 0}}})
    images = tmp_path / "images"
    images.mkdir()
    for stem in ("a", "b"):
        Image.new("RGB", (64, 64), (90, 110, 70)).save(images / f"{stem}.png")

    r = prioritize_review_queue(tmp_path, checkpoint_path=ckpt, images_dir=str(images), method="combined")

    assert "error" not in r, r
    assert len(r["queue"]) == 2, r
    assert all(isinstance(entry["score"], float) for entry in r["queue"])


def test_prioritize_review_queue_unbound_run_carries_no_marks(tmp_path, monkeypatch):
    """A checkpoint no run of the project produced has nothing bound to check against: no
    ``reference_member`` on any entry."""
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    ckpt = foreign_checkpoint(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt, images_dir=str(images))
    assert "error" not in r, r
    assert r["queue"], r
    assert all("reference_member" not in entry for entry in r["queue"])


# -- rail: calibration marks are decided by each sample's own recorded source ------------------

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
    bucket), the layout whose selection records each sample's source under that flat root."""
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


def _draw_flat(project: Path, root: Path, out: Path, *, seed: int = 2):
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(project, str(root), output_path=str(out), subject=_FLAT_SUBJECT, seed=seed,
                         train_ratio=0.4, val_ratio=0.3, calibration_ratio=0.15, holdout_ratio=0.15)
    assert "error" not in result, result
    return read_selection(out, project=project)


def test_prioritize_review_queue_marks_a_flat_images_tree_dataset_correctly(tmp_path, monkeypatch):
    """A dataset whose labels are bucketed by date but whose images live in the flat images/ root
    (no images/<date>/ bucket) still marks its calibration side correctly: each sample's own
    recorded source names the flat root, never a date guessed from images_dir's path shape (which
    cannot tell a flat root apart from a dateless one)."""
    date = "2026-03-01"
    root = _bucketed_labels_flat_images_dataset(tmp_path / "data", date)
    manifest_dir = tmp_path / "manifest"
    drawn = _draw_flat(tmp_path, root, manifest_dir)
    reference_stems = {
        Path(s.source).stem for s in _reference_samples(drawn)
        if Path(s.ground_truth).parent.name == date
    }
    assert reference_stems  # the fixture's own three-way ratio gives this date some

    _run_dir, ckpt_path = _bound_checkpoint(tmp_path, manifest_dir, "exp-pq-flat")
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt_path, images_dir=str(root / "images"))
    assert "error" not in r, r
    assert r["queue"], r
    marked_true = {Path(e["image"]).stem for e in r["queue"] if e["reference_member"]}
    assert marked_true == reference_stems


def test_prioritize_review_queue_a_bound_run_never_marks_another_dates_calibration_side(
    tmp_path, monkeypatch,
):
    """A selection spanning two dates (one canonical, the other bucketed-labels-flat-images like
    the single-date rail above) marks only the calibration samples whose own source sits in the
    queue's images directory: a sample the selection holds under the other date's own images
    bucket must never read as a member here, even though the same stem name recurs under both."""
    canonical_date, flat_date = "2026-03-01", "2026-03-15"
    root = _mixed_two_date_dataset(tmp_path / "data", canonical_date, flat_date)
    manifest_dir = tmp_path / "manifest"
    # A seed whose draw puts calibration samples under both dates, a stem among them under one only.
    drawn = _draw_flat(tmp_path, root, manifest_dir, seed=3)
    here = (root / "images").resolve()
    bound_stems = {Path(s.source).stem for s in _reference_samples(drawn)
                   if Path(s.source).parent.resolve() == here}
    other_stems = {Path(s.source).stem for s in _reference_samples(drawn)
                   if Path(s.source).parent.resolve() != here}
    assert bound_stems and other_stems
    leaked = other_stems - bound_stems
    assert leaked  # a stem calibration-only under the other date; the case that must not leak

    _run_dir, ckpt_path = _bound_checkpoint(tmp_path, manifest_dir, "exp-pq-two-dates")
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt_path, images_dir=str(root / "images"))
    assert "error" not in r, r
    assert r["queue"], r
    marked_true = {Path(e["image"]).stem for e in r["queue"] if e["reference_member"]}
    assert marked_true == bound_stems
    assert not (marked_true & leaked)


def test_prioritize_review_queue_signature_drops_the_triage_only_parameters():
    """prioritize_review_queue carries no strategy flag and none of triage_predictions's own
    knobs."""
    import inspect

    params = inspect.signature(prioritize_review_queue).parameters
    assert "strategy" not in params
    assert "low" not in params
    assert "high" not in params
    assert "auto_threshold" not in params


def test_triage_predictions_signature_carries_the_triage_only_parameters():
    """triage_predictions carries the confidence-triage parameters: low, high and
    auto_threshold."""
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
             "iscrowd": False, "reviewed_by": "", "conf": None, "class_id": None, "producer_identity": None, "conf_threshold": None, "missed_object_attested": False, "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
        (bucket, "imgB.png"): {"img_status": "completed", "detections": [
            {"action": "rejected", "class_name": "diseased",
             "iscrowd": False, "reviewed_by": "", "conf": None, "class_id": None, "producer_identity": None, "conf_threshold": None, "missed_object_attested": False, "gt_bbox_norm": None, "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]}]},
    }}
    from tcip_annotation.review_engine import ReviewEngine

    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update(state)
    engine.save_review_state()
    return state_dir


def _published_classified_bucket(project: Path, dataset_root: Path) -> str:
    """The classified bucket the seeded verdicts were recorded against, published under the scope
    its checkpoint records; its directory."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    bucket_dir = dataset_root / CLASSIFIED_BUCKET
    published(project, bucket_dir, [], scope={
        "subject": CLASSIFIED_SUBJECT, "attribute": CLASSIFIED_ATTRIBUTE,
        "id_map": {"healthy": 0, "diseased": 1}})
    return str(bucket_dir)


def _source_dataset_with_registry(root: Path) -> Path:
    """A dataset root carrying a real subject registry, with the two reviewed images under its
    own images/ (the segment dataset_root_of needs to locate the root back from it)."""
    dataset_root = root / "source_dataset"
    images = dataset_root / "images"
    _source_images(images)
    (dataset_root / "subjects.json").write_text(
        '{"leaf": {"attributes": {"condition": {"type": "categorical", '
        '"values": ["healthy", "diseased"]}}}}',
        encoding="utf-8",
    )
    return dataset_root


def test_materialize_writes_positives_under_a_classified_scope_in_the_ground_truth_shape(tmp_path):
    """The object class lands in subject, the confirmed value under the scope's own attribute,
    never the verdict-name-derived subject a detector review would write."""
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(project_state_dir(dataset_root))
    bucket = _published_classified_bucket(tmp_path, dataset_root)
    source = _source_dataset_with_registry(tmp_path)

    r = materialize_review_dataset(
        tmp_path, str(dataset_root), str(source / "images"), str(tmp_path / "out"),
        predictions_dir=bucket)

    assert "error" not in r
    assert r["positive"] == 1
    assert r["subject"] == CLASSIFIED_SUBJECT
    assert r["attribute"] == CLASSIFIED_ATTRIBUTE
    from tcip_annotation.json_io import read_annotations

    anns = read_annotations(str(tmp_path / "out" / "annotations" / "imgA.json"))
    assert anns[0].subject == CLASSIFIED_SUBJECT
    assert anns[0].attributes == {CLASSIFIED_ATTRIBUTE: "healthy"}


def test_materialize_never_confirms_a_negative_under_a_classified_scope(tmp_path):
    """A rejected value call names the model's wrong-state guess, never the object's absence, so
    the rejected-only image is named in unconfirmed_negatives and no confirmed-negative status
    is ever recorded for it, even though its label file is still an empty background."""
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(project_state_dir(dataset_root))
    bucket = _published_classified_bucket(tmp_path, dataset_root)
    source = _source_dataset_with_registry(tmp_path)
    out = tmp_path / "out"

    r = materialize_review_dataset(
        tmp_path, str(dataset_root), str(source / "images"), str(out), predictions_dir=bucket)

    assert "error" not in r
    assert len(r["unconfirmed_negatives"]) == 1
    assert r["unconfirmed_negatives"][0]["image"] == "imgB.png"
    from tcip_mcp.dataset_layout import read_image_status_store

    assert read_image_status_store(out) == {}


def test_materialize_copies_the_source_registry_under_a_classified_scope(tmp_path):
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(project_state_dir(dataset_root))
    bucket = _published_classified_bucket(tmp_path, dataset_root)
    source = _source_dataset_with_registry(tmp_path)
    out = tmp_path / "out"

    r = materialize_review_dataset(
        tmp_path, str(dataset_root), str(source / "images"), str(out), predictions_dir=bucket)

    assert "error" not in r
    assert (out / "subjects.json").is_file()
    assert (out / "subjects.json").read_text(encoding="utf-8") == (
        (source / "subjects.json").read_text(encoding="utf-8"))


def test_materialize_refuses_a_classified_scope_with_no_source_registry(tmp_path):
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(project_state_dir(dataset_root))
    bucket = _published_classified_bucket(tmp_path, dataset_root)
    src = _source_images(tmp_path / "src")  # a bare directory, no dataset root to derive from

    r = materialize_review_dataset(
        tmp_path, str(dataset_root), str(src), str(tmp_path / "out"), predictions_dir=bucket)

    assert "error" in r
    assert "register_dataset" in r["error"]


def test_materialize_refuses_a_classified_scope_into_a_populated_output(tmp_path):
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(project_state_dir(dataset_root))
    bucket = _published_classified_bucket(tmp_path, dataset_root)
    source = _source_dataset_with_registry(tmp_path)
    out = tmp_path / "out"
    out.mkdir(parents=True)
    (out / "subjects.json").write_text('{"other": {}}', encoding="utf-8")

    r = materialize_review_dataset(
        tmp_path, str(dataset_root), str(source / "images"), str(out), predictions_dir=bucket)

    assert "error" in r
    assert "already holds a subject registry" in r["error"]
    assert (out / "subjects.json").read_text(encoding="utf-8") == '{"other": {}}'


@pytest.mark.parametrize("subject", [CLASSIFIED_SUBJECT, ""])
def test_materialize_refuses_a_subject_stated_beside_a_published_bucket(tmp_path, subject):
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(project_state_dir(dataset_root))
    bucket = _published_classified_bucket(tmp_path, dataset_root)
    source = _source_dataset_with_registry(tmp_path)

    r = materialize_review_dataset(
        tmp_path, str(dataset_root), str(source / "images"), str(tmp_path / "out"),
        predictions_dir=bucket, subject=subject)

    assert "would be a second one" in r.get("error", ""), r


def test_materialize_refuses_an_undecodable_bucket_record(tmp_path):
    dataset_root = tmp_path / "dataset"
    _seed_classified_verdicts(project_state_dir(dataset_root))
    bucket = _published_classified_bucket(tmp_path, dataset_root)
    (Path(bucket) / "bucket.json").write_bytes(b"{not json")
    src = _source_images(tmp_path / "src")

    r = materialize_review_dataset(
        tmp_path, str(dataset_root), str(src), str(tmp_path / "out"), predictions_dir=bucket)

    assert "does not decode" in r["error"]


def test_materialize_names_the_reviewed_buckets_when_none_is_named(tmp_path):
    """A store holding a bucket's verdicts, read with no bucket named, refuses naming the bucket
    rather than guessing which review to curate."""
    dataset_root, src = _setup(tmp_path)

    r = materialize_review_dataset(tmp_path, str(dataset_root), str(src), str(tmp_path / "out"))

    assert repr(BUCKET) in r["error"] and "predictions_dir" in r["error"]
