"""Review-queue MCP tools: the ranked queue and triage, each skipping what the label documents
mark finished."""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES
from tcip_mcp.tools.feedback_tools import prioritize_review_queue
from tests._verified_checkpoint_fixtures import SAMPLE_MAX_DETS

DATE = "2026-03-04"


def _dataset(tmp_path: Path, *, marked: tuple[str, ...]) -> Path:
    """A dataset of two images on one date, those named in ``marked`` marked complete for ``bud``
    through the editor's own save door; its images directory."""
    from PIL import Image

    from tcip_mcp.dataset_layout import image_dir
    from tests._producer_fixtures import label_image, mark_complete

    root = tmp_path / "dataset"
    images = image_dir(root, DATE)
    images.mkdir(parents=True)
    for stem in ("imgA", "imgB"):
        Image.new("RGB", (64, 64), (120, 120, 120)).save(images / f"{stem}.png")
        label_image(images / f"{stem}.png", [], 64, 64, keep_empty=True)
        if stem in marked:
            mark_complete(images / f"{stem}.png", "bud", project=tmp_path)
    return images


def test_prioritize_review_queue_checkpoint_missing(tmp_path):
    r = prioritize_review_queue(tmp_path, str(tmp_path / "nope.pt"), str(tmp_path))
    assert "error" in r  # early guard, no torch import needed


def test_prioritize_review_queue_skips_the_images_marked_finished(tmp_path):
    """Both marked images drop out before any scorer runs, the plumbing triage_predictions
    shares."""
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images = _dataset(tmp_path, marked=("imgA", "imgB"))
    ckpt = registered_checkpoint(tmp_path)

    r = prioritize_review_queue(tmp_path, checkpoint_path=ckpt, images_dir=str(images),
                                subject="bud")
    assert r["reviewed_skipped"] == 2
    assert r["total_candidates"] == 0
    assert r["queue"] == []


def test_triage_predictions_skips_only_the_images_marked_finished(tmp_path, monkeypatch):
    """A marked image drops out of the queue; an unmarked one, and every image of a subject nobody
    marked, is still triaged."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images = _dataset(tmp_path, marked=("imgA",))
    ckpt = registered_checkpoint(tmp_path)
    monkeypatch.setattr(predmod, "GenericPredictor", lambda *a, **k: SimpleNamespace(
        predict_batch=lambda sources, **kw: [{"image": Path(s).name, "scores": [0.5]}
                                             for s in sources]))

    r = triage_predictions(tmp_path, checkpoint_path=str(ckpt), images_dir=str(images),
                           subject="bud", max_dets=SAMPLE_MAX_DETS)
    assert r["reviewed_skipped"] == 1
    assert r["review_images"] == ["imgB.png"]

    other = triage_predictions(tmp_path, checkpoint_path=str(ckpt), images_dir=str(images),
                               subject="leaf", max_dets=SAMPLE_MAX_DETS)
    assert other["reviewed_skipped"] == 0
    assert other["review_images"] == ["imgA.png", "imgB.png"]


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
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")

    predictions = [{"image": "a.jpg", "width": 4, "height": 4, "head0_values": [0.42]}]
    monkeypatch.setattr(
        predmod, "GenericPredictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources, **kw: predictions))

    r = triage_predictions(
        tmp_path, checkpoint_path=str(ckpt), images_dir=str(images), max_dets=SAMPLE_MAX_DETS)
    assert r["needs_review"] == 1
    assert r["review_images"] == ["a.jpg"]
    assert r["unscoreable_images"] == ["a.jpg"]


def _stubbed_triage_predictions(tmp_path, monkeypatch, predictions: list[dict], **kwargs):
    """triage_predictions against a real registered checkpoint with predict_batch stubbed to a
    fixed set of confidence-bearing and unscoreable predictions, one call site both door-level
    triage tests share."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    for pred in predictions:
        (images / pred["image"]).write_bytes(b"x")
    monkeypatch.setattr(
        predmod, "GenericPredictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources, **kw: predictions))

    return triage_predictions(tmp_path, checkpoint_path=str(ckpt), images_dir=str(images),
                              **{"max_dets": SAMPLE_MAX_DETS, **kwargs})


def _triage_over_one_image(tmp_path, monkeypatch) -> tuple[str, str, list]:
    """A registered detector, a capture holding one image, and the execution records its stubbed
    predictor is run under, appended as it runs."""
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")
    ran_at: list = []

    def _predict(sources, **kw):
        ran_at.append(kw["execution"])
        return [{"image": "a.jpg", "scores": [0.35]}]

    monkeypatch.setattr(predmod, "GenericPredictor",
                        lambda *a, **k: SimpleNamespace(predict_batch=_predict))
    return str(ckpt), str(images), ran_at


def test_triage_refuses_a_detector_call_stating_no_cap(tmp_path, monkeypatch):
    """No default stands behind a detector's cap: a triage call stating no ``max_dets`` refuses
    naming it and predicts nothing."""
    from tcip_mcp.tools.feedback_tools import triage_predictions

    ckpt, images, ran_at = _triage_over_one_image(tmp_path, monkeypatch)

    refused = triage_predictions(tmp_path, checkpoint_path=ckpt, images_dir=images, low=0.3)

    assert "max_dets" in refused.get("error", ""), refused
    assert ran_at == []


def test_triage_predicts_a_detector_at_its_low_bound(tmp_path, monkeypatch):
    """Every box the band can hold exists: a detector predicts at ``low`` itself, never at a
    default above it, under the cap the call states."""
    from tcip_mcp.tools.feedback_tools import triage_predictions

    ckpt, images, ran_at = _triage_over_one_image(tmp_path, monkeypatch)

    triaged = triage_predictions(tmp_path, checkpoint_path=ckpt, images_dir=images, low=0.3,
                                 max_dets=SAMPLE_MAX_DETS)

    assert [(e.conf, e.max_dets) for e in ran_at] == [(0.3, SAMPLE_MAX_DETS)]
    assert triaged["review_images"] == ["a.jpg"]


def test_triage_predictions_routes_the_band_and_the_unscoreable_and_accepts_nothing(
        tmp_path, monkeypatch):
    """A confident prediction leaves the queue without being accepted: the door writes nothing,
    while the mid-confidence and unscoreable predictions are routed to review."""
    predictions = [
        {"image": "high.jpg", "scores": [0.9]},
        {"image": "mid.jpg", "scores": [0.5]},
        {"image": "unscoreable.jpg", "head0_values": [0.42]},
    ]
    r = _stubbed_triage_predictions(tmp_path, monkeypatch, predictions)
    assert r == {"total_images": 3, "reviewed_skipped": 0, "needs_review": 2,
                 "review_images": ["mid.jpg", "unscoreable.jpg"],
                 "unscoreable_images": ["unscoreable.jpg"]}


def test_unresolvable_scorer_raises_valueerror_not_an_import_error():
    """The refusal is a ValueError whatever the name looks like.

    ``resolve_scorer``'s callers catch ``ValueError`` to turn a refusal into an error dict: a
    dotted name that fails to import must raise ``ValueError`` too, never
    ``ModuleNotFoundError`` straight out of the audited MCP tool.
    """
    import pytest

    from tcip_mcp.pipelines.active_learning.scorer import resolve_scorer

    with pytest.raises(ValueError):
        resolve_scorer("no_such_scorer", "detection")
    with pytest.raises(ValueError, match="Could not import scorer"):
        resolve_scorer("not_a_module.at_all:make", "detection")


def test_unresolvable_proposal_engine_raises_valueerror():
    import pytest

    from tcip_mcp.pipelines.proposal import resolve_proposer

    with pytest.raises(ValueError):
        resolve_proposer("no_such_engine")
    with pytest.raises(ValueError, match="Could not import proposal engine"):
        resolve_proposer("not_a_module.at_all:make")


# -- rail: prioritize_review_queue marks a bound run's calibration-side candidates ------------


def _bespoke_checkpoint_payload() -> dict:
    from tcip_mcp.pipelines.model_build import CONFIG_KEY, STATE_DICT_KEY
    from tests._chain_fixtures import built_model, training_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    config = training_config(BUILT_DETECTOR,
                             {"num_channels": 3, "scope": {"subject": "bud", "attributes": []}})
    model = built_model(config)
    return {CONFIG_KEY: config, STATE_DICT_KEY: model.state_dict()}


def _bound_checkpoint(project: Path, manifest_dir: Path, experiment_id: str) -> tuple[Path, str]:
    """A run of ``project`` bound to the selection at ``manifest_dir``, run through the child's
    own entry (its data resolved and recorded, its body saving the model its config builds), so
    its completed checkpoint's producer is that bound run. Returns ``(run directory, checkpoint
    path)``."""
    from tcip_mcp.experiments import observe
    from tests._chain_fixtures import SAVE_BUILT_WEIGHTS, run_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, worker_run

    run_dir = worker_run(project, run_config(manifest_dir, BUILT_DETECTOR,
                                             training_source=SAVE_BUILT_WEIGHTS),
                         experiment_id=experiment_id)
    checkpoint = observe(run_dir).checkpoint
    assert checkpoint is not None
    return run_dir, checkpoint["path"]


def _stub_scorer(monkeypatch) -> None:
    """Scores every candidate 1.0 in order: the real scorer reads model logits, which this rail
    has no need to exercise, since the calibration mark is computed from the candidate's own
    path and the selection, never from a score."""
    import tcip_mcp.pipelines.active_learning.scorer as al_scorer

    class _Scorer:
        def score(self, sources, predictor):
            return [(s, 1.0) for s in sources]

    monkeypatch.setattr(al_scorer, "resolve_scorer", lambda method, task: _Scorer())


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

    from tests._verified_checkpoint_fixtures import ONE_BAND_DETECTOR, registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, model_source=ONE_BAND_DETECTOR,
                                 data={"num_channels": 1, "scope": {"subject": "bud"}})
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    for stem in ("a", "b"):
        Image.new("RGB", (64, 64), (90, 110, 70)).save(images / f"{stem}.png")

    r = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt, images_dir=str(images), method="combined"
    )

    assert "error" not in r, r
    assert len(r["queue"]) == 2, r
    assert all(isinstance(entry["score"], float) for entry in r["queue"])


def test_prioritize_review_queue_unbound_run_carries_no_marks(tmp_path, monkeypatch):
    """A checkpoint no run of the project produced has nothing bound to check against: no
    ``reference_member`` on any entry."""
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    ckpt = foreign_checkpoint(tmp_path)
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
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


def _labeled_flat_image(path: Path) -> None:
    """A gray image at ``path`` whose label document holds one ``leaf`` box."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import label_image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (_FLAT_IMG, _FLAT_IMG), color=(128, 128, 128)).save(path)
    label_image(path, [Annotation(subject=_FLAT_SUBJECT, geometry=BBox(2, 2, 10, 10))],
                _FLAT_IMG, _FLAT_IMG)


def _flat_images_dataset(root: Path, stems=_FLAT_STEMS) -> Path:
    """Labeled images in the undated capture alone, so the selection records each sample's
    source under it."""
    for stem in stems:
        _labeled_flat_image(root / "images" / UNDATED_BUCKET / f"{stem}.jpg")
    return root


def _mixed_dated_and_flat_dataset(root: Path, date: str, stems=_FLAT_STEMS) -> Path:
    """One capture's labeled images under ``images/<date>/`` and the same stems' labeled images
    in the undated capture beside it."""
    for stem in stems:
        _labeled_flat_image(root / "images" / date / f"{stem}.jpg")
    _flat_images_dataset(root, stems)
    return root


def _draw_flat(project: Path, root: Path, out: Path, *, seed: int = 2):
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(project, str(root), output_path=str(out), subject=_FLAT_SUBJECT, seed=seed,
                         val_ratio=0.3, calibration_ratio=0.15, holdout_ratio=0.15)
    assert "error" not in result, result
    return read_selection(out, project=project)


def test_prioritize_review_queue_marks_an_undated_capture_correctly(tmp_path, monkeypatch):
    """A dataset whose images are its undated capture alone marks its calibration side
    correctly: each sample's own recorded source names that capture."""
    root = _flat_images_dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    drawn = _draw_flat(tmp_path, root, manifest_dir)
    reference_stems = {Path(s.source).stem for s in _reference_samples(drawn)}
    assert reference_stems  # the fixture's own three-way ratio gives it some

    _run_dir, ckpt_path = _bound_checkpoint(tmp_path, manifest_dir, "exp-pq-flat")
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt_path, images_dir=str(root / "images" / UNDATED_BUCKET))
    assert "error" not in r, r
    assert r["queue"], r
    marked_true = {Path(e["image"]).stem for e in r["queue"] if e["reference_member"]}
    assert marked_true == reference_stems


def test_prioritize_review_queue_a_bound_run_never_marks_another_dates_calibration_side(
    tmp_path, monkeypatch,
):
    """A selection spanning a dated and the undated capture marks only the calibration
    samples whose own source sits in the queue's images directory: a sample the selection holds
    under the dated capture's tree must never read as a member here, even though the same stem
    name recurs under both."""
    root = _mixed_dated_and_flat_dataset(tmp_path / "data", "2026-03-01")
    manifest_dir = tmp_path / "manifest"
    # A seed whose draw puts calibration samples under both dates, a stem among them under one only.
    drawn = _draw_flat(tmp_path, root, manifest_dir, seed=3)
    here = (root / "images" / UNDATED_BUCKET).resolve()
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
        tmp_path, checkpoint_path=ckpt_path, images_dir=str(root / "images" / UNDATED_BUCKET))
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
    """triage_predictions carries the confidence-triage band, low and high, and accepts
    nothing."""
    import inspect

    from tcip_mcp.tools.feedback_tools import triage_predictions

    params = inspect.signature(triage_predictions).parameters
    assert "low" in params
    assert "high" in params
    assert "auto_threshold" not in params


def test_feedback_tools_register_in_manifest():
    from tcip_mcp.server import list_registered_tools
    names = list_registered_tools()
    assert "materialize_review_dataset" not in names
    assert "prioritize_review_queue" in names
