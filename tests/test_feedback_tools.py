"""The review-queue MCP tools: the ranked queue and confidence triage.

Each skips what the label documents mark finished.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir, parse_image_path
from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES
from tcip_mcp.tools.feedback_tools import prioritize_review_queue
from tests._training_values import COMBINED_SCORER, RATIONALE, drawn_split

if TYPE_CHECKING:
    import pytest

    from tcip_mcp.pipelines.data.selection import Selection
    from tcip_mcp.pipelines.execution import Execution
    from tcip_mcp.pipelines.image_utils import BandGroupRef
    from tcip_mcp.pipelines.inference.generic_predictor import (
        GenericPredictor,
    )

DATE = "2026-03-04"
QUEUE = {"scorer": COMBINED_SCORER, "budget": 50, "rationale": RATIONALE}
"""What a test's review queue states: its scorer block, its budget and their reason."""
BAND = {"low": 0.3, "high": 0.8, "rationale": RATIONALE}
"""The needs-review band a test's triage states, with its reason."""


def _dataset(tmp_path: Path, *, marked: tuple[str, ...]) -> Path:
    """Return the images directory of a dataset of two images on one date.

    Each image holds an empty label document, and the images named in ``marked`` are marked
    complete for ``bud`` through ``save_label_document``.
    """
    from tests._producer_fixtures import (
        label_image,
        mark_complete,
        write_image,
    )

    root = tmp_path / "dataset"
    images = image_dir(root, DATE)
    for stem in ("imgA", "imgB"):
        image = write_image(images / f"{stem}.png", (64, 64), (120, 120, 120))
        label_image(image, [], 64, 64, keep_empty=True)
        if stem in marked:
            mark_complete(image, "bud", project=tmp_path)
    return images


def test_prioritize_review_queue_checkpoint_missing(tmp_path: Path) -> None:
    r = prioritize_review_queue(
        tmp_path, str(tmp_path / "nope.pt"), str(tmp_path), **QUEUE
    )
    assert "error" in r  # early guard, no torch import needed


def test_prioritize_review_queue_skips_the_images_marked_finished(
    tmp_path: Path,
) -> None:
    """``prioritize_review_queue`` skips both marked images and queues no candidate."""
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images = _dataset(tmp_path, marked=("imgA", "imgB"))
    ckpt = registered_checkpoint(tmp_path)

    r = prioritize_review_queue(
        tmp_path,
        checkpoint_path=ckpt,
        images_dir=str(images),
        subject="bud",
        **QUEUE,
    )
    assert r["reviewed_skipped"] == 2
    assert r["total_candidates"] == 0
    assert r["queue"] == []


def test_triage_predictions_skips_only_the_images_marked_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Triage skips the image marked finished for its subject and triages the rest.

    The marked image drops out of the queue and the unmarked one is triaged; for a subject
    nobody marked, both images are triaged.
    """
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images = _dataset(tmp_path, marked=("imgA",))
    ckpt = registered_checkpoint(tmp_path)
    monkeypatch.setattr(
        predmod,
        "GenericPredictor",
        lambda *a, **k: SimpleNamespace(
            predict_batch=lambda sources, **kw: [
                {"image": Path(s).name, "scores": [0.5]} for s in sources
            ]
        ),
    )

    r = triage_predictions(
        tmp_path,
        checkpoint_path=str(ckpt),
        images_dir=str(images),
        subject="bud",
        **BAND,
    )
    assert r["reviewed_skipped"] == 1
    assert r["review_images"] == ["imgB.png"]

    other = triage_predictions(
        tmp_path,
        checkpoint_path=str(ckpt),
        images_dir=str(images),
        subject="leaf",
        **BAND,
    )
    assert other["reviewed_skipped"] == 0
    assert other["review_images"] == ["imgA.png", "imgB.png"]


def test_triage_predictions_surfaces_unscoreable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Triage routes predictions carrying no confidence to review, tagged unscoreable.

    A prediction carrying ``head0_values`` and no scores is routed to review and listed in
    ``unscoreable_images``.
    """
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)
    images = image_dir(tmp_path, UNDATED_BUCKET)
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")

    predictions = [
        {"image": "a.jpg", "width": 4, "height": 4, "head0_values": [0.42]}
    ]
    monkeypatch.setattr(
        predmod,
        "GenericPredictor",
        lambda *a, **k: SimpleNamespace(
            predict_batch=lambda sources, **kw: predictions
        ),
    )

    r = triage_predictions(
        tmp_path, checkpoint_path=str(ckpt), images_dir=str(images), **BAND
    )
    assert r["needs_review"] == 1
    assert r["review_images"] == ["a.jpg"]
    assert r["unscoreable_images"] == ["a.jpg"]


def _stubbed_triage_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predictions: list[dict],
    **kwargs: float | str,
) -> dict:
    """Run triage_predictions over a real registered checkpoint with predict_batch stubbed.

    The stubbed ``predict_batch`` answers ``predictions`` as given, and ``kwargs`` override the
    sample band.
    """
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)
    images = image_dir(tmp_path, UNDATED_BUCKET)
    images.mkdir(parents=True)
    for pred in predictions:
        (images / pred["image"]).write_bytes(b"x")
    monkeypatch.setattr(
        predmod,
        "GenericPredictor",
        lambda *a, **k: SimpleNamespace(
            predict_batch=lambda sources, **kw: predictions
        ),
    )

    return triage_predictions(
        tmp_path,
        checkpoint_path=str(ckpt),
        images_dir=str(images),
        **{**BAND, **kwargs},
    )


def _triage_over_one_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str, list]:
    """Return a registered detector, a one-image capture, and its predictor's executions.

    The execution records are those the stubbed predictor is run under, appended as it runs.
    """
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)
    images = image_dir(tmp_path, UNDATED_BUCKET)
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")
    ran_at: list = []

    def _predict(
        sources: list[str | Path | BandGroupRef], **kw: Execution | int
    ) -> list[dict]:
        ran_at.append(kw["execution"])
        return [{"image": "a.jpg", "scores": [0.35]}]

    monkeypatch.setattr(
        predmod,
        "GenericPredictor",
        lambda *a, **k: SimpleNamespace(predict_batch=_predict),
    )
    return str(ckpt), str(images), ran_at


def test_triage_predicts_a_detector_at_its_low_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Triage predicts a detector at the band's low bound.

    The detector predicts at ``low`` itself, each frame capped at the density its checkpoint
    recorded, and the answer carries the band.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.tools.feedback_tools import triage_predictions

    ckpt, images, ran_at = _triage_over_one_image(tmp_path, monkeypatch)

    triaged = triage_predictions(
        tmp_path, checkpoint_path=ckpt, images_dir=images, **BAND
    )

    density = load_registered_checkpoint(
        ckpt, project=tmp_path
    ).geometry.object_density
    assert [(e.conf, e.density) for e in ran_at] == [(0.3, density)]
    assert triaged["review_images"] == ["a.jpg"]
    assert {name: triaged[name] for name in BAND} == BAND


def test_triage_predictions_routes_the_band_and_the_unscoreable_and_accepts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Triage routes the band and the unscoreable to review and answers no acceptance.

    A confident prediction leaves the queue without being accepted: the answer names no accepted
    image, while the mid-confidence and unscoreable predictions are routed to review.
    """
    predictions = [
        {"image": "high.jpg", "scores": [0.9]},
        {"image": "mid.jpg", "scores": [0.5]},
        {"image": "unscoreable.jpg", "head0_values": [0.42]},
    ]
    r = _stubbed_triage_predictions(tmp_path, monkeypatch, predictions)
    assert r == {
        **BAND,
        "total_images": 3,
        "reviewed_skipped": 0,
        "needs_review": 2,
        "review_images": ["mid.jpg", "unscoreable.jpg"],
        "unscoreable_images": ["unscoreable.jpg"],
    }


def test_unresolvable_scorer_raises_valueerror_not_an_import_error() -> None:
    """An unknown scorer name and a dotted one that fails to import each raise ``ValueError``."""
    import pytest

    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.pipelines.active_learning.scorer import resolve_scorer

    with pytest.raises(ValueError):
        resolve_scorer(ScorerSpec(name="no_such_scorer"), "detection")
    with pytest.raises(ValueError, match="Could not import scorer"):
        resolve_scorer(
            ScorerSpec(name="not_a_module.at_all:make"), "detection"
        )


def test_a_scorer_block_states_every_value_its_factory_requires_and_nothing_else() -> (
    None
):
    """A scorer block refuses a missing or untaken factory value and builds at its stated ones.

    A ``combined`` block without its weights refuses naming each missing one, a value the
    factory does not take refuses naming it, and a stated block builds its scorer with the values
    it states.
    """
    import pytest

    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.pipelines.active_learning.scorer import (
        CombinedScorer,
        resolve_scorer,
    )

    with pytest.raises(TypeError, match="uncertainty_weight"):
        resolve_scorer(ScorerSpec(name="combined"), "detection")
    with pytest.raises(TypeError, match="diversity_weight"):
        resolve_scorer(
            ScorerSpec.model_validate(
                {"name": "combined", "uncertainty_weight": 0.5}
            ),
            "detection",
        )
    with pytest.raises(TypeError, match="uncertainty_weight"):
        resolve_scorer(
            ScorerSpec.model_validate(
                {"name": "uncertainty", "uncertainty_weight": 0.5}
            ),
            "detection",
        )
    built = resolve_scorer(COMBINED_SCORER, "detection")
    assert isinstance(built, CombinedScorer)
    assert (built.uw, built.dw) == (0.6, 0.4)


def test_unresolvable_proposal_engine_raises_valueerror() -> None:
    import pytest

    from tcip_mcp.pipelines.proposal import resolve_proposer

    with pytest.raises(ValueError):
        resolve_proposer("no_such_engine")
    with pytest.raises(ValueError, match="Could not import proposal engine"):
        resolve_proposer("not_a_module.at_all:make")


# -- rail: prioritize_review_queue marks a bound run's calibration-side candidates ------------


def _bound_checkpoint(
    project: Path, manifest_dir: Path, experiment_id: str
) -> tuple[Path, str]:
    """Run a detector bound to the selection at ``manifest_dir`` through the child's own entry.

    The run belongs to ``project`` and its body is ``SAVE_BUILT_WEIGHTS``. Return
    ``(run directory, checkpoint path)``.
    """
    from tcip_mcp.experiments import observe
    from tests._chain_fixtures import SAVE_BUILT_WEIGHTS, run_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, worker_run

    run_dir = worker_run(
        project,
        run_config(
            project,
            manifest_dir,
            BUILT_DETECTOR,
            training_source=SAVE_BUILT_WEIGHTS,
        ),
        experiment_id=experiment_id,
    )
    checkpoint = observe(run_dir).checkpoint
    assert checkpoint is not None
    return run_dir, checkpoint.path


def _stub_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the scorer to score each candidate 1.0, in the order handed."""
    import tcip_mcp.pipelines.active_learning.scorer as al_scorer

    class _Scorer:
        def score(
            self, sources: list[str], predictor: GenericPredictor
        ) -> list[tuple[str, float]]:
            return [(s, 1.0) for s in sources]

    monkeypatch.setattr(
        al_scorer, "resolve_scorer", lambda block, task: _Scorer()
    )


def _reference_stems(drawn: Selection, capture: str) -> set[str]:
    """Return the stems of ``drawn``'s calibration and holdout samples imaged in ``capture``."""
    return {
        stem
        for side in REFERENCE_SIDES
        for _root, sample_capture, stem in (
            parse_image_path(s.source) for s in drawn.on(side)
        )
        if sample_capture == capture
    }


def test_prioritize_review_queue_marks_a_bound_runs_reference_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound run's review queue marks candidates on its selection's reference sides.

    A checkpoint whose run was bound to a selection marks each ranked candidate against that
    selection's own calibration and holdout samples under the queue's images directory.
    """
    from tests.test_selection_ground_truth_digests import (
        DATES,
        _dataset,
        _draw,
    )

    root = tmp_path / "data"
    captures = _dataset(root)
    manifest_dir = tmp_path / "manifest"
    drawn = _draw(tmp_path, root, manifest_dir)
    date = DATES[0]
    reference_stems = _reference_stems(drawn, date)
    assert (
        reference_stems
    )  # the fixture's own three-way ratio gives this date some

    _run_dir, ckpt_path = _bound_checkpoint(
        tmp_path, manifest_dir, "exp-pq-marks"
    )
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path,
        checkpoint_path=ckpt_path,
        images_dir=str(captures[date].images_dir),
        **QUEUE,
    )
    assert "error" not in r, r
    assert r["queue"], r
    for entry in r["queue"]:
        stem = Path(entry["image"]).stem
        assert entry["reference_member"] == (stem in reference_stems), entry


def test_the_review_queue_marks_the_runs_frozen_calibration_side_after_the_selection_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The review queue marks the run's frozen reference sides after the selection changed.

    With the selection rewritten after the run, its calibration and holdout samples moved to
    train, the queue still marks the reference samples the drawn selection held under its date.
    """
    import tcip_store as ts
    from tcip_mcp.pipelines.data.selection import (
        selection_key,
        stored_selection_document,
    )
    from tests.test_selection_ground_truth_digests import (
        DATES,
        _dataset,
        _draw,
    )

    root = tmp_path / "data"
    captures = _dataset(root)
    manifest_dir = tmp_path / "manifest"
    drawn = _draw(tmp_path, root, manifest_dir)
    date = DATES[0]
    reference_stems = _reference_stems(drawn, date)
    assert reference_stems

    _run_dir, ckpt_path = _bound_checkpoint(
        tmp_path, manifest_dir, "exp-pq-frozen"
    )

    document = stored_selection_document(drawn, tmp_path)
    assert isinstance(document, dict) and isinstance(document["samples"], list)
    for sample in document["samples"]:
        assert isinstance(sample, dict)
        if sample["side"] in REFERENCE_SIDES:
            sample["side"] = "train"
    ts.replace(selection_key(manifest_dir), document)
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path,
        checkpoint_path=ckpt_path,
        images_dir=str(captures[date].images_dir),
        **QUEUE,
    )
    assert "error" not in r, r
    marked = {
        Path(e["image"]).stem for e in r["queue"] if e["reference_member"]
    }
    assert marked == reference_stems


def test_the_review_queue_scores_candidates_at_the_checkpoints_own_read_width(
    tmp_path: Path,
) -> None:
    """The review queue scores each three-band candidate under a one-channel checkpoint."""
    from tests._producer_fixtures import write_image
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        registered_checkpoint,
    )

    ckpt = registered_checkpoint(
        tmp_path,
        model_source=BUILT_DETECTOR,
        data={"num_channels": 1, "scope": {"subject": "bud"}},
    )
    images = image_dir(tmp_path, UNDATED_BUCKET)
    for stem in ("a", "b"):
        write_image(images / f"{stem}.png", (64, 64), (90, 110, 70))

    r = prioritize_review_queue(
        tmp_path,
        checkpoint_path=ckpt,
        images_dir=str(images),
        **QUEUE,
    )

    assert "error" not in r, r
    assert len(r["queue"]) == 2, r
    assert all(isinstance(entry["score"], float) for entry in r["queue"])


def test_prioritize_review_queue_unbound_run_carries_no_marks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unbound run's review queue carries no reference marks.

    A checkpoint whose registry entry names no producing run has nothing bound to check
    against: no ``reference_member`` on any entry.
    """
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    ckpt = foreign_checkpoint(tmp_path)
    images = image_dir(tmp_path, UNDATED_BUCKET)
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt, images_dir=str(images), **QUEUE
    )
    assert "error" not in r, r
    assert r["queue"], r
    assert all("reference_member" not in entry for entry in r["queue"])


def test_the_queue_operation_reads_the_locations_its_arrival_established(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ranked_review_queue`` does not locate its checkpoint or images directory again.

    The queue the web route launches over the checkpoint and images directory its own arrival
    located ranks them as given: nothing beneath the arrival locates either again.
    """
    import tcip_mcp.model_registry as model_registry
    import tcip_mcp.tools.feedback_tools as feedback_tools
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = Path(registered_checkpoint(tmp_path))
    images = image_dir(tmp_path, UNDATED_BUCKET)
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")
    _stub_scorer(monkeypatch)
    located_again: list = []

    def located(path: str | Path, project: str | Path | None) -> Path:
        located_again.append(path)
        return Path(path)

    for module in (feedback_tools, model_registry):
        monkeypatch.setattr(module, "located", located)

    r = feedback_tools.ranked_review_queue(
        tmp_path,
        ckpt,
        images,
        scorer=COMBINED_SCORER,
        budget=5,
        rationale=RATIONALE,
        subject=None,
    )

    assert "error" not in r, r
    assert r["queue"], r
    assert located_again == []
    assert (r["scorer"], r["budget"], r["rationale"]) == (
        COMBINED_SCORER.model_dump(),
        5,
        RATIONALE,
    )


# -- rail: calibration marks are decided by each sample's own recorded source ------------------

_FLAT_SUBJECT = "leaf"
_FLAT_IMG = 32
_FLAT_STEMS = ("a", "b", "c", "d", "e", "f")


def _labeled_flat_capture(images_dir: Path, stems: tuple[str, ...]) -> Path:
    """Write a gray :func:`labeled_frame` of one ``leaf`` box per stem into ``images_dir``.

    Returns ``images_dir``.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import labeled_frame

    for stem in stems:
        labeled_frame(
            images_dir / f"{stem}.jpg",
            [Annotation(subject=_FLAT_SUBJECT, geometry=BBox(2, 2, 10, 10))],
            _FLAT_IMG,
            _FLAT_IMG,
            (128, 128, 128),
        )
    return images_dir


def _flat_images_dataset(
    root: Path, stems: tuple[str, ...] = _FLAT_STEMS
) -> Path:
    """Write labeled images into ``root``'s undated capture alone; return that capture."""
    return _labeled_flat_capture(image_dir(root, UNDATED_BUCKET), stems)


def _mixed_dated_and_flat_dataset(
    root: Path, date: str, stems: tuple[str, ...] = _FLAT_STEMS
) -> Path:
    """Write labeled images under ``images/<date>/`` and the same stems in the undated capture.

    Returns the undated capture.
    """
    _labeled_flat_capture(image_dir(root, date), stems)
    return _flat_images_dataset(root, stems)


def _draw_flat(
    project: Path, root: Path, out: Path, *, seed: int = 2
) -> Selection:
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(
        project,
        str(root),
        output_path=str(out),
        subject=_FLAT_SUBJECT,
        **drawn_split(
            seed=seed,
            val_ratio=0.3,
            calibration_ratio=0.15,
            holdout_ratio=0.15,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in result, result
    return read_selection(out, project=project)


def test_prioritize_review_queue_marks_an_undated_capture_correctly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run bound over an undated capture alone marks its reference sides correctly.

    Over a dataset whose images are its undated capture alone, the queue marks exactly the stems
    of the drawn selection's calibration and holdout samples.
    """
    root = tmp_path / "data"
    undated = _flat_images_dataset(root)
    manifest_dir = tmp_path / "manifest"
    drawn = _draw_flat(tmp_path, root, manifest_dir)
    reference_stems = _reference_stems(drawn, UNDATED_BUCKET)
    assert reference_stems  # the fixture's own three-way ratio gives it some

    _run_dir, ckpt_path = _bound_checkpoint(
        tmp_path, manifest_dir, "exp-pq-flat"
    )
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path,
        checkpoint_path=ckpt_path,
        images_dir=str(undated),
        **QUEUE,
    )
    assert "error" not in r, r
    assert r["queue"], r
    marked_true = {
        Path(e["image"]).stem for e in r["queue"] if e["reference_member"]
    }
    assert marked_true == reference_stems


def test_prioritize_review_queue_a_bound_run_never_marks_another_dates_calibration_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound run does not mark a reference sample from another capture's directory.

    A selection spanning a dated and the undated capture marks exactly the calibration samples
    whose source sits in the queue's images directory, and no stem that is a calibration sample
    under the dated capture alone.
    """
    dated = "2026-03-01"
    root = tmp_path / "data"
    undated = _mixed_dated_and_flat_dataset(root, dated)
    manifest_dir = tmp_path / "manifest"
    # A seed whose draw puts calibration samples under both dates, a stem among them under one only.
    drawn = _draw_flat(tmp_path, root, manifest_dir, seed=3)
    bound_stems = _reference_stems(drawn, UNDATED_BUCKET)
    other_stems = _reference_stems(drawn, dated)
    assert bound_stems and other_stems
    leaked = other_stems - bound_stems
    assert leaked  # a stem calibration-only under the other date; the case that must not leak

    _run_dir, ckpt_path = _bound_checkpoint(
        tmp_path, manifest_dir, "exp-pq-two-dates"
    )
    _stub_scorer(monkeypatch)

    r = prioritize_review_queue(
        tmp_path,
        checkpoint_path=ckpt_path,
        images_dir=str(undated),
        **QUEUE,
    )
    assert "error" not in r, r
    assert r["queue"], r
    marked_true = {
        Path(e["image"]).stem for e in r["queue"] if e["reference_member"]
    }
    assert marked_true == bound_stems
    assert not (marked_true & leaked)


def test_prioritize_review_queue_signature_drops_the_triage_only_parameters() -> (
    None
):
    """``prioritize_review_queue`` carries no strategy flag and none of triage's own knobs."""
    import inspect

    params = inspect.signature(prioritize_review_queue).parameters
    assert "strategy" not in params
    assert "low" not in params
    assert "high" not in params
    assert "auto_threshold" not in params


def test_the_review_queue_takes_its_scorer_budget_and_reason_with_no_default() -> (
    None
):
    """The review queue takes its scorer, budget and reason with no default.

    ``prioritize_review_queue`` and ``ranked_review_queue`` give ``scorer``, ``budget`` and
    ``rationale`` no default, and the web payload requires all three.
    """
    import inspect

    from tcip_mcp.tools.feedback_tools import ranked_review_queue
    from tcip_web.routes.annotate import LaunchPriorityQueuePayload

    for door in (prioritize_review_queue, ranked_review_queue):
        params = inspect.signature(door).parameters
        for name in ("scorer", "budget", "rationale"):
            assert params[name].default is inspect.Parameter.empty, (
                door,
                name,
            )
    fields = LaunchPriorityQueuePayload.model_fields
    assert all(
        fields[name].is_required()
        for name in ("scorer", "budget", "rationale")
    )


def test_triage_predictions_signature_carries_the_triage_only_parameters() -> (
    None
):
    """``triage_predictions`` takes its band and reason with no default.

    ``triage_predictions`` and ``review_queue`` give ``low`` and ``high`` no default, and
    ``triage_predictions`` gives ``rationale`` none and takes no ``auto_threshold``.
    """
    import inspect

    from tcip_mcp.pipelines.active_learning.selector import review_queue
    from tcip_mcp.tools.feedback_tools import triage_predictions

    params = inspect.signature(triage_predictions).parameters
    for door in (triage_predictions, review_queue):
        for name in ("low", "high"):
            assert (
                inspect.signature(door).parameters[name].default
                is inspect.Parameter.empty
            ), (door, name)
    assert params["rationale"].default is inspect.Parameter.empty
    assert "auto_threshold" not in params


def test_feedback_tools_register_in_manifest() -> None:
    from tcip_web.cli import tool_listing

    names = {t.name for t in tool_listing()}
    assert "materialize_review_dataset" not in names
    assert "prioritize_review_queue" in names
