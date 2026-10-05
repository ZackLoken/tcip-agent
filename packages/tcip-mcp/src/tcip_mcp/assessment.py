"""An assessment: a run measuring how well one checkpoint, under one execution record, reproduces a
held-out reference, at one delivered kind of one trait revision.

An assessment is a directory under the project's ``.tcip/assessments/``, opened once, holding
``assessment.json`` and, under ``reference/``, a copy of every reference ground truth it measured
from, label document, mask, table and physical-extent table alike. It records the reference, the
source digest of every reference sample, the disjointness and criterion evidence and the trait
revision it was judged under; its ``passed`` says a number from that checkpoint, under that
execution record, is validated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from operator import methodcaller
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from tcip_store import Key

from tcip_mcp.pipelines.block_calibration import DEFAULT_K_CAL, DEFAULT_K_TEST
from tcip_mcp.pipelines.data.selection import (
    GROUND_TRUTH_PATHS, ground_truth_of, ground_truth_record,
)
from tcip_mcp.pipelines.execution import DEFAULT_TILE_BATCH_SIZE, Execution, Stated
from tcip_mcp.registry_paths import PathFields, recorded_paths, runtime_paths, within
from tcip_mcp.traits import (
    DETECTOR_KINDS,
    PER_IMAGE_COUNT,
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_ORDINAL_AGGREGATE,
    STATE_CROSSING_DATES,
)

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import Sample, Selection
    from tcip_mcp.pipelines.execution import Pass
    from tcip_mcp.traits import TraitEntry, TraitRevision

ASSESSMENTS_DIR = Path(".tcip/assessments")
ASSESSMENT_FILE = "assessment.json"
REFERENCE_DIR = "reference"

_PATHS: PathFields = (
    ("reference", "selection_dir"), ("reference", "samples", "[]", "source"),
    *within(("reference", "samples", "[]", "ground_truth"), GROUND_TRUTH_PATHS),
    *within(("reference", "ground_truth", "[]", "ground_truth"), GROUND_TRUTH_PATHS),
)


class AssessmentRefused(ValueError):
    """An assessment that cannot run as asked: a selection without both reference sides, a
    checkpoint whose head does not produce what the delivered kind measures, or a reference the
    criterion cannot read."""


@dataclass(frozen=True)
class RetainedFile:
    """One reference ground truth: the label document's key or the file's path it was read at,
    its :func:`~tcip_mcp.pipelines.data.selection.ground_truth_digest` when retained, and its copy
    under the assessment's ``reference/``."""

    ground_truth: Key | str
    digest: str
    copy: str


@dataclass(frozen=True)
class ReferenceSample:
    """One reference sample: its side, member, source, source digest, group, ground truth, the row
    its table names it by and the region of its source it is."""

    side: str
    member: str
    source: str
    source_digest: str
    group: str
    ground_truth: Key | str
    row_key: str | None
    rect: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class Reference:
    """What an assessment measured: the selection it came from, or the mosaic's
    ``raster_identity``, the captures it spans, each retained ground-truth file and every sample."""

    selection_dir: str | None
    raster_identity: dict | None
    captures: list[tuple[str | None, str | None]]
    ground_truth: list[RetainedFile]
    samples: list[ReferenceSample]

    def covers(self, *, dataset_id: str | None, date: str | None,
               raster_identity: dict | None) -> bool:
        """Whether this reference spans a capture: the same mosaic for a raster, one of its
        captures for per-image predictions."""
        if raster_identity is not None:
            return self.raster_identity == raster_identity
        return (dataset_id, date) in self.captures

    def moved(self, run_dir: Path) -> list[str]:
        """Each retained ground truth whose original, or whose copy under ``run_dir``, no longer
        digests to the digest recorded for it, by its stored spelling, then each reference source
        whose pixels no longer digest to the source digest recorded for its sample
        (:func:`~tcip_mcp.pipelines.data.selection.source_digests`), or a source file gone, by
        its path."""
        from tcip_mcp.pipelines.data.selection import Sample, moved_ground_truth, source_digests

        moved = [str(ground_truth_record(f.ground_truth)) for f in self.ground_truth
                 if moved_ground_truth({f.ground_truth: f.digest,
                                        str(run_dir / f.copy): f.digest})]
        samples = {Sample(member=s.member, source=s.source, ground_truth=s.ground_truth,
                          group=s.group, side=s.side, rect=s.rect,
                          row_key=s.row_key): s.source_digest for s in self.samples}
        try:
            now = source_digests(samples)
        except FileNotFoundError as exc:
            return [*moved, str(exc.filename)]
        return [*moved, *dict.fromkeys(sample.source for sample, digest in samples.items()
                                       if now[sample.location] != digest)]


@dataclass(frozen=True)
class Assessment:
    """One finished assessment, as its ``assessment.json`` states it.

    ``delivery_kind``, ``producer`` (:attr:`~tcip_mcp.model_registry.VerifiedCheckpoint.producer`)
    and ``execution`` are ``None`` for a physical-scale assessment. ``disjointness`` and
    ``criterion`` are the evidence the failures were read from, ``scale`` the scale a
    physical-scale assessment derived. ``revision`` is the trait revision it was judged under, as
    :attr:`~tcip_mcp.traits.TraitRevision.ref` names it."""

    assessment_id: str
    delivery_kind: str | None
    revision: dict[str, Any]
    producer: dict[str, str | None] | None
    execution: Execution | None
    reference: Reference
    disjointness: dict[str, Any]
    criterion: dict[str, Any]
    scale: dict[str, Any] | None
    failures: list[str]
    passed: bool


def assessment_dir(project: Path | str, assessment_id: str) -> Path:
    """One assessment's directory. Refuses an id that is not a single directory name."""
    from tcip_mcp.experiments import run_name

    return Path(project) / ASSESSMENTS_DIR / run_name(assessment_id)


def read_assessment(project: Path | str, assessment_id: str) -> Assessment:
    """Assessment ``assessment_id`` of ``project``, its paths resolved against the project. An id
    naming no finished assessment refuses (``ValueError``) naming it."""
    from tcip_mcp.experiments import read_record

    path = assessment_dir(project, assessment_id) / ASSESSMENT_FILE
    if not path.is_file():
        raise ValueError(f"no assessment {assessment_id!r} is recorded under {project}: nothing "
                         "states what it measured.")
    record = runtime_paths(read_record(path), _PATHS, project)
    execution = record.pop("execution")
    raw = record.pop("reference")
    reference = Reference(
        selection_dir=raw["selection_dir"], raster_identity=raw["raster_identity"],
        captures=[(c["dataset_id"], c["date"]) for c in raw["captures"]],
        ground_truth=[RetainedFile(**{**f, "ground_truth": ground_truth_of(f["ground_truth"])})
                      for f in raw["ground_truth"]],
        samples=[ReferenceSample(**{**s, "ground_truth": ground_truth_of(s["ground_truth"]),
                                    "rect": tuple(s["rect"]) if s["rect"] else None})
                 for s in raw["samples"]])
    return Assessment(**record, reference=reference,
                      execution=Execution.of(execution) if execution else None)


def _open_run(project: Path) -> Path:
    """A fresh assessment's directory, named for its id, created once, with its empty reference
    folder."""
    from tcip_mcp.experiments import create_run_directory, mint_experiment_id

    run_dir = create_run_directory(assessment_dir(project, mint_experiment_id("assessment")))
    (run_dir / REFERENCE_DIR).mkdir()
    return run_dir


def _finish(project: Path, run_dir: Path, revision: TraitRevision,
            record: dict[str, Any]) -> Assessment:
    """Write ``record``, judged under ``revision``, as the run's ``assessment.json`` once, then
    its one audit line, ``assessment_recorded``; return it as read back."""
    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.experiments import write_once

    record = {"assessment_id": run_dir.name, "revision": revision.ref, **record,
              "passed": not record["failures"]}
    write_once(run_dir / ASSESSMENT_FILE, recorded_paths(record, _PATHS, project))
    record_event_or_raise(
        "assessment_recorded",
        {"assessment_id": run_dir.name, "trait": revision.entry.name,
         "delivery_kind": record["delivery_kind"], "passed": record["passed"]},
        actor=None, scope=project)
    return read_assessment(project, run_dir.name)


def _reference_reads(samples: list[Sample], extra: tuple[str, ...] = ()
                     ) -> dict[Key | str, Any]:
    """What the store answered (``Versioned``) for each distinct ground truth of the admitted
    ``samples``, their admission's own read (``stored``), and for each file of ``extra``, read
    once here; an ``extra`` file that is not there refuses (:class:`AssessmentRefused`) with the
    store's own ``NotFound`` message."""
    import tcip_store

    reads: dict[Key | str, Any] = {s.ground_truth: s.stored for s in samples}
    for path in extra:
        try:
            reads[path] = tcip_store.read_blob_versioned(Path(path))
        except tcip_store.NotFound as exc:
            raise AssessmentRefused(f"the reference cannot be read, so it cannot be assessed: "
                                    f"{exc}") from exc
    return reads


def _retained(run_dir: Path, samples: list[Sample], reads: dict[Key | str, Any]
              ) -> tuple[list[RetainedFile], list[Sample]]:
    """Each of ``reads`` (:func:`_reference_reads`) copied once into the run's reference folder
    before anything is measured, a label document as its stored record
    (:func:`tcip_store.encode_record`); and ``samples`` measuring what was retained: a mask's or a
    table's copy, and a document's record at the retained version (its ``ground_truth_digest``)."""
    import tcip_store

    from tcip_mcp.experiments import publish_once

    files: dict[Key | str, RetainedFile] = {}
    for i, (gt, stored) in enumerate(reads.items()):
        data = tcip_store.encode_record(stored.value) if isinstance(gt, Key) else stored.value
        name = f"{i:04d}_{gt.parts[-1]}.json" if isinstance(gt, Key) else f"{i:04d}_{Path(gt).name}"
        publish_once(run_dir / REFERENCE_DIR / name, methodcaller("write", data))
        files[gt] = RetainedFile(ground_truth=gt, digest=stored.version.token,
                                 copy=f"{REFERENCE_DIR}/{name}")
    measured = [replace(s, ground_truth_digest=files[s.ground_truth].digest)
                if isinstance(s.ground_truth, Key)
                else replace(s, ground_truth=str(run_dir / files[s.ground_truth].copy))
                for s in samples]
    return list(files.values()), measured


def _reference_record(selection_dir: str | None, raster_identity: dict | None,
                      samples: list[Sample], digest_of: dict[str, str],
                      retained: list[RetainedFile]) -> dict[str, Any]:
    """The one reference record every assessment stores: where the reference came from, the
    captures its samples span, every retained file, and each sample's side, member, source, source
    digest (``digest_of``), group, ground truth, row key and region."""
    from tcip_mcp.dataset_layout import capture_of

    captures = sorted({capture_of(s.source) for s in samples}, key=str)
    return {
        "selection_dir": selection_dir, "raster_identity": raster_identity,
        "captures": [{"dataset_id": dataset_id, "date": date} for dataset_id, date in captures],
        "ground_truth": [{**vars(f), "ground_truth": ground_truth_record(f.ground_truth)}
                         for f in retained],
        "samples": [{"side": s.side, "member": s.member, "source": s.source,
                     "source_digest": digest_of[s.location], "group": s.group,
                     "ground_truth": ground_truth_record(s.ground_truth), "row_key": s.row_key,
                     "rect": list(s.rect) if s.rect is not None else None} for s in samples],
    }


def _reference_sides(project: Path, selection_dir: str
                     ) -> tuple[Selection, list[Sample], list[Sample]]:
    """The selection at ``selection_dir`` and its calibration and holdout sides; a selection
    holding either side empty refuses (:class:`AssessmentRefused`)."""
    from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES, read_selection

    selection = read_selection(selection_dir, project=project)
    cal, hold = (selection.on(side) for side in REFERENCE_SIDES)
    empty = [side for side, samples in zip(REFERENCE_SIDES, (cal, hold)) if not samples]
    if empty:
        raise AssessmentRefused(f"the selection at {selection_dir} holds no {empty} side: an "
                                "assessment fits on the calibration side and checks on the "
                                "holdout side. Draw a selection with both (draw_splits).")
    return selection, cal, hold


def _prepared(project: Path, *, checkpoint_path: str, trait: str, delivery_kind: str,
              stated: Stated, device: str | None,
              tile_batch_size: int) -> tuple[TraitRevision, Pass]:
    """``trait``'s latest confirmed revision stating a ``delivery_kind`` operationalization, and the
    pass the registered checkpoint runs under ``stated``. A checkpoint whose head does not produce
    what the kind measures refuses (:class:`AssessmentRefused`)."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.operationalization import confirmed_revision
    from tcip_mcp.pipelines.execution import prepare_pass

    revision = confirmed_revision(delivery_kind, project=project, trait=trait)
    checkpoint = load_registered_checkpoint(checkpoint_path, project=project)
    from tcip_mcp.pipelines.model_contract import DETECTION_TASKS

    expected = (tuple(sorted(DETECTION_TASKS)) if delivery_kind in DETECTOR_KINDS
                else ("ordinal",) if delivery_kind == PER_PLANT_ORDINAL_AGGREGATE
                else ("regression",))
    if checkpoint.task not in expected:
        raise AssessmentRefused(f"{checkpoint_path} is a {checkpoint.task!r} checkpoint, and a "
                                f"{delivery_kind} delivery is measured off a {expected} head.")
    return revision, prepare_pass(checkpoint, stated, device=device,
                                  tile_batch_size=tile_batch_size)


def _admit_reference(samples: list[Sample], scope: Any) -> list[Sample]:
    """``samples`` as the platform's own admission re-admits them under ``scope``
    (:func:`~tcip_mcp.pipelines.data.label_queries.readmitted_samples`), each carrying its read;
    refuses a reference that admission would not admit, and one whose label documents only the
    model stands behind (:func:`~tcip_annotation.json_io.require_reference_ground_truth`)."""
    from tcip_annotation.json_io import require_reference_ground_truth

    from tcip_mcp.pipelines.data.label_queries import readmitted_samples

    admitted = readmitted_samples(samples, scope)
    require_reference_ground_truth([a for s in admitted if isinstance(s.ground_truth, Key)
                                    for a in s.read.annotations])
    return admitted


def _run_sides(project: Path, experiment_id: str | None) -> dict[str, list[Sample]] | None:
    """The producing run's ``train`` and ``val`` samples, or ``None`` when no run of ``project``
    produced the checkpoint, so no training membership is known."""
    if experiment_id is None:
        return None
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines.data.split_construction import partition_samples

    run = partition_samples(run_resolution(experiment_id, project=project)["partition"])
    return {side: [s for s in run if s.side == side] for side in ("train", "val")}


def _disjointness(digest_of: dict[str, str], cal: list[Sample], hold: list[Sample],
                  run: dict[str, list[Sample]] | None) -> tuple[dict[str, Any], list[str]]:
    """Whether the reference is held out: its calibration and holdout sides share no source digest
    (``digest_of``), and neither shares a group key or a source digest with any of the producing
    ``run``'s sides. ``(evidence, failures)``; ``run`` ``None`` is a run whose membership is
    unknown, a failure of its own, and ``{}`` a reference no model is measured against."""
    from tcip_mcp.pipelines.data.selection import source_digests

    evidence: dict[str, Any] = {"holdout_shares_calibration": sorted(
        {digest_of[s.location] for s in cal} & {digest_of[s.location] for s in hold})}
    failures = ["holdout_shares_calibration_image"] if evidence["holdout_shares_calibration"] else []
    if run is None:
        evidence["training"] = None
        return evidence, [*failures, "training_membership_unknown"]
    groups = {s.group for s in cal + hold}
    digests = {digest_of[s.location] for s in cal + hold}
    for side, name, failure in (("train", "training", "reference_shares_training"),
                                ("val", "selection", "reference_shares_selection")):
        if side not in run:
            continue
        shared = {"groups": sorted(groups & {s.group for s in run[side]}),
                  "source_digests": sorted(digests & set(source_digests(run[side]).values()))}
        evidence[name] = shared
        if shared["groups"] or shared["source_digests"]:
            failures.append(failure)
    return evidence, failures


def assess(
    project: Path, *, checkpoint_path: str, trait: str, delivery_kind: str, selection_dir: str,
    stated: Stated = Stated(), device: str | None = None,
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
) -> Assessment:
    """Assess ``checkpoint_path`` against the calibration and holdout sides of the selection at
    ``selection_dir``, for a ``delivery_kind`` delivery of ``trait``'s latest confirmed revision,
    and record the result as a new assessment; return it.

    ``stated`` is what the caller states of the execution record; the rest is the checkpoint's own
    geometry, a derivation over the calibration side's ground truth (the merge threshold, the
    detection cap), or the operating point the trait's count objective fits there (the conf).

    Refuses before any inference when the selection holds no calibration or no holdout side, when
    the checkpoint's head does not produce what the kind measures, and when a state-crossing
    checkpoint classifies no positive state (:class:`AssessmentRefused`); and before any inference
    over it, a reference the admission would not admit or only the model stands behind
    (:func:`_admit_reference`).
    """
    from tcip_mcp.pipelines.data.selection import source_digests

    revision, p = _prepared(project, checkpoint_path=checkpoint_path, trait=trait,
                            delivery_kind=delivery_kind, stated=stated, device=device,
                            tile_batch_size=tile_batch_size)
    _selection, cal, hold = _reference_sides(project, selection_dir)
    state = revision.entry.positive_state
    if delivery_kind == STATE_CROSSING_DATES and p.scope.state_ids(state) is None:
        raise AssessmentRefused(
            f"{checkpoint_path} classifies no {state}: a state fraction is measured off a "
            "classifier of the trait's positive state.")
    admitted = _admit_reference(cal + hold, p.scope)
    cal, hold = admitted[:len(cal)], admitted[len(cal):]
    reads = _reference_reads(cal + hold)
    digest_of = source_digests(cal + hold)
    disjointness, failures = _disjointness(digest_of, cal, hold,
                                           _run_sides(project, p.checkpoint.experiment_id))
    run_dir = _open_run(project)
    retained, measured = _retained(run_dir, cal + hold, reads)
    m_cal, m_hold = measured[:len(cal)], measured[len(cal):]
    if delivery_kind in DETECTOR_KINDS:
        criterion, criterion_failures = _detection(p, m_cal, m_hold, revision.entry,
                                                   delivery_kind, digest_of)
    else:
        criterion, criterion_failures = _scalar(p, m_hold, revision.entry, digest_of)
    return _finish(project, run_dir, revision, {
        "delivery_kind": delivery_kind, "producer": p.checkpoint.producer,
        "execution": p.execution.record(),
        "reference": _reference_record(str(selection_dir), None, cal + hold, digest_of, retained),
        "disjointness": disjointness, "criterion": criterion, "scale": None,
        "failures": [*failures, *criterion_failures],
    })


def _count_fit(p: Pass, entry: TraitEntry, counts: list[int],
               merge_boxes: Callable[[], list[list[list[float]]]],
               collect: Callable[[Execution], tuple[list[dict], list[dict]]]
               ) -> tuple[dict, list[str], list[dict], list[dict]]:
    """The count criterion over a reference: the cap derived from ``counts`` (the calibration
    side's per-image object counts) where unstated, the merge threshold derived from
    ``merge_boxes`` (the calibration side's ground truth) where unstated, the calibration and
    holdout records ``collect`` predicts under the pass's record at the staged conf floor, and the
    conf the count objective fits there (:func:`~tcip_mcp.pipelines.operating_point.
    count_criterion`) written into the pass's record.
    ``(evidence, failures, calibration records, holdout records)``."""
    from tcip_mcp.pipelines.derivations import MAX_DETS_DERIVATION, derive_max_dets_from_counts
    from tcip_mcp.pipelines.operating_point import (
        STAGED_CONF_FLOOR, STAGED_CONF_FLOOR_SOURCE, count_criterion,
        detector_operating_point_holder,
    )

    if p.execution.sources.get("max_dets") != "explicit":
        p.execution = p.execution.with_value(
            "max_dets", derive_max_dets_from_counts(counts), MAX_DETS_DERIVATION)
    if p.execution.tiled:
        p.derive_merge(merge_boxes())
    staged = p.execution.with_value("conf", STAGED_CONF_FLOOR, STAGED_CONF_FLOOR_SOURCE)
    cal_records, hold_records = collect(staged)
    holder, path = detector_operating_point_holder(p.predictor.model)
    conf, evidence, failures = count_criterion(
        cal_records, hold_records, entry, staged_conf_floor_attribute_path=path,
        staged_conf_floor=STAGED_CONF_FLOOR if hasattr(holder, "score_thresh") else None)
    p.execution = p.execution.with_value("conf", conf, evidence["conf_derived_from"])
    return evidence, failures, cal_records, hold_records


def _reference_dataset(p: Pass, samples: list[Sample]) -> Any:
    """The detection loader over ``samples`` under the pass's scope, at the width it reads at."""
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes

    return build_dataset("detection", tiling=None, samples=samples, scope=p.scope,
                         sizes=resolve_sizes("detection", {"num_channels": p.predictor.in_chans},
                                             samples))


def _records(p: Pass, ds: Any, digest_of: dict[str, str], execution: Execution) -> list[dict]:
    """One evaluation record per sample of the loader ``ds``, predicted under ``execution``
    (:func:`~tcip_mcp.pipelines.training.evaluation.prediction_record`), the sample's source digest
    as its ``image_id``."""
    from tcip_mcp.pipelines.training.evaluation import gt_records, prediction_record

    results = (p.predict([ds.image_of(k) for k in ds.stems], execution=execution)
               if ds.stems else [])
    return [prediction_record(r, gt_records(ds.det_targets(ds.document(k))), image_id=digest_of[k])
            for k, r in zip(ds.stems, results, strict=True)]


def _detection(p: Pass, cal: list[Sample], hold: list[Sample], entry: TraitEntry,
               delivery_kind: str, digest_of: dict[str, str]) -> tuple[dict, list[str]]:
    """The count criterion over the reference (:func:`_count_fit`), each side's loader built once,
    every fitted value from the calibration side, plus the classifier agreement over matched
    instances for a state-fraction delivery. ``digest_of`` is keyed by each sample's location,
    which reading the retained copies leaves unchanged."""
    from tcip_mcp.pipelines.data.splits import count_label_lines
    from tcip_mcp.pipelines.training.evaluation import gt_objects, gt_records

    cal_ds, hold_ds = _reference_dataset(p, cal), _reference_dataset(p, hold)
    evidence, failures, cal_records, hold_records = _count_fit(
        p, entry, [count_label_lines(cal_ds.document(k), p.scope) for k in cal_ds.stems],
        lambda: [[a["bbox"] for a in gt_objects(
            {"gt": gt_records(cal_ds.det_targets(cal_ds.document(k)))})] for k in cal_ds.stems],
        lambda execution: (_records(p, cal_ds, digest_of, execution),
                           _records(p, hold_ds, digest_of, execution)))
    measured: dict[str, Any] = {"count": evidence}
    if delivery_kind == STATE_CROSSING_DATES:
        from tcip_mcp.pipelines.operating_point import classifier_criterion

        classifier, classifier_failures = classifier_criterion(
            _classification_items(cal_records, p.scope, entry, evidence["localization"],
                                  evidence["conf"]),
            _classification_items(hold_records, p.scope, entry, evidence["localization"],
                                  evidence["conf"]), entry)
        measured["classifier"] = classifier
        failures = [*failures, *classifier_failures]
    return measured, failures


def _classification_items(records: list[dict], scope: Any, entry: TraitEntry, criterion: dict,
                          conf: float) -> list[dict]:
    """One item per reference instance assessed for the positive state's attribute and matched
    to one detection at ``conf`` (:func:`~tcip_mcp.pipelines.training.evaluation.attribute_pairs`
    under the trait's localization ``criterion``): whether the reference's and the model's value
    id of that attribute is the positive state's (:meth:`~tcip_mcp.pipelines.data.selection.
    ClassScope.state_ids`)."""
    from tcip_mcp.pipelines.training.evaluation import attribute_pairs

    column, positive = cast(tuple, scope.state_ids(entry.positive_state))
    return [{"image_id": image_id, "is_true_positive": truth == positive,
             "is_pred_positive": predicted == positive}
            for image_id, truth, predicted in attribute_pairs(records, criterion, conf=conf,
                                                              column=column)]


def _scalar(p: Pass, hold: list[Sample], entry: TraitEntry,
            digest_of: dict[str, str]) -> tuple[dict, list[str]]:
    """The scalar criterion over the holdout side: each sample's one predicted rank or value
    against its table row (:func:`~tcip_mcp.pipelines.data.datasets.table_values`). A pass that
    returns other than one prediction per sample, and a prediction carrying no output, refuse by
    name; a regression delivery is scored by the revision's own ``regression_criterion``, one the
    platform does not register refusing by name."""
    from tcip_mcp.pipelines.data.datasets import table_values
    from tcip_mcp.pipelines.operating_point import REGRESSION_CRITERIA, scalar_criterion
    from tcip_mcp.pipelines.training.evaluation import quadratic_weighted_kappa

    ordinal = p.checkpoint.task == "ordinal"
    if not ordinal and entry.regression_criterion not in REGRESSION_CRITERIA:
        raise AssessmentRefused(f"the revision assesses its regression delivery by "
                                f"{entry.regression_criterion!r}, and a regression delivery is "
                                f"assessed by one of {sorted(REGRESSION_CRITERIA)}.")
    suffix, cast_to = ("_ranks", int) if ordinal else ("_values", float)
    predictions = p.predict([s.image for s in hold])
    if len(predictions) != len(hold):
        raise AssessmentRefused(f"the checkpoint returned {len(predictions)} predictions for "
                                f"{len(hold)} holdout samples: the reference is scored one "
                                "prediction per sample, so it cannot be scored whole.")
    held = []
    for sample, truth, prediction in zip(hold, table_values(hold), predictions, strict=True):
        values = [v for k, v in prediction.items() if k.endswith(suffix) and isinstance(v, list)]
        if not values or not values[0]:
            raise AssessmentRefused(
                f"the prediction for {sample.source} carries no {suffix} output: the checkpoint "
                "produced nothing to measure for it, so the reference cannot be scored whole.")
        held.append({"image_id": digest_of[sample.location], "true": cast_to(truth),
                     "predicted": cast_to(values[0][0])})
    if ordinal:
        num_ranks = p.predictor.dims["num_ranks"]
        return scalar_criterion(
            held, score=lambda pred, gt: quadratic_weighted_kappa(pred, gt, num_ranks),
            floor=cast(float, entry.ordinal_agreement_floor),
            criterion="quadratic_weighted_kappa")
    return scalar_criterion(held, score=REGRESSION_CRITERIA[entry.regression_criterion],
                            floor=cast(float, entry.regression_skill_floor),
                            criterion=entry.regression_criterion)


def assess_reserved_regions(
    project: Path, *, checkpoint_path: str, trait: str, delivery_kind: str,
    k_cal: int = DEFAULT_K_CAL, k_test: int = DEFAULT_K_TEST, stated: Stated = Stated(),
    device: str | None = None, tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
) -> Assessment:
    """Assess a checkpoint trained on one mosaic against that mosaic's own reserved calibration
    and holdout regions (its run's within-image split with ``calibration_ratio``), cut
    into ``k_cal`` and ``k_test`` buffered bands, for a count delivery of ``trait``; record the
    result as a new assessment and return it.

    The bands are predicted through the tiled pass the whole mosaic is later published under, the
    tile edge the split was drawn at, the count criterion fitted as :func:`assess` fits it, every
    derived value from the calibration region alone. Each band is a reference sample of the
    mosaic, its region named. The reference's scope is the training mosaic's recorded content
    identity. Refuses (:class:`AssessmentRefused`) a delivery that is not a count, a checkpoint no
    run of this project produced, a run with no reserved regions, a stated tile edge other than the
    split's, regions not attested complete, too few
    bands carrying ground truth, a band not held out from the run's training regions, a mosaic that
    changed since the split, and a mosaic label document only the model stands behind.
    """
    import numpy as np

    from tcip_mcp.experiments import run_resolution
    from tcip_annotation.json_io import xywh
    from tcip_annotation.state import object_rows

    from tcip_mcp.pipelines import block_calibration as blocks
    from tcip_mcp.pipelines.data.datasets import PER_BOX_KEYS, crowd_of
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.data.selection import DOCUMENT, Sample, source_digests
    from tcip_mcp.pipelines.data.split_construction import partition_samples
    from tcip_mcp.pipelines.derivations import derive_block_scale_px
    from tcip_mcp.pipelines.operating_point import spatial_disjointness
    from tcip_mcp.pipelines.raster_source import BandGroupRef, open_raster
    from tcip_mcp.pipelines.training.evaluation import gt_records

    if delivery_kind not in (PER_IMAGE_COUNT, PER_PLANT_COUNT_AGGREGATE):
        raise AssessmentRefused(f"a mosaic's reserved regions answer for a count, not a "
                                f"{delivery_kind} delivery.")
    revision, p = _prepared(project, checkpoint_path=checkpoint_path, trait=trait,
                            delivery_kind=delivery_kind,
                            stated=stated.model_copy(update={"tile": True}), device=device,
                            tile_batch_size=tile_batch_size)
    experiment_id = p.checkpoint.experiment_id
    if experiment_id is None:
        raise AssessmentRefused("no run of this project produced this checkpoint, so no mosaic's "
                                "reserved regions are known to have been held out from it.")
    resolved = run_resolution(experiment_id, project=project)
    spatial = blocks.reserved_spatial_regions(resolved)
    if spatial is None:
        raise AssessmentRefused(
            f"run {experiment_id!r} resolved no within-image spatial split with a reserved "
            "calibration and holdout region (train it with data.split.calibration_ratio set).")
    (mosaic,) = partition_samples(resolved["partition"])
    stem = mosaic.member
    tile_size, overlap = int(spatial["tile_size"]), float(spatial["overlap"])
    if p.execution.tile_size != tile_size:
        raise AssessmentRefused(
            f"the run's reserved regions were tiled at {tile_size}px and this pass runs at "
            f"{p.execution.tile_size}px; the assessed pass and the published one run at one tile "
            "edge.")
    scope = p.scope.admitted_for(DOCUMENT, f"experiment {experiment_id!r}")
    (mosaic,) = _admit_reference([mosaic], scope)
    document = mosaic.read
    reads = _reference_reads([mosaic])
    cal_rect, test_rect = (tuple(spatial[k][0]) for k in ("calibration_region", "holdout_region"))
    blocks.check_completeness(document, f"{stem}'s label document", cast(str, scope.subject),
                              {"calibration_region": cal_rect, "holdout_region": test_rect})
    source = mosaic.image
    run_dir = _open_run(project)
    retained, _measured = _retained(run_dir, [mosaic], reads)
    target = json_det_targets(document.annotations, scope)
    gt = {k: np.asarray(target[k]) for k in PER_BOX_KEYS if k in target}
    gt["boxes"] = gt["boxes"].astype(np.float32).reshape(-1, 4)
    objects = gt["boxes"][object_rows(crowd_of(gt))]
    calibration_objects = objects[blocks.centered_in(objects, cal_rect)]
    plants = None
    if resolved["data"].get("plant_csv_paths"):
        from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs

        plants = read_plant_csvs([Path(c) for c in resolved["data"]["plant_csv_paths"]]) or None
    try:
        buffer_px, scale_source = derive_block_scale_px(
            tile_size=tile_size, plants=plants,
            gt_boxes_per_image=[[xywh(*box) for box in calibration_objects.tolist()]],
            raster_path=None if isinstance(source, BandGroupRef) else source)
        bands = {"calibration": blocks.band_rects(cal_rect, k_cal, tile_size, overlap, buffer_px,
                                                  "cal"),
                 "holdout": blocks.band_rects(test_rect, k_test, tile_size, overlap, buffer_px,
                                              "test")}
    except ValueError as exc:
        raise AssessmentRefused(f"no feasible band layout over the reserved regions "
                                f"(k_cal={k_cal}, k_test={k_test}): {exc}") from exc
    leaks = spatial_disjointness(spatial, [r for b in bands.values() for r in b.values()])
    if leaks:
        raise AssessmentRefused(f"band(s) {leaks} are not held out from run {experiment_id!r}'s "
                                "training regions, so they cannot stand as its reference.")
    band_counts = {side: {name: sum(object_rows(crowd_of(blocks.select_gt_for_band(gt, rect))))
                          for name, rect in side_bands.items()}
                   for side, side_bands in bands.items()}
    for side, counts in band_counts.items():
        blocks.check_feasibility(counts, side=side)
    samples = [Sample(member=name, source=mosaic.source, ground_truth=mosaic.ground_truth,
                      group=name, side=side, rect=cast(Any, tuple(rect)), image=source)
               for side, side_bands in bands.items() for name, rect in side_bands.items()]
    digest_of = source_digests(samples)

    def collect(execution: Execution) -> tuple[list[dict], list[dict]]:
        with open_raster(source, p.predictor.in_chans) as reader:
            if (reader.width, reader.height) != (int(spatial["width"]), int(spatial["height"])):
                raise AssessmentRefused(
                    f"the run recorded a {spatial['width']}x{spatial['height']} mosaic and "
                    f"{source} now reads {reader.width}x{reader.height}; retrain or re-split "
                    "against the current file.")
            return (_band_records(reader, bands["calibration"], p, execution, gt=gt,
                                  digest_of=digest_of, source=str(source)),
                    _band_records(reader, bands["holdout"], p, execution, gt=gt,
                                  digest_of=digest_of, source=str(source)))

    evidence, failures, _cal, _hold = _count_fit(
        p, revision.entry, list(band_counts["calibration"].values()),
        lambda: [[a["bbox"] for a in gt_records(blocks.select_gt_for_band(gt, rect))]
                 for rect in bands["calibration"].values()],
        collect)
    disjointness, disjoint_failures = _disjointness(
        digest_of, [s for s in samples if s.side == "calibration"],
        [s for s in samples if s.side == "holdout"], _run_sides(project, experiment_id))
    return _finish(project, run_dir, revision, {
        "delivery_kind": delivery_kind, "producer": p.checkpoint.producer,
        "execution": p.execution.record(),
        "reference": _reference_record(None, spatial["raster_content_identity"], samples,
                                       digest_of, retained),
        "disjointness": disjointness,
        "criterion": {"count": {**evidence, "block_scale_px": buffer_px,
                                "block_scale_source": scale_source, "band_gt_counts": band_counts,
                                "density_uniformity_flags": sorted(
                                    f for counts in band_counts.values()
                                    for f in blocks.density_uniformity_flags(counts))}},
        "scale": None,
        "failures": [*disjoint_failures, *failures],
    })


def _band_records(reader: Any, bands: dict[str, tuple[int, int, int, int]], p: Pass,
                  execution: Execution, *, gt: dict, digest_of: dict[str, str],
                  source: str) -> list[dict]:
    """One evaluation record per band (:func:`~tcip_mcp.pipelines.training.evaluation.
    prediction_record`): the tiled prediction under ``execution`` over the band widened on every
    side (clipped to the mosaic) by the overlap its lattice states between neighbors, keeping the
    detections and the ground truth centered in the band itself, in band coordinates, the band
    sample's source digest as its ``image_id``."""
    import numpy as np

    from tcip_mcp.pipelines import block_calibration as blocks
    from tcip_mcp.pipelines.data.selection import Sample
    from tcip_mcp.pipelines.data.datasets import PER_BOX_KEYS
    from tcip_mcp.pipelines.raster_source import Rect, _RegionView
    from tcip_mcp.pipelines.slicing import slice_lattice
    from tcip_mcp.pipelines.training.evaluation import gt_records, prediction_record

    tile_size = cast(int, execution.tile_size)
    halo = tile_size - slice_lattice(tile_size, 2 * tile_size, tile_size,
                                     cast(float, execution.overlap))[1][0]
    records = []
    for name, inner in sorted(bands.items()):
        ix0, iy0, ix1, iy1 = inner
        hx0, hy0 = max(0, ix0 - halo), max(0, iy0 - halo)
        hx1, hy1 = min(reader.width, ix1 + halo), min(reader.height, iy1 + halo)
        location = Sample(member=name, source=source, ground_truth="", group=name, side="",
                          rect=inner).location
        result = p.predictor.predict_sliced(
            _RegionView(reader, Rect(hx0, hy0, hx1, hy1)), execution=execution,
            tile_batch_size=p.tile_batch_size, require_masks=False, source_label=name)
        boxes = np.asarray(result["boxes"], dtype=np.float64).reshape(-1, 4) + [hx0, hy0, hx0, hy0]
        kept = blocks.centered_in(boxes, inner)
        rows = {key: [v for v, k in zip(result[key], kept) if k]
                for key in PER_BOX_KEYS if key in result}
        records.append(prediction_record(
            {**result, **rows, "width": ix1 - ix0, "height": iy1 - iy0,
             "boxes": (boxes[kept] - [ix0, iy0, ix0, iy0]).tolist()},
            gt_records(blocks.select_gt_for_band(gt, inner)), image_id=digest_of[location]))
    return records


def _read_reference_csv(data: bytes, csv_path: str) -> dict[str, dict[str, Any]]:
    """``stem -> {"physical_extent": float, "unit": str}`` from the bytes ``data`` of the
    breeder's reference CSV at ``csv_path``, its ``image_stem``, ``physical_extent`` and ``unit``
    columns read by name. A missing column, a short row, a non-numeric or non-positive extent and
    a repeated stem each refuse naming the line."""
    import csv
    import io

    out: dict[str, dict[str, Any]] = {}
    with io.StringIO(data.decode("utf-8"), newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        columns = ("image_stem", "physical_extent", "unit")
        missing = [name for name in columns if name not in header]
        if missing:
            raise AssessmentRefused(f"{csv_path} is missing column(s) {missing} in its header "
                                    f"{header!r}; expected {list(columns)}.")
        index = {name: header.index(name) for name in columns}
        for line_no, row in enumerate(reader, start=2):
            if len(row) < len(header):
                raise AssessmentRefused(f"{csv_path}:{line_no} has {len(row)} column(s), fewer "
                                        f"than the header's {len(header)}.")
            stem = row[index["image_stem"]].strip()
            try:
                extent = float(row[index["physical_extent"]].strip())
            except ValueError:
                raise AssessmentRefused(f"{csv_path}:{line_no} has a non-numeric "
                                        f"physical_extent for stem {stem!r}.") from None
            if not extent > 0:
                raise AssessmentRefused(f"{csv_path}:{line_no} states physical_extent {extent} "
                                        f"for stem {stem!r}: a physical length is positive.")
            if stem in out:
                raise AssessmentRefused(f"{csv_path}:{line_no} repeats stem {stem!r}.")
            out[stem] = {"physical_extent": extent, "unit": row[index["unit"]].strip()}
    return out


def assess_physical_scale(
    project: Path, *, trait: str, selection_dir: str, reference_csv: str, unit: str,
    reference_subject: str,
) -> Assessment:
    """Derive a per-pixel physical scale in ``unit`` on the calibration side of the selection at
    ``selection_dir`` and check it against the holdout side, for ``trait``'s latest confirmed
    revision; record the result as a new assessment and return it.

    The reference is admitted as :func:`assess` admits one, under the selection's own scope, and
    its sides must share no source digest. Each reference image carries exactly one
    ``reference_subject`` polygon or mask, whose principal-axis extent is its pixel extent;
    ``reference_csv`` (``image_stem, physical_extent, unit``) is the breeder's physical extent of
    the same object, read once, retained with the reference and measured as read. The scale is the mean implied scale of
    the calibration side; it passes when the holdout's own relative dispersion and the scale's
    relative deviation from the holdout mean are both within ``scale_tolerance_frac``. Refuses an
    unauthored tolerance, a unit that is not a linear length unit crops.yml declares, a selection
    without both reference sides, a reference the admission would not admit or only the model
    stands behind, a reference image with no row in the CSV, a non-positive extent, and a
    reference annotation that is not exactly one polygon.
    """
    import math
    import statistics

    from tcip_annotation.state import polygonal

    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.pipelines.data.selection import source_digests
    from tcip_mcp.pipelines.measurement.mask_geometry import principal_axis_extent_of_points
    from tcip_mcp.traits import authored, crops_length_units

    revision = latest_confirmed(trait, project)
    authored(revision.entry, ("scale_tolerance_frac",))
    if unit not in crops_length_units():
        raise AssessmentRefused(f"a per-pixel scale is a length per pixel, and {unit!r} is not a "
                                f"linear length unit crops.yml declares "
                                f"({sorted(crops_length_units())}).")
    selection, cal, hold = _reference_sides(project, selection_dir)
    admitted = _admit_reference(cal + hold, selection.scope)
    cal, hold = admitted[:len(cal)], admitted[len(cal):]
    reads = _reference_reads(cal + hold, (reference_csv,))
    digest_of = source_digests(cal + hold)
    disjointness, failures = _disjointness(digest_of, cal, hold, {})
    run_dir = _open_run(project)
    retained, measured = _retained(run_dir, cal + hold, reads)
    rows = _read_reference_csv(reads[reference_csv].value, reference_csv)
    implied: dict[str, dict[str, float]] = {"calibration": {}, "holdout": {}}
    for sample in measured:
        row = rows.get(sample.member)
        if row is None:
            raise AssessmentRefused(f"{reference_csv} names no physical extent for reference "
                                    f"image {sample.member!r}.")
        if row["unit"] != unit:
            raise AssessmentRefused(f"{reference_csv} states {sample.member!r} in "
                                    f"{row['unit']!r}, not {unit!r}; a reference is never "
                                    "converted between units.")
        found = [a for a in sample.read.annotations if a.subject == reference_subject]
        if len(found) != 1 or not polygonal(found[0].geometry):
            raise AssessmentRefused(
                f"{sample.member!r} carries {len(found)} {reference_subject!r} annotation(s); a "
                "reference image carries exactly one, as a polygon or mask, since a box's long "
                "side is its projected extent, not the object's length.")
        points = [pt for ring in cast(Any, found[0].geometry).rings for pt in ring]
        pixels = principal_axis_extent_of_points(points)
        if not pixels > 0:
            raise AssessmentRefused(f"{sample.member!r}'s {reference_subject!r} polygon has no "
                                    "extent, so no scale is implied by it.")
        implied[sample.side][sample.member] = row["physical_extent"] / pixels
    tolerance = cast(float, revision.entry.scale_tolerance_frac)
    failures += [f"insufficient_{side}_references" for side in ("calibration", "holdout")
                 if len(implied[side]) < 2]
    evidence: dict[str, Any] = {"implied_scales": implied, "tolerance_frac": tolerance}
    value = None
    if not any(f.startswith("insufficient_") for f in failures):
        value = statistics.mean(implied["calibration"].values())
        held = list(implied["holdout"].values())
        mean = statistics.mean(held)
        dispersion = abs(statistics.stdev(held) / mean) if mean else math.inf
        deviation = abs(value - mean) / abs(mean) if mean else math.inf
        evidence.update(calibration_mean=value, holdout_mean=mean,
                        holdout_relative_dispersion=dispersion, relative_deviation=deviation)
        failures += (["holdout_dispersion_exceeds_tolerance"] if dispersion > tolerance else []) + (
            ["holdout_mean_outside_tolerance"] if deviation > tolerance else [])
    return _finish(project, run_dir, revision, {
        "delivery_kind": None, "producer": None, "execution": None,
        "reference": _reference_record(str(selection_dir), None, cal + hold, digest_of, retained),
        "disjointness": disjointness, "criterion": {"scale": evidence},
        "scale": {"unit": unit, "value": value, "reference_subject": reference_subject},
        "failures": failures,
    })
