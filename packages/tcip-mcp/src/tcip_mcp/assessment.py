"""An assessment: a run measuring how well one checkpoint reproduces a held-out reference.

The checkpoint runs under one execution record, at one delivered kind of one trait revision. An
assessment is a directory under the project's ``.tcip/assessments/``, opened once, holding
``assessment.json`` and, under ``reference/``, a copy of every reference ground truth it measured
from, label document, mask, table and physical-extent table alike. It records the reference, the
source digest of every reference sample, the disjointness and criterion evidence and the trait
revision it was judged under; one naming no failure says a number from that checkpoint, under
that execution record, is validated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from operator import methodcaller
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import ConfigDict, JsonValue, TypeAdapter

from tcip_mcp.model_registry import Producer
from tcip_mcp.pipelines.data.selection import (
    GROUND_TRUTH_PATHS,
    SAMPLE_PATHS,
    GroundTruth,
    Sample,
    Side,
    ground_truth_record,
)
from tcip_mcp.pipelines.execution import (
    DEFAULT_TILE_BATCH_SIZE,
    Execution,
    MaskCut,
    Stated,
)
from tcip_mcp.pipelines.raster_source import RasterIdentity
from tcip_mcp.pipelines.schemas import TraitRevisionRef
from tcip_mcp.registry_paths import (
    PathFields,
    located,
    recorded_paths,
    runtime_paths,
    within,
)
from tcip_mcp.traits import (
    DETECTOR_KINDS,
    PER_IMAGE_COUNT,
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_ORDINAL_AGGREGATE,
    STATE_CROSSING_DATES,
)
from tcip_store import Key

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.datasets import DocumentDataset
    from tcip_mcp.pipelines.data.selection import ClassScope, Selection
    from tcip_mcp.pipelines.execution import Pass, Preparation
    from tcip_mcp.pipelines.raster_source import RasterSource
    from tcip_mcp.pipelines.schemas import TilingSpec
    from tcip_mcp.traits import TraitEntry, TraitRevision

ASSESSMENTS_DIR = Path(".tcip/assessments")
ASSESSMENT_FILE = "assessment.json"
REFERENCE_DIR = "reference"

_PATHS: PathFields = (
    ("reference", "selection_dir"),
    *within(("reference", "samples", "[]", "sample"), SAMPLE_PATHS),
    *within(
        ("reference", "ground_truth", "[]", "ground_truth"), GROUND_TRUTH_PATHS
    ),
)


class AssessmentRefusedError(ValueError):
    """An assessment that cannot run as asked.

    The causes are a selection without both reference sides, a checkpoint whose head does not
    produce what the delivered kind measures, or a reference the criterion cannot read.
    """


@dataclass(frozen=True)
class RetainedFile:
    """One reference ground truth, its digest when retained, and its copy.

    It holds the label document's key or the file's path it was read at, its
    :func:`~tcip_mcp.pipelines.data.selection.ground_truth_digest` when retained, and its copy
    under the assessment's ``reference/``.
    """

    ground_truth: GroundTruth
    digest: str
    copy: str


@dataclass(frozen=True)
class ReferenceSample:
    """One reference sample and the digest its source's pixels had when it was measured."""

    sample: Sample
    source_digest: str


@dataclass(frozen=True)
class Reference:
    """What an assessment measured.

    It holds the selection it came from, or the mosaic's ``raster_identity``, the identity id
    the samples' dataset root recorded when it was measured (``None`` for a root that recorded
    none), each retained ground-truth file and every sample.
    """

    selection_dir: str | None
    raster_identity: RasterIdentity | None
    dataset_id: str | None
    ground_truth: list[RetainedFile]
    samples: list[ReferenceSample]

    def covers(
        self,
        *,
        dataset_id: str | None,
        date: str | None,
        raster_identity: RasterIdentity | None,
    ) -> bool:
        """Return whether this reference spans a capture.

        For a raster it is the same mosaic; for per-image predictions, the recorded
        ``dataset_id`` and the capture of one of its samples' sources
        (:func:`~tcip_mcp.dataset_layout.image_capture`).
        """
        from tcip_mcp.dataset_layout import image_capture

        if raster_identity is not None:
            return self.raster_identity == raster_identity
        dates = {image_capture(s.sample.source)[1] for s in self.samples}
        return dataset_id == self.dataset_id and date in dates

    def moved(self, run_dir: Path) -> list[str]:
        """Return what of this reference no longer digests as recorded.

        First each retained ground truth whose original, or whose copy under ``run_dir``, no
        longer digests to the digest recorded for it, by its stored spelling, then each reference
        source whose pixels no longer digest to the source digest recorded for its sample
        (:func:`~tcip_mcp.pipelines.data.selection.source_digests`), or a source file gone, by
        its path.
        """
        from tcip_mcp.pipelines.data.selection import (
            moved_ground_truth,
            source_digests,
        )

        moved = [
            str(ground_truth_record(f.ground_truth))
            for f in self.ground_truth
            if moved_ground_truth(
                {f.ground_truth: f.digest, str(run_dir / f.copy): f.digest}
            )
        ]
        samples = {s.sample: s.source_digest for s in self.samples}
        try:
            now = source_digests(samples)
        except FileNotFoundError as exc:
            return [*moved, str(exc.filename)]
        return [
            *moved,
            *dict.fromkeys(
                sample.source
                for sample, digest in samples.items()
                if now[sample.location] != digest
            ),
        ]


@dataclass(frozen=True)
class Assessment:
    """One finished assessment, as its ``assessment.json`` states it.

    ``delivery_kind``, ``producer`` (:attr:`~tcip_mcp.model_registry.VerifiedCheckpoint.producer`),
    ``execution`` and ``rationale`` are ``None`` for a physical-scale assessment. ``disjointness``
    and ``criterion`` are the evidence the failures were read from, ``scale`` the scale a
    physical-scale assessment derived. ``revision`` is the trait revision it was judged under, and
    ``rationale`` why the assessment states the test settings it does.
    """

    assessment_id: str
    delivery_kind: str | None
    revision: TraitRevisionRef
    producer: Producer | None
    execution: Execution | None
    rationale: str | None
    reference: Reference
    disjointness: dict[str, JsonValue]
    criterion: dict[str, JsonValue]
    scale: dict[str, JsonValue] | None
    failures: list[str]
    __pydantic_config__ = ConfigDict(
        ser_json_inf_nan="constants", extra="forbid"
    )

    @property
    def passed(self) -> bool:
        """Whether the assessment names no failure."""
        return not self.failures


ASSESSMENT_RECORD = TypeAdapter(Assessment)
"""The assessment record's validation and its JSON form."""


def assessment_dir(project: Path | str, assessment_id: str) -> Path:
    """Return one assessment's directory, refusing an id that is not a single directory name."""
    from tcip_mcp.experiments import run_name

    return Path(project) / ASSESSMENTS_DIR / run_name(assessment_id)


def read_assessment(project: Path | str, assessment_id: str) -> Assessment:
    """Return assessment ``assessment_id`` of ``project``, its paths resolved against the project.

    An id naming no finished assessment refuses (``ValueError``) naming it, and a record that
    does not validate as an :class:`Assessment` (``pydantic.ValidationError``) naming each field.
    """
    from tcip_mcp.experiments import read_record

    path = assessment_dir(project, assessment_id) / ASSESSMENT_FILE
    if not path.is_file():
        raise ValueError(
            f"no assessment {assessment_id!r} is recorded under {project}: nothing "
            "states what it measured."
        )
    return ASSESSMENT_RECORD.validate_python(
        runtime_paths(read_record(path), _PATHS, project)
    )


def _open_run(project: Path) -> Path:
    """Create a fresh assessment's directory, named for its id, once, and return it.

    It holds an empty reference folder.
    """
    from tcip_mcp.experiments import create_run_directory, mint_experiment_id

    run_dir = create_run_directory(
        assessment_dir(project, mint_experiment_id("assessment"))
    )
    (run_dir / REFERENCE_DIR).mkdir()
    return run_dir


def _finish(
    project: Path,
    run_dir: Path,
    revision: TraitRevision,
    *,
    delivery_kind: str | None,
    producer: Producer | None,
    execution: Execution | None,
    rationale: str | None,
    reference: Reference,
    disjointness: dict[str, Any],
    criterion: dict[str, Any],
    scale: dict[str, Any] | None,
    failures: list[str],
) -> Assessment:
    """Write the assessment judged under ``revision`` as the run's ``assessment.json`` once.

    Then its one audit line, ``assessment_recorded``, is written; returns it as read back.
    """
    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.experiments import write_once

    assessment = Assessment(
        assessment_id=run_dir.name,
        delivery_kind=delivery_kind,
        revision=revision.ref,
        producer=producer,
        execution=execution,
        rationale=rationale,
        reference=reference,
        disjointness=disjointness,
        criterion=criterion,
        scale=scale,
        failures=failures,
    )
    write_once(
        run_dir / ASSESSMENT_FILE,
        recorded_paths(
            ASSESSMENT_RECORD.dump_python(assessment, mode="json"),
            _PATHS,
            project,
        ),
    )
    record_event_or_raise(
        "assessment_recorded",
        {
            "assessment_id": run_dir.name,
            "trait": revision.entry.name,
            "delivery_kind": delivery_kind,
            "passed": assessment.passed,
        },
        actor=None,
        scope=project,
    )
    return read_assessment(project, run_dir.name)


def _reference_reads(
    samples: list[Sample], extra: tuple[str, ...] = ()
) -> dict[Key | str, Any]:
    """Return what the store answered (``Versioned``) for each reference ground truth.

    These are each distinct ground truth of the admitted ``samples``, their admission's own read
    (``stored``), and each file of ``extra``, read once here; an ``extra`` file that is not there
    refuses (:class:`AssessmentRefusedError`) with the store's own ``NotFoundError`` message.
    """
    import tcip_store

    reads: dict[Key | str, Any] = {s.ground_truth: s.stored for s in samples}
    for path in extra:
        try:
            reads[path] = tcip_store.read_blob_versioned(Path(path))
        except tcip_store.NotFoundError as exc:
            raise AssessmentRefusedError(
                f"the reference cannot be read, so it cannot be assessed: "
                f"{exc}"
            ) from exc
    return reads


def _retained(
    run_dir: Path, samples: list[Sample], reads: dict[Key | str, Any]
) -> tuple[list[RetainedFile], list[Sample]]:
    """Copy each of ``reads`` once into the run's reference folder; return them and the samples.

    ``reads`` are :func:`_reference_reads`'s, copied before anything is measured, a label document
    as its stored record (:func:`tcip_store.encode_record`). The returned ``samples`` measure what
    was retained: a mask's or a table's copy, and a document's record at the retained version
    (its ``ground_truth_digest``).
    """
    import tcip_store
    from tcip_mcp.experiments import publish_once

    files: dict[Key | str, RetainedFile] = {}
    for i, (gt, stored) in enumerate(reads.items()):
        data = (
            tcip_store.encode_record(stored.value)
            if isinstance(gt, Key)
            else stored.value
        )
        name = (
            f"{i:04d}_{gt.parts[-1]}.json"
            if isinstance(gt, Key)
            else f"{i:04d}_{Path(gt).name}"
        )
        publish_once(
            run_dir / REFERENCE_DIR / name, methodcaller("write", data)
        )
        files[gt] = RetainedFile(
            ground_truth=gt,
            digest=stored.version.token,
            copy=f"{REFERENCE_DIR}/{name}",
        )
    measured = [
        replace(s, ground_truth_digest=files[s.ground_truth].digest)
        if isinstance(s.ground_truth, Key)
        else replace(s, ground_truth=str(run_dir / files[s.ground_truth].copy))
        for s in samples
    ]
    return list(files.values()), measured


def _reference(
    selection_dir: str | None,
    raster_identity: RasterIdentity | None,
    samples: list[Sample],
    digest_of: dict[str, str],
    retained: list[RetainedFile],
) -> Reference:
    """Return what an assessment measured.

    It holds where the reference came from, the identity id the one dataset root of its samples
    records as it is measured (:func:`~tcip_mcp.dataset_layout.capture_of` of its first sample),
    every retained file, and each sample beside its source digest (``digest_of``).
    """
    from tcip_mcp.dataset_layout import capture_of

    return Reference(
        selection_dir=selection_dir,
        raster_identity=raster_identity,
        dataset_id=capture_of(samples[0].source)[0],
        ground_truth=retained,
        samples=[
            ReferenceSample(sample=s, source_digest=digest_of[s.location])
            for s in samples
        ],
    )


def _reference_sides(
    project: Path, selection_dir: str
) -> tuple[Selection, list[Sample], list[Sample]]:
    """Return the selection at ``selection_dir`` and its calibration and holdout sides.

    A selection holding either side empty refuses (:class:`AssessmentRefusedError`).
    """
    from tcip_mcp.pipelines.data.selection import (
        REFERENCE_SIDES,
        read_selection,
    )

    selection = read_selection(selection_dir, project=project)
    cal, hold = (selection.on(side) for side in REFERENCE_SIDES)
    empty = [
        side
        for side, samples in zip(REFERENCE_SIDES, (cal, hold))
        if not samples
    ]
    if empty:
        raise AssessmentRefusedError(
            f"the selection at {selection_dir} holds no {empty} side: an "
            "assessment fits on the calibration side and checks on the "
            "holdout side. Draw a selection with both (draw_splits)."
        )
    return selection, cal, hold


def _prepared(
    project: Path,
    *,
    checkpoint_path: str,
    trait: str,
    delivery_kind: str,
    stated: Stated,
    device: str | None,
    tile_batch_size: int,
) -> tuple[TraitRevision, Preparation]:
    """Return ``trait``'s confirmed revision and the registered checkpoint readied for a pass.

    The revision is ``trait``'s latest confirmed one stating a ``delivery_kind``
    operationalization; the checkpoint is the registered one at ``checkpoint_path``
    (``registry_paths.located`` against ``project``), readied for a pass under ``stated``, a
    detector's at the stated conf the count fit collects at. A checkpoint whose head does not
    produce what the kind measures refuses (:class:`AssessmentRefusedError`).
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.operationalization import confirmed_revision
    from tcip_mcp.pipelines.execution import prepare

    revision = confirmed_revision(delivery_kind, project=project, trait=trait)
    checkpoint = load_registered_checkpoint(
        located(checkpoint_path, project), project=project
    )
    from tcip_mcp.pipelines.model_contract import DETECTOR_TASKS

    expected = (
        tuple(sorted(DETECTOR_TASKS))
        if delivery_kind in DETECTOR_KINDS
        else ("ordinal",)
        if delivery_kind == PER_PLANT_ORDINAL_AGGREGATE
        else ("regression",)
    )
    if checkpoint.task not in expected:
        raise AssessmentRefusedError(
            f"{checkpoint_path} is a {checkpoint.task!r} checkpoint, and a "
            f"{delivery_kind} delivery is measured off a {expected} head."
        )
    return revision, prepare(
        checkpoint, stated, device=device, tile_batch_size=tile_batch_size
    )


def _admit_reference(samples: list[Sample], scope: ClassScope) -> list[Sample]:
    """Return ``samples`` as the platform's own admission re-admits them under ``scope``.

    The admission is :func:`~tcip_mcp.pipelines.data.label_queries.readmitted_samples`, each
    sample carrying its read; refuses a reference that admission would not admit, and one whose
    label documents only the model stands behind
    (:func:`~tcip_annotation.json_io.require_reference_ground_truth`).
    """
    from tcip_annotation.json_io import require_reference_ground_truth
    from tcip_mcp.pipelines.data.label_queries import readmitted_samples

    admitted = readmitted_samples(samples, scope)
    require_reference_ground_truth(
        [
            a
            for s in admitted
            if isinstance(s.ground_truth, Key)
            for a in s.read.annotations
        ]
    )
    return admitted


def _run_sides(
    project: Path, experiment_id: str | None
) -> dict[str, list[Sample]] | None:
    """Return the producing run's ``train`` and ``val`` samples, or ``None``.

    ``None`` when no run of ``project`` produced the checkpoint, so no training membership is
    known.
    """
    if experiment_id is None:
        return None
    from tcip_mcp.experiments import run_resolution

    run = run_resolution(experiment_id, project=project).partition.samples
    return {
        side: [s for s in run if s.side == side] for side in ("train", "val")
    }


def _disjointness(
    digest_of: dict[str, str],
    cal: list[Sample],
    hold: list[Sample],
    run: dict[str, list[Sample]] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(evidence, failures)`` of whether the reference is held out.

    It is held out when its calibration and holdout sides share no source digest (``digest_of``),
    and neither shares a group key or a source digest with any of the producing ``run``'s sides.
    ``run`` ``None`` is a run whose membership is unknown, a failure of its own, and ``{}`` a
    reference no model is measured against.
    """
    from tcip_mcp.pipelines.data.selection import source_digests

    evidence: dict[str, Any] = {
        "holdout_shares_calibration": sorted(
            {digest_of[s.location] for s in cal}
            & {digest_of[s.location] for s in hold}
        )
    }
    failures = (
        ["holdout_shares_calibration_image"]
        if evidence["holdout_shares_calibration"]
        else []
    )
    if run is None:
        evidence["training"] = None
        return evidence, [*failures, "training_membership_unknown"]
    groups = {s.group for s in cal + hold}
    digests = {digest_of[s.location] for s in cal + hold}
    for side, name, failure in (
        ("train", "training", "reference_shares_training"),
        ("val", "selection", "reference_shares_selection"),
    ):
        if side not in run:
            continue
        shared = {
            "groups": sorted(groups & {s.group for s in run[side]}),
            "source_digests": sorted(
                digests & set(source_digests(run[side]).values())
            ),
        }
        evidence[name] = shared
        if shared["groups"] or shared["source_digests"]:
            failures.append(failure)
    return evidence, failures


def assess(
    project: Path,
    *,
    checkpoint_path: str,
    trait: str,
    delivery_kind: str,
    selection_dir: str,
    stated: Stated,
    equivalence_z: float,
    dispersion_quantile: float,
    rationale: str,
    device: str | None = None,
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
) -> Assessment:
    """Assess ``checkpoint_path`` against a selection's sides and record a new assessment.

    The sides are the calibration and holdout sides of the selection at ``selection_dir``; the
    assessment is for a ``delivery_kind`` delivery of ``trait``'s latest confirmed revision, and
    is returned. Each path is :func:`~tcip_mcp.registry_paths.located` against ``project``.

    ``stated`` is what the caller states of the execution record, a detector's the conf its
    predictions are collected at; the rest is the checkpoint's own geometry, a derivation over
    the calibration side's ground truth (the object density, any tile geometry the checkpoint
    does not record, the merge threshold), or the operating point the trait's count objective
    fits there (the conf). A count delivery's held-out check runs at the stated ``equivalence_z``
    and ``dispersion_quantile`` (:func:`~tcip_mcp.pipelines.operating_point.count_criterion`); the
    record carries ``rationale``, why the assessment states them.

    Refuses before any inference when the selection holds no calibration or no holdout side, when
    the checkpoint's head does not produce what the kind measures, and when a state-crossing
    checkpoint classifies no positive state (:class:`AssessmentRefusedError`); and before any
    inference over it, a reference the admission would not admit or only the model stands behind
    (:func:`_admit_reference`).
    """
    from tcip_mcp.pipelines.data.selection import source_digests

    selection_dir = str(located(selection_dir, project))
    revision, prep = _prepared(
        project,
        checkpoint_path=checkpoint_path,
        trait=trait,
        delivery_kind=delivery_kind,
        stated=stated,
        device=device,
        tile_batch_size=tile_batch_size,
    )
    _selection, cal, hold = _reference_sides(project, selection_dir)
    state = revision.entry.positive_state
    positive = (
        prep.scope.state_ids(state)
        if delivery_kind == STATE_CROSSING_DATES
        else None
    )
    if delivery_kind == STATE_CROSSING_DATES and positive is None:
        raise AssessmentRefusedError(
            f"{checkpoint_path} classifies no {state}: a state fraction is measured off a "
            "classifier of the trait's positive state."
        )
    admitted = _admit_reference(cal + hold, prep.scope)
    cal, hold = admitted[: len(cal)], admitted[len(cal) :]
    reads = _reference_reads(cal + hold)
    digest_of = source_digests(cal + hold)
    disjointness, failures = _disjointness(
        digest_of,
        cal,
        hold,
        _run_sides(project, prep.checkpoint.experiment_id),
    )
    run_dir = _open_run(project)
    retained, measured = _retained(run_dir, cal + hold, reads)
    m_cal, m_hold = measured[: len(cal)], measured[len(cal) :]
    if delivery_kind in DETECTOR_KINDS:
        p, criterion, criterion_failures = _detection(
            prep,
            m_cal,
            m_hold,
            revision.entry,
            positive,
            digest_of,
            equivalence_z=equivalence_z,
            dispersion_quantile=dispersion_quantile,
        )
    else:
        p = prep.runnable()
        criterion, criterion_failures = _scalar(
            p, m_hold, revision.entry, digest_of
        )
    return _finish(
        project,
        run_dir,
        revision,
        delivery_kind=delivery_kind,
        producer=p.checkpoint.producer,
        execution=p.execution,
        rationale=rationale,
        reference=_reference(
            str(selection_dir), None, cal + hold, digest_of, retained
        ),
        disjointness=disjointness,
        criterion=criterion,
        scale=None,
        failures=[*failures, *criterion_failures],
    )


def _count_fit(
    p: Pass,
    entry: TraitEntry,
    collect: Callable[[Execution], tuple[list[dict], list[dict]]],
    *,
    equivalence_z: float,
    dispersion_quantile: float,
) -> tuple[dict, list[str], list[dict], list[dict]]:
    """Fit the count criterion over a reference at ``equivalence_z`` and ``dispersion_quantile``.

    It reads the calibration and holdout records ``collect`` predicts under the pass's record, at
    the conf its statement carries, and writes into the pass's record the conf the count
    objective fits there, that stated conf its floor
    (:func:`~tcip_mcp.pipelines.operating_point.count_criterion`); an
    :attr:`~tcip_mcp.pipelines.execution.Execution.uncut` record then takes the mask cut its
    calibration side derives over the matching that fit made (:func:`_mask_fit`). Returns
    ``(evidence, failures, calibration records, holdout records)``.
    """
    from tcip_mcp.pipelines.operating_point import (
        count_criterion,
        detector_operating_point_holder,
    )

    cal_records, hold_records = collect(p.execution)
    _, path = detector_operating_point_holder(p.predictor.model)
    held: dict = {}
    conf, evidence, failures = count_criterion(
        cal_records,
        hold_records,
        entry,
        staged_conf_floor_attribute_path=path,
        staged_conf_floor=cast(float, p.execution.conf),
        equivalence_z=equivalence_z,
        dispersion_quantile=dispersion_quantile,
        held=held,
    )
    p.execution = replace(
        p.execution,
        conf=conf,
        sources={**p.execution.sources, "conf": evidence["conf_derived_from"]},
    )
    if p.execution.uncut:
        p.execution = _mask_fit(p.execution, cal_records, evidence, held)
    return evidence, failures, cal_records, hold_records


def _records(
    p: Pass,
    ds: DocumentDataset,
    digest_of: dict[str, str],
    execution: Execution,
) -> list[dict]:
    """Return one evaluation record per sample of the loader ``ds``, predicted under ``execution``.

    Each record is :func:`~tcip_mcp.pipelines.training.evaluation.result_record`'s, the sample's
    source digest as its ``image_id``.
    """
    from tcip_mcp.pipelines.training.evaluation import (
        gt_records,
        result_record,
    )

    results = (
        p.predict([ds.image_of(k) for k in ds.stems], execution=execution)
        if ds.stems
        else []
    )
    return [
        result_record(
            r,
            gt_records(ds.det_targets(ds.document(k))),
            image_id=digest_of[k],
        )
        for k, r in zip(ds.stems, results, strict=True)
    ]


def _detection(
    prep: Preparation,
    cal: list[Sample],
    hold: list[Sample],
    entry: TraitEntry,
    positive: tuple[int, int] | None,
    digest_of: dict[str, str],
    *,
    equivalence_z: float,
    dispersion_quantile: float,
) -> tuple[Pass, dict, list[str]]:
    """Return the runnable pass, its measured criteria and failures over the reference.

    The pass is made runnable from the calibration side's ground truth (its frames' object
    density, and its merge threshold and tile geometry where unstated and unrecorded); the count
    criterion is measured over the reference at ``equivalence_z`` and ``dispersion_quantile``
    (:func:`_count_fit`, which derives an uncut pass's mask cut), each side's loader built once,
    every fitted value from the calibration side, plus, given the positive state's
    ``(column, value id)`` for a state-fraction delivery, the classifier agreement over matched
    instances. ``digest_of`` is keyed by each sample's location, which reading the retained copies
    leaves unchanged.
    """
    from tcip_mcp.pipelines.data.datasets import DocumentDataset
    from tcip_mcp.pipelines.data.split_construction import predictor_dataset
    from tcip_mcp.pipelines.execution import Reference

    cal_ds, hold_ds = (
        predictor_dataset(
            prep.predictor.task, side, prep.scope, prep.predictor, None
        )
        for side in (cal, hold)
    )
    assert isinstance(cal_ds, DocumentDataset) and isinstance(
        hold_ds, DocumentDataset
    ), "a detector's build over samples is one of these"
    p = prep.runnable(Reference(regions=cal_ds.regions))
    evidence, failures, cal_records, hold_records = _count_fit(
        p,
        entry,
        lambda execution: (
            _records(p, cal_ds, digest_of, execution),
            _records(p, hold_ds, digest_of, execution),
        ),
        equivalence_z=equivalence_z,
        dispersion_quantile=dispersion_quantile,
    )
    measured: dict[str, Any] = {"count": evidence}
    if positive is not None:
        from tcip_mcp.pipelines.operating_point import classifier_criterion

        classifier, classifier_failures = classifier_criterion(
            _classification_items(
                cal_records,
                positive,
                evidence["localization"],
                evidence["conf"],
            ),
            _classification_items(
                hold_records,
                positive,
                evidence["localization"],
                evidence["conf"],
            ),
            entry,
            equivalence_z=equivalence_z,
        )
        measured["classifier"] = classifier
        failures = [*failures, *classifier_failures]
    return p, measured, failures


def _mask_fit(
    execution: Execution, cal_records: list[dict], evidence: dict, held: dict
) -> Execution:
    """Return ``execution`` with its ``mask_cut`` derived over the calibration records.

    Each detection's uncut ``soft_mask`` is paired with the rings of the object it matched at the
    fitted conf, under the calibration curve's own criterion (the matching the count criterion's
    sweep made, taken from ``held``). The cut is the ``mask_binarize``
    (:func:`~tcip_mcp.pipelines.derivations.derive_mask_binarize`), then the ``vertex_spacing``
    of the masks cut at it (:func:`~tcip_mcp.pipelines.derivations.derive_vertex_spacing`), each
    field's derivation label recorded as its own ``sources`` entry.
    """
    from dataclasses import replace

    from tcip_mcp.pipelines.derivations import (
        MASK_BINARIZE_DERIVATION,
        VERTEX_SPACING_DERIVATION,
        derive_mask_binarize,
        derive_vertex_spacing,
    )
    from tcip_mcp.pipelines.training.evaluation import matched

    pairs = [
        (*rec["dt"][d]["soft_mask"], rec["gt"][g]["rings"])
        for rec, by_class in zip(
            cal_records,
            matched(
                cal_records,
                evidence["calibration_curve"]["criterion"],
                conf=evidence["conf"],
                held=held,
            ),
        )
        for m in by_class.values()
        for g, d in m.pairs
    ]
    threshold = derive_mask_binarize(pairs)
    derived = {
        "mask_binarize": (threshold, MASK_BINARIZE_DERIVATION),
        "vertex_spacing": (
            derive_vertex_spacing(pairs, threshold),
            VERTEX_SPACING_DERIVATION,
        ),
    }
    return replace(
        execution,
        mask_cut=MaskCut(**{k: value for k, (value, _) in derived.items()}),
        sources=execution.sources
        | {k: label for k, (_, label) in derived.items()},
    )


def _classification_items(
    records: list[dict],
    state: tuple[int, int],
    criterion: dict,
    conf: float,
) -> list[dict]:
    """Return one item per reference instance assessed for the state's attribute and matched.

    ``state`` is the positive state's ``(column, value id)``
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.state_ids`). An instance is matched to
    one detection at ``conf`` (:func:`~tcip_mcp.pipelines.training.evaluation.attribute_pairs`
    under the trait's localization ``criterion``); its item says whether the reference's and the
    model's value id of that attribute is the positive state's.
    """
    from tcip_mcp.pipelines.training.evaluation import attribute_pairs

    column, positive = state
    return [
        {
            "image_id": image_id,
            "is_true_positive": truth == positive,
            "is_pred_positive": predicted == positive,
        }
        for image_id, truth, predicted in attribute_pairs(
            records, criterion, conf=conf, column=column
        )
    ]


def _scalar(
    p: Pass, hold: list[Sample], entry: TraitEntry, digest_of: dict[str, str]
) -> tuple[dict, list[str]]:
    """Return the scalar criterion over the holdout side and its failures.

    Each sample's one predicted rank or value is scored against its table row
    (:func:`~tcip_mcp.pipelines.data.datasets.table_values`). A pass that returns other than one
    prediction per sample, and a prediction carrying no output, refuse by name; a regression
    delivery is scored by the revision's own ``regression_criterion``.
    """
    from tcip_mcp.pipelines.data.datasets import table_values
    from tcip_mcp.pipelines.operating_point import (
        REGRESSION_CRITERIA,
        scalar_criterion,
    )
    from tcip_mcp.pipelines.training.evaluation import quadratic_weighted_kappa

    ordinal = p.checkpoint.task == "ordinal"
    suffix, cast_to = ("_ranks", int) if ordinal else ("_values", float)
    predictions = p.predict([s.image for s in hold])
    if len(predictions) != len(hold):
        raise AssessmentRefusedError(
            f"the checkpoint returned {len(predictions)} predictions for "
            f"{len(hold)} holdout samples: the reference is scored one "
            "prediction per sample, so it cannot be scored whole."
        )
    held = []
    for sample, truth, prediction in zip(
        hold, table_values(hold), predictions, strict=True
    ):
        values = [
            v
            for k, v in prediction.items()
            if k.endswith(suffix) and isinstance(v, list)
        ]
        if not values or not values[0]:
            raise AssessmentRefusedError(
                f"the prediction for {sample.source} carries no {suffix} output: the checkpoint "
                "produced nothing to measure for it, so the reference cannot be scored whole."
            )
        held.append(
            {
                "image_id": digest_of[sample.location],
                "true": cast_to(truth),
                "predicted": cast_to(values[0][0]),
            }
        )
    if ordinal:
        num_ranks = p.predictor.dims["num_ranks"]
        return scalar_criterion(
            held,
            score=lambda pred, gt: quadratic_weighted_kappa(
                pred, gt, num_ranks
            ),
            floor=cast(float, entry.ordinal_agreement_floor),
            criterion="quadratic_weighted_kappa",
        )
    return scalar_criterion(
        held,
        score=REGRESSION_CRITERIA[entry.regression_criterion],
        floor=cast(float, entry.regression_skill_floor),
        criterion=entry.regression_criterion,
    )


def assess_reserved_regions(
    project: Path,
    *,
    checkpoint_path: str,
    trait: str,
    delivery_kind: str,
    k_cal: int,
    k_test: int,
    stated: Stated,
    equivalence_z: float,
    dispersion_quantile: float,
    rationale: str,
    device: str | None = None,
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
) -> Assessment:
    """Assess a checkpoint trained on one mosaic against that mosaic's own reserved regions.

    The regions are its run's within-image calibration and holdout split with
    ``calibration_ratio``, cut into ``k_cal`` and ``k_test`` buffered bands, for a count delivery
    of ``trait``; the result is recorded as a new assessment, the band counts and ``rationale``
    with it, and returned.

    The bands are laid at the lattice the run trained at (its ``data.tiling``) and predicted
    through a tiled pass, the count criterion fitted as :func:`assess` fits it at
    ``equivalence_z`` and ``dispersion_quantile``, every derived value from the calibration region
    alone, the object density among them, which caps each frame a pass under it predicts at that
    frame's own pixels. Each band is a reference sample of the mosaic, its region named. The
    reference's scope is the training mosaic's recorded content identity. Refuses
    (:class:`AssessmentRefusedError`) a delivery that is not a count, a checkpoint no run of this
    project produced, a run with no reserved regions, regions not attested complete, too few
    bands carrying ground truth, a band not held out from the run's training regions, a mosaic
    that changed since the split, and a mosaic label document only the model stands behind.
    """
    import numpy as np

    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines import block_calibration as blocks
    from tcip_mcp.pipelines.data.datasets import PER_BOX_KEYS, object_region
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.data.selection import source_digests
    from tcip_mcp.pipelines.derivations import derive_block_scale_px
    from tcip_mcp.pipelines.execution import Reference
    from tcip_mcp.pipelines.operating_point import spatial_disjointness
    from tcip_mcp.pipelines.raster_source import BandGroupRef, open_raster

    if delivery_kind not in (PER_IMAGE_COUNT, PER_PLANT_COUNT_AGGREGATE):
        raise AssessmentRefusedError(
            f"a mosaic's reserved regions answer for a count, not a "
            f"{delivery_kind} delivery."
        )
    revision, prep = _prepared(
        project,
        checkpoint_path=checkpoint_path,
        trait=trait,
        delivery_kind=delivery_kind,
        stated=stated.model_copy(update={"tile": True}),
        device=device,
        tile_batch_size=tile_batch_size,
    )
    experiment_id = prep.checkpoint.experiment_id
    if experiment_id is None:
        raise AssessmentRefusedError(
            "no run of this project produced this checkpoint, so no mosaic's "
            "reserved regions are known to have been held out from it."
        )
    resolved = run_resolution(experiment_id, project=project)
    spatial = blocks.reserved_spatial_regions(resolved.partition.spatial)
    if spatial is None:
        raise AssessmentRefusedError(
            f"run {experiment_id!r} resolved no within-image spatial split with a reserved "
            "calibration and holdout region (train it with data.split.calibration_ratio set)."
        )
    (mosaic,) = resolved.partition.samples
    stem = mosaic.member
    lattice = cast("TilingSpec", resolved.data.tiling)
    tile_size, overlap = (
        cast(int, lattice.tile_size),
        cast(float, lattice.overlap),
    )
    scope = prep.scope
    subject = scope.document_attributes(
        f"experiment {experiment_id!r}"
    ).subject
    (mosaic,) = _admit_reference([mosaic], scope)
    document = mosaic.read
    reads = _reference_reads([mosaic])
    cal_rect, test_rect = (
        spatial.calibration_region[0],
        spatial.holdout_region[0],
    )
    blocks.check_completeness(
        document,
        f"{stem}'s label document",
        subject,
        {"calibration_region": cal_rect, "holdout_region": test_rect},
    )
    source = mosaic.image
    run_dir = _open_run(project)
    retained, _measured = _retained(run_dir, [mosaic], reads)
    target = json_det_targets(document.annotations, scope)
    gt = {k: np.asarray(target[k]) for k in PER_BOX_KEYS if k in target}
    gt["boxes"] = gt["boxes"].astype(np.float32).reshape(-1, 4)
    plants = None
    if resolved.data.plant_csv_paths:
        from tcip_mcp.pipelines.postprocessing.plant_mapping import (
            read_plant_csvs,
        )

        plants = (
            read_plant_csvs([Path(c) for c in resolved.data.plant_csv_paths])
            or None
        )
    try:
        cx0, cy0, cx1, cy1 = cal_rect
        buffer_px, scale_source = derive_block_scale_px(
            tile_size=tile_size,
            plants=plants,
            objects=object_region(
                blocks.select_gt_for_band(gt, cal_rect),
                (cx1 - cx0) * (cy1 - cy0),
            ),
            raster_path=None if isinstance(source, BandGroupRef) else source,
        )
        bands: dict[Side, dict[str, tuple[int, int, int, int]]] = {
            "calibration": blocks.band_rects(
                cal_rect, k_cal, tile_size, overlap, buffer_px, "cal"
            ),
            "holdout": blocks.band_rects(
                test_rect, k_test, tile_size, overlap, buffer_px, "test"
            ),
        }
    except ValueError as exc:
        raise AssessmentRefusedError(
            f"no feasible band layout over the reserved regions "
            f"(k_cal={k_cal}, k_test={k_test}): {exc}"
        ) from exc
    leaks = spatial_disjointness(
        spatial, [r for b in bands.values() for r in b.values()]
    )
    if leaks:
        raise AssessmentRefusedError(
            f"band(s) {leaks} are not held out from run {experiment_id!r}'s "
            "training regions, so they cannot stand as its reference."
        )
    band_regions = {
        side: {
            name: object_region(
                blocks.select_gt_for_band(gt, (x0, y0, x1, y1)),
                (x1 - x0) * (y1 - y0),
            )
            for name, (x0, y0, x1, y1) in side_bands.items()
        }
        for side, side_bands in bands.items()
    }
    band_counts = {
        side: {name: len(region.boxes) for name, region in regions.items()}
        for side, regions in band_regions.items()
    }
    for side, counts in band_counts.items():
        blocks.check_feasibility(counts, side=side)
    samples = [
        Sample(
            member=name,
            source=mosaic.source,
            ground_truth=mosaic.ground_truth,
            group=name,
            side=side,
            rect=rect,
            image=source,
        )
        for side, side_bands in bands.items()
        for name, rect in side_bands.items()
    ]
    digest_of = source_digests(samples)
    p = prep.runnable(
        Reference(regions=list(band_regions["calibration"].values()))
    )

    def collect(execution: Execution) -> tuple[list[dict], list[dict]]:
        with open_raster(source, p.predictor.in_chans) as reader:
            if (reader.width, reader.height) != (
                spatial.width,
                spatial.height,
            ):
                raise AssessmentRefusedError(
                    f"the run recorded a {spatial.width}x{spatial.height} mosaic and "
                    f"{source} now reads {reader.width}x{reader.height}; retrain or re-split "
                    "against the current file."
                )
            return (
                _band_records(
                    reader,
                    "calibration",
                    bands["calibration"],
                    p,
                    execution,
                    gt=gt,
                    digest_of=digest_of,
                    source=str(source),
                ),
                _band_records(
                    reader,
                    "holdout",
                    bands["holdout"],
                    p,
                    execution,
                    gt=gt,
                    digest_of=digest_of,
                    source=str(source),
                ),
            )

    evidence, failures, _cal, _hold = _count_fit(
        p,
        revision.entry,
        collect,
        equivalence_z=equivalence_z,
        dispersion_quantile=dispersion_quantile,
    )
    disjointness, disjoint_failures = _disjointness(
        digest_of,
        [s for s in samples if s.side == "calibration"],
        [s for s in samples if s.side == "holdout"],
        _run_sides(project, experiment_id),
    )
    return _finish(
        project,
        run_dir,
        revision,
        delivery_kind=delivery_kind,
        producer=p.checkpoint.producer,
        execution=p.execution,
        rationale=rationale,
        reference=_reference(
            None,
            spatial.raster_content_identity,
            samples,
            digest_of,
            retained,
        ),
        disjointness=disjointness,
        criterion={
            "count": {
                **evidence,
                "block_scale_px": buffer_px,
                "block_scale_source": scale_source,
                "k_cal": k_cal,
                "k_test": k_test,
                "band_gt_counts": band_counts,
                "density_uniformity_flags": sorted(
                    f
                    for counts in band_counts.values()
                    for f in blocks.density_uniformity_flags(counts)
                ),
            }
        },
        scale=None,
        failures=[*disjoint_failures, *failures],
    )


def _band_records(
    reader: RasterSource,
    side: Side,
    bands: dict[str, tuple[int, int, int, int]],
    p: Pass,
    execution: Execution,
    *,
    gt: dict,
    digest_of: dict[str, str],
    source: str,
) -> list[dict]:
    """Return one evaluation record per band of ``side``, in band-name order.

    Each is :func:`~tcip_mcp.pipelines.training.evaluation.result_record`'s over the tiled
    prediction under ``execution`` over the band widened on every side (clipped to the mosaic) by
    the overlap its lattice states between neighbors, the frame that forward ran on, keeping the
    detections and the ground truth centered in the band itself, in that frame's coordinates, the
    band as the record's scored ``region``, the band sample's source digest as its ``image_id``.
    """
    import numpy as np

    from tcip_mcp.pipelines import block_calibration as blocks
    from tcip_mcp.pipelines.data.datasets import PER_BOX_KEYS
    from tcip_mcp.pipelines.raster_source import Rect, _RegionView
    from tcip_mcp.pipelines.slicing import slice_lattice
    from tcip_mcp.pipelines.training.evaluation import (
        gt_records,
        result_record,
    )

    tile_size = cast(int, execution.tile_size)
    halo = (
        tile_size
        - slice_lattice(
            tile_size, 2 * tile_size, tile_size, cast(float, execution.overlap)
        )[1][0]
    )
    records = []
    for name, inner in sorted(bands.items()):
        ix0, iy0, ix1, iy1 = inner
        hx0, hy0 = max(0, ix0 - halo), max(0, iy0 - halo)
        hx1, hy1 = (
            min(reader.width, ix1 + halo),
            min(reader.height, iy1 + halo),
        )
        location = Sample(
            member=name,
            source=source,
            ground_truth="",
            group=name,
            side=side,
            rect=inner,
        ).location
        result = p.predictor.predict_sliced(
            _RegionView(reader, Rect(hx0, hy0, hx1, hy1)),
            execution=execution,
            tile_batch_size=p.tile_batch_size,
            source_label=name,
        )
        region = (ix0 - hx0, iy0 - hy0, ix1 - hx0, iy1 - hy0)
        kept = blocks.centered_in(
            np.asarray(result["boxes"], dtype=np.float64).reshape(-1, 4),
            region,
        )
        rows = {
            key: [v for v, k in zip(result[key], kept) if k]
            for key in PER_BOX_KEYS
            if key in result
        }
        band = blocks.select_gt_for_band(gt, inner)
        band["boxes"] += [region[0], region[1], region[0], region[1]]
        records.append(
            result_record(
                {**result, **rows},
                gt_records(band),
                image_id=digest_of[location],
                region=region,
            )
        )
    return records


def _read_reference_csv(
    data: bytes, csv_path: str
) -> dict[str, dict[str, Any]]:
    """Return ``stem -> {"physical_extent": float, "unit": str}`` from a reference CSV's bytes.

    ``data`` are the bytes of the breeder's reference CSV at ``csv_path``, its ``image_stem``,
    ``physical_extent`` and ``unit`` columns read by name. A missing column, a short row, a
    non-numeric or non-positive extent and a repeated stem each refuse naming the line.
    """
    import csv
    import io

    out: dict[str, dict[str, Any]] = {}
    with io.StringIO(data.decode("utf-8"), newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        columns = ("image_stem", "physical_extent", "unit")
        missing = [name for name in columns if name not in header]
        if missing:
            raise AssessmentRefusedError(
                f"{csv_path} is missing column(s) {missing} in its header "
                f"{header!r}; expected {list(columns)}."
            )
        index = {name: header.index(name) for name in columns}
        for line_no, row in enumerate(reader, start=2):
            if len(row) < len(header):
                raise AssessmentRefusedError(
                    f"{csv_path}:{line_no} has {len(row)} column(s), fewer "
                    f"than the header's {len(header)}."
                )
            stem = row[index["image_stem"]].strip()
            try:
                extent = float(row[index["physical_extent"]].strip())
            except ValueError:
                raise AssessmentRefusedError(
                    f"{csv_path}:{line_no} has a non-numeric "
                    f"physical_extent for stem {stem!r}."
                ) from None
            if not extent > 0:
                raise AssessmentRefusedError(
                    f"{csv_path}:{line_no} states physical_extent {extent} "
                    f"for stem {stem!r}: a physical length is positive."
                )
            if stem in out:
                raise AssessmentRefusedError(
                    f"{csv_path}:{line_no} repeats stem {stem!r}."
                )
            out[stem] = {
                "physical_extent": extent,
                "unit": row[index["unit"]].strip(),
            }
    return out


def assess_physical_scale(
    project: Path,
    *,
    trait: str,
    selection_dir: str,
    reference_csv: str,
    unit: str,
    reference_subject: str,
) -> Assessment:
    """Derive a per-pixel physical scale in ``unit`` and check it, recording a new assessment.

    The scale is derived on the calibration side of the selection at ``selection_dir`` and
    checked against the holdout side, for ``trait``'s latest confirmed revision; the result is
    recorded as a new assessment and returned.

    The reference is admitted as :func:`assess` admits one, under the selection's own scope, and
    its sides must share no source digest. Each reference image carries exactly one
    ``reference_subject`` polygon or mask, whose principal-axis extent is its pixel extent;
    ``reference_csv`` (``image_stem, physical_extent, unit``) is the breeder's physical extent of
    the same object, read once, retained with the reference and measured as read; each path is
    :func:`~tcip_mcp.registry_paths.located` against ``project``. The scale is the mean implied
    scale of the calibration side; it passes when the holdout's own relative dispersion and the
    scale's relative deviation from the holdout mean are both within ``scale_tolerance_frac``.
    Refuses an unauthored tolerance, a unit that is not a linear length unit crops.yml declares, a
    selection without both reference sides, a reference the admission would not admit or only the
    model stands behind, a reference image with no row in the CSV, a non-positive extent, and a
    reference annotation that is not exactly one polygon.
    """
    import math
    import statistics

    from tcip_annotation.state import polygonal
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.pipelines.data.selection import source_digests
    from tcip_mcp.pipelines.measurement.mask_geometry import (
        principal_axis_extent_of_points,
    )
    from tcip_mcp.traits import authored, crops_length_units

    selection_dir = str(located(selection_dir, project))
    reference_csv = str(located(reference_csv, project))
    revision = latest_confirmed(trait, project)
    authored(revision.entry, ("scale_tolerance_frac",))
    if unit not in crops_length_units():
        raise AssessmentRefusedError(
            f"a per-pixel scale is a length per pixel, and {unit!r} is not a "
            f"linear length unit crops.yml declares "
            f"({sorted(crops_length_units())})."
        )
    selection, cal, hold = _reference_sides(project, selection_dir)
    admitted = _admit_reference(cal + hold, selection.scope)
    cal, hold = admitted[: len(cal)], admitted[len(cal) :]
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
            raise AssessmentRefusedError(
                f"{reference_csv} names no physical extent for reference "
                f"image {sample.member!r}."
            )
        if row["unit"] != unit:
            raise AssessmentRefusedError(
                f"{reference_csv} states {sample.member!r} in "
                f"{row['unit']!r}, not {unit!r}; a reference is never "
                "converted between units."
            )
        found = [
            a
            for a in sample.read.annotations
            if a.subject == reference_subject
        ]
        if len(found) != 1 or not polygonal(found[0].geometry):
            raise AssessmentRefusedError(
                f"{sample.member!r} carries {len(found)} {reference_subject!r} annotation(s); a "
                "reference image carries exactly one, as a polygon or mask, since a box's long "
                "side is its projected extent, not the object's length."
            )
        points = [
            pt for ring in cast(Any, found[0].geometry).rings for pt in ring
        ]
        pixels = principal_axis_extent_of_points(points)
        if not pixels > 0:
            raise AssessmentRefusedError(
                f"{sample.member!r}'s {reference_subject!r} polygon has no "
                "extent, so no scale is implied by it."
            )
        implied[sample.side][sample.member] = row["physical_extent"] / pixels
    tolerance = cast(float, revision.entry.scale_tolerance_frac)
    failures += [
        f"insufficient_{side}_references"
        for side in ("calibration", "holdout")
        if len(implied[side]) < 2
    ]
    evidence: dict[str, Any] = {
        "implied_scales": implied,
        "tolerance_frac": tolerance,
    }
    value = None
    if not any(f.startswith("insufficient_") for f in failures):
        value = statistics.mean(implied["calibration"].values())
        held = list(implied["holdout"].values())
        mean = statistics.mean(held)
        dispersion = abs(statistics.stdev(held) / mean) if mean else math.inf
        deviation = abs(value - mean) / abs(mean) if mean else math.inf
        evidence.update(
            calibration_mean=value,
            holdout_mean=mean,
            holdout_relative_dispersion=dispersion,
            relative_deviation=deviation,
        )
        failures += (
            ["holdout_dispersion_exceeds_tolerance"]
            if dispersion > tolerance
            else []
        ) + (
            ["holdout_mean_outside_tolerance"] if deviation > tolerance else []
        )
    return _finish(
        project,
        run_dir,
        revision,
        delivery_kind=None,
        producer=None,
        execution=None,
        rationale=None,
        reference=_reference(
            str(selection_dir), None, cal + hold, digest_of, retained
        ),
        disjointness=disjointness,
        criterion={"scale": evidence},
        scale={
            "unit": unit,
            "value": value,
            "reference_subject": reference_subject,
        },
        failures=failures,
    )
