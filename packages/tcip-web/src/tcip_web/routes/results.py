"""Results routes: plant-mapping, per-plant phenology curves, CSV export.

The delivered target is a per-plant CSV of a registered trait's own milestone-date columns
(``<phenology_prefix>_<NN>per_date`` for each milestone its ``TraitSpec`` declares; every
registered trait has its own prefix and milestone set, resolved from the spec, with no code
change here). That pipeline looks like:

    predictions(date) -> per-plant detections of the trait's object (via plant mapping)
                       -> classify each detection positive vs not (validated classifier)
                       -> positive fraction / total per (plant, date)
                       -> find the dates that fraction crosses each declared milestone

The positive-state fraction is the share of a plant's detections of the trait's object that are
in its positive/measured state: an expert-defined morphological stage from a validated classifier, never a
geometric proxy such as bbox height (see the ``phenology`` skill + the CLAUDE.md
measurement-integrity invariant). The milestone math lives once in
``tcip_mcp...postprocessing.phenology``; this module is the HTTP surface the Results tab calls
and delegates to it.

This module also serves the operationalization record: what a trait's delivered number means, who
recorded it, and the breeder's confirmation of it. The confirmation is written here and nowhere
else, because an agent able to write one would be confirming its own definition of the measurement.

The backend owns everything except the model inference (which is driven by the Inference tab).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from tcip_mcp.pipelines.postprocessing import phenology, plant_mapping

from tcip_web.paths import assert_path_allowed, assert_project_root_allowed, exposed_arrival, within
from tcip_web.state import store

if TYPE_CHECKING:
    from tcip_mcp.class_registry import ClassRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])


def _guarded_path(path: str) -> Path:
    """Confine a client-supplied path to the allowed roots; the resolved path every read uses."""
    try:
        return assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _open_project_root(stated: Optional[str] = None) -> Path:
    """The project the GUI has open: the only project a Results door exports into or audits under.

    Only the guarded selection route sets it, so a delivery cannot be pointed at a project by
    naming one in a payload. ``stated`` is a payload's own ``project_root``; when given it must be
    the same filesystem object as the open project, so the request record a browser replays into
    the CSV door cannot disagree with what it showed on screen.
    """
    root = store.project_root
    if root is None:
        raise HTTPException(
            409, "no project is open; open a project in the GUI before using the Results doors")
    if stated:
        try:
            same = os.path.samefile(_guarded_path(stated), root)
        except OSError as exc:
            raise HTTPException(403, f"project_root {stated} cannot be compared with the open "
                                     f"project {root}: {exc}") from exc
        if not same:
            raise HTTPException(
                403, f"project_root {stated} is not the open project {root}; a delivery belongs "
                     "to the project the GUI has open")
    return root


def _evidence_roots(root: Path) -> list[Path]:
    """The open project's own tree plus every dataset root registered to it."""
    from tcip_mcp.tools.project_tools import read_datasets

    return [root, *(Path(e["path"]) for e in read_datasets(root) if e.get("path"))]


def _belonging(root: Path, *paths: Optional[str]) -> list[Optional[Path]]:
    """Confine evidence paths to the open project: its tree or a dataset registered to it.

    Resolved paths come back in the order given (``None`` for an omitted one), and every later
    read uses them. A path inside the managed allow-set but outside this project's own set
    refuses, naming the project and the roots it checked: an allow-list proves a managed
    filesystem, not that the evidence belongs to the project being delivered from.
    """
    roots = _evidence_roots(root)
    out: list[Optional[Path]] = []
    for p in paths:
        if not p:
            out.append(None)
            continue
        resolved = _resolved(p)
        if not any(within(resolved, r) for r in roots):
            raise HTTPException(
                403, f"{p} does not belong to project {root}: it is under neither the project "
                     f"nor a dataset registered to it ({', '.join(str(r) for r in roots)})")
        out.append(resolved)
    return out


def _resolved(path: str) -> Path:
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(400, f"cannot resolve {path}: {exc}") from exc


def _reference_file(path: str, request: Request) -> Path:
    """A breeder-supplied input file read by a door: unconfined from this machine, confined from
    a routable connection, the same rule the folder picker applies."""
    if exposed_arrival(request.scope):
        return _guarded_path(path)
    return _resolved(path)


def _under_project(root: Path, path: str) -> Path:
    """A path a Results door writes, confined to the open project's own tree (never a dataset)."""
    resolved = _resolved(path)
    if not within(resolved, root):
        raise HTTPException(
            403, f"{path} is outside the open project {root}; a mapping or a delivery is project "
                 "state and is written under the project itself")
    return resolved


def _guarded_project_root(project_root: str) -> Path:
    """Confine a request's project root and hand back the resolved path every later read uses.

    The returned path is the load-bearing half: a door that guards the raw string and then
    resolves that string again reopens exactly what the guard closed.
    """
    try:
        return assert_project_root_allowed(project_root)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _audit(project_root: str, tool: str, arguments: dict, **extra: object) -> None:
    """Record a GUI results mutation in the project's audit log.

    A delivery and the plant mapping behind it are project state, not dataset state: the
    dataset can be read by more than one project, but the export is this project's own
    outward action. ``extra`` carries a fact the entry shape has no home for, such as the
    person an action is recorded as coming from. Never fails the request.
    """
    if not project_root:
        return
    from tcip_mcp.audit import record_event

    record_event(tool, arguments, source="gui", scope=project_root, **extra)


# ── Plant mapping ──────────────────────────────────────────────────────


class BuildMappingPayload(BaseModel):
    name: str
    images_root: str
    plant_csv_paths: list[str]
    dates: Optional[list[str]] = None
    nn_tolerance_m: Optional[float] = None


@router.post("/plant_mapping/build")
def build_plant_mapping(payload: BuildMappingPayload, request: Request) -> dict:
    """Build the image-to-plant mapping from the open project's images and the breeder's plant files.

    The mapping is project state, so it always persists under the open project, by name, and is
    audited there; ``images_root`` must be a registered dataset's own ``images/`` directory (its
    identity record minted by ``register_dataset``), so a mapping cannot be built for one project
    from another project's captures, or over a directory that merely ends in ``images``. The
    plant-location files are the breeder's own reference data, picked from wherever they keep
    them: from this machine they are read from any path, like the picker that finds them; from a
    routable connection they are confined to the allowed roots. A receipt that cannot be written
    answers 409: the record it would have named is left on disk, refused until a rebuild replaces
    it.
    """
    from tcip_store.layout_claims import NAME_SEGMENT

    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.dataset_layout import dataset_root_of, image_root, require_dataset_identity
    from tcip_mcp.pipelines.data.splits import same_directory
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem

    if not NAME_SEGMENT.fullmatch(payload.name):
        raise HTTPException(
            400,
            f"name {payload.name!r} is not lowercase letters, digits and single hyphens "
            f"({NAME_SEGMENT.pattern})")

    root = _open_project_root()
    (images_root,) = _belonging(root, payload.images_root)
    assert images_root is not None
    plant_csvs = [_reference_file(p, request) for p in payload.plant_csv_paths]

    candidate = dataset_root_of(images_root)
    if candidate is None or not same_directory(image_root(candidate), images_root):
        raise HTTPException(
            400,
            f"{payload.images_root} is not a dataset's own images/ root; build_plant_mapping "
            "maps a registered dataset's image tree")
    try:
        identity = require_dataset_identity(candidate)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        build = plant_mapping.build_mapping(
            images_root, plant_csvs,
            name=payload.name, dataset_root=candidate, dataset_id=identity["id"],
            project_root=root, built_by="gui_build_plant_mapping",
            dates=payload.dates, nn_tolerance_m=payload.nn_tolerance_m,
        )
    except AmbiguousImageStem as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        plant_mapping.persist_mapping(build, root, payload.name)
    except AuditEntryNotWritten as exc:
        raise HTTPException(409, str(exc)) from exc

    _audit(
        str(root),
        "gui_build_plant_mapping",
        {
            "name": payload.name,
            "images_root": str(images_root),
            "dataset_root": str(candidate),
            "n_dates": len(build.dates),
        },
    )

    summary = {}
    for date, assignments in build.assignments.items():
        matched = sum(1 for a in assignments if a.plot_name)
        summary[date] = {
            "n_images": len(assignments),
            "n_mapped": matched,
            "avg_distance_m": (
                sum(a.distance_m for a in assignments if a.distance_m is not None)
                / max(1, sum(1 for a in assignments if a.distance_m is not None))
            ),
        }

    return {
        "mapping": build.rows(),
        "summary": summary,
        "unreadable": build.unreadable,
    }


class LoadMappingPayload(BaseModel):
    name: str


@router.post("/plant_mapping/load")
def load_plant_mapping(payload: LoadMappingPayload) -> dict:
    from tcip_store import StoreError
    from tcip_store.layout_claims import NAME_SEGMENT

    if not NAME_SEGMENT.fullmatch(payload.name):
        raise HTTPException(
            400,
            f"name {payload.name!r} is not lowercase letters, digits and single hyphens "
            f"({NAME_SEGMENT.pattern})")

    root = _open_project_root()
    try:
        build = plant_mapping.load_mapping(root, payload.name)
    except (StoreError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"mapping": build.rows() if build is not None else {}}


@router.get("/plant_mapping/list")
def list_plant_mappings() -> dict:
    """Every mapping name persisted under the open project, for the Results tab's picker."""
    root = _open_project_root()
    return {"names": plant_mapping.plant_mapping_names(root)}


# ── Per-plant curves ───────────────────────────────────────────────────


class PhenologyPayload(BaseModel):
    """The inputs a phenology measurement is computed from: never the measurement itself.

    Every Results door takes this shape. A caller-composed ``rows`` table was tried here before,
    with the server inferring what it meant (inference predicates over row shape, an
    ``export_kind`` declaration), but both were defeated because the caller controls the column
    names and the declaration. The real problem: the server was classifying data it did not
    produce, and that information is not in the payload. A table with a ``ratio`` column is a
    phenology curve or an unrelated QC table depending on where it came from, which is exactly what a
    caller-supplied payload erases. So no door accepts rows: they accept a request to compute
    rows, and the server knows what it produced because it produced it.
    """

    project_root: str
    mapping_name: str  # a name persisted under the open project's own plant-mapping store
    # map date → predictions directory for that date. A trait's positive class id is resolved
    # server-side from each bucket's own recorded id_map: a client-supplied
    # class id is never honored, closing the bypass a caller-chosen id would otherwise open.
    predictions_by_date: dict[str, str]
    trait: str
    # Show provisional numbers on screen rather than refusing outright: the same escape
    # ``compute_phenology`` offers, so a breeder whose operating point is not yet calibrated can see
    # what they have instead of a dead end. It never applies to a file leaving the platform.
    acknowledge_unvalidated: bool = False


class _PhenologyMeasurement:
    """One trait's per-plant phenology measurement plus the on-disk evidence that qualifies it."""

    def __init__(
        self, spec, plants: dict, validity: dict, gate, positive_class_id, project_root: Path,
        basis, bindings: dict, predictions_by_date: dict[str, str], flags: dict[str, str | None],
        plant_mapping_disclosure: dict,
    ) -> None:
        self.spec, self.plants, self.validity, self.gate = spec, plants, validity, gate
        self.positive_class_id = positive_class_id
        # What the count reconciliation verified per bucket, and the buckets it verified, by their
        # resolved paths.
        self.bindings, self.predictions_by_date = bindings, predictions_by_date
        self.pred_dirs = list(predictions_by_date.values())
        # The guarded, resolved root every later write and audit entry resolves from.
        self.project_root = project_root
        # What the precondition rested on, re-checked before this measurement reaches a caller.
        self.basis = basis
        # The dimension flags the gate above was computed from; validity["tile_size"] is None for
        # an untiled delivery, not the same as the key being absent from the gate's own flags.
        self.flags = flags
        # The mapping this delivery attributed detections through, plus what verify_mapping_inputs
        # could not verify: threaded to the event and the CSV's own columns.
        self.plant_mapping_disclosure = plant_mapping_disclosure

    @property
    def positive_class_assessed(self) -> bool:
        """Whether the trait's positive-class axis was assessed at all.

        Requires the bucket-level fact ``compute_phenology`` refuses on: some bucket's recorded
        ``id_map`` actually contains the trait's positive class, as well as a fully-classified date.
        ``per_plant_phenology``'s flag alone reads True for a date with zero detections even in a
        bucket that never had the axis, because zero detections are trivially "all classified".
        """
        return self.positive_class_id is not None and bool(self.plants["positive_class_assessed"])

    def curve_rows(self) -> list[dict]:
        """Per-(plant, date) rows: the milestone rows' own series, not a second aggregation."""
        return [
            {"plant_id": row["plant_id"], "accession": row["accession"], **point}
            for row in self.plants["rows"] for point in row["series"]
        ]

    def milestone_rows(self) -> list[dict]:
        return [{k: v for k, v in row.items() if k != "series"} for row in self.plants["rows"]]


def _delivered_registry(pred_dirs: Sequence[str]) -> "ClassRegistry | None":
    """The class registry for the single dataset this delivery's prediction buckets belong to.

    Resolved from the buckets themselves, the way ``compute_phenology`` resolves it, never from the
    open project's own root: a project's dataset commonly lives outside its own tree. ``None`` when
    ``pred_dirs`` is empty: with no bucket named yet there is no delivered dataset to check against,
    the same case a caller-supplied ``predictions_by_date`` of ``{}`` reaches the mapping-not-found
    refusal through, unchecked here. Otherwise refuses by name when the buckets span more than one
    dataset root, or resolve to none with a registry, since a state_crossing_dates delivery naming
    real buckets cannot check its positive class with none in reach.
    """
    if not pred_dirs:
        return None
    from tcip_mcp.class_registry import RegistryError, registry_for_pred_dirs

    try:
        registry = registry_for_pred_dirs(pred_dirs)
    except RegistryError as exc:
        raise HTTPException(400, str(exc)) from exc
    if registry is None:
        raise HTTPException(
            400,
            f"no class registry is reachable for the dataset behind {list(pred_dirs)}: register "
            "the dataset (register_dataset) or write its classes.json before a state_crossing_dates "
            "delivery can check the positive class",
        )
    return registry


def _measure_phenology(payload: PhenologyPayload) -> _PhenologyMeasurement:
    """Compute a trait's phenology measurement and reconcile the evidence behind it: one producer.

    Every Results door routes through here, so the numbers and the validity that qualifies them are
    read in one place from the buckets' own sidecars: the curve and milestone doors share the same
    gate as CSV, so a breeder never sees an unvalidated phenology date on screen only to have the
    refusal surface later on Download. Rows come from the canonical ``per_plant_phenology`` (the same
    function ``compute_phenology`` delivers from) rather than a second aggregation loop, so the two
    surfaces cannot diverge.

    The spec and the operationalization record come from one call against the guarded root, so the
    two can never be read from different projects, and the precondition runs before anything else
    this door does: a breeder must not see a curve on screen that Download would then refuse.
    """
    from tcip_mcp.operationalization import (
        STATE_CROSSING_DATES,
        check_operationalization,
        resolve_trait_and_record,
    )
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity,
        check_delivery_gate,
        reconcile_classifier_validity,
        reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )
    from tcip_mcp.traits import TraitUnknownError

    root = _open_project_root(payload.project_root)
    try:
        spec, record, _specs_dir = resolve_trait_and_record(
            payload.trait, STATE_CROSSING_DATES, project_root=root)
    except TraitUnknownError as e:
        raise HTTPException(400, str(e)) from e

    resolved_dirs = _belonging(root, *payload.predictions_by_date.values())
    predictions_by_date = {
        date: str(p) for date, p in zip(payload.predictions_by_date, resolved_dirs) if p is not None
    }

    # The delivered dataset's own registry, resolved from the buckets this delivery actually reads.
    registry = _delivered_registry(list(predictions_by_date.values()))
    stated = check_operationalization(spec, record, STATE_CROSSING_DATES, registry=registry)
    if not stated.ok:
        raise HTTPException(400, stated.as_detail())

    try:
        mapping_build, verified = plant_mapping.resolve_delivery_mapping(
            root, payload.mapping_name, predictions_by_date)
    except plant_mapping.MappingDeliveryRefusal as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    mapping_raw = mapping_build.rows()

    pred_dirs = list(predictions_by_date.values())
    recon = reconcile_operating_point_validity(pred_dirs, trait=payload.trait)
    classifier_recon = reconcile_classifier_validity(pred_dirs)
    # The same binding compute_phenology applies, from the same shared owner rather than a second
    # copy: a classifier stamp calibrated for another trait or against a run that did not produce
    # these predictions does not validate this delivery. Without it the web door accepted a stamp
    # the MCP door rejects, and this route writes that stamp into the delivered CSV.
    classifier_state, binding_note = bind_classifier_validity(
        classifier_recon["validated"], pred_dirs, pred_dirs, trait=payload.trait,
    )
    # The tile scale is the other gating dimension: a tile edge with no real basis at all is as
    # untrustworthy here as an uncalibrated conf, operative only for tiled buckets.
    tile_recon = reconcile_tile_size_validity(pred_dirs)
    validity = {
        "operating_point": recon["validated"],
        "classifier": classifier_state,
        "operating_point_conf": recon["conf"],
        "missing_operating_point_sidecars": recon["missing_sidecars"],
        "unvalidated_buckets": recon["unvalidated_buckets"],
        "binding_notes": recon["binding_notes"],
        "missing_classifier_sidecars": classifier_recon["missing_sidecars"],
        "classifier_binding_note": binding_note,
        "tile_size": tile_recon["validated"],
        "unvalidated_tile_size_buckets": tile_recon["unvalidated_buckets"],
    }
    flags = phenology.phenology_delivery_flags(classifier_state, recon["validated"], tile_recon)
    gate = check_delivery_gate(flags, acknowledge_unvalidated=payload.acknowledge_unvalidated)
    plants = phenology.per_plant_phenology(
        mapping_raw, predictions_by_date,
        positive_class_name=spec.positive_class_name, spec=spec,
    )
    positive_class_id, _msg = phenology.resolve_positive_class_id(spec, predictions_by_date)
    return _PhenologyMeasurement(
        spec, plants, validity, gate, positive_class_id, root, stated.basis,
        recon["bindings"], predictions_by_date, flags,
        mapping_build.delivery_disclosure(verified))


def _still_stated(measurement: _PhenologyMeasurement, trait: str) -> None:
    """Refuse when the confirmation this measurement was produced under moved while it was produced.

    Called immediately before a response body is composed and before the export door writes its
    file, so a withdrawal or a spec edit mid-delivery leaves nothing delivered and nothing written.
    Two keys in two stores cannot be read atomically together, so this closes that window instead.
    """
    from tcip_mcp.operationalization import (
        STATE_CROSSING_DATES,
        check_operationalization,
        resolve_trait_and_record,
    )

    spec, record, _specs_dir = resolve_trait_and_record(
        trait, STATE_CROSSING_DATES, project_root=measurement.project_root)
    registry = _delivered_registry(measurement.pred_dirs)
    check = check_operationalization(
        spec, record, STATE_CROSSING_DATES, registry=registry, basis=measurement.basis)
    if not check.ok:
        raise HTTPException(400, check.as_detail())


def _refusal(measurement: _PhenologyMeasurement) -> str:
    from tcip_mcp.pipelines.resolution import binding_notes_text

    tile_note = ""
    if measurement.validity["unvalidated_tile_size_buckets"]:
        tile_note = (
            f" Tiled bucket(s) {measurement.validity['unvalidated_tile_size_buckets']} carry a "
            "tile_size with no persisted training geometry, no recoverable native-frame edge, and "
            "no explicit caller override, so the scale their counts were produced at has no basis. "
            "Re-export with an explicit tile_size, or from a checkpoint whose training tile "
            "geometry was persisted."
        )
    return (
        "phenology delivery requires a validated classifier and count operating point, reconciled from "
        "the prediction buckets' own sidecars (never a caller-asserted string). Unvalidated: "
        f"{list(measurement.gate.unvalidated)} (operating_point="
        f"{measurement.validity['operating_point']!r}, classifier={measurement.validity['classifier']!r}, "
        f"tile_size={measurement.validity['tile_size']!r}; "
        f"missing operating_point.json: {measurement.validity['missing_operating_point_sidecars']}; "
        f"unvalidated buckets: {measurement.validity['unvalidated_buckets']}; missing "
        f"classifier_operating_point.json: {measurement.validity['missing_classifier_sidecars']}). "
        "Produce the predictions via a calibrated export_predictions and calibrate the classifier "
        "via calibrate_classifier_operating_point."
        + tile_note
        + (f" {binding_notes_text(measurement.validity['binding_notes'])}"
           if measurement.validity["binding_notes"] else "")
        + (f" {measurement.validity['classifier_binding_note']}"
           if measurement.validity["classifier_binding_note"] else "")
    )


def _disclosure(measurement: _PhenologyMeasurement) -> dict:
    """What qualifies these numbers, returned beside them so no surface can render them bare."""
    return {
        "validated": measurement.gate.stamp,
        # True whenever a dimension lacked on-disk evidence: including when the caller acknowledged
        # it, which is exactly when a surface must not render these numbers as valid.
        "provisional": bool(measurement.gate.unvalidated),
        "validity_detail": measurement.validity,
        # Honest signal: was anything actually classified along the trait's axis? If false, the
        # ratios are not a valid phenology measurement: do not deliver curves from them.
        "positive_class_assessed": measurement.positive_class_assessed,
        "captures_unverified": measurement.plant_mapping_disclosure["captures_unverified"],
        "plant_csvs_unverified": measurement.plant_mapping_disclosure["plant_csvs_unverified"],
    }


@router.post("/per_plant_curves")
def per_plant_curves(payload: PhenologyPayload) -> dict:
    """Per-(plant, date) positive-fraction curve from classified predictions.

    Gated on the same reconciled evidence as the CSV door (see ``_measure_phenology``): a curve IS the delivered
    phenology measurement, just un-summarised, so it is refused on unvalidated evidence unless the caller
    explicitly acknowledges, in which case it ships marked provisional rather than bare.
    """
    measurement = _measure_phenology(payload)
    if not measurement.gate.ok:
        raise HTTPException(400, _refusal(measurement))
    _still_stated(measurement, payload.trait)
    from tcip_mcp.operationalization import STATE_CROSSING_DATES
    from tcip_mcp.pipelines.resolution import record_delivery_binding_event

    record_delivery_binding_event("results.per_plant_curves", None,
                                  measurement.pred_dirs, measurement.bindings,
                                  measurement_documents=["operating_point",
                                                         "classifier_operating_point"],
                                  scale_document=None,
                                  trait=payload.trait, delivery_kind=STATE_CROSSING_DATES,
                                  project_root=measurement.project_root,
                                  plant_mapping=measurement.plant_mapping_disclosure)
    return {
        "rows": measurement.curve_rows(),
        "n_plants": len(measurement.plants["rows"]),
        "positive_class_id": measurement.positive_class_id,
        **_disclosure(measurement),
    }


# ── Milestone dates + CSV export ───────────────────────────────────────


@router.post("/onset_dates")
def onset_dates(payload: PhenologyPayload) -> dict:
    """Each plant's phenology milestones, computed from the buckets rather than from caller rows.

    Takes the same inputs as ``per_plant_curves``: both doors project one ``per_plant_phenology``
    result rather than accepting a caller-composed table, so a milestone date and the curve it was
    read off can never come from different numbers.
    """
    measurement = _measure_phenology(payload)
    if not measurement.gate.ok:
        raise HTTPException(400, _refusal(measurement))
    _still_stated(measurement, payload.trait)
    from tcip_mcp.operationalization import STATE_CROSSING_DATES
    from tcip_mcp.pipelines.resolution import record_delivery_binding_event

    record_delivery_binding_event("results.onset_dates", None,
                                  measurement.pred_dirs, measurement.bindings,
                                  measurement_documents=["operating_point",
                                                         "classifier_operating_point"],
                                  scale_document=None,
                                  trait=payload.trait, delivery_kind=STATE_CROSSING_DATES,
                                  project_root=measurement.project_root,
                                  plant_mapping=measurement.plant_mapping_disclosure)
    return {"rows": measurement.milestone_rows(), **_disclosure(measurement)}


class ExportCsvPayload(PhenologyPayload):
    # Which server computation to export: a choice of producer, never a claim about what the rows
    # mean or whether they are valid. Picking the "wrong" one yields a correctly-gated CSV of the
    # other thing, so there is nothing here for a caller to defeat.
    payload: Literal["curves", "milestones"]
    filename: Optional[str] = None


@router.post("/export_csv")
def export_csv(payload: ExportCsvPayload) -> Response:
    """Write the CSV for a phenology measurement this route computes itself.

    The gate is unconditional because there is nothing else to branch on: this door computes
    the rows from the buckets (``_measure_phenology``) instead of accepting a caller-composed table, so the only
    question left is whether the evidence on disk supports delivering them. ``acknowledge_unvalidated``
    is deliberately ignored here: it lets the breeder look at provisional numbers on screen, never
    write them to a file that leaves the platform without its evidence, so the writer below is
    always called with ``acknowledge_unvalidated=False``.

    Delegates to ``write_phenology_csv``/``write_phenology_curve_csv``, the same writer(s)
    ``compute_phenology`` calls: the gate, the provenance cells and the recorded delivery event are
    all theirs, so a web-delivered CSV and the MCP door's cannot disagree about what a phenology
    delivery carries. The response body is the file read back as bytes, never a second composition
    of the same rows.
    """
    measurement = _measure_phenology(payload)
    if measurement.gate.unvalidated:
        raise HTTPException(400, _refusal(measurement))
    # The same refusal compute_phenology makes: if no bucket anywhere ever assessed the trait's
    # positive-class axis, the fraction is not a measurement and there is nothing valid to deliver.
    if not measurement.positive_class_assessed:
        raise HTTPException(
            400,
            f"predictions carry no {measurement.spec.positive_class_name!r} class anywhere in this "
            "delivery. The classifier that produced them never assessed this trait's positive "
            "class, so the positive fraction is not a valid measurement. Run and validate the "
            "classifier first.",
        )

    if payload.payload == "milestones":
        rows = measurement.milestone_rows()
        write_csv = phenology.write_phenology_csv
    else:
        rows = measurement.curve_rows()
        write_csv = phenology.write_phenology_curve_csv
    if not rows:
        raise HTTPException(400, "no rows to export")

    filename = payload.filename or f"{payload.trait}_{payload.payload}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    _still_stated(measurement, payload.trait)
    from tcip_mcp.tools.phenology_tools import _resolve_producer_identity

    producer = _resolve_producer_identity(measurement.predictions_by_date)
    # The browser download lands wherever the breeder's browser puts it; the delivery itself
    # belongs to the project, so the same bytes are written to <project>/results_export/, audited.
    saved_path = measurement.project_root / "results_export" / Path(filename).name
    write_csv(
        "results.export_csv", rows, saved_path, measurement.spec,
        flags=measurement.flags, acknowledge_unvalidated=False, basis=measurement.basis,
        operating_point_conf=measurement.validity["operating_point_conf"], producer=producer,
        bindings=measurement.bindings, pred_dirs=measurement.pred_dirs,
        project_root=measurement.project_root,
        plant_mapping=measurement.plant_mapping_disclosure)
    body = saved_path.read_bytes()
    _audit(str(measurement.project_root), "results.export_csv", {
        "trait": payload.trait, "payload": payload.payload, "saved_path": str(saved_path),
        "rows": len(rows),
    })
    headers["X-TCIP-Saved-To"] = str(saved_path)
    return Response(content=body, media_type="text/csv", headers=headers)


# ── What a delivered number means: the record, and the breeder's confirmation ──


class ConfirmOperationalizationPayload(BaseModel):
    """One breeder confirmation, or one withdrawal, for one trait's one delivery kind.

    ``record_seen`` is the content hash of the record the surface rendered. Confirming is an act
    over what the breeder read, so a record rewritten while the panel was open refuses rather than
    taking the click for text nobody displayed. ``user`` is the name the surface carries; when it is
    absent the backend falls back to its own process identity and records that it did.
    """

    project_root: str
    trait: str
    delivery_kind: str
    record_seen: str
    user: Optional[str] = None
    confirmed: bool = True


def _operationalization_body(project_root: Path, trait: str, delivery_kind: str) -> dict:
    """One record as the confirming surface reads it: what is stated, what covers it, what moved.

    ``confirmed_current`` and ``superseded`` come from the same ``check_operationalization`` the
    delivery doors run, so the panel and the door cannot disagree about whether a confirmation still
    holds, and no surface re-derives that comparison for itself. ``delivers`` quotes the crop
    vocabulary's own wording, so the breeder reads their definition before the agent's statement.
    Nothing stated yet reads as null statement fields with ``confirmed_current`` false.
    """
    from tcip_mcp import operationalization as op
    from tcip_mcp.traits import crops_definitions

    spec, record, _specs_dir = op.resolve_trait_and_record(
        trait, delivery_kind, project_root=project_root
    )
    registry = None
    if delivery_kind == op.STATE_CROSSING_DATES:
        # No prediction buckets are in scope for a record display, so this resolves the same
        # project-root-or-single-registered-dataset the statement writer resolves against.
        registry = op.resolve_statement_registry(str(project_root), "")
    check = op.check_operationalization(spec, record, delivery_kind, registry=registry)
    stated = record.value or {}
    definitions = crops_definitions()
    return {
        "trait": trait,
        "delivery_kind": delivery_kind,
        **{field: stated.get(field) for field in op.STATEMENT_FIELDS},
        "confirmed_by": stated.get("confirmed_by"),
        "confirmed_at": stated.get("confirmed_at"),
        "identity_from_request": stated.get("identity_from_request"),
        "confirmed_current": check.ok,
        "superseded": [dict(entry) for entry in check.superseded],
        "registry_problem": check.registry_problem,
        "delivers": [{"name": name, "definition": definitions.get(name)} for name in spec.delivers],
        "record_seen": op.record_seen_hash(stated),
    }


@router.get("/operationalization")
def get_operationalization(project_root: str, trait: str, delivery_kind: str) -> dict:
    """What this trait's delivered number is recorded to mean, and whether it is confirmed now.

    The panel renders every field this returns and posts ``record_seen`` back with the confirmation,
    so what the breeder authorizes is what they were shown.
    """
    from tcip_mcp.traits import TraitUnknownError

    root = _guarded_project_root(project_root)
    try:
        return _operationalization_body(root, trait, delivery_kind)
    except (TraitUnknownError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@router.get("/operationalizations")
def list_operationalizations(project_root: str) -> dict:
    """Every operationalization record this project holds, one row per trait and delivery kind.

    The panel enumerates rather than selects: the record's key is a pair, so a surface that knew
    only a trait could not address one, and a kind selector would hardcode today's kinds into the
    browser. Rows for a kind the Results tab cannot compute are still listed, because a count or an
    aggregate record is confirmable whether or not this tab renders that delivery.

    ``unresolved`` names a record whose trait is no longer registered, or whose delivery kind is not
    one this platform declares, rather than dropping it: an unlistable record and no record at all
    would otherwise read identically.

    ``statement_fields`` names the fields ``record_seen`` hashes, in the order a surface shows them,
    so the confirming surface renders the set the record module owns rather than its own copy of it.
    """
    import tcip_store as ts
    from tcip_mcp import operationalization as op
    from tcip_mcp.traits import TraitUnknownError

    root = _guarded_project_root(project_root)
    records: list[dict] = []
    unresolved: list[dict] = []
    for key in ts.keys(op.OPERATIONALIZATIONS_STORE, str(op.operationalizations_scope(root))):
        trait, delivery_kind = key.parts
        try:
            records.append(_operationalization_body(root, trait, delivery_kind))
        except (TraitUnknownError, ValueError) as e:
            unresolved.append(
                {"trait": trait, "delivery_kind": delivery_kind, "reason": str(e)}
            )
    return {
        "records": records,
        "unresolved": unresolved,
        "statement_fields": list(op.STATEMENT_FIELDS),
    }


@router.post("/operationalization/confirm")
def confirm_operationalization(payload: ConfirmOperationalizationPayload) -> dict:
    """Record the breeder's confirmation of what is on file, or withdraw one they gave before.

    The one path a confirmation is ever written by. No MCP tool reaches this writer: an agent that
    could confirm its own statement would be confirming its own definition of the measurement.

    Refused with 400 when nothing is stated for this trait and kind, or the trait is not registered
    for this project, and with 409 when the record moved since the surface read it, the body then
    carrying what is on file so the panel re-renders that and the breeder confirms what they see.
    ``confirmed`` false withdraws, clearing the four confirmation fields and leaving the statement.

    Nothing verifies that the person at the keyboard is the person named. What this records is a
    name the request supplied, whether the request supplied one at all, and an audit entry saying
    the act happened; it is not authentication and nothing here may be read as one.
    """
    from tcip_mcp import operationalization as op
    from tcip_mcp.traits import TraitUnknownError

    from tcip_web.identity import resolve_user, user_id

    root = _guarded_project_root(payload.project_root)
    # The writer applies the user: convention to whatever name it is given, so it is passed bare.
    actor = resolve_user(payload.user)
    identity_from_request = bool((payload.user or "").strip())
    try:
        record = op.confirm_trait_operationalization(
            root,
            payload.trait,
            payload.delivery_kind,
            user=actor,
            record_seen=payload.record_seen,
            identity_from_request=identity_from_request,
            confirmed=payload.confirmed,
        )
    except op.RecordMoved as e:
        raise HTTPException(
            409,
            {
                "message": str(e),
                "record": _operationalization_body(root, payload.trait, payload.delivery_kind),
            },
        ) from e
    except (TraitUnknownError, ValueError) as e:
        raise HTTPException(400, str(e)) from e

    from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise

    audit_warning: Optional[str] = None
    try:
        record_event_or_raise(
            "results.confirm_trait_operationalization",
            {
                "trait": payload.trait,
                "delivery_kind": payload.delivery_kind,
                "confirmed": payload.confirmed,
                "identity_from_request": identity_from_request,
            },
            source="gui",
            scope=str(root),
            user=user_id(actor),
        )
    except AuditEntryNotWritten as e:
        audit_warning = str(e)

    body = {field: record[field] for field in op.CONFIRMATION_FIELDS}
    body["audit_warning"] = audit_warning
    return body


# ── What a trait's own semantics mean: the authoring statement, and its confirmation ──


def _trait_spec_statement_body(project_root: Path, trait: str) -> dict:
    """One trait's authoring statement as the confirming surface reads it.

    Mirrors ``_operationalization_body``: ``confirmed_current`` is read-time drift detection
    against the live spec (``traits.trait_spec_statement_current``, A1's own mechanism), so the
    panel and any future delivery precondition cannot disagree about whether a confirmation still
    holds. ``statement_fields`` carries the authored ``TraitSpec`` fields exactly as the statement
    recorded them, a flat mapping the record itself already holds, never re-nested or re-derived
    here. Nothing stated yet reads as null statement fields with ``confirmed_current`` false.
    """
    import tcip_store as ts
    from tcip_mcp import traits

    spec = traits.get_trait_for(trait, project_root)
    scope = traits.trait_spec_statements_scope(project_root)
    key = traits.trait_spec_statement_key(scope, trait)
    stored = ts.read_versioned(key, default=None)
    stated = stored.value or {}
    return {
        "trait": trait,
        **{field: stated.get(field) for field in traits.TRAIT_SPEC_STATEMENT_FIELDS},
        "confirmed_by": stated.get("confirmed_by"),
        "confirmed_at": stated.get("confirmed_at"),
        "identity_from_request": stated.get("identity_from_request"),
        "confirmed_current": traits.trait_spec_statement_current(spec, stated),
        "record_seen": traits.trait_spec_statement_seen_hash(stated),
    }


@router.get("/trait-spec-statement")
def get_trait_spec_statement(project_root: str, trait: str) -> dict:
    """What this trait's own semantics were authored to mean, and whether that is confirmed now.

    The panel renders every field this returns and posts ``record_seen`` back with the
    confirmation, so what the breeder authorizes is what they were shown.
    """
    from tcip_mcp.traits import TraitUnknownError

    root = _guarded_project_root(project_root)
    try:
        return _trait_spec_statement_body(root, trait)
    except TraitUnknownError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/trait-spec-statements")
def list_trait_spec_statements(project_root: str) -> dict:
    """Every trait-spec authoring statement this project holds, one row per trait.

    ``unresolved`` names a statement whose trait is no longer registered, rather than dropping it:
    an unlistable statement and no statement at all would otherwise read identically.
    """
    import tcip_store as ts
    from tcip_mcp import traits
    from tcip_mcp.traits import TraitUnknownError

    root = _guarded_project_root(project_root)
    records: list[dict] = []
    unresolved: list[dict] = []
    scope = traits.trait_spec_statements_scope(root)
    for key in ts.keys(traits.TRAIT_SPEC_STATEMENTS_STORE, str(scope)):
        (trait,) = key.parts
        try:
            records.append(_trait_spec_statement_body(root, trait))
        except TraitUnknownError as e:
            unresolved.append({"trait": trait, "reason": str(e)})
    return {
        "records": records,
        "unresolved": unresolved,
        "statement_fields": list(traits.TRAIT_SPEC_STATEMENT_FIELDS),
    }


class ConfirmTraitSpecPayload(BaseModel):
    """One breeder confirmation, or one withdrawal, for one trait's authoring statement.

    ``record_seen`` is the content hash of the statement the surface rendered, the same
    read-what-was-shown discipline the operationalization confirmation already uses.
    """

    project_root: str
    trait: str
    record_seen: str
    user: Optional[str] = None
    confirmed: bool = True


@router.post("/trait-spec-statement/confirm")
def confirm_trait_spec_statement(payload: ConfirmTraitSpecPayload) -> dict:
    """Record the breeder's confirmation of a trait's own authored semantics, or withdraw one.

    The one path a trait-spec confirmation is ever written by. No MCP tool reaches this writer:
    an agent that could confirm its own authoring statement would be confirming its own definition
    of the trait.

    Refused with 400 when nothing is stated for this trait, and with 409 when the statement moved
    since the surface read it, the body then carrying what is on file so the panel re-renders that
    and the breeder confirms what they see.

    The confirmation write itself must not land silently unrecorded (A8): a failed audit append
    does not refuse an otherwise-successful confirmation, it rides back as ``audit_warning`` on an
    ordinary 200 body, since the confirmation itself already committed.
    """
    from tcip_mcp import traits
    from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise

    from tcip_web.identity import resolve_user, user_id

    root = _guarded_project_root(payload.project_root)
    actor = resolve_user(payload.user)
    identity_from_request = bool((payload.user or "").strip())
    try:
        record = traits.confirm_trait_spec(
            root,
            payload.trait,
            user=actor,
            record_seen=payload.record_seen,
            identity_from_request=identity_from_request,
            confirmed=payload.confirmed,
        )
    except traits.TraitSpecStatementMoved as e:
        raise HTTPException(
            409,
            {
                "kind": "trait_spec_authoring",
                "message": str(e),
                "record": _trait_spec_statement_body(root, payload.trait),
            },
        ) from e
    except (traits.TraitSpecStatementNotFound, ValueError) as e:
        raise HTTPException(400, str(e)) from e

    audit_warning: Optional[str] = None
    try:
        record_event_or_raise(
            "results.confirm_trait_spec",
            {"trait": payload.trait, "confirmed": payload.confirmed,
             "identity_from_request": identity_from_request},
            source="gui",
            scope=str(root),
            user=user_id(actor),
        )
    except AuditEntryNotWritten as e:
        audit_warning = str(e)

    body = {field: record[field] for field in traits.TRAIT_SPEC_CONFIRMATION_FIELDS}
    body["audit_warning"] = audit_warning
    return body


# ── What has shipped: the delivery-event record, read-only ─────────────


@router.get("/delivery-events")
def list_delivery_events(project_root: str) -> dict:
    """Every delivery event this project holds: what shipped, under which trait and kind, and the
    real per-bucket verification evidence the delivering door reconciled at the time.

    Read-only, no confirmation action: a delivery event is a fact recorded after an artifact
    already shipped under a meaning the breeder already confirmed elsewhere, not a statement of
    its own to confirm.
    """
    import tcip_store as ts
    from tcip_mcp.pipelines.resolution import DELIVERY_EVENTS_STORE, delivery_events_scope

    root = _guarded_project_root(project_root)
    scope = delivery_events_scope(root)
    records = [ts.read(key) for key in ts.keys(DELIVERY_EVENTS_STORE, str(scope))]
    return {"records": records}


# ── Registered traits (drives the Results tab's trait selection) ───────


@router.get("/traits")
def list_traits(project_root: str) -> dict:
    """Traits registered for this project, so the Results tab resolves which trait it is computing
    for from the project's own registry instead of assuming one.

    ``milestone_fractions_by_trait`` carries each trait's declared milestone fractions verbatim
    rather than a derived category, so a surface can tell whether a trait has milestones to compute
    without the server deciding what that means for it.

    ``invalid_specs`` names every spec file under this project's registry that failed to load and
    why, so a breeder facing zero (or fewer than expected) traits can tell "nothing registered" from
    "something is registered but broken" instead of the two reading identically.
    """
    root = _guarded_project_root(project_root)
    from tcip_mcp.traits import load_trait_specs_with_errors

    specs, errors = load_trait_specs_with_errors(project_root=root)
    return {
        "traits": sorted(spec.name for spec in specs),
        "milestone_fractions_by_trait": {
            spec.name: list(spec.milestone_fractions) for spec in specs
        },
        "invalid_specs": errors,
    }


# ── List registered models (used by Inference tab) ─────────────────────


@router.get("/models/registered")
def registered_models(project_path: str, tag: Optional[str] = None) -> dict:
    root = _guarded_project_root(project_path)
    from tcip_mcp.tools.model_tools import list_registered_models

    return list_registered_models(str(root), tag)
