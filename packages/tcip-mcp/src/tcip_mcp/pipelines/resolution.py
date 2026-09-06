"""Runtime parameter resolution, the "derive, don't pin" currency.

Every result-affecting parameter that varies by dataset/model/trait is resolved at runtime into a
``ResolvedParam`` carrying not just a value but *how it was derived* and *whether it is trustworthy*.
The point (CLAUDE.md "Parameters: derive, don't pin"): the agent derives operating points from the
data in hand, per dataset, and the provenance travels with every result so a phenotype can always be
traced to the operating point that produced it.

The measurement-integrity firewall lives here: a parameter that ``requires_validation`` (an operating
point like a confidence threshold, or a physical scale) is *structurally un-consumable as a bare
number* unless it was checked against the right kind of real-world reference for what it is.
``.value`` raises; a caller that genuinely means to ship an unvalidated value must go through
``unvalidated_value(...)`` and say so explicitly. This makes an unvalidated measurement value
physically un-shippable rather than merely discouraged.

No torch, safe to import anywhere; the storage seam (``tcip_store``) and ``tcip_annotation``'s
``json_io`` (the sidecar filename set) are the only dependencies beyond the standard library,
since the prediction buckets' provenance stamps are declared here.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tcip_store
from tcip_annotation.json_io import SIDECAR_FILENAMES as _SIDECAR_FILENAMES
from tcip_store import (
    RECORD_JSON,
    Key,
    StoreDescriptor,
    StoreError,
    check_json_value,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

logger = logging.getLogger(__name__)

# --- vocabularies ---------------------------------------------------------
SOURCES = ("explicit", "derived", "default")

# validated_against records which reference confirmed a value that requires validation, the
# shared-reference principle (CLAUDE.md): a reference sized to the trait, not dense GT for every
# trait. "false" is a real, distinct value (not Python None) so a sidecar/provenance record can say
# "checked, and it wasn't" rather than leaving the field merely absent.
VALIDATED_HELD_OUT = "held_out_annotations"          # a disjoint held-out split of this dataset's GT
VALIDATED_REVIEW_CONFIRMED = "reviewer_confirmed_annotations"  # a breeder-confirmed output sample
VALIDATED_PHYSICAL_MEASUREMENT = "physical_measurement"  # checked against a known physical dimension
# A tile scale's own real basis for trust: persisted tiled training geometry, a checkpoint's own
# recorded uniform untiled frame (native-ratio), or a caller's override the checkpoint doesn't contradict.
VALIDATED_PERSISTED_GEOMETRY = "persisted_training_geometry"
VALIDATED_NATIVE_FRAME_GEOMETRY = "persisted_native_frame_geometry"
VALIDATED_EXPLICIT_GEOMETRY = "explicit_caller_stated_geometry"
# A raster export target matched the mosaic a block-calibrated bundle was validated against; two
# references, one per comparison the export door actually had the basis to run.
VALIDATED_SAME_MOSAIC_IDENTITY = "same_mosaic_georeferenced_identity"
VALIDATED_SAME_MOSAIC_CONTENT_IDENTITY = "same_mosaic_content_identity"
CLAIM_SCOPE_REFERENCES = (VALIDATED_SAME_MOSAIC_IDENTITY, VALIDATED_SAME_MOSAIC_CONTENT_IDENTITY)
"""Which references legitimately clear the claim-scope dimension, stated once for the door that
writes one and the door that reads it back. Narrower than :data:`VALIDATED_SHIPPABLE` on purpose:
an annotation or physical reference says nothing about which raster a bucket's predictions were
produced on, so a sidecar recording one there clears nothing.

``VALIDATED_SAME_MOSAIC_IDENTITY`` asserts content identity and geotransform identity to the
training raster (:func:`~tcip_mcp.pipelines.raster_source.georeferenced_raster_identity_mismatch`),
earned when the recorded training identity carries a geotransform to compare against.
``VALIDATED_SAME_MOSAIC_CONTENT_IDENTITY`` asserts content identity alone
(:func:`~tcip_mcp.pipelines.raster_source.raster_identity_matches`), earned when the training
identity has none (a band-group source, or an unprojected raster): the weaker claim, for the
comparison the export door actually had the basis to run."""
VALIDATED_FALSE = "false"

# Which validated_against values legitimately clear validation for which kind of thing being
# validated, an annotation-count operating point (conf, a mask-binarize threshold), a physical
# scale, and a tile geometry are checked against fundamentally different references, and none may
# satisfy another's requirement.
VALIDATION_KINDS = ("annotations", "physical", "geometry")
_GEOMETRY_REFERENCE_BY_SOURCE: dict[str, str] = {
    "derived": VALIDATED_PERSISTED_GEOMETRY,
    "native_ratio": VALIDATED_NATIVE_FRAME_GEOMETRY,
    "explicit": VALIDATED_EXPLICIT_GEOMETRY,
}
"""Which ``tile_size_source`` earns which geometry reference, stated once for both directions:
:func:`resolve_tile_size_param` stamps a reference from a source and :func:`tile_size_source_of`
recovers the source behind a recorded one, so a door reading a persisted stamp back cannot disagree
with the door that wrote it about what counts as a real basis for a tile scale."""

GEOMETRY_REFERENCE_STRENGTH: tuple[str, ...] = (
    VALIDATED_PERSISTED_GEOMETRY, VALIDATED_NATIVE_FRAME_GEOMETRY, VALIDATED_EXPLICIT_GEOMETRY,
)
"""The three geometry references, strongest first: ``derived`` is the exact regime the model
trained under, the native-frame tier a regime mechanically inferred from the checkpoint's own
recorded frame, and ``explicit`` a caller's statement checked against the checkpoint's own recorded
geometry for contradiction only (:func:`~tcip_mcp.pipelines.inference.predictor.resolve_tile_regime`
refuses one that contradicts), never elevated to the strength of a value the checkpoint produced.
A mixed delivery travels under the weakest member present (:func:`reconcile_tile_size_validity`),
never the strongest; not a floor rank (:func:`_validity_rank` already means that binary check)."""
_ACCEPTED_REFERENCES: dict[str, tuple[str, ...]] = {
    "annotations": (VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED),
    "physical": (VALIDATED_PHYSICAL_MEASUREMENT,),
    "geometry": GEOMETRY_REFERENCE_STRENGTH,
}


def accepted_references(validation_kind: str) -> tuple[str, ...]:
    """The validated_against values that legitimately clear validation for this kind."""
    return _ACCEPTED_REFERENCES[validation_kind]


def cleared_reference(reference: str | None, *, validation_kind: str) -> str:
    """``reference`` when it legitimately clears ``validation_kind``, else :data:`VALIDATED_FALSE`.

    The one place "did this reference clear this kind" is decided, so the reader floor
    (:func:`_sidecar_reference`, reading a persisted stamp) and the earning gate
    (:func:`open_validation`, reading a resolver's own live result) cannot drift into disagreeing
    about which references count. An absent, unrecognized or wrong-kind reference floors rather
    than being read as validated.
    """
    return reference if reference in accepted_references(validation_kind) else VALIDATED_FALSE


def tile_size_source_of(reference: str | None, *, tile_size: int | None) -> str:
    """The ``tile_size_source`` a recorded geometry reference was earned by.

    The inverse of what :func:`resolve_tile_size_param` stamps, for a caller holding a persisted
    stamp rather than a live resolution (the review-promotion path, reading a bucket's sidecar back
    to re-resolve an operating point from it). Reading the reference, never the bare ``source``
    field a sidecar also carries, is what stops a native-ratio tile edge (source ``"derived"``, same
    as a real persisted geometry) from being re-read as the stronger tier: each real reference maps
    back to its own source through the lookup. A tile size present with no accepted reference behind
    it (a stamp nothing in the current vocabulary answers for, e.g. more than one accepted reference
    among the buckets a review promotion reads) comes back as ``"recorded"``, kept rather than
    dropped: the edge is real, only its reference is unrecognized. ``"default"`` is only for no tile
    size at all.
    """
    for source, accepted in _GEOMETRY_REFERENCE_BY_SOURCE.items():
        if reference == accepted:
            return source
    return "recorded" if tile_size is not None else "default"


# Every real (non-"false") reference across every kind, used only where the question is "is this
# a real reference at all": kind-correctness belongs to accepted_references/_DIMENSION_REFERENCES.
VALIDATED_SHIPPABLE = (
    VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED, VALIDATED_PHYSICAL_MEASUREMENT,
    VALIDATED_PERSISTED_GEOMETRY, VALIDATED_NATIVE_FRAME_GEOMETRY, VALIDATED_EXPLICIT_GEOMETRY,
    VALIDATED_SAME_MOSAIC_IDENTITY, VALIDATED_SAME_MOSAIC_CONTENT_IDENTITY,
)

# Shared inference operating-point defaults, referenced by both run_inference and the web route so the
# same model+images can't give a different count by entry point.
DEFAULT_CONF = 0.5
DEFAULT_NMS_IOU = 0.3
DEFAULT_OVERLAP = 0.2
# A high full-frame detection cap so dense scenes (hundreds of objects) aren't silently truncated
# at a framework default (torchvision 100 / ultralytics 300). Enforced after any tiled merge.
DEFAULT_MAX_DETS = 1000


def applied_operating_point(
    conf_threshold: float | None, global_nms_iou: float | None, max_dets: int | None,
) -> tuple[float, float, int]:
    """The stated-vs-platform-default resolution for conf/NMS/max_dets, shared by every caller
    that applies one directly: ``run_inference``'s ``dry_run`` preview, its verified body and its
    raster branch (``inference_tools.py``), and the full-frame delivery-grade evaluation
    (``run_full_frame_evaluation``), so none of them can resolve an unstated parameter differently."""
    applied_nms_iou = DEFAULT_NMS_IOU if global_nms_iou is None else float(global_nms_iou)
    applied_max_dets = DEFAULT_MAX_DETS if max_dets is None else int(max_dets)
    applied_conf = DEFAULT_CONF if conf_threshold is None else float(conf_threshold)
    return applied_conf, applied_nms_iou, applied_max_dets


class UnvalidatedOperatingPointError(RuntimeError):
    """Raised when a param that requires validation is consumed as if it were trustworthy.

    This is the firewall: it means a value that defines the measurement (a confidence threshold that
    decides the object count, or a physical scale that decides every dimensional number) is about to
    flow into a result without having been checked against the right kind of real-world reference. Do
    not silence it by reaching for ``_raw``; either validate the value, or consume it via
    ``unvalidated_value(...)`` and stamp the result ``validated=false`` so the un-trustworthiness
    travels downstream.
    """


@dataclass(frozen=True)
class ResolvedParam:
    """One parameter, resolved from the data in hand, with provenance and a validation status.

    Most parameters never need validation, a fact read from the data (``in_chans``), a statistic
    computed from this dataset's own spread (``cross_tile_nms``), or a plain configuration default
    (``tiled``) are all trustworthy by construction (``requires_validation=False``, the default).
    ``derived_from`` (free text) is where *how it was produced* is described, there is no separate
    category label to learn on top of that.

    A parameter that does need validation (a confidence threshold, a physical scale) sets
    ``requires_validation=True`` and a real ``validation_kind`` (what kind of reference can validate
    it, see ``VALIDATION_KINDS``); it is shippable only once ``validated_against`` names a reference
    ``accepted_references(validation_kind)`` recognizes for that kind.
    """

    name: str
    _raw: Any
    source: str  # one of SOURCES
    derived_from: str = ""  # human-readable: what artifact/analysis produced it
    requires_validation: bool = False
    validation_kind: str | None = None  # one of VALIDATION_KINDS, required iff requires_validation
    validated_against: str | None = None  # None | a member of VALIDATED_SHIPPABLE | VALIDATED_FALSE
    dataset_scoped: bool = False  # True => only valid for the dataset named by dataset_hash
    dataset_hash: str | None = None
    capture_scoped: bool = False  # True => only valid for the single capture named by capture_id
    capture_id: str | None = None
    gate_evidence: dict | None = None  # sensitivity data (e.g. count-vs-conf curve) for a validated param

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.requires_validation and self.validation_kind not in VALIDATION_KINDS:
            raise ValueError(
                f"{self.name!r}: requires_validation=True needs a real validation_kind in "
                f"{VALIDATION_KINDS}, got {self.validation_kind!r}, a param that needs validation "
                f"must say what kind of reference can validate it, never left to a silent default."
            )
        if not self.requires_validation and self.validation_kind is not None:
            raise ValueError(
                f"{self.name!r}: validation_kind={self.validation_kind!r} set but "
                f"requires_validation=False, a kind with nothing to validate is a contradiction."
            )
        if self.validated_against is not None and self.validated_against not in (
                *VALIDATED_SHIPPABLE, VALIDATED_FALSE):
            raise ValueError(f"validated_against invalid: {self.validated_against!r}")

    @property
    def is_shippable(self) -> bool:
        """Can this value flow into a delivered result as a trustworthy number?

        A param that doesn't require validation is always shippable. One that does is shippable only
        once ``validated_against`` is a reference ``accepted_references(validation_kind)`` recognizes
        for its own kind, a physical scale can never be satisfied by an annotation-based reference,
        or vice versa.
        """
        if not self.requires_validation:
            return True
        assert self.validation_kind is not None, (
            "requires_validation=True guarantees a real validation_kind, enforced in __post_init__"
        )
        return self.validated_against in accepted_references(self.validation_kind)

    @property
    def value(self) -> Any:
        """The trustworthy value, raises the firewall for an unvalidated param that requires it."""
        if not self.is_shippable:
            raise UnvalidatedOperatingPointError(
                f"{self.name!r} requires validation ({self.validation_kind}) and has "
                f"validated_against={self.validated_against!r}, cannot be consumed as a trustworthy "
                f"value. Validate it against a real reference for its kind, or consume it via "
                f"unvalidated_value(acknowledge_unvalidated=True) and stamp the result validated=false."
            )
        return self._raw

    def unvalidated_value(self, *, acknowledge_unvalidated: bool) -> Any:
        """Escape hatch: read the raw value while explicitly acknowledging it is unvalidated.

        The caller must pass ``acknowledge_unvalidated=True`` and is responsible for stamping the
        resulting output ``validated=false`` (carrying ``self.gate_evidence``) so the uncertainty travels on.
        """
        if not acknowledge_unvalidated:
            raise UnvalidatedOperatingPointError(
                f"{self.name!r}: pass acknowledge_unvalidated=True to read an unvalidated operating point."
            )
        return self._raw

    def to_provenance(self) -> dict:
        """Serializable record for stamping into the experiment config / predictions / CSV."""
        return {
            "name": self.name,
            "value": self._raw,
            "source": self.source,
            "derived_from": self.derived_from,
            "requires_validation": self.requires_validation,
            "validation_kind": self.validation_kind,
            "validated_against": self.validated_against,
            "dataset_scoped": self.dataset_scoped,
            "dataset_hash": self.dataset_hash,
            "capture_scoped": self.capture_scoped,
            "capture_id": self.capture_id,
            # the full gate evidence can be large; keep a marker, callers attach it separately if wanted
            "has_gate_evidence": self.gate_evidence is not None,
        }


def derived(name: str, value: Any, *, derived_from: str,
            requires_validation: bool = False, validation_kind: str | None = None,
            validated_against: str | None = None, dataset_scoped: bool = False,
            dataset_hash: str | None = None, capture_scoped: bool = False,
            capture_id: str | None = None, gate_evidence: dict | None = None) -> ResolvedParam:
    """Convenience constructor for a data/model-derived param (``source="derived"``)."""
    return ResolvedParam(
        name=name, _raw=value, source="derived", derived_from=derived_from,
        requires_validation=requires_validation, validation_kind=validation_kind,
        validated_against=validated_against, dataset_scoped=dataset_scoped, dataset_hash=dataset_hash,
        capture_scoped=capture_scoped, capture_id=capture_id, gate_evidence=gate_evidence,
    )


def default(name: str, value: Any, *, derived_from: str = "documented default") -> ResolvedParam:
    """Convenience constructor for a documented default (``source="default"``), never requires
    validation; a param that does is always constructed explicitly with ``requires_validation=True``."""
    return ResolvedParam(name=name, _raw=value, source="default", derived_from=derived_from)


@dataclass
class ResolvedBundle:
    """The phenotype-gating operating point for one (trait, dataset, checkpoint), fully resolved.

    Holds the operating-point params (conf, cross_tile_nms, tiled, tile_size, max_dets) plus the
    structural facts (in_chans, num_classes) and the produced count / classifier-validation status.
    Stamped into the immutable experiment config and every prediction/CSV.
    """

    trait: str
    dataset_hash: str | None
    params: dict[str, ResolvedParam] = field(default_factory=dict)
    classifier_validated_vs_gt: str | None = None  # for a trait's positive-class classifier

    def get(self, name: str) -> ResolvedParam:
        return self.params[name]

    def value(self, name: str) -> Any:
        """Firewalled value accessor (raises for an unvalidated calibration param)."""
        return self.params[name].value

    @property
    def is_shippable(self) -> bool:
        """True only if every param requiring validation cleared a reference of its own kind."""
        return all(p.is_shippable for p in self.params.values())

    def shippable_issues(self, *, target_dataset_hash: str | None = None,
                         target_capture_id: str | None = None) -> list[str]:
        """Reasons this bundle cannot ship a trustworthy phenotype (empty = shippable)."""
        issues: list[str] = []
        for p in self.params.values():
            if not p.is_shippable:
                issues.append(
                    f"{p.name}: requires validation ({p.validation_kind}), not validated "
                    f"(validated_against={p.validated_against})"
                )
            if p.dataset_scoped and target_dataset_hash is not None and p.dataset_hash != target_dataset_hash:
                issues.append(
                    f"{p.name}: dataset-scoped value derived on {p.dataset_hash} inherited across a "
                    f"different dataset {target_dataset_hash}, re-resolve per dataset, never inherit"
                )
            if p.capture_scoped and p.capture_id is None:
                # No deriver for capture_id exists yet, an honest "not comparable" marker rather
                # than silently passing a check that cannot actually run.
                issues.append(
                    f"{p.name}: capture-scoped but no capture_id was ever derived for it, "
                    "cross-capture reuse cannot be checked (not comparable, not confirmed safe)"
                )
            elif p.capture_scoped and target_capture_id is not None and p.capture_id != target_capture_id:
                issues.append(
                    f"{p.name}: capture-scoped value derived for capture {p.capture_id} inherited "
                    f"across a different capture {target_capture_id}, re-resolve per capture, "
                    "never inherit"
                )
        return issues

    def to_provenance(self) -> dict:
        return {
            "trait": self.trait,
            "dataset_hash": self.dataset_hash,
            "classifier_validated_vs_gt": self.classifier_validated_vs_gt,
            "operating_point": {name: p.to_provenance() for name, p in self.params.items()},
        }


def resolve_tile_size_param(
    tile_size: int | None, *, tiled: bool, tile_size_source: str,
    tile_size_derived_from: str | None,
) -> ResolvedParam:
    """The ``tile_size`` dimension, gated the same shape ``conf`` already is, the shared
    construction site both :func:`raw_operating_point` (here) and
    :func:`tcip_mcp.pipelines.operating_point.resolve_operating_point` (the calibrated path) call,
    so the two doors can't drift into disagreeing about when a tile scale is trustworthy.

    ``tile_size_derived_from`` is required (no default) so every caller states it explicitly; it is
    read only for ``tile_size_source == "explicit"``, where it becomes the stamped ``derived_from``
    (see :func:`~tcip_mcp.pipelines.inference.predictor.explicit_edge_provenance`, which composes it
    from what the checkpoint's own recorded geometry says about the stated edge), never a placeholder
    like "caller override" doing double duty inside the stamped claim.

    Only meaningful when ``tiled``: an untiled run's count never depends on tile_size, so it stays a
    plain non-gating fact there (mirrors ``in_chans``), gating it anyway would refuse legitimate
    untiled work over a dimension that was never operative. When tiled, the value is shippable when
    its source names a real basis for trusting the scale, ranked strongest to weakest (see
    :data:`GEOMETRY_REFERENCE_STRENGTH`): the checkpoint's own persisted training geometry
    (``"derived"``); a tile edge mechanically derived from a checkpoint's own uniform untiled
    training frame (``"native_ratio"``), never stated by a caller; or a caller's deliberate explicit
    override (``"explicit"``, already checked for contradiction against the checkpoint's own
    recorded geometry by the caller before this function ever sees it, but a stated decision, not a
    guess). ``"recorded"`` (:func:`tile_size_source_of`'s own fallback, a
    real edge read back off a persisted stamp whose reference the current vocabulary does not
    accept) keeps the edge, floored to unvalidated. Any other source (``"unavailable"``, no
    persisted geometry and nothing explicit) means nothing justifies a scale at all, so ``tile_size``
    itself is ``None``: never a fabricated number, closing the asymmetry with the delivery-gating
    path (``run_full_frame_evaluation``), which already refuses outright for this exact case.
    """
    if not tiled:
        return default("tile_size", None)
    if tile_size_source == "derived" and tile_size is not None:
        return derived(
            "tile_size", int(tile_size), derived_from="persisted training tile geometry",
            requires_validation=True, validation_kind="geometry",
            validated_against=_GEOMETRY_REFERENCE_BY_SOURCE["derived"],
        )
    if tile_size_source == "native_ratio" and tile_size is not None:
        return derived(
            "tile_size", int(tile_size),
            derived_from="the checkpoint's own uniform untiled training frame",
            requires_validation=True, validation_kind="geometry",
            validated_against=_GEOMETRY_REFERENCE_BY_SOURCE["native_ratio"],
        )
    if tile_size_source == "explicit" and tile_size is not None:
        if tile_size_derived_from is None:
            raise ValueError(
                "tile_size_derived_from is required when tile_size_source is 'explicit': the "
                "caller must compose it (see explicit_edge_provenance) before an explicit tile "
                "edge can be stamped into the operating-point claim."
            )
        return ResolvedParam(
            "tile_size", int(tile_size), source="explicit",
            derived_from=tile_size_derived_from,
            requires_validation=True, validation_kind="geometry",
            validated_against=_GEOMETRY_REFERENCE_BY_SOURCE["explicit"],
        )
    if tile_size_source == "recorded" and tile_size is not None:
        # SOURCES has no token for a read-back edge; "derived" is the closest, as for native_ratio.
        return derived(
            "tile_size", int(tile_size),
            derived_from="read back from the bucket's own stamp with no accepted geometry "
                         "reference behind it",
            requires_validation=True, validation_kind="geometry", validated_against=VALIDATED_FALSE,
        )
    return ResolvedParam(
        "tile_size", None, source="default",
        derived_from="no persisted training geometry, no explicit override, and no basis to "
                     "derive a tile edge from",
        requires_validation=True, validation_kind="geometry",
        validated_against=VALIDATED_FALSE,
    )


def raw_operating_point(
    *, conf: float, cross_tile_nms: float | None, tiled: bool, tile_size: int | None,
    max_dets: int | None, tile_size_source: str = "default", tile_size_derived_from: str | None = None,
    tiled_source: str = "default", conf_stated: bool = False, max_dets_stated: bool = False,
) -> ResolvedBundle:
    """The operating point for raw (uncalibrated) inference, the one every caller resolves through.

    ``conf`` is a documented default with no per-dataset GT behind it, so it requires validation and
    is stamped ``validated_against=false``: reading it requires ``unvalidated_value(...)`` and the
    caller must stamp its output ``validated=false``. This is what stops the MCP tool and the web job
    giving a different count (the phenotype) for the same model + images by entry point.

    Every caller of this function resolves ``max_dets`` to a concrete cap before reaching here
    (the shared platform default when the caller stated nothing); an uncapped ``max_dets=None`` is
    a different regime's own deliberate value, built by
    :func:`block_calibrated_export_operating_point` for the block-calibrated whole-mosaic pass,
    never by this function.

    ``conf_stated``/``max_dets_stated`` say whether the caller explicitly chose that value, mapped
    to the same explicit-vs-default provenance vocabulary ``tile_size_source``/``tiled_source``
    already carry for their own params: a caller-chosen value that happens to equal the platform
    default is stamped ``"explicit"``, never silently read back as an untouched default.

    ``tile_size_source`` records whether the tile edge was ``derived`` from the checkpoint's training
    geometry, ``native_ratio`` (the checkpoint's own uniform untiled frame), ``explicit`` (caller
    override), ``recorded`` (read back off a stamp with no accepted reference behind it), or has no
    real basis at all (``"unavailable"``), so a train/infer scale mismatch is visible in the
    provenance rather than silent: see :func:`resolve_tile_size_param`, a tiled run with no basis
    at all is a real, gating-firewalled unvalidated dimension (``tile_size`` itself ``None``, never
    a fabricated number), not silently shippable engineering trivia.
    ``tiled_source`` is the same provenance vocabulary for the boolean itself: a caller that
    explicitly chose to tile (or not) stamps ``"explicit"``; a caller who passed nothing gets
    ``"default"``. Every caller of this function derives its own concrete ``tiled`` bool (no shared
    fallback constant) before reaching here, so ``tiled`` itself is always a real bool, never ``None``.

    ``tile_size_derived_from`` is forwarded to :func:`resolve_tile_size_param` unchanged; it matters
    only for ``tile_size_source == "explicit"``, where the caller composed it through
    :func:`~tcip_mcp.pipelines.inference.predictor.explicit_edge_provenance` against the checkpoint
    it already checked the stated edge with :func:`~tcip_mcp.pipelines.inference.predictor.
    resolve_tile_regime`.
    """
    tile_param = resolve_tile_size_param(
        tile_size, tiled=tiled, tile_size_source=tile_size_source,
        tile_size_derived_from=tile_size_derived_from)
    if tiled_source == "explicit":
        tiled_param = ResolvedParam("tiled", tiled, source="explicit", derived_from="caller override")
    else:
        tiled_param = default("tiled", tiled)
    if conf_stated:
        conf_param = ResolvedParam(
            "conf", conf, source="explicit", derived_from="caller override",
            requires_validation=True, validation_kind="annotations", validated_against=VALIDATED_FALSE,
        )
    else:
        conf_param = ResolvedParam(
            "conf", conf, source="default", requires_validation=True,
            validation_kind="annotations", validated_against=VALIDATED_FALSE,
        )
    if max_dets_stated:
        max_dets_param = ResolvedParam(
            "max_dets", max_dets, source="explicit", derived_from="caller override")
    else:
        max_dets_param = default("max_dets", max_dets)
    return ResolvedBundle(trait="", dataset_hash=None, params={
        "conf": conf_param,
        "cross_tile_nms": default("cross_tile_nms", cross_tile_nms if tiled else None),
        "tiled": tiled_param,
        "tile_size": tile_param,
        "max_dets": max_dets_param,
    })


def block_calibrated_export_operating_point(
    block_bundle: ResolvedBundle, *, trait: str, tile_size: int | None, tile_size_source: str,
    tile_size_derived_from: str | None = None,
) -> ResolvedBundle:
    """The whole-mosaic export operating point a block-calibrated bundle ships at.

    The third regime beside :func:`raw_operating_point` and
    :func:`tcip_mcp.pipelines.operating_point.resolve_operating_point`, resolved here rather than
    assembled at the export door, so what carries over from a block calibration and what does not
    is stated once. ``conf`` and ``cross_tile_nms`` carry over unchanged: those are what the mosaic's
    reserved calibration and test bands measured. ``max_dets`` deliberately does not, and is
    committed to ``None`` (uncapped): the block bundle's own cap is derived from the density of one
    reserved band, and adopting it wholesale would truncate the count over the whole mosaic, which
    is the phenotype. Tiling is always on, since a raster too large to load whole has no untiled
    alternative, and the tile scale is gated through the same :func:`resolve_tile_size_param` every
    other door resolves through.
    """
    return ResolvedBundle(trait=trait, dataset_hash=block_bundle.dataset_hash, params={
        "conf": block_bundle.get("conf"),
        "cross_tile_nms": block_bundle.get("cross_tile_nms"),
        "tiled": default("tiled", True),
        "tile_size": resolve_tile_size_param(
            tile_size, tiled=True, tile_size_source=tile_size_source,
            tile_size_derived_from=tile_size_derived_from),
        "max_dets": default(
            "max_dets", None,
            derived_from="block calibration: not transferred, uncapped for the whole-mosaic pass"),
    })


# --- dataset identity -----------------------------------------------------

def _label_bytes(lp: Path) -> bytes:
    """One label file's bytes, empty for a missing file: valid negatives are hashed as empty, not
    skipped, so their presence/absence still contributes to identity."""
    return lp.read_bytes() if lp.is_file() else b""


def dataset_hash(labels_dir: str | Path, stems: list[str] | None = None) -> str:
    """A content-addressed hash identifying a dataset's ground truth.

    Two datasets with the same labels hash equal (so a calibration is valid iff its hash matches the
    inference dataset's). Content-based (label bytes), so it is machine-independent, a path can move
    between machines but the GT identity does not. Missing labels are hashed as empty (they are valid
    negatives), so their presence/absence still contributes to identity. Streams one label's bytes
    at a time into the running hash rather than buffering the whole labels directory in memory.
    """
    from tcip_annotation.json_io import prediction_documents

    labels_dir = Path(labels_dir)
    if stems is None:
        # Canonical labels are per-image JSON; a bucket's own sidecar stamps are not labels.
        stems = sorted(p.stem for p in prediction_documents(labels_dir))
    else:
        stems = sorted(stems)
    h = hashlib.sha256()
    for stem in stems:
        h.update(stem.encode("utf-8"))
        h.update(b"\0")
        h.update(_label_bytes(labels_dir / f"{stem}.json"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def label_digests(labels_dir: str | Path, stems: list[str]) -> dict[str, str]:
    """Each stem's own content-addressed digest, ``sha256(label bytes)[:16]``, the same
    empty-bytes-for-a-missing-file convention :func:`dataset_hash` folds into its one combined
    hash, surfaced here per stem instead: a withdrawn label reads as the digest of empty bytes,
    not as an absent key. Streams one label's bytes at a time; a caller wanting both this and
    :func:`dataset_hash` over the same stems calls :func:`dataset_hash_and_label_digests` instead,
    which reads each label once for both.
    """
    labels_dir = Path(labels_dir)
    return {
        stem: hashlib.sha256(_label_bytes(labels_dir / f"{stem}.json")).hexdigest()[:16]
        for stem in sorted(stems)
    }


def dataset_hash_and_label_digests(
    labels_dir: str | Path, stems: list[str],
) -> tuple[str, dict[str, str]]:
    """:func:`dataset_hash` and :func:`label_digests` over the same stem set, in one pass: each
    label's bytes read once and folded into both the combined hash and its own per-stem digest,
    rather than through two separate calls that would each open every file. ``draw_splits`` calls
    this for a members block's ``dataset_hash``/``label_digests`` pair.
    """
    labels_dir = Path(labels_dir)
    stems = sorted(stems)
    h = hashlib.sha256()
    per_stem: dict[str, str] = {}
    for stem in stems:
        b = _label_bytes(labels_dir / f"{stem}.json")
        h.update(stem.encode("utf-8"))
        h.update(b"\0")
        h.update(b)
        h.update(b"\0")
        per_stem[stem] = hashlib.sha256(b).hexdigest()[:16]
    return h.hexdigest()[:16], per_stem


def manifest_digest(manifest: dict) -> str:
    """The one digest a split-manifest record earns: sha256 over ``RECORD_JSON.encode(manifest)``.

    Called at bind time (``split_construction.persist_split_manifest``, stamping ``split.json``'s
    ``label_digests.manifest_sha256``) and at calibration time (``inference_tools``'s
    ``split_manifest_sha256``, and the operator script that passes the same fact) so the two
    sides that must agree on a manifest's identity can never spell the digest differently.
    """
    return hashlib.sha256(RECORD_JSON.encode(manifest)).hexdigest()


def csv_dataset_hash(csv_path: str | Path) -> str:
    """A content-addressed hash identifying a CSV-sourced GT identity (``OrdinalDataset``/
    ``RegressionDataset``'s ``(stem, value)`` rows), the same content-addressed principle as
    :func:`dataset_hash` (which is hardcoded to a directory of per-image JSON label files and is not
    reusable as-is for a single flat CSV).

    Rows are sorted by stem before hashing so row order in the file doesn't spuriously change the
    identity; two CSVs with the same (stem, value) pairs hash equal regardless of how they were
    written.
    """
    rows: list[tuple[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0].strip(), row[1].strip()))
    h = hashlib.sha256()
    for stem, value in sorted(rows):
        h.update(stem.encode("utf-8"))
        h.update(b"\0")
        h.update(value.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


# --- the prediction bucket's provenance stamps (the delivery gate reads these, not a caller string) ---

_SIDECAR_LOCATOR = RootedFileLocator(suffix=".json")
"""A stamp sits directly in the bucket it describes, addressed by its own document name."""

_SIDECAR_STORES: dict[str, str] = {
    document: register_store(
        StoreDescriptor(
            name=f"{document}_sidecar",
            kind="record",
            key_fields=("document",),
            frozen=True,
            codec=RECORD_JSON,
            concurrency="cas",
            locator=_SIDECAR_LOCATOR,
        )
    ).name
    for document in (filename[: -len(".json")] for filename in sorted(_SIDECAR_FILENAMES))
}
"""One store per measurement dimension, never one store holding every dimension's fields: the
dimensions are structurally independent (a physical scale is a fact about the imagery, a classifier
stamp is about a state call, the count operating point is about a threshold), and a single document
is exactly what would let a generic writer conflate them. The document names come from
:data:`tcip_annotation.json_io.SIDECAR_FILENAMES`, the one declared set, rather than a second
enumeration of them here."""


def sidecar_key(pred_dir: str | Path, document: str = "operating_point") -> Key:
    """One prediction bucket's provenance stamp for one measurement dimension.

    ``cas`` on every one of them: the review-promotion path merges its own validation fields into a
    stamp the producing run already wrote, from a different process, so an unconditional replace
    there would drop the producer's own record of what made the predictions.
    """
    try:
        store = _SIDECAR_STORES[document]
    except KeyError:
        raise ValueError(
            f"{document!r} is not a prediction-bucket stamp; declared documents are "
            f"{sorted(_SIDECAR_STORES)}"
        ) from None
    return Key(store, str(pred_dir), (document,))


def _read_sidecar(pred_dir: str | Path, document: str, *, strict: bool = False) -> dict | None:
    """One bucket's stamp for one dimension, or ``None`` when absent.

    At its default (``strict=False``), a stamp that will not decode also reads as ``None``: an
    unreadable stamp floors the dimension it describes to unvalidated at every reconciler below,
    which is the safe direction, where a raised decode error would take down a delivery gate that
    has a well-defined answer for a stamp it cannot trust.

    Under ``strict=True``, which :func:`bucket_scope` passes, the seam's own decode error
    (``StoreError``, covering ``DecodeError`` and ``SchemaVersionRefused``) propagates instead: a
    caller that must tell an absent stamp from a present one that will not decode cannot read the
    second as the first, since a classified bucket whose stamp will not decode would otherwise be
    reviewed as a bare directory and its value-keyed records accepted as object classes.
    """
    try:
        return tcip_store.read(sidecar_key(pred_dir, document), default=None)
    except StoreError:
        if strict:
            raise
        return None


def well_formed_validated_by(stamp: dict | None) -> dict | None:
    """The stamp's pointer at the validation record behind it, or ``None`` when it has no usable one.

    A pointer is usable only with both halves present as non-empty strings: an experiment to look in
    and the identity of one row inside it. Read here for the writer refusal and the reader
    verification alike, so a shape one side accepts can never be a shape the other rejects.
    """
    pointer = (stamp or {}).get("validated_by")
    if not isinstance(pointer, dict):
        return None
    experiment_id = pointer.get("experiment_id")
    record_digest = pointer.get("record_digest")
    if not isinstance(experiment_id, str) or not experiment_id:
        return None
    if not isinstance(record_digest, str) or not record_digest:
        return None
    return pointer


def scope_consistent_with_map(
    subject: str | None, attribute: str | None, id_map: dict | None,
) -> str | None:
    """Whether a bucket's ``(subject, attribute)`` claim is consistent with its own recorded
    ``id_map``, or the reason it is not (``None`` when it is).

    A detector pair (``attribute`` ``None``) needs a map that is absent or keyed by exactly the
    subject, the shape a detector run records (``class_registry.assign_class_ids`` with no
    attribute); a map keyed otherwise says the bucket classified. A classified pair (``attribute``
    not ``None``) needs a map that is not keyed by the subject alone, since a run that decoded
    along an attribute never records that shape.

    The stamp write rail (:func:`_check_stamp_claim`) and the conform script's own rule 3 call
    this one predicate rather than holding the rule twice.

    The one blind spot: an attribute declaring exactly one value whose name equals the subject
    records ``{subject: 0}``, indistinguishable here from a detector map. No registry in the two
    projects on the share declares such a value; a caller stamping a detector pair over a one-key
    map this way should report the case to its operator rather than trust it silently.

    An empty recorded ``id_map`` (``{}``, distinct from ``None``, absent) names no vocabulary
    either pair could agree or disagree with: refused by name under both pairs, rather than read
    as the detector branch's "not keyed by the subject" (a reason meant for a real, non-empty,
    differently-keyed map) or silently admitted by the classified branch, which would otherwise
    treat naming nothing as naming a value.
    """
    if id_map is not None and not id_map:
        return (
            f"the pair ({subject!r}, {attribute!r}) claims a scope over a bucket whose recorded "
            "id_map is empty: an empty map names no vocabulary a pair could agree or disagree with."
        )
    keyed_by_subject_alone = id_map is None or set(id_map) == {subject}
    if attribute is None:
        if keyed_by_subject_alone:
            return None
        # Reached only when keyed_by_subject_alone is False, which is impossible for id_map=None.
        assert id_map is not None
        return (
            f"the pair ({subject!r}, None) claims a detector bucket, but its recorded id_map is "
            f"keyed by {sorted(id_map)}, not just {subject!r}: this looks like a classified bucket"
        )
    if keyed_by_subject_alone:
        return (
            f"the pair ({subject!r}, {attribute!r}) claims a classified bucket, but its recorded "
            f"id_map is keyed by exactly {subject!r}, the shape a detector run records"
        )
    return None


class StampScopeUnstated(ValueError):
    """A bucket's ``operating_point.json`` decodes but carries no usable ``(subject, attribute)``
    pair: either key absent, or one present with a type other than ``str`` or ``None``.

    Raised by :func:`bucket_scope` for a stamp written before this platform recorded the pair;
    the remedy is ``tcip repair-classified-predictions`` over the bucket, named in the
    message every time this is raised.
    """


def _check_stamp_claim(
    stamp: dict, document: str, pred_dir: str | Path, *, introduced_keys: set[str] | None = None,
) -> None:
    """Refuse a stamp whose shape or claim a reader could not trust.

    For the ``operating_point`` document, every top-level key this write actually introduces must
    be declared: one of ``operating_point_stamp``'s own (:data:`STAMP_KEYS`) or a named producer
    addition (:data:`STAMP_EXTENSION_KEYS`). ``introduced_keys`` is ``None`` for a fresh
    :func:`write_sidecar` (the whole stamp is new, so every key is checked) and the merged-minus-
    stored key set for :func:`update_sidecar` (a promotion over a stamp that already carries a
    foreign top-level key, from a direct store write or a hand-authored stamp, is not refused for a
    key it did not itself write). A producer that needs a new key declares it in
    ``STAMP_EXTENSION_KEYS``; this rail exists so the declared union stays the one place a reader
    can learn the whole stamp shape, rather than each producer inventing its own beside it. Other
    documents (classifier/ordinal/regression/scale) carry no such declared shape and are not
    checked here.

    The same document also carries the writer-side scope rail: the body being written (the whole
    fresh stamp, or the merged body for an update) must carry both ``subject`` and ``attribute``,
    each a string or ``None``; an ``attribute`` that is not ``None`` needs a ``subject`` that is
    not ``None``; and when the body also carries an ``id_map`` the pair must agree with it through
    :func:`scope_consistent_with_map`. A fresh stamp failing this is refused naming the producer's
    own obligation to state the pair; a merged body missing the pair is refused naming the stored
    stamp as one written before the keys existed, and the conform script as the remedy. No live
    producer can mint a stamp without the pair after this rail exists; the pair is provenance, not
    claim (:data:`_CLAIM_KEYS` does not carry it), so a stamp's scope can still be edited by the
    conform script without flooring a sealed count claim.

    A validated stamp also names the record it was earned from and the trait it was earned for.
    Neither is defaultable: a pointer this writer filled in would point at nothing, and a trait it
    guessed would be a claim nobody made. Both rails are writer-side and close nothing on their own,
    since a file written straight to disk never passes here; they exist so a platform producer
    cannot omit what every reader compares.
    """
    if document == "operating_point":
        checked = set(stamp) if introduced_keys is None else introduced_keys
        unknown = checked - STAMP_KEYS - set(STAMP_EXTENSION_KEYS)
        if unknown:
            raise ValueError(
                f"{document}.json at {str(pred_dir)!r} carries undeclared top-level key(s) "
                f"{sorted(unknown)}. Declare a new producer addition in STAMP_EXTENSION_KEYS "
                "(resolution.py), naming which producer writes it, before writing it here."
            )
        fresh = introduced_keys is None
        if "subject" not in stamp or "attribute" not in stamp:
            if fresh:
                raise ValueError(
                    f"{document}.json at {str(pred_dir)!r} carries no subject/attribute pair. "
                    "Every producer must call operating_point_stamp with both: the object class "
                    "every prediction record in this bucket is of, and the attribute each "
                    "record's value sits under (None for a detector bucket)."
                )
            raise ValueError(
                f"{document}.json at {str(pred_dir)!r} has a stored stamp with no subject/"
                "attribute pair, written before this platform recorded the pair. Run "
                "tcip repair-classified-predictions over this bucket before merging "
                "into its stamp."
            )
        subject, attribute = stamp.get("subject"), stamp.get("attribute")
        if subject is not None and not isinstance(subject, str):
            raise ValueError(
                f"{document}.json at {str(pred_dir)!r}: subject must be a string or None, "
                f"got {type(subject).__name__}")
        if attribute is not None and not isinstance(attribute, str):
            raise ValueError(
                f"{document}.json at {str(pred_dir)!r}: attribute must be a string or None, "
                f"got {type(attribute).__name__}")
        if attribute is not None and subject is None:
            raise ValueError(
                f"{document}.json at {str(pred_dir)!r} declares attribute {attribute!r} with no "
                "subject: a value with no object class names nothing a reader could hold it to."
            )
        recorded_map = stamp.get("id_map")
        if recorded_map is not None:
            reason = scope_consistent_with_map(subject, attribute, recorded_map)
            if reason is not None:
                raise ValueError(f"{document}.json at {str(pred_dir)!r}: {reason}")
    if not stamp.get("validated"):
        return
    if well_formed_validated_by(stamp) is None:
        raise ValueError(
            f"{document}.json at {str(pred_dir)!r} claims validated with no well-formed "
            "validated_by (an experiment_id and a record_digest, both non-empty strings). A "
            "validated claim is earned through open_validation/seal_validation, which append the "
            "record and return the stamp with its pointer merged in; a stamp cannot assert one."
        )
    if not stamp.get("trait"):
        raise ValueError(
            f"{document}.json at {str(pred_dir)!r} claims validated with no trait. Pass the trait "
            "the claim was earned for to open_validation and stamp the body seal_validation "
            "returns; a claim with no trait names nothing a delivery can be checked against."
        )


def write_sidecar(pred_dir: str | Path, stamp: dict, document: str = "operating_point") -> None:
    """Write one bucket's stamp whole, under the stamp's own lock.

    A stamp is assembled by its producer, and ``operating_point_stamp`` carries a producer's
    own extra fields through ``**fields``, so what it holds is checked here, where every
    sidecar write passes, rather than at each producer.
    """
    check_json_value(stamp, path="stamp")
    _check_stamp_claim(stamp, document, pred_dir)
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    key = sidecar_key(pred_dir, document)
    with tcip_store.transaction(key) as txn:
        txn.write(key, stamp)


def update_sidecar(
    pred_dir: str | Path, updater: Callable[[dict], dict | None],
    document: str = "operating_point",
) -> bool:
    """Merge into one bucket's existing stamp, reading and writing inside one lock hold.

    ``updater`` receives the stored stamp (``{}`` when there is none) and returns the value to
    store, or ``None`` to leave it exactly as it was. Returns whether anything was written. The
    read and the write are one transaction, so a promotion can never overwrite fields another
    process stamped between them, and a no-downgrade decision the updater makes is made against
    what is actually stored rather than against a value read before the lock.
    """
    key = sidecar_key(pred_dir, document)
    with tcip_store.transaction(key) as txn:
        current = txn.read(key, default={})
        current = current if isinstance(current, dict) else {}
        updated = updater(current)
        if updated is None:
            return False
        check_json_value(updated, path="stamp")
        _check_stamp_claim(
            updated, document, pred_dir, introduced_keys=set(updated) - set(current))
        txn.write(key, updated)
    return True


def fold_tile_validation(validated: bool, tile_size_validated: str | None) -> bool:
    """The one floor a stamp's ``validated`` bit answers to: no tile-geometry claim, no validated
    bucket, whatever the door's own dimension resolved. Called by :func:`operating_point_stamp` for
    every fresh stamp and by a caller that merges a later claim onto one (``calibrate_count_operating_point``),
    so the floor is applied identically wherever a bucket's overall ``validated`` bit is computed.
    """
    return bool(validated) and tile_size_validated != VALIDATED_FALSE


def operating_point_stamp(
    operating_point: dict | None,
    *,
    validated: bool,
    validated_by: dict | None,
    tile_size_validated: str | None,
    shippable_issues: list[str],
    id_map: dict | None,
    subject: str | None,
    attribute: str | None,
    trait: str | None,
    dataset_hash: str | None,
    checkpoint: str | None,
    checkpoint_sha256: str | None,
    experiment_id: str | None,
    images_dir: str | None,
    raster_path: str | None,
    produced_at: str | None,
    **fields: Any,
) -> dict:
    """The ``operating_point.json`` stamp every path that writes predictions records beside them.

    One constructor for every producer (the agent's image and raster export doors, the GUI's own
    inference worker), so a provenance key one path needs exists on all of them and a reader can
    ask the same question of any bucket. Every field is required, with no default that would let a
    door quietly omit what stands behind its counts; the per-path additions a single producer has
    (persisted gate evidence, a mask-binarize threshold, a block calibration's own record) travel through
    ``fields``.

    ``subject`` and ``attribute`` are the run's own scope: the object class every prediction
    record in the bucket is of, and the attribute each record's value sits under (``attribute``
    ``None`` for a detector bucket). Required, with no default, the same run-scope pair the
    training run recorded on its experiment config; every reader below resolves a bucket's scope
    from these two fields through :func:`bucket_scope`, never from the records' own vocabulary.

    ``validated`` is the producing door's own verdict over the dimensions it resolved. The tile
    scale is floored in here rather than at each door: a bucket whose tile geometry has no real
    basis produced its counts at a scale nothing justifies, so it is not a validated bucket no
    matter what the conf dimension earned.

    ``validated_by`` is the pointer at the validation record the claim was earned from, the mapping
    :func:`seal_validation` returns, and ``None`` for a stamp that claims nothing. It has no default
    on purpose: a producer that stamps a validated bucket must have earned a record to name, and a
    producer that stamps an unvalidated one says so at its own call site.
    """
    return {
        "trait": trait,
        "dataset_hash": dataset_hash,
        "operating_point": operating_point,
        "id_map": id_map,
        "subject": subject,
        "attribute": attribute,
        "validated": fold_tile_validation(validated, tile_size_validated),
        "validated_by": validated_by,
        "tile_size_validated": tile_size_validated,
        "shippable_issues": list(shippable_issues),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "experiment_id": experiment_id,
        "images_dir": images_dir,
        "raster_path": raster_path,
        "produced_at": produced_at,
        **fields,
    }


def prediction_producer(checkpoint_path: str, sha256: str) -> str:
    """The one ``created_by`` spelling for a prediction written behind a resolved checkpoint.

    Every checkpoint-backed door (the image and raster export regimes, the web inference worker)
    resolves a checkpoint's identity before it writes anything, so the hash is always in hand by
    the time this is called; the parameter is required rather than defaulted so the bare,
    hash-less form cannot be spelled through it, and a caller that reaches here without the hash
    is refused by name rather than stamping a producer with no identity. ``stage_proposals``
    writes its own producer strings, in either input regime (the caller's model name, or the
    proposal engine), and is outside this helper by design: neither regime is backed by a
    resolved checkpoint.
    """
    if not sha256:
        raise ValueError(
            f"no checkpoint hash for {checkpoint_path}: a prediction's producer is stamped only "
            "from a resolved checkpoint identity (resolve_model_identity)")
    return f"model:{Path(checkpoint_path).stem}@{sha256[:12]}"


STAMP_KEYS: frozenset[str] = frozenset((
    "trait", "dataset_hash", "operating_point", "id_map", "subject", "attribute",
    "validated", "validated_by", "tile_size_validated", "shippable_issues", "checkpoint",
    "checkpoint_sha256", "experiment_id", "images_dir", "raster_path", "produced_at",
))
"""``operating_point_stamp``'s own sixteen keys: the ones it returns unconditionally, before a
producer's own ``**fields``. Declared literally rather than derived from the signature, since a
parameter name matching its returned key is this constructor's own convention, not a guarantee;
``tests/test_operating_point_sidecar_seam.py`` pins the two against each other."""

STAMP_EXTENSION_KEYS: dict[str, str] = {
    "validated_reference": "the review-promotion path (routes/review.py's _stamp_body)",
    "validation_source": "the review-promotion path (routes/review.py's _stamp_body)",
    "review_reference_hash": "the review-promotion path (routes/review.py's _stamp_body)",
    "review_image_count": "the review-promotion path (routes/review.py's _stamp_body)",
    "validated_at": "the review-promotion path (routes/review.py's _stamp_body)",
    "mask_binarize": "the image-export and raster-export doors and the web inference worker, "
                     "for a bucket carrying masks",
    "claim_scope_validated": "the raster-export door's block-calibration branch",
    "block_calibration": "the raster-export door's block-calibration branch",
    "raster_content_identity": "the raster-export door, recorded for every run of that regime",
    "overlap": "the web inference worker",
    "overlap_source": "the web inference worker",
    "calibration_curve_path": "the shared per-image bucket publisher behind run_inference "
                              "and deliver_per_image_counts's live path, for a calibrated run that "
                              "persisted a curve",
    "gate_evidence_summary": "the shared per-image bucket publisher behind run_inference "
                             "and deliver_per_image_counts's live path, for a calibrated run that "
                             "persisted a curve, and calibration_tools.calibrate_count_operating_point, "
                             "which earns a claim over an already-published bucket rather than "
                             "publishing one",
    "image_filenames": "the per-image bucket publishers (the shared image-bucket publisher behind "
                       "run_inference and deliver_per_image_counts's live path, and the web inference "
                       "worker): each prediction document stem mapped to its source image's "
                       "basename with extension",
}
"""Every top-level key a producer adds beside ``operating_point_stamp``'s own sixteen, one entry
per key naming which producer writes it. :func:`write_sidecar` refuses a fresh ``operating_point``
stamp whose body carries a top-level key outside ``STAMP_KEYS | STAMP_EXTENSION_KEYS``;
:func:`update_sidecar` refuses only a key the update itself introduces relative to the stored
stamp, never one the stored stamp already carried. Either refusal names the key and this
declaration; a new producer addition is admitted by declaring it here, not by the writer silently
accepting whatever a caller assembled."""


def read_operating_point_sidecar(pred_dir: str | Path) -> dict | None:
    """The bucket's ``operating_point.json`` stamp, or ``None`` if absent/unreadable (never raises)."""
    return _read_sidecar(pred_dir, "operating_point")


@dataclass(frozen=True)
class BucketScope:
    """A prediction bucket's own recorded ``(subject, attribute)`` claim.

    ``subject`` is the object class every prediction record in the bucket is of; ``attribute`` is
    the attribute each record's value sits under, ``None`` for a detector bucket.
    """

    subject: str | None
    attribute: str | None

    @property
    def classified(self) -> bool:
        return self.attribute is not None


def bucket_scope(pred_dir: str | Path) -> BucketScope | None:
    """A prediction bucket's own recorded scope, or ``None`` for a bucket with no stamp at all.

    ``None`` means a bare directory (a staged bucket, a hand-split copy, a directory under no
    producer's own layout): its records are read under the caller's own statement, never as a
    proven detector bucket. A stamp that decodes but carries no usable ``(subject, attribute)``
    pair raises :class:`StampScopeUnstated`, naming the conform script; a stamp that will not
    decode at all propagates the seam's own error (the strict read, :func:`_read_sidecar`). Both
    are refusals rather than a bare-directory read: an undecodable or pre-scope classified stamp
    read as a bare directory would let its value-keyed records be reviewed as object classes, the
    laundering this function exists to remove.
    """
    stamp = _read_sidecar(pred_dir, "operating_point", strict=True)
    if stamp is None:
        return None
    if "subject" not in stamp or "attribute" not in stamp:
        raise StampScopeUnstated(
            f"{pred_dir}: operating_point.json carries no subject/attribute pair. Run "
            "tcip repair-classified-predictions over this bucket before reading its scope."
        )
    subject, attribute = stamp["subject"], stamp["attribute"]
    if (subject is not None and not isinstance(subject, str)) or (
        attribute is not None and not isinstance(attribute, str)
    ):
        raise StampScopeUnstated(
            f"{pred_dir}: operating_point.json's subject/attribute pair is not a string or None. "
            "Run tcip repair-classified-predictions over this bucket."
        )
    return BucketScope(subject=subject, attribute=attribute)


def read_classifier_operating_point_sidecar(pred_dir: str | Path) -> dict | None:
    """The bucket's ``classifier_operating_point.json`` stamp, or ``None`` if absent/unreadable.

    A file distinct from ``operating_point.json``: the classifier-validity dimension is structurally
    independent from the count operating point's, so the two are never written to the
    same fields a generic writer could conflate (``_sidecar_reference`` reads exactly
    ``validated``/``operating_point.conf.validated_against``, which must stay the count dimension's
    alone).
    """
    return _read_sidecar(pred_dir, "classifier_operating_point")


def read_ordinal_operating_point_sidecar(pred_dir: str | Path) -> dict | None:
    """The bucket's ``ordinal_operating_point.json`` stamp, or ``None`` if absent/unreadable.

    A file distinct from ``operating_point.json``/``classifier_operating_point.json``: the ordinal
    compensating-error dimension is structurally independent from both, same reasoning as
    :func:`read_classifier_operating_point_sidecar`, a generic writer must not conflate these
    dimensions.
    """
    return _read_sidecar(pred_dir, "ordinal_operating_point")


def read_regression_operating_point_sidecar(pred_dir: str | Path) -> dict | None:
    """The bucket's ``regression_operating_point.json`` stamp, or ``None`` if absent/unreadable.

    A file distinct from every other operating-point sidecar, same reasoning as
    :func:`read_classifier_operating_point_sidecar`/:func:`read_ordinal_operating_point_sidecar`.
    """
    return _read_sidecar(pred_dir, "regression_operating_point")


def read_scale_sidecar(pred_dir: str | Path) -> dict | None:
    """The bucket's ``resolve_scale.json`` stamp, or ``None`` if absent/unreadable (never raises).

    A file distinct from ``operating_point.json``: the physical-scale dimension
    (:func:`tcip_mcp.pipelines.measurement.mask_geometry.resolve_scale`) has no production writer
    folding it into the count operating point today, and is structurally independent from it anyway
    (a physical scale is a fact about the imagery, not a count calibration), the same reasoning that
    keeps ``classifier_operating_point.json`` its own file rather than a field inside this one.
    """
    return _read_sidecar(pred_dir, "resolve_scale")


def stamp_names_raster(sidecar: dict | None) -> bool:
    """Whether a stamp names a whole-raster bucket (its own ``raster_path`` field set).

    One mosaic total is not a per-image count, so a raster bucket is refused wherever a door
    reasons over per-image predictions: shared by the per-image delivery door's own refusal
    (``inference_tools._deliver_per_image_counts_from_bucket``) and
    ``calibration_tools.calibrate_count_operating_point``, rather than each spelling the same
    ``raster_path is not None`` check on its own.
    """
    return sidecar is not None and sidecar.get("raster_path") is not None


def _sidecar_reference(
    sidecar: dict | None, *, param_key: str = "conf", validation_kind: str = "annotations",
) -> str:
    """Which reference the sidecar's named param cleared, for its ``validation_kind``, or
    ``VALIDATED_FALSE``.

    Never upgrades a missing/unrecognized/wrong-kind value to a shippable reference, a param whose
    own recorded reference is absent, or belongs to a different validation kind than this one, floors
    to ``false`` rather than being read as validated. (Upgrading on a bare top-level ``validated``
    bool was a real laundering path: a physical-measurement reference could read back as an
    annotation-based one purely because the sidecar's overall flag was true.) ``param_key`` lets a
    differently-shaped sidecar (e.g. a classifier stamp's ``classifier`` param) reuse this same read.
    """
    if not sidecar or not sidecar.get("validated"):
        return VALIDATED_FALSE
    param = (sidecar.get("operating_point") or {}).get(param_key) or {}
    return cleared_reference(param.get("validated_against"), validation_kind=validation_kind)


_LABEL_MOVEMENT_KEYS: tuple[str, ...] = (
    "labels_moved_draw_to_run", "labels_moved_run_to_now", "calibration_labels_moved",
    "manifest_redrawn", "calibration_labels_dir",
)
"""The five keys :func:`resolver_selection_disjointness` copies onto an applicable row beside
the leak fields: presence is required there, their value is not gated on (a moved label is
visible on the row, not a floor)."""

# --- the claim a stamp asserts, and the record that has to answer for it -------------------

_CLAIM_KEYS: dict[str, tuple[str, ...]] = {
    "operating_point": ("operating_point", "tile_size_validated", "claim_scope_validated",
                        "shippable_issues", "id_map", "mask_binarize"),
    "classifier_operating_point": ("operating_point",),
    "ordinal_operating_point": ("operating_point",),
    "regression_operating_point": ("operating_point",),
    "resolve_scale": ("operating_point",),
}
"""Which of a stamp's fields *are* the claim, per document: the values a delivery consumes, as
opposed to the provenance describing where they came from. Stated once and nowhere restated, since
the side that mints a record and the side that verifies one must subset a stamp identically or a
verification compares two different things and always agrees. ``operating_point``'s
``subject``/``attribute`` pair is deliberately absent: it is provenance, not claim, so the conform
script can edit a bucket's scope without flooring a count claim already sealed over it, and
:func:`verify_stamp_binding` never sees it move."""

_DOCUMENT_PARAM: dict[str, tuple[str, str]] = {
    "operating_point": ("conf", "annotations"),
    "classifier_operating_point": ("classifier", "annotations"),
    "ordinal_operating_point": ("ordinal", "annotations"),
    "regression_operating_point": ("regression", "annotations"),
    "resolve_scale": ("scale", "physical"),
}
"""The parameter each document's claim hangs on, and the kind of reference that can validate it.
Read by every reconciler and by the verifier, so a document's validity is decided against one
parameter of one kind wherever it is read."""

MEASUREMENT_DOCUMENTS: tuple[str, ...] = (
    "operating_point", "ordinal_operating_point", "regression_operating_point",
)
"""The sidecar documents a per-plant/per-image delivery may state as its own
``measurement_document``: every :data:`_DOCUMENT_PARAM` entry except ``classifier_operating_point``
(no per-plant aggregate rests on a classifier alone today) and ``resolve_scale`` (a physical scale
is never itself the measurement it states the unit of)."""


def claim_payload(sidecar: dict | None, *, document: str) -> dict:
    """The part of a stamp that constitutes the claim, for the document it is a stamp of.

    The one extractor both sides of the binding call: :func:`seal_validation` runs it over the stamp
    body a door is about to publish and stores the result in the validation record, and
    :func:`verify_stamp_binding` runs it over the stamp it is reading and compares. Compared whole,
    never field by field, so a key added to a stamp after its record was minted reads as a
    disagreement rather than as an ignored extra. A key the stamp does not carry is absent from the
    payload rather than defaulted, since a defaulted value is a claim nobody made.
    """
    try:
        keys = _CLAIM_KEYS[document]
    except KeyError:
        raise ValueError(
            f"{document!r} is not a prediction-bucket stamp; declared documents are "
            f"{sorted(_CLAIM_KEYS)}"
        ) from None
    stamp = sidecar or {}
    return {key: stamp[key] for key in keys if key in stamp}


@dataclass(frozen=True)
class _Resolver:
    """One resolver a document's claim may be earned through, and how to read its result."""

    module: str
    function: str
    trait_param: str | None
    experiment_param: str | None


_DOCUMENT_RESOLVERS: dict[str, dict[str, _Resolver]] = {
    "operating_point": {
        "resolve_operating_point": _Resolver(
            "tcip_mcp.pipelines.operating_point", "resolve_operating_point",
            "trait_name", "experiment_id"),
        "resolve_operating_point_from_review": _Resolver(
            "tcip_mcp.pipelines.feedback.review_calibration", "resolve_operating_point_from_review",
            "trait_name", "experiment_id"),
    },
    "classifier_operating_point": {
        "resolve_classifier_operating_point": _Resolver(
            "tcip_mcp.pipelines.operating_point", "resolve_classifier_operating_point",
            "trait_name", "experiment_id"),
    },
    "ordinal_operating_point": {
        "resolve_ordinal_operating_point": _Resolver(
            "tcip_mcp.pipelines.operating_point", "resolve_ordinal_operating_point",
            "trait_name", "experiment_id"),
    },
    "regression_operating_point": {
        "resolve_regression_operating_point": _Resolver(
            "tcip_mcp.pipelines.operating_point", "resolve_regression_operating_point",
            "trait_name", "experiment_id"),
    },
    "resolve_scale": {
        "resolve_physical_scale": _Resolver(
            "tcip_mcp.pipelines.measurement.scale_calibration", "resolve_physical_scale",
            None, None),
    },
}
"""Which resolvers may earn which document's claim, named rather than handed in. A caller supplies
the resolver's *name* and its inputs and never a callable: a primitive that ran whatever function it
was given would be a primitive a caller could hand a verdict to, which is the signature this seam
exists to remove. A document with more than one entry has genuinely different evidence shapes behind
one claim (ground-truth records against reviewer-confirmed verdicts), and the caller says which."""

_REFERENCE_INPUT_GROUPS = ("label_dirs", "label_csvs", "reference_buckets", "scope_roots",
                           "label_stems", "stated_values")
"""The kinds of evidence a reference identity is built from. Each names locations the platform can
hash for itself, except ``stated_values``, which holds what another primitive already computed (a
split lock's identity, a review reference's hash and image count) and this one cannot recompute.
``label_stems`` is ``label_dirs``' narrower sibling: a directory hashed over a named subset of its
stems (``{role: {"path": dir, "stems": [...]}}``) rather than whole, for a calibration restricted
to a split manifest's calibration side, where the whole directory was never what the reference
swept."""

_UNCOMPARED = object()
"""A resolver result that publishes no value of its own for its parameter, so the claim's value has
nothing here to be compared against and the reference alone carries the agreement."""


def _resolver_reference(result: Any, param_key: str) -> str | None:
    """The reference a resolver's own result recorded for the document's parameter."""
    if isinstance(result, ResolvedBundle):
        return result.get(param_key).validated_against
    if isinstance(result, ResolvedParam):
        return result.validated_against
    return (result or {}).get("validated_against")


def _resolver_value(result: Any, param_key: str) -> Any:
    """The value a resolver's own result carries for the document's parameter, if it carries one."""
    if isinstance(result, ResolvedBundle):
        return result.get(param_key)._raw
    if isinstance(result, ResolvedParam):
        return result._raw
    if isinstance(result, Mapping) and "value" in result:
        return result["value"]
    return _UNCOMPARED


def _disjointness_evidence(result: Any, document: str, caller: str) -> dict | None:
    """The gate evidence dict a disjointness-reading resolver checks, for one declared document
    (:data:`_DOCUMENT_PARAM`): the document guard and the live-result-to-evidence extraction
    :func:`resolver_train_disjointness` and :func:`resolver_selection_disjointness` both need,
    in one place so declaring a fifth checked document, or changing how evidence is pulled out of
    a live result, edits one function rather than two that must agree.

    ``None`` for ``resolve_scale`` (no training run to check against) and for a result carrying
    no gate evidence at all. A document neither resolver knows how to read from raises, naming
    ``caller``, rather than silently sealing ``null`` for a check nobody ran.
    """
    if document == "resolve_scale":
        return None
    if document not in (
        "operating_point", "classifier_operating_point", "ordinal_operating_point",
        "regression_operating_point",
    ):
        raise ValueError(
            f"{document!r} is not a document {caller} knows how to read a disjointness check "
            f"from; declared documents are {sorted(_DOCUMENT_PARAM)}"
        )
    param_key, _ = _DOCUMENT_PARAM[document]
    if isinstance(result, ResolvedBundle):
        return result.get(param_key).gate_evidence
    if isinstance(result, Mapping):
        return result.get("gate_evidence")
    return None


def resolver_train_disjointness(result: Any, document: str) -> dict | None:
    """Whether and how a resolver's own live result checked train-disjointness: the two facts the
    gate itself records, ``{"checked": bool, "group_check": str | None}``, never a bare ``true``
    over a check the gate's own record says did not run.
    """
    evidence = _disjointness_evidence(result, document, "resolver_train_disjointness")
    td = (evidence or {}).get("train_disjointness")
    if not isinstance(td, dict):
        return None
    return {"checked": bool(td.get("checked")), "group_check": td.get("group_check")}


def resolver_selection_disjointness(result: Any, document: str) -> dict | None:
    """Whether and how a resolver's own live result checked selection-disjointness (the
    checkpoint's own selection side, ``split.json``'s ``val``, disjoint from the reference): the
    same shape live gate evidence carries, ``applicable``, ``reason``, ``checked``, ``unresolvable``,
    ``leaked_groups``, ``leaked_stems``, ``group_check`` and, when the calibration read a label
    directory, the four label-movement keys plus ``calibration_labels_dir`` beside them, so the
    row a delivery door reads carries the leak fields and the movement facts its floor and its
    breeder sentence read, not only the pass/fail booleans.
    """
    evidence = _disjointness_evidence(result, document, "resolver_selection_disjointness")
    sd = (evidence or {}).get("selection_disjointness")
    if not isinstance(sd, dict):
        return None
    return {
        "applicable": bool(sd.get("applicable")), "reason": sd.get("reason"),
        "checked": bool(sd.get("checked")), "unresolvable": bool(sd.get("unresolvable")),
        "leaked_groups": list(sd.get("leaked_groups") or []),
        "leaked_stems": list(sd.get("leaked_stems") or []),
        "group_check": sd.get("group_check"),
        # Preserved as-is, never coerced: None means "not checked", an empty list means
        # "checked, nothing moved", and the two must stay distinguishable on the row.
        "labels_moved_draw_to_run": sd.get("labels_moved_draw_to_run"),
        "labels_moved_run_to_now": sd.get("labels_moved_run_to_now"),
        "calibration_labels_moved": sd.get("calibration_labels_moved"),
        "manifest_redrawn": sd.get("manifest_redrawn"),
        "calibration_labels_dir": sd.get("calibration_labels_dir"),
    }


def _relative_location(path: str | Path, dataset_root: Path) -> str:
    """Where an input sits, expressed against the dataset root the record hangs off.

    Recorded so an auditor who wants to recompute a reference can find it, which no delivery does
    (:func:`verify_stamp_binding` compares the recorded hashes, it does not re-read the labels,
    verdicts or lock behind them). An input outside the dataset root is legitimate work (a CSV over a
    loose images directory), and comes back as a path stepping out of the root, or as an absolute
    path when the two share no anchor at all.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(dataset_root, walk_up=True).as_posix()
    except ValueError:
        return resolved.as_posix()


def _reference_identity(reference_inputs: dict, dataset_root: Path) -> dict:
    """The identity of the evidence a claim was earned against, hashed here rather than stated.

    Every location the platform can hash for itself is hashed here (labels through
    :func:`dataset_hash`, a CSV-sourced reference through :func:`csv_dataset_hash`, a reference
    prediction bucket through ``bucket_content_digest``), so a caller cannot hand over an identity
    for evidence it did not present. Each entry records where the input was as well as what it
    hashed to, so the reference stays resolvable to someone auditing offline.
    """
    from tcip_mcp.prediction_buckets import bucket_content_digest

    unknown = sorted(set(reference_inputs) - {"dataset_root", *_REFERENCE_INPUT_GROUPS})
    if unknown:
        raise ValueError(
            f"reference_inputs holds {', '.join(unknown)}, which name no kind of evidence this "
            f"seam can identify; the kinds are {', '.join(_REFERENCE_INPUT_GROUPS)}."
        )

    identity: dict[str, Any] = {}
    for role, d in sorted((reference_inputs.get("label_dirs") or {}).items()):
        identity.setdefault("label_dirs", {})[role] = {
            "path": _relative_location(d, dataset_root), "dataset_hash": dataset_hash(d)}
    for role, p in sorted((reference_inputs.get("label_csvs") or {}).items()):
        identity.setdefault("label_csvs", {})[role] = {
            "path": _relative_location(p, dataset_root), "dataset_hash": csv_dataset_hash(p)}
    for role, d in sorted((reference_inputs.get("reference_buckets") or {}).items()):
        identity.setdefault("reference_buckets", {})[role] = {
            "path": _relative_location(d, dataset_root), "content_digest": bucket_content_digest(d)}
    for role, d in sorted((reference_inputs.get("scope_roots") or {}).items()):
        identity.setdefault("scope_roots", {})[role] = _relative_location(d, dataset_root)
    for role, spec in sorted((reference_inputs.get("label_stems") or {}).items()):
        stems = sorted(spec["stems"])
        identity.setdefault("label_stems", {})[role] = {
            "path": _relative_location(spec["path"], dataset_root),
            "dataset_hash": dataset_hash(spec["path"], stems=stems),
            "count": len(stems),
        }
    stated = reference_inputs.get("stated_values") or {}
    if stated:
        identity["stated_values"] = {k: stated[k] for k in sorted(stated)}

    if not identity:
        raise ValueError(
            "reference_inputs names no evidence at all; a validation record has to say what the "
            f"claim was checked against, through one of {', '.join(_REFERENCE_INPUT_GROUPS)}."
        )
    return identity


@dataclass(frozen=True)
class ValidationDraft:
    """A gate that passed, before there is anything on disk for it to cover.

    Not a record and not appendable: it carries the resolver's own result and the identity of the
    evidence that produced it, and it is handed to :func:`seal_validation` once the predictions it
    covers have been written. The split is what lets the gate run exactly once, before anything is
    published, while the content identity is taken over the files as they actually landed.
    """

    document: str
    trait: str
    validated_against: str
    checkpoint_sha256: str | None
    producing_experiment_id: str | None
    reference_identity: dict
    dataset_root: str
    result: Any
    token: str


_OPEN_DRAFTS: set[str] = set()
"""Drafts this process minted. A draft assembled by any other means is refused at the seal, so the
resolver result a record is minted from is one this process watched a gate produce."""


def open_validation(
    *,
    document: str,
    evidence: dict,
    trait: str,
    checkpoint_sha256: str | None,
    producing_experiment_id: str | None,
    reference_inputs: dict,
) -> ValidationDraft:
    """Run a document's own gate over the evidence, and return the draft a record is sealed from.

    The first of the two phases a validated claim is earned in. It takes the evidence, never a
    verdict: it looks up the named resolver among those the document admits, runs it, and refuses
    unless the resolver's own result cleared a reference ``accepted_references`` recognizes for the
    document's kind. Nothing is written here, and a caller holding a draft holds a passed gate, not
    a claim.

    ``evidence`` is ``{"resolver": <name>, "inputs": {...}}``: which of the document's resolvers ran
    the gate, and the arguments it ran over. The trait and the producing experiment are passed here
    rather than through ``inputs``, so the trait a record is earned for is the trait the gate was
    run for, and the run whose training split disjointness is checked is the run that produced the
    predictions. ``inputs`` restating either is refused rather than allowed to disagree.

    ``reference_inputs`` names the evidence's own locations (see :func:`_reference_identity`), and
    must include the ``dataset_root`` the claim, its covered buckets and its reference all hang off.

    ``checkpoint_sha256`` is the identity the evidence carried, and ``producing_experiment_id`` the
    run that produced the predictions; both may be ``None`` for a bespoke or unregistered checkpoint,
    and both are recorded as they are rather than re-derived from a file on disk, which would prove
    a file with that content exists somewhere and not that these predictions came from it.
    """
    import importlib

    try:
        resolvers = _DOCUMENT_RESOLVERS[document]
    except KeyError:
        raise ValueError(
            f"{document!r} is not a prediction-bucket stamp; declared documents are "
            f"{sorted(_DOCUMENT_RESOLVERS)}"
        ) from None
    if not isinstance(trait, str) or not trait:
        raise ValueError(
            f"a {document} claim needs the trait it is earned for; a validated stamp with no trait "
            "names nothing a delivery can be checked against."
        )

    unknown = sorted(set(evidence) - {"resolver", "inputs"})
    if unknown or "resolver" not in evidence:
        raise ValueError(
            f"evidence for {document} must be {{'resolver': <name>, 'inputs': {{...}}}}; got "
            f"{sorted(evidence)}. The resolver is named, never handed in, and the admitted names "
            f"for this document are {sorted(resolvers)}."
        )
    name = evidence["resolver"]
    if name not in resolvers:
        raise ValueError(
            f"{name!r} does not earn a {document} claim; the resolvers this document admits are "
            f"{sorted(resolvers)}."
        )
    spec = resolvers[name]
    inputs = dict(evidence.get("inputs") or {})
    owned = {p for p in (spec.trait_param, spec.experiment_param) if p is not None} & set(inputs)
    if owned:
        raise ValueError(
            f"evidence inputs for {document} restate {', '.join(sorted(owned))}: the trait and the "
            "producing run are open_validation's own arguments, so a second spelling of them could "
            "disagree with the record they are written into."
        )
    if spec.trait_param:
        inputs[spec.trait_param] = trait
    if spec.experiment_param:
        inputs[spec.experiment_param] = producing_experiment_id

    root = reference_inputs.get("dataset_root")
    if not root:
        raise ValueError(
            f"reference_inputs for {document} names no dataset_root; a record's covered buckets and "
            "reference locations are recorded against one, so the claim cannot be placed without it."
        )
    dataset_root = Path(root).resolve()
    reference_identity = _reference_identity(reference_inputs, dataset_root)

    result = getattr(importlib.import_module(spec.module), spec.function)(**inputs)
    param_key, validation_kind = _DOCUMENT_PARAM[document]
    reported = _resolver_reference(result, param_key)
    cleared = cleared_reference(reported, validation_kind=validation_kind)
    if cleared == VALIDATED_FALSE:
        raise ValueError(
            f"{name} reported {param_key} validated_against={reported!r} for trait {trait!r}, which "
            f"clears nothing for a {validation_kind} claim; a {document} claim is earned only "
            f"against {list(accepted_references(validation_kind))}. Deliver provisionally through "
            "an acknowledged delivery, or calibrate against a reference sized to the trait."
        )

    import secrets

    token = secrets.token_hex(16)
    _OPEN_DRAFTS.add(token)
    return ValidationDraft(
        document=document, trait=trait, validated_against=cleared,
        checkpoint_sha256=checkpoint_sha256, producing_experiment_id=producing_experiment_id,
        reference_identity=reference_identity, dataset_root=str(dataset_root), result=result,
        token=token,
    )


_CALIBRATION_EXPERIMENT_DERIVATION: dict[str | None, str] = {
    None: "a claim earned at a delivery door for predictions no run in this platform's "
         "experiment record produced",
    "resolve_scale": "a physical-scale claim earned against a breeder-supplied reference, not "
                     "predictions any run produced",
}
"""What a calibration experiment minted by :func:`seal_validation` says it is, by document, read
by ``None`` for every document with no more specific sentence of its own (every document but
``resolve_scale`` today)."""


def bucket_relative_key(bucket: str | Path, root: str | Path, *, document: str) -> str:
    """``bucket``'s path relative to ``root``, or refuse: the one under-root check every caller
    that records a claim against a bucket applies.

    Shared by :func:`seal_validation` (each bucket a claim covers) and
    ``calibration_tools.calibrate_count_operating_point``'s own pre-check over ``pred_dir``, so a
    bucket outside the dataset root is refused with one wording wherever a claim would try to
    place it, rather than two independently-worded checks that could drift.
    """
    resolved = Path(bucket).resolve()
    root_resolved = Path(root).resolve()
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError:
        raise ValueError(
            f"prediction bucket {str(resolved)!r} is not under dataset_root {str(root_resolved)!r}, "
            f"so a {document} claim covering it has no dataset-relative key to record. Write the "
            "predictions into the dataset's own predictions layout (resolve_prediction_bucket) to "
            "earn a validated claim."
        ) from None


def seal_validation(
    draft: ValidationDraft,
    *,
    dataset_root: str | Path,
    bucket_dirs: list[str | Path] | tuple[str | Path, ...],
    stamp_body: dict,
    images_dir: str | Path | None = None,
) -> tuple[str, dict]:
    """Append the record a passed gate earned, over the files as they are now, and stamp the pointer.

    The second phase. It takes the content identity of every bucket the claim covers from the files
    on disk at this moment (so the claim covers what was actually written, not what the run set out
    to write), takes the claim itself from the stamp body about to be published, appends the row, and
    returns the digest together with that stamp body with ``validated_by`` merged in. The caller
    writes the returned body last.

    There is no transaction across the experiment store and the bucket, and none is invented: the
    order is chosen so every partial state fails closed. A crash before this call leaves prediction
    files with no stamp, which floors. A crash after it leaves a record no stamp names, which is
    inert. Only a stamp that names a row a reader can find and recompute delivers.

    ``covered_buckets`` is keyed by each bucket's path relative to ``dataset_root``, so a dataset
    moved or copied whole still verifies while a bucket moved to a different place inside it does
    not. A bucket outside the stated dataset root cannot be keyed that way and is refused here, the
    same claim being unverifiable on the reading side.

    For the ``operating_point`` document the digest is over the bucket's prediction bytes
    (:func:`~tcip_mcp.prediction_buckets.bucket_content_digest`), since that claim is about what was
    predicted. For ``resolve_scale`` the digest is over the bytes of the bucket's own images, read
    from ``images_dir`` (:func:`~tcip_mcp.prediction_buckets.bucket_stems_digest`) instead: a scale
    claim is a fact about the bucket's imagery, not its predictions, so re-exporting predictions over
    the same images must not floor it, while an image added to, removed from, or replaced in the
    bucket must. ``images_dir`` is required for a ``resolve_scale`` draft, and unused otherwise.
    """
    from tcip_mcp.experiments import _append_validation, ensure_calibration_experiment
    from tcip_mcp.prediction_buckets import bucket_content_digest, bucket_stems_digest

    if draft.token not in _OPEN_DRAFTS:
        raise ValueError(
            "seal_validation was handed a draft this process did not mint; a record is appended "
            "only for a gate open_validation itself ran over the evidence."
        )
    root = Path(dataset_root).resolve()
    if str(root) != draft.dataset_root:
        raise ValueError(
            f"seal_validation was given dataset_root {str(root)!r} for a draft opened against "
            f"{draft.dataset_root!r}; the covered buckets and the reference identity are recorded "
            "against one root, so they cannot be sealed against another."
        )
    if draft.document == "resolve_scale" and bucket_dirs and images_dir is None:
        raise ValueError(
            "seal_validation needs images_dir to hash a resolve_scale claim's covered bucket(s): "
            "the claim binds to the imagery, not the predictions."
        )

    covered: dict[str, str] = {}
    if draft.document in ("operating_point", "resolve_scale"):
        def digest_fn(d: Path) -> str:
            if draft.document == "operating_point":
                return bucket_content_digest(d)
            assert images_dir is not None
            return bucket_stems_digest(d, images_dir=images_dir)

        for d in bucket_dirs:
            resolved = Path(d).resolve()
            key = bucket_relative_key(resolved, root, document=draft.document)
            covered[key] = digest_fn(resolved)
    elif bucket_dirs:
        raise ValueError(
            f"a {draft.document} claim covers no prediction bucket's content: it is earned against a "
            "reference and legitimately applies to later buckets. Name the calibration and holdout "
            "buckets in reference_inputs['reference_buckets'] instead of in bucket_dirs."
        )

    claim = claim_payload(stamp_body, document=draft.document)
    param_key, _ = _DOCUMENT_PARAM[draft.document]
    stamped = (claim.get("operating_point") or {}).get(param_key) or {}
    if not stamp_body.get("validated"):
        raise ValueError(
            f"the {draft.document} stamp handed to seal_validation is not stamped validated, so the "
            "record would answer for a claim the stamp does not make."
        )
    if stamped.get("validated_against") != draft.validated_against:
        raise ValueError(
            f"the {draft.document} stamp records {param_key} validated_against="
            f"{stamped.get('validated_against')!r} while the gate cleared "
            f"{draft.validated_against!r}; stamp the reference the resolver reported."
        )
    resolved_value = _resolver_value(draft.result, param_key)
    if resolved_value is not _UNCOMPARED and stamped.get("value") != resolved_value:
        raise ValueError(
            f"the {draft.document} stamp records {param_key}={stamped.get('value')!r} while the gate "
            f"resolved {resolved_value!r}; a record cannot answer for a value its gate never saw."
        )

    experiment_id = draft.producing_experiment_id or ensure_calibration_experiment(
        document=draft.document, checkpoint_sha256=draft.checkpoint_sha256,
        reference_identity=draft.reference_identity, trait=draft.trait,
        config={"derived_from": _CALIBRATION_EXPERIMENT_DERIVATION.get(
            draft.document, _CALIBRATION_EXPERIMENT_DERIVATION[None])},
    )
    body = {
        "document": draft.document,
        "trait": draft.trait,
        "claim": claim,
        "validated_against": draft.validated_against,
        "checkpoint_sha256": draft.checkpoint_sha256,
        "producing_experiment_id": draft.producing_experiment_id,
        "reference_identity": draft.reference_identity,
        "covered_buckets": covered,
        "dataset_root": str(root),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "train_disjointness": resolver_train_disjointness(draft.result, draft.document),
        "selection_disjointness": resolver_selection_disjointness(draft.result, draft.document),
    }
    appended = _append_validation(experiment_id, body)
    if "error" in appended:
        raise ValueError(f"the {draft.document} claim was not recorded: {appended['error']}")
    record_digest = appended["record_digest"]
    return record_digest, {**stamp_body,
                           "validated_by": {"experiment_id": experiment_id,
                                            "record_digest": record_digest}}


@dataclass(frozen=True)
class StampBinding:
    """Whether a stamp's claim is answered for by a record outside the bucket that made it.

    ``claimed`` says whether the stamp asserts validation at all: a stamp that asserts nothing has
    nothing to bind, so ``ok`` stays true and the dimension floors on its own merits exactly as it
    did before. ``experiment_id`` is the experiment the record lives in, which for a calibration
    earned outside a training run is not the run that produced the predictions;
    ``producing_experiment_id`` is that run, and is what a delivery's producer column reports.
    """

    ok: bool
    claimed: bool
    experiment_id: str | None = None
    producing_experiment_id: str | None = None
    checkpoint_sha256: str | None = None
    record_digest: str | None = None
    train_disjointness: dict | None = None
    selection_disjointness: dict | None = None
    note: str = ""


def verify_stamp_binding(
    sidecar: dict | None, pred_dir: str | Path, *, document: str, trait: str | None = None,
    digest_memo: dict[str, str] | None = None, images_dir: str | Path | None = None,
) -> StampBinding:
    """Check that a stamp's validation claim is answered for by a record it cannot itself write.

    Called from inside the reconcilers rather than at each delivery door, so no door can deliver
    without it. Every check is cheap: a stamp read, a log read, and for the count and scale
    documents one pass over the bucket's own prediction files or imagery the claim covers. No model
    is loaded and no gate is re-run.

    In order: the stamp's own parameter cleared a reference of the document's kind (unchanged, and
    still first); it names an experiment and a row; that experiment exists; that row is in it and
    still hashes to the identity the stamp committed to; the row agrees with the stamp on document,
    reference, checkpoint identity (absence equal to absence), trait and the whole claim payload; and
    for the count and scale documents, every bucket being read is in the covered set at its
    dataset-relative key with the content (or imagery) identity it was earned over, recomputed now.
    ``images_dir`` is required to reach that last check for ``resolve_scale`` and unused otherwise.
    When the reference identity carries a ``split_manifest_dir``, the row must also carry a
    ``selection_disjointness`` that is either not-applicable (with a reason) or checked with no
    leak; a manifest-scoped reference earned before that field existed, or earned against a
    checkpoint whose own run is unknown, floors here rather than reading as cleared.

    Verification is per stamp file, not per parameter. One failed check floors every dimension that
    stamp carries, so a count operating point, a tile geometry, a claim scope and a review upgrade
    written into one ``operating_point.json`` stand or fall together.

    ``digest_memo`` is a caller-owned dict living for the span of one delivery, so a bucket several
    reconcilers read is hashed once. There is deliberately no cache beyond it: recomputation is what
    detects a replacement whose size and timestamp were restored.
    """
    from tcip_mcp.experiments import experiment_exists, experiments_scope, find_validation
    from tcip_mcp.prediction_buckets import bucket_content_digest, bucket_stems_digest

    param_key, validation_kind = _DOCUMENT_PARAM[document]
    stamp = sidecar or {}
    reference = _sidecar_reference(stamp, param_key=param_key, validation_kind=validation_kind)
    if reference == VALIDATED_FALSE:
        return StampBinding(ok=True, claimed=False)

    bucket = str(pred_dir)

    def floored(note: str, **known: Any) -> StampBinding:
        return StampBinding(ok=False, claimed=True, note=note, **known)

    pointer = well_formed_validated_by(stamp)
    if pointer is None:
        return floored(
            f"{document}.json at {bucket!r} claims validated with no well-formed validated_by, so "
            "no record answers for it. A validated claim is earned through the calibrated export "
            "door, the calibration tool for this document, or the review validation action; the "
            "bucket delivers unvalidated until one of them earns a record."
        )
    experiment_id = pointer["experiment_id"]
    record_digest = pointer["record_digest"]

    if not experiment_exists(experiment_id):
        return floored(
            f"{document}.json at {bucket!r} names experiment {experiment_id!r}, which the experiment "
            f"store at {experiments_scope()} does not hold. Earn the claim through the calibration "
            "door for this document, which creates the record it names.",
            experiment_id=experiment_id, record_digest=record_digest,
        )

    row = find_validation(experiment_id, record_digest)
    if row is None:
        return floored(
            f"{document}.json at {bucket!r} names record {record_digest!r} in experiment "
            f"{experiment_id!r}, and no row in that experiment's validations hashes to it (searched "
            f"the experiment store at {experiments_scope()}). Re-earn the claim through the "
            "calibration door for this document.",
            experiment_id=experiment_id, record_digest=record_digest,
        )

    known = {"experiment_id": experiment_id, "record_digest": record_digest,
             "producing_experiment_id": row.get("producing_experiment_id"),
             "checkpoint_sha256": row.get("checkpoint_sha256"),
             "train_disjointness": row.get("train_disjointness"),
             "selection_disjointness": row.get("selection_disjointness")}

    if row.get("document") != document:
        return floored(
            f"record {record_digest!r} was earned for {row.get('document')!r}, not for the "
            f"{document} being read at {bucket!r}. Re-calibrate for the document this delivery "
            "reads.", **known)
    if row.get("validated_against") != reference:
        return floored(
            f"{document}.json at {bucket!r} records validated_against={reference!r} while record "
            f"{record_digest!r} was earned against {row.get('validated_against')!r}. Re-calibrate "
            "against the reference the stamp claims.", **known)
    if row.get("checkpoint_sha256") != stamp.get("checkpoint_sha256"):
        return floored(
            f"{document}.json at {bucket!r} records checkpoint {stamp.get('checkpoint_sha256')!r} "
            f"while record {record_digest!r} was earned under {row.get('checkpoint_sha256')!r}. "
            "Re-calibrate for the checkpoint that produced these predictions.", **known)
    if row.get("trait") != stamp.get("trait"):
        return floored(
            f"{document}.json at {bucket!r} records trait {stamp.get('trait')!r} while record "
            f"{record_digest!r} was earned for {row.get('trait')!r}. Re-calibrate for the delivered "
            "trait.", **known)
    if trait is not None and row.get("trait") != trait:
        return floored(
            f"record {record_digest!r} behind {document}.json at {bucket!r} was earned for trait "
            f"{row.get('trait')!r}, not {trait!r}. Re-calibrate for the delivered trait.", **known)
    if claim_payload(stamp, document=document) != row.get("claim"):
        return floored(
            f"{document}.json at {bucket!r} asserts a claim record {record_digest!r} was not earned "
            f"for: the stamp's {', '.join(_CLAIM_KEYS[document])} disagree with the values the gate "
            "was run over. Re-calibrate to earn a record for the values being delivered.", **known)

    row_split_manifest_dir = (row.get("reference_identity") or {}).get(
        "stated_values", {}).get("split_manifest_dir")
    if row_split_manifest_dir is not None:
        sd = row.get("selection_disjointness")
        sd_applicable = isinstance(sd, dict) and sd.get("applicable") is True
        sd_ok = isinstance(sd, dict) and (
            (sd.get("applicable") is False and sd.get("reason"))
            or (sd_applicable and sd.get("checked") is True
                and not sd.get("leaked_groups") and not sd.get("leaked_stems")
                and all(k in sd for k in _LABEL_MOVEMENT_KEYS))
        )
        if not sd_ok:
            return floored(
                f"{document}.json at {bucket!r} claims a validated reference under split "
                f"manifest {row_split_manifest_dir!r}, and record {record_digest!r} carries no "
                "selection_disjointness check that is either not-applicable (with a reason) or "
                "checked with no leak and the label-movement keys sealed. Calibrate again under "
                "the manifest's calibration side with a checkpoint whose run is on record.",
                **known)

    if document in ("operating_point", "resolve_scale"):
        resolved = Path(bucket).resolve()
        dataset_root = _dataset_root_of(resolved)
        noun = "count" if document == "operating_point" else "scale"
        if dataset_root is None:
            return floored(
                f"{document}.json at {bucket!r} claims a validated {noun} from a bucket under no "
                "dataset root, so the covered set cannot be located. Write the predictions into the "
                "dataset's own predictions layout (resolve_prediction_bucket) to earn a validated "
                f"{noun}.", **known)
        key = resolved.relative_to(dataset_root).as_posix()
        covered = row.get("covered_buckets") or {}
        if key not in covered:
            return floored(
                f"{document}.json at {bucket!r} claims a validated {noun}, and record "
                f"{record_digest!r} covers {sorted(covered)} rather than {key!r}. Write to a fresh "
                "bucket variant and re-validate.", **known)
        if document == "resolve_scale" and images_dir is None:
            raise ValueError(
                f"verify_stamp_binding needs images_dir to recompute the imagery digest a "
                f"resolve_scale claim at {bucket!r} covers."
            )
        if document == "operating_point":
            recomputed = bucket_content_digest(resolved, memo=digest_memo)
        else:
            # document == "resolve_scale" here; the check above already raised if images_dir
            # were None for that document.
            assert images_dir is not None
            recomputed = bucket_stems_digest(resolved, images_dir=images_dir)
        if recomputed != covered[key]:
            what = "prediction files" if document == "operating_point" else "image set"
            return floored(
                f"the {what} in {bucket!r} hash to {recomputed!r}, and record {record_digest!r} "
                f"was earned over {covered[key]!r}: a file has been added, replaced or removed "
                "since the claim was earned. Write to a fresh bucket variant and re-validate.",
                **known)

    return StampBinding(ok=True, claimed=True, **known)


def _dataset_root_of(path: Path) -> Path | None:
    """The dataset root a bucket sits under, resolved from where it is now rather than from the
    record, so a dataset moved or copied whole still keys its covered buckets the same way."""
    from tcip_mcp.dataset_layout import dataset_root_of

    root = dataset_root_of(path)
    return root.resolve() if root is not None else None


def experiment_recorded_checkpoint(experiment_id: str) -> str | None:
    """The checkpoint identity this experiment record answers for, or ``None`` when it records none.

    Read from the run's own lineage, and only for a record whose status is ``completed``: a
    file-backend crash between the lineage apply and the status apply could otherwise leave a
    still-``running`` record carrying a digest, and a run that never completed vouches for
    nothing. ``complete_run`` writes the digest and the terminal status together in one
    transaction, so this is the one fact the run itself recorded of its own output.
    """
    from tcip_mcp.experiments import lineage_key, read_member, status_key

    status = read_member(status_key(experiment_id), {})
    if not isinstance(status, dict) or status.get("state") != "completed":
        return None
    lineage = read_member(lineage_key(experiment_id), {})
    weights_sha256 = lineage.get("model_weights_sha256") if isinstance(lineage, dict) else None
    return str(weights_sha256) if weights_sha256 else None


def corroborated_producer(
    checkpoint_sha256: str | None, experiment_id: str | None
) -> tuple[str | None, str | None]:
    """The producing checkpoint and run a delivery may name, from what a record outside the stamp
    confirms of them.

    Validity and producer identity rest on different evidence, so an honestly unvalidated bucket
    keeps the identity it really has. A stamp naming no experiment stands on its checkpoint hash
    alone, which came from resolving the checkpoint and not from any validation claim. A stamp
    naming an experiment is emitted only when that experiment exists and the checkpoint it recorded
    is the one the stamp names, absence equal to absence; otherwise the delivery says the producer
    is unknown rather than repeating names nothing answers for.
    """
    if not experiment_id:
        return checkpoint_sha256, None
    from tcip_mcp.experiments import experiment_exists

    if not experiment_exists(experiment_id):
        return None, None
    if experiment_recorded_checkpoint(experiment_id) != checkpoint_sha256:
        return None, None
    return checkpoint_sha256, experiment_id


def delivered_provenance(
    asserted: Mapping[str, Any] | None,
    bindings: Mapping[str, StampBinding],
    *,
    columns: Sequence[str],
) -> dict[str, Any]:
    """The provenance cells a delivered CSV carries, for one door's own column list.

    The one builder behind every delivered producer column, so two deliverables with different
    column lists cannot disagree on what a given column holds. ``bindings`` is one entry per bucket
    the delivery read, as the reconciler verified them; ``asserted`` is what the stamps and the
    producing call claimed, which is a starting point and never the last word.

    ``validation_record`` names the experiment and row every bucket's claim was answered for by,
    and is empty unless every bucket read is bound, since one cell cannot name a record for buckets
    that have none. ``producer_model_sha256`` and ``producing_experiment_id`` are corroborated
    through :func:`corroborated_producer`, preferring the identity the verified records carry over
    the identity the stamps assert. An ``operating_point_validated`` named in ``columns`` passes
    through whatever ``asserted`` carried for it, unchanged; :func:`delivered_tail` is what
    actually stamps that column from the gate.
    """
    values = dict(asserted or {})
    bound = bool(bindings) and all(b.ok and b.claimed for b in bindings.values())
    values["validation_record"] = "; ".join(sorted(
        {f"{b.experiment_id}:{b.record_digest}" for b in bindings.values()})) if bound else ""

    identities = {(b.checkpoint_sha256, b.producing_experiment_id) for b in bindings.values()}
    if bound and len(identities) == 1:
        checkpoint, producing_experiment_id = next(iter(identities))
    else:
        checkpoint = values.get("producer_model_sha256")
        producing_experiment_id = values.get("producing_experiment_id")
    values["producer_model_sha256"], values["producing_experiment_id"] = corroborated_producer(
        checkpoint, producing_experiment_id)
    return {c: values.get(c) for c in columns}


_DIMENSION_TO_COLUMN: dict[str, str] = {
    "operating_point": "operating_point_validated",
    "classifier": "positive_state_classifier_validated",
}
"""Which delivery-gate dimension owns which CSV validity column, wherever a delivered tail carries
one. :func:`delivered_tail` reads this to derive a door's own ``own_column`` set structurally from
its own column list, rather than a second list stating the same fact and free to drift from it."""


def delivered_tail(
    asserted: Mapping[str, Any] | None,
    bindings: Mapping[str, StampBinding],
    gate: DeliveryGateResult,
    *,
    columns: Sequence[str],
) -> dict[str, Any]:
    """One delivered CSV row's full producer-plus-validity tail, for one door's own column list.

    The one composition behind every delivered tail: producer identity and ``validation_record``
    come from :func:`delivered_provenance`; ``produced_at`` is this call's own write time, computed
    once here rather than accepted from ``asserted``, which is refused when it carries a real
    (non-``None``) ``produced_at`` of its own rather than silently overridden, since the column is
    this composition's own fact and a second source pretending to it is exactly the drift this
    removes -- a ``None``-valued key (a caller that composed ``{"produced_at": x.get(...)}`` over
    something carrying none) is absence, not an assertion, the same convention
    :func:`corroborated_producer` uses; and every validity column ``columns`` actually carries
    (``_DIMENSION_TO_COLUMN``'s owned columns present in ``columns``, ``operating_point_validated``
    included) is stamped through ``gate.column_stamp`` here, never left to ``delivered_provenance``,
    with ``own_column`` derived from that same membership check, so a dimension without a column of
    its own floors every column that does exist and no door can drift into disagreeing about what a
    validated column means. An ``unvalidated_dimensions`` named in ``columns`` carries
    :meth:`DeliveryGateResult.unvalidated_cell`, every gated dimension that did not validate (blank
    when none), so a reader can always recover the gate's full outcome behind a floored validity
    column without a second spelling of it. ``acknowledged_by``/``acknowledgement_reason`` named in
    ``columns`` carry ``gate.acknowledged_by``/``gate.acknowledgement_reason`` verbatim (``None``
    when nothing was acknowledged): every door whose own column list names them carries the pair,
    the phenology writer and the two count writers (``export_detection_csv``,
    ``export_aggregated_csv``) alike.
    """
    if asserted and asserted.get("produced_at") is not None:
        raise ValueError(
            "delivered_tail: asserted carries its own produced_at; produced_at is this "
            "composition's own write-time fact, never a caller-supplied one."
        )
    values = delivered_provenance(asserted, bindings, columns=columns)
    if "produced_at" in columns:
        values["produced_at"] = datetime.now(timezone.utc).isoformat()
    owned = tuple(dim for dim, col in _DIMENSION_TO_COLUMN.items() if col in columns)
    for dim in owned:
        values[_DIMENSION_TO_COLUMN[dim]] = gate.column_stamp(dim, own_column=owned)
    if "unvalidated_dimensions" in columns:
        values["unvalidated_dimensions"] = gate.unvalidated_cell()
    if "acknowledged_by" in columns:
        values["acknowledged_by"] = gate.acknowledged_by
    if "acknowledgement_reason" in columns:
        values["acknowledgement_reason"] = gate.acknowledgement_reason
    return values


DELIVERY_EVENTS_STORE = "delivery_events"
_DELIVERY_EVENTS_LOCATOR = RootedFileLocator(prefix=("delivery_events",), suffix=".json")
register_store(
    StoreDescriptor(
        name=DELIVERY_EVENTS_STORE,
        kind="record",
        key_fields=("event_id",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        enumerable=True,
        locator=_DELIVERY_EVENTS_LOCATOR,
    )
)
"""One record per completed delivery, enumerable, keyed by a content-derived id so two deliveries
never collide and neither ever needs to read the other before writing (``last_writer_wins``: each
key is written exactly once, so there is nothing to compare-and-set against)."""


def delivery_events_scope(project_root: str | Path | None = None) -> Path:
    """Where a project's delivery-event records live: ``<root>/.tcip/state``.

    Scoped under the project root the same way ``operationalizations_scope`` is
    (``operationalization.py``'s own pattern): this store is the project's enumerable document of
    what shipped, a different concern from the dataset-scoped audit-log line
    ``record_delivery_binding_event`` already writes below, over a scope of its own.
    """
    if project_root is not None:
        return Path(project_root) / ".tcip" / "state"
    from tcip_mcp.project_paths import resolve_state

    return resolve_state(Path(".tcip") / "state")


def delivery_event_key(scope: str | Path, event_id: str) -> Key:
    """One delivery event's record, addressed by its own content-derived id."""
    return Key(DELIVERY_EVENTS_STORE, str(scope), (event_id,))


DELIVERY_SUPERSESSIONS_STORE = "delivery_supersessions"
_DELIVERY_SUPERSESSIONS_LOCATOR = RootedFileLocator(
    prefix=("delivery_supersessions",), suffix=".json")
register_store(
    StoreDescriptor(
        name=DELIVERY_SUPERSESSIONS_STORE,
        kind="record",
        key_fields=("event_id",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_DELIVERY_SUPERSESSIONS_LOCATOR,
    )
)
"""One record per superseded delivery event, keyed by the superseded event's own id (so an event
can carry at most one supersession); ``concurrency="cas"`` since ``supersede_delivery`` writes
create-only (``expect=Version.ABSENT``) rather than overwriting a standing withdrawal."""


def delivery_supersession_key(scope: str | Path, event_id: str) -> Key:
    """The supersession filed against ``event_id``, if any: same scope as the event it names."""
    return Key(DELIVERY_SUPERSESSIONS_STORE, str(scope), (event_id,))


def load_delivery_supersessions(project_root: str | Path | None = None) -> dict[str, dict]:
    """Every delivery-event id under this project that carries a supersession, mapped to its own
    stored record: the input :func:`~tcip_mcp.pipelines.delivery_events_schema.with_supersessions`
    joins against a delivery-events listing, read once here rather than per event."""
    scope = delivery_events_scope(project_root)
    out: dict[str, dict] = {}
    for key in tcip_store.keys(DELIVERY_SUPERSESSIONS_STORE, str(scope)):
        record = tcip_store.read(key, default=None)
        if isinstance(record, dict):
            out[key.parts[-1]] = record
    return out


class DeliveryEventShapeError(ValueError):
    """One project's ``delivery_events`` record does not validate against ``DeliveryEventRecord``.

    ``event_id`` names the offending record when it could be read from the raw document, ``None``
    when the document is not even a dict."""

    def __init__(self, message: str, *, event_id: str | None) -> None:
        super().__init__(message)
        self.event_id = event_id


def _validated_delivery_event(record: Any, event_id: str | None, scope: Path) -> None:
    """Validate one stored ``delivery_events`` record against
    :class:`~tcip_mcp.pipelines.delivery_events_schema.DeliveryEventRecord`, raising
    :class:`DeliveryEventShapeError` naming ``event_id`` on a shape error. The one check
    :func:`read_delivery_events` and :func:`read_one_delivery_event` both run, so a record
    either reader meets refuses the same way rather than one tolerating what the other would
    refuse."""
    from pydantic import ValidationError

    from tcip_mcp.pipelines.delivery_events_schema import (
        DeliveryEventRecord,
        validation_error_detail,
    )

    try:
        DeliveryEventRecord.model_validate(record)
    except ValidationError as exc:
        raise DeliveryEventShapeError(
            f"delivery event {event_id!r} under {scope} does not validate against the "
            f"current delivery_events shape: {validation_error_detail(exc)}; no operator door "
            "rewrites an existing delivery_events record, so this project's stored events must "
            "be corrected to the current shape before they can be read",
            event_id=event_id,
        ) from exc


def read_delivery_events(project_root: str | Path | None = None) -> list[dict]:
    """Every ``delivery_events`` record stored under this project, each validated against
    :class:`~tcip_mcp.pipelines.delivery_events_schema.DeliveryEventRecord`.

    Raises :class:`DeliveryEventShapeError`, naming the offending ``event_id``, on the first
    stored record that does not validate, rather than silently dropping or half-trusting it. The
    Results tab's delivery panel (``routes/results.py``'s ``list_delivery_events``) and
    :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping._citing_delivery_event_ids` both read
    delivery events through this one function, so a shape refusal reads the same wherever it is
    met, and a rebuild's own citing-events check can never silently skip a record it cannot
    decode.
    """
    scope = delivery_events_scope(project_root)
    records: list[dict] = []
    for key in tcip_store.keys(DELIVERY_EVENTS_STORE, str(scope)):
        record = tcip_store.read(key, default=None)
        event_id = record.get("event_id") if isinstance(record, dict) else None
        _validated_delivery_event(record, event_id, scope)
        records.append(record)
    return records


def read_one_delivery_event(project_root: str | Path | None, event_id: str) -> dict | None:
    """One ``delivery_events`` record by its own id, validated against
    :class:`~tcip_mcp.pipelines.delivery_events_schema.DeliveryEventRecord` the same way
    :func:`read_delivery_events` validates every record it lists.

    ``None`` when nothing is stored under ``event_id``; raises :class:`DeliveryEventShapeError`
    when a stored record does not validate. ``supersede_delivery`` reads the event it supersedes
    and any replacement event through this, so it never quietly supersedes a record the Results
    tab's panel would refuse to list.
    """
    scope = delivery_events_scope(project_root)
    record = tcip_store.read(delivery_event_key(scope, event_id), default=None)
    if record is None:
        return None
    _validated_delivery_event(record, event_id, scope)
    return record


def _delivery_event_id(door: str, output_path: str | None, now: str) -> str:
    """A stable, Windows-safe id for one delivery event: a hex digest carries no colon and no
    timestamp-collision risk a caller-supplied nonce would otherwise need, and is computed here
    rather than accepted from a caller so no door can name its own event twice."""
    return hashlib.sha256(f"{door}|{output_path}|{now}".encode()).hexdigest()


def _delivered_file_sha256(output_path: str | None) -> str | None:
    """The delivered file's own digest, read after the writer already wrote it: ``None`` only for
    a fileless event (the two ``phenology_measurement`` calls below, which compute a curve rather
    than write a CSV). A stated ``output_path`` the writer just produced that cannot be read back
    is a failed delivery, not a fileless one, and raises rather than recording a blank digest for
    bytes that were supposedly just written."""
    if not output_path:
        return None
    return hashlib.sha256(Path(output_path).read_bytes()).hexdigest()


def record_delivery_binding_event(
    door: str,
    output_path: str | None,
    pred_dirs: Sequence[str] | None,
    bindings: Mapping[str, StampBinding],
    *,
    measurement_documents: Sequence[str],
    scale_document: str | None,
    acknowledgement: Acknowledgement | None,
    trait: str | None = None,
    delivery_kind: str | None = None,
    project_root: str | Path | None = None,
    plant_mapping: dict | None = None,
) -> bool:
    """Record what verification found for each bucket a delivery read, in that dataset's own log.

    The ``@audited`` decorator records a door's arguments and its status, which is not what a later
    reader of a delivered number needs: they need which buckets stood behind it, which of their
    claims were answered for, and by which records. That cannot be obtained by changing the
    decorator's inputs, so a door emits it alongside. The event files against the dataset root the
    buckets share, since it describes records that travel with the data; a delivery whose buckets
    share no dataset root files against the platform log instead. This mutation already committed
    (the artifact shipped before this call runs) with no tool body of its own for ``@audited`` to
    bracket, so a dropped append raises ``AuditEntryNotWritten`` rather than passing silently,
    per the platform's audited-mutation invariant, instead of leaving the artifact's own delivery
    unrecorded on the platform's canonical log.

    Beside that dataset-scoped audit line, a project-scoped ``delivery_events`` record is also
    written, carrying the real per-bucket ``StampBinding`` evidence this call already computed
    rather than a coarse gate stamp, so a delivery can be found again by the same ``trait``/
    ``delivery_kind`` vocabulary an operationalization or a trait-spec statement is found by. This
    second write, unlike the audit line above, stays best-effort on its own terms: a delivery
    event is a fact recorded after the artifact it describes already shipped, not a confirmation,
    so a lost line here is a provenance gap surfaced by a warning, never a reason to make an
    already-completed delivery look retryable.

    ``project_root`` names the project this event belongs to, for a caller (a web route) whose
    process can serve more than one project: an MCP tool leaves it unset and gets the process-pinned
    root, correct since that process serves exactly one project, but a web route already holding its
    own guarded, resolved root passes it explicitly, the same divergence D11 already closes for the
    operationalization record.

    ``measurement_documents`` names which sidecar document(s) the delivery's own gate reconciled
    (the count-delivery door's single-element statement, or the phenology doors' fixed
    ``["operating_point", "classifier_operating_point"]``), and ``scale_document`` names
    ``"resolve_scale"`` when the delivery also rests on a physical scale, ``None`` otherwise. Both
    are required, never defaulted, so a caller cannot silently omit what its own gate actually
    reconciled.

    ``acknowledgement`` is the breeder's own act of shipping this delivery unvalidated (the same
    ``Acknowledgement`` a passing ``check_delivery_gate`` call may have taken), or ``None`` when
    nothing needed acknowledging. Required, never defaulted, for the same reason
    ``measurement_documents`` is: a caller cannot silently omit whether this delivery rests on a
    breeder's acknowledgement. Recorded as the record's own ``acknowledged_by``/
    ``acknowledgement_reason`` fields, both present and null together when ``acknowledgement`` is
    ``None``.

    ``plant_mapping`` is the delivery's own plant-mapping binding, door-conditional: the phenology
    doors always pass a walked mapping's own disclosure (name, project and dataset roots, the
    record's digest, its per-date capture identity, and the two unverified disclosures),
    ``deliver_per_plant_csv`` passes one only when its own caller verified a named mapping against
    the buckets this delivery reads, ``deliver_orthomosaic_plant_counts`` passes a whole-raster
    frame's own registry disclosure instead (no walked mapping exists for it), and every other
    delivery door passes ``None`` since none reads a mapping or a registry.

    The assembled record is validated against ``DeliveryEventRecord`` (``delivery_events_schema.py``,
    the same shape ``list_delivery_events`` reads back through) before the write. A shape violation
    is a deterministic defect in the caller, never an environmental failure the way an unwritable
    disk is, so it raises ``pydantic.ValidationError`` to the caller instead of falling into this
    call's own best-effort warning path below, which covers only the project-scoped store write.

    Raises ``AuditEntryNotWritten`` (``tcip_mcp.audit``) when the dataset-scoped audit line
    itself cannot be appended: the caller already delivered the artifact and cannot un-deliver
    on this, but it must not read a caller-visible success out of the project-scoped record's
    own return value below with the platform's canonical log silently missing the event.

    Returns whether the project-scoped ``delivery_events`` write actually landed: ``True`` on a
    successful store write, ``False`` when that best-effort write failed (logged, never raised).
    A delivered file already exists by the time this runs, so a caller cannot un-deliver on a
    ``False`` here; it can only disclose the gap to whoever asked for the delivery.
    """
    from tcip_mcp.audit import record_event_or_raise

    roots = {_dataset_root_of(Path(d)) for d in (pred_dirs or [])}
    scope = roots.pop() if len(roots) == 1 else None
    record_event_or_raise(
        door,
        {"output_path": output_path, "pred_dirs": list(pred_dirs or [])},
        scope=scope,
        verified_buckets={
            bucket: {"verified": b.ok,
                     "record": f"{b.experiment_id}:{b.record_digest}" if b.claimed and b.ok else "",
                     "note": b.note}
            for bucket, b in bindings.items()
        },
        record_digests=sorted({b.record_digest for b in bindings.values()
                               if b.ok and b.claimed and b.record_digest}),
    )

    now = datetime.now(timezone.utc).isoformat()
    event_id = _delivery_event_id(door, output_path, now)
    record = {
        "event_id": event_id,
        "trait": trait,
        "delivery_kind": delivery_kind,
        "door": door,
        "output_path": output_path,
        "output_sha256": _delivered_file_sha256(output_path),
        "measurement_documents": list(measurement_documents),
        "scale_document": scale_document,
        "acknowledged_by": acknowledgement.acknowledged_by if acknowledgement is not None else None,
        "acknowledgement_reason": acknowledgement.reason if acknowledgement is not None else None,
        "plant_mapping": plant_mapping,
        "documents": {
            bucket: {
                "ok": b.ok, "claimed": b.claimed, "experiment_id": b.experiment_id,
                "producing_experiment_id": b.producing_experiment_id,
                "checkpoint_sha256": b.checkpoint_sha256, "record_digest": b.record_digest,
                "note": b.note,
            }
            for bucket, b in bindings.items()
        },
        "produced_at": now,
    }
    from tcip_mcp.pipelines.delivery_events_schema import DeliveryEventRecord

    DeliveryEventRecord.model_validate(record)
    key = delivery_event_key(delivery_events_scope(project_root), event_id)
    try:
        tcip_store.replace(key, record)
    except Exception:
        logger.warning("Failed to write delivery_events record for door %r", door, exc_info=True)
        return False
    return True


def binding_notes_text(notes: Mapping[str, str]) -> str:
    """Render a reconciler's ``binding_notes`` as one refusal-ready line naming each bucket and why.

    The one join every delivery door renders a floored binding's notes through, so a refusal names
    the failing sidecar and the reason exactly once; a door that also reconciles the same buckets
    for its own bindings must not append a second, separately-derived copy of what another door's
    exception already carries.
    """
    return " ".join(f"{bucket}: {note}" for bucket, note in sorted(notes.items()) if note)


def _validity_rank(state: str | None, accepted: tuple[str, ...]) -> int:
    """Floor ordering: unvalidated (0) < a reference ``accepted`` recognizes for the dimension
    being reconciled (1). ``None`` = no assertion (never lowers). Kind-aware on purpose: a
    wrong-kind asserted string is not a real assertion about this dimension, so it floors the
    result rather than being silently ignored as if it outranked the on-disk state."""
    if state is None:
        return 99
    return 1 if state in accepted else 0


def _reconcile_validity(
    pred_dirs: list[str] | tuple[str, ...], *, asserted: str | None, document: str,
    trait: str | None, digest_memo: dict[str, str] | None = None,
) -> dict:
    """Floor a validity dimension against every bucket's on-disk sidecar, generalized.

    Shared by :func:`reconcile_operating_point_validity` and :func:`reconcile_classifier_validity`,
    the flooring logic (read on-disk, never trust a caller string, an asserted value may only lower
    the result) is identical for both dimensions; only which document is read differs, threaded in by
    the thin public wrappers below, which is also what says which parameter and which kind of
    reference that document's claim rests on.

    ``trait`` is the delivery's own registry trait, required (never defaulted) so a caller cannot
    silently skip stating it; passed straight through to :func:`verify_stamp_binding`, which compares
    it against the record's own trait only when it is not ``None``. Threaded through by the three
    measurement wrappers (count/ordinal/regression); :func:`reconcile_classifier_validity` passes
    ``None``, since the classifier's own trait comparison already runs once, in
    :func:`bind_classifier_validity`, and duplicating it here would lose the note the delivery doors
    render from that function's own return.

    A stamp whose claim no validation record answers for floors here, with the reason recorded per
    bucket: a claim a bucket wrote for itself is not evidence, and the check lives inside this shared
    body so no delivery door can reach a validated result without it.

    Returns ``{validated, on_disk_validated, missing_sidecars, unvalidated_buckets, binding_notes,
    bindings, conf, per_bucket}``. ``bindings`` carries one verified result per bucket read, so a
    delivery door stamps its provenance columns from the verification this body already ran rather
    than repeating it.
    """
    param_key, validation_kind = _DOCUMENT_PARAM[document]
    per_bucket: dict[str, str] = {}
    missing: list[str] = []
    unvalidated: list[str] = []
    binding_notes: dict[str, str] = {}
    bindings: dict[str, StampBinding] = {}
    refs: set[str] = set()
    confs: list[float] = []
    all_validated = bool(pred_dirs)
    accepted = accepted_references(validation_kind)
    memo = digest_memo if digest_memo is not None else {}
    for d in pred_dirs:
        sc = _read_sidecar(d, document)
        if sc is None:
            missing.append(str(d))
            per_bucket[str(d)] = VALIDATED_FALSE
            bindings[str(d)] = StampBinding(ok=True, claimed=False)
            all_validated = False
            continue
        ref = _sidecar_reference(sc, param_key=param_key, validation_kind=validation_kind)
        binding = verify_stamp_binding(sc, d, document=document, digest_memo=memo, trait=trait)
        bindings[str(d)] = binding
        if not binding.ok:
            binding_notes[str(d)] = binding.note
            ref = VALIDATED_FALSE
        per_bucket[str(d)] = ref
        if ref in accepted:
            refs.add(ref)
            conf_val = ((sc.get("operating_point") or {}).get(param_key) or {}).get("value")
            if isinstance(conf_val, (int, float)):
                confs.append(float(conf_val))
        else:
            unvalidated.append(str(d))
            all_validated = False

    if all_validated and refs:
        on_disk = VALIDATED_HELD_OUT if VALIDATED_HELD_OUT in refs else next(iter(refs))
    else:
        on_disk = VALIDATED_FALSE

    # Floor: the asserted string may only lower the on-disk result (prefer on-disk).
    validated = (on_disk if _validity_rank(asserted, accepted) >= _validity_rank(on_disk, accepted)
                 else VALIDATED_FALSE)
    return {
        "validated": validated,
        "on_disk_validated": all_validated and bool(refs),
        "missing_sidecars": missing,
        "unvalidated_buckets": unvalidated,
        "binding_notes": binding_notes,
        "bindings": bindings,
        "conf": (confs[0] if len(set(confs)) == 1 else None),
        "per_bucket": per_bucket,
    }


def reconcile_operating_point_validity(
    pred_dirs: list[str] | tuple[str, ...], *, trait: str, asserted: str | None = None,
    digest_memo: dict[str, str] | None = None,
) -> dict:
    """Floor the count operating-point validity against every bucket's ``operating_point.json``.

    The delivery gate must not trust a caller's asserted string: it reads each prediction bucket's
    on-disk sidecar and takes the floor of asserted-vs-on-disk. A missing/unreadable sidecar, or any
    bucket stamped ``validated=false``, floors the whole curve to ``false``, never a crash. See
    :func:`_reconcile_validity` for the shared mechanism.

    ``trait`` is the trait this delivery is actually being produced for, required so a count claim
    earned for one trait cannot silently answer for a delivery of another; compared against each
    bucket's own record via :func:`verify_stamp_binding`.
    """
    return _reconcile_validity(
        pred_dirs, asserted=asserted, document="operating_point", trait=trait,
        digest_memo=digest_memo,
    )


def reconcile_classifier_validity(
    pred_dirs: list[str] | tuple[str, ...], *, asserted: str | None = None,
    digest_memo: dict[str, str] | None = None,
) -> dict:
    """Floor the classifier validity against every bucket's ``classifier_operating_point.json``.

    Structurally the same reconciliation :func:`reconcile_operating_point_validity` performs for the
    count operating point, the same function, parameterized to a different sidecar file and param
    key, never a hand-written sibling. A bucket with no persisted classifier-calibration run floors
    to ``false``: there is no legitimate way to earn a classifier-validated stamp without one, so this
    never falls back to a caller-asserted string.

    Threads no ``trait`` into the shared mechanism (unlike the three measurement reconcilers): the
    classifier's own trait comparison already runs once, in :func:`bind_classifier_validity`, and
    duplicating it here would lose the breeder-facing note the delivery doors render from that
    function's own return.
    """
    return _reconcile_validity(
        pred_dirs, asserted=asserted, document="classifier_operating_point", trait=None,
        digest_memo=digest_memo,
    )


def reconcile_ordinal_validity(
    pred_dirs: list[str] | tuple[str, ...], *, trait: str, asserted: str | None = None,
    digest_memo: dict[str, str] | None = None,
) -> dict:
    """Floor the ordinal compensating-error validity against every bucket's
    ``ordinal_operating_point.json``.

    Structurally the same reconciliation :func:`reconcile_classifier_validity` performs for the
    classifier dimension, the same shared mechanism, parameterized to a different sidecar file and
    param key. A bucket with no persisted ordinal-calibration run floors to ``false``: there is no
    legitimate way to earn an ordinal-validated stamp without one.

    ``trait`` is required, the same measurement-reconciler trait binding
    :func:`reconcile_operating_point_validity` performs.
    """
    return _reconcile_validity(
        pred_dirs, asserted=asserted, document="ordinal_operating_point", trait=trait,
        digest_memo=digest_memo,
    )


def reconcile_regression_validity(
    pred_dirs: list[str] | tuple[str, ...], *, trait: str, asserted: str | None = None,
    digest_memo: dict[str, str] | None = None,
) -> dict:
    """Floor the regression compensating-error validity against every bucket's
    ``regression_operating_point.json``. Same shape as :func:`reconcile_ordinal_validity`, for the
    regression dimension's own sidecar/param key, including the required ``trait`` binding.
    """
    return _reconcile_validity(
        pred_dirs, asserted=asserted, document="regression_operating_point", trait=trait,
        digest_memo=digest_memo,
    )


def tile_size_gate_flag(operating_point: dict | None) -> str | None:
    """The tile-geometry dimension's delivery-gate flag for one resolved operating point.

    ``operating_point`` is a bundle's ``to_provenance()["operating_point"]`` mapping, the same shape
    a run returns in-memory and ``run_inference`` persists into ``operating_point.json``.
    Returns ``None`` when tiling was not operative for that run (``resolve_tile_size_param`` only
    sets ``requires_validation`` when ``tiled``), so an untiled run's tile_size never manufactures a
    refusal over a dimension that was never operative. Otherwise it returns the reference the tile
    scale actually cleared, or ``VALIDATED_FALSE``.

    Reads the tile_size param's own recorded reference and nothing else: the sidecar's top-level
    ``validated`` bool is the whole bundle's shippability, so consulting it here would report a
    genuinely persisted tile geometry as fabricated whenever some other dimension (conf) was the
    thing that failed. Every delivery door that gates on tile geometry resolves it through this one
    function so the doors cannot drift into disagreeing about when a tile scale is trustworthy.
    """
    prov = (operating_point or {}).get("tile_size") or {}
    if not prov.get("requires_validation"):
        return None
    ref = prov.get("validated_against")
    return ref if ref in accepted_references("geometry") else VALIDATED_FALSE


def reconcile_tile_size_validity(
    pred_dirs: list[str] | tuple[str, ...], *, digest_memo: dict[str, str] | None = None,
) -> dict:
    """Floor the tile-geometry dimension across every prediction bucket's ``operating_point.json``.

    The sidecar-reading counterpart of :func:`tile_size_gate_flag`, for the delivery doors that
    assemble a phenotype from already-written prediction buckets rather than from a live run. A
    delivery spanning several buckets is only as grounded as its least-grounded tiled bucket, so any
    operative bucket whose tile scale has no real basis floors the whole dimension to
    ``VALIDATED_FALSE``; when the cleared references differ across buckets the weakest member of
    :data:`GEOMETRY_REFERENCE_STRENGTH` present is what travels, never a stronger one some other
    bucket earned.

    Returns ``{operative, validated, per_bucket, unvalidated_buckets, binding_notes}``. ``operative``
    is False (and ``validated`` ``None``) when no bucket ran tiled, in which case the caller adds
    nothing to its gate. A bucket with no readable sidecar contributes nothing here; that bucket's
    missing stamp already floors the count operating point via
    :func:`reconcile_operating_point_validity`. A tiled bucket whose stamp claims a validation no
    record answers for floors here too: the tile geometry rides in the same stamp file as the count
    claim, so the two stand or fall together.
    """
    per_bucket: dict[str, str] = {}
    unvalidated: list[str] = []
    binding_notes: dict[str, str] = {}
    refs: set[str] = set()
    accepted = accepted_references("geometry")
    memo = digest_memo if digest_memo is not None else {}
    for d in pred_dirs:
        sc = read_operating_point_sidecar(d)
        flag = tile_size_gate_flag((sc or {}).get("operating_point"))
        if flag is None:
            continue
        binding = verify_stamp_binding(sc, d, document="operating_point", digest_memo=memo)
        if not binding.ok:
            binding_notes[str(d)] = binding.note
            flag = VALIDATED_FALSE
        per_bucket[str(d)] = flag
        if flag in accepted:
            refs.add(flag)
        else:
            unvalidated.append(str(d))
    if not per_bucket:
        return {"operative": False, "validated": None, "per_bucket": {}, "unvalidated_buckets": [],
                "binding_notes": binding_notes}
    if unvalidated:
        validated = VALIDATED_FALSE
    else:
        # The weakest member of GEOMETRY_REFERENCE_STRENGTH present travels (last match, strongest
        # first).
        validated = next(ref for ref in reversed(GEOMETRY_REFERENCE_STRENGTH) if ref in refs)
    return {"operative": True, "validated": validated, "per_bucket": per_bucket,
            "unvalidated_buckets": unvalidated, "binding_notes": binding_notes}


def reconcile_claim_scope_validity(
    pred_dirs: list[str] | tuple[str, ...], *, digest_memo: dict[str, str] | None = None,
) -> dict:
    """Floor the claim-scope dimension across every prediction bucket's ``operating_point.json``.

    The sidecar-reading counterpart of the export-time claim-scope check, for the delivery doors
    that assemble a phenotype from already-written buckets. A bucket whose sidecar records no
    ``claim_scope_validated`` is skipped, the same way :func:`reconcile_tile_size_validity` skips an
    untiled bucket: the dimension was never operative for it, so nothing here manufactures a
    refusal over it. Once any bucket does record one, the dimension is operative for the whole
    delivery, and a bucket whose recorded value is not a member of :data:`CLAIM_SCOPE_REFERENCES`
    floors it to ``VALIDATED_FALSE``.

    Returns ``{operative, validated, per_bucket, unvalidated_buckets, binding_notes}``, the same
    shape :func:`reconcile_tile_size_validity` returns, and floors on an unanswered-for stamp for the
    same reason: the claim scope rides in the same stamp file as the count claim.
    """
    per_bucket: dict[str, str] = {}
    unvalidated: list[str] = []
    binding_notes: dict[str, str] = {}
    refs: set[str] = set()
    memo = digest_memo if digest_memo is not None else {}
    for d in pred_dirs:
        sc = read_operating_point_sidecar(d)
        recorded = (sc or {}).get("claim_scope_validated")
        if recorded is None:
            continue
        flag = recorded if recorded in CLAIM_SCOPE_REFERENCES else VALIDATED_FALSE
        binding = verify_stamp_binding(sc, d, document="operating_point", digest_memo=memo)
        if not binding.ok:
            binding_notes[str(d)] = binding.note
            flag = VALIDATED_FALSE
        per_bucket[str(d)] = flag
        if flag in CLAIM_SCOPE_REFERENCES:
            refs.add(flag)
        else:
            unvalidated.append(str(d))
    if not per_bucket:
        return {"operative": False, "validated": None, "per_bucket": {}, "unvalidated_buckets": [],
                "binding_notes": binding_notes}
    validated = VALIDATED_FALSE if unvalidated else sorted(refs)[0]
    return {"operative": True, "validated": validated, "per_bucket": per_bucket,
            "unvalidated_buckets": unvalidated, "binding_notes": binding_notes}


def reconcile_scale_validity(
    pred_dirs: list[str] | tuple[str, ...], *, unit: str, trait: str, images_dir: str | Path,
    capture_id: str | None = None, asserted: str | None = None,
    digest_memo: dict[str, str] | None = None,
) -> dict:
    """Floor the physical-scale dimension across every prediction bucket's ``resolve_scale.json``.

    Structurally the sidecar-reading counterpart of :func:`reconcile_tile_size_validity`, but the two
    dimensions differ in what "not operative" means: tiling is legitimately absent from an untiled
    run (that bucket's own ``operating_point.json`` records it as non-gating, read straight from a
    file every bucket always has), so an untiled bucket is skipped rather than floored. A physical
    scale has no such always-present file to read "not applicable" from: whether it is relevant at
    all is a fact about the *trait* (does the delivery carry a dimensional value), which only the
    caller (holding the results) can know. A caller that calls this at all has already decided the
    dimension is relevant, so once called, a bucket with no readable ``resolve_scale.json`` floors
    the whole dimension exactly as a missing ``operating_point.json`` floors the count operating
    point, never silently skipped the way an untiled bucket's tile_size is.

    ``capture_id``, when given, is the delivery's own requested scope. A bucket's recorded scale may
    itself be scoped to a single capture (``resolve_scale``'s own ``capture_id``, real for a handheld
    standoff that varies image to image within one dataset): a caller-supplied ``capture_id`` that
    disagrees with a bucket's recorded one must not silently validate, the same principle
    :func:`bind_classifier_validity` applies to a trait mismatch, so that bucket floors to
    ``VALIDATED_FALSE`` even though its own sidecar says validated. A bucket whose scale was never
    capture-scoped (recorded ``capture_id`` is ``None``) applies to any capture, since nothing in it
    claims to be capture-specific. A caller that passes no ``capture_id`` is not asking for
    cross-capture scoping to be checked at all, so no bucket is floored on this basis.

    ``unit`` is the linear basis the delivery's own value_key implies
    (:func:`~tcip_mcp.pipelines.measurement.mask_geometry.unit_from_value_key`'s second element): a
    bucket whose stamped ``operating_point.scale.unit`` differs floors, naming both, so a scale
    stamped in centimetres cannot clear a delivery in millimetres. ``trait`` is the delivery's own
    registry trait, threaded into :func:`verify_stamp_binding` the same way the three measurement
    reconcilers thread it, so a scale earned for another trait floors. ``images_dir`` is the
    delivery's own images directory, threaded into :func:`verify_stamp_binding` to recompute the
    imagery digest a scale claim's covered bucket names.

    ``asserted``, mirroring :func:`reconcile_operating_point_validity`, may only lower the on-disk
    result, never raise it: a caller string can never launder an ungrounded scale into a shippable one.

    Returns ``{operative, validated, per_bucket, unvalidated_buckets, binding_notes}``, the same
    shape :func:`reconcile_tile_size_validity` returns. ``operative`` is False (``validated``
    ``None``) only when ``pred_dirs`` itself is empty, there is nothing to reconcile against. A
    bucket whose ``resolve_scale.json`` no validation record answers for floors here, whether the
    stamp was hand-authored or produced by :func:`~tcip_mcp.tools.scale_tools.calibrate_physical_scale`.
    """
    if not pred_dirs:
        return {"operative": False, "validated": None, "per_bucket": {}, "unvalidated_buckets": [],
                "binding_notes": {}}
    param_key, validation_kind = _DOCUMENT_PARAM["resolve_scale"]
    accepted = accepted_references(validation_kind)
    per_bucket: dict[str, str] = {}
    unvalidated: list[str] = []
    binding_notes: dict[str, str] = {}
    refs: set[str] = set()
    memo = digest_memo if digest_memo is not None else {}
    for d in pred_dirs:
        sc = read_scale_sidecar(d)
        ref = _sidecar_reference(sc, param_key=param_key, validation_kind=validation_kind)
        if ref in accepted and capture_id is not None:
            # ref clears only when _sidecar_reference found sc truthy and validated.
            assert sc is not None
            recorded = ((sc.get("operating_point") or {}).get(param_key) or {}).get("capture_id")
            if recorded is not None and recorded != capture_id:
                ref = VALIDATED_FALSE
        if ref in accepted:
            assert sc is not None
            stamped_unit = ((sc.get("operating_point") or {}).get(param_key) or {}).get("unit")
            if stamped_unit != unit:
                ref = VALIDATED_FALSE
                binding_notes[str(d)] = (
                    f"resolve_scale.json at {str(d)!r} is stamped in {stamped_unit!r}, not the "
                    f"delivered unit {unit!r}; a scale stamped in one linear unit cannot clear a "
                    "delivery in another."
                )
        binding = verify_stamp_binding(sc, d, document="resolve_scale", trait=trait,
                                       digest_memo=memo, images_dir=images_dir)
        if not binding.ok:
            binding_notes[str(d)] = binding.note
            ref = VALIDATED_FALSE
        per_bucket[str(d)] = ref
        if ref in accepted:
            refs.add(ref)
        else:
            unvalidated.append(str(d))
    on_disk = VALIDATED_FALSE if unvalidated or not refs else next(iter(refs))
    validated = (on_disk if _validity_rank(asserted, accepted) >= _validity_rank(on_disk, accepted)
                 else VALIDATED_FALSE)
    return {"operative": True, "validated": validated, "per_bucket": per_bucket,
            "unvalidated_buckets": unvalidated, "binding_notes": binding_notes}


def bind_classifier_validity(
    classifier_state: str | None,
    classifier_dirs: list[str] | tuple[str, ...] | None,
    producing_dirs: list[str] | tuple[str, ...],
    *,
    trait: str,
    digest_memo: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    """Floor a reconciled classifier stamp to the delivery it is being used to validate.

    Unlike the count dimension (which reconciles from the same buckets it delivers),
    :func:`reconcile_classifier_validity` alone cannot see whether a genuinely-validated stamp was
    calibrated for an unrelated model or trait, it reads only the validity field. What the stamp must
    agree with is checked through :func:`verify_stamp_binding`, so the trait comparison is the same
    one every document gets and the run being compared is the run each stamp's own record names.

    The producing runs are taken from the count buckets' verified bindings, never from what those
    sidecars declare for themselves: an id a bucket wrote beside its own predictions is the id
    whoever wrote the bucket chose, so a set built from it would let one file decide what the other
    is checked against. A bucket whose binding does not hold contributes no id at all.

    Both sides recording no producing run is agreement, not a gap to be tolerated: a bespoke or
    unregistered checkpoint has no run to name, and its documents legitimately live in calibration
    experiments of their own.

    Returns ``(state, note)``, ``state`` floored to ``VALIDATED_FALSE`` on a mismatch, and a
    breeder-readable ``note`` naming which sidecar failed and why (empty when nothing was floored).

    Every delivery door must call this after reconciling, not just the one that first needed it:
    ``deliver_phenology_milestones`` and the web Results doors share it rather than each composing the flag,
    so the two surfaces cannot disagree about what a classifier stamp means.
    """
    if classifier_state in (None, VALIDATED_FALSE):
        return classifier_state, ""
    memo = digest_memo if digest_memo is not None else {}
    producing_experiment_ids: set[str] = set()
    for d in producing_dirs:
        binding = verify_stamp_binding(
            read_operating_point_sidecar(d), d, document="operating_point", digest_memo=memo)
        if binding.ok and binding.claimed and binding.producing_experiment_id:
            producing_experiment_ids.add(binding.producing_experiment_id)
    for d in (classifier_dirs or []):
        binding = verify_stamp_binding(
            read_classifier_operating_point_sidecar(d), d,
            document="classifier_operating_point", trait=trait, digest_memo=memo)
        if not binding.ok:
            return VALIDATED_FALSE, binding.note
        stamped_exp = binding.producing_experiment_id
        if producing_experiment_ids and stamped_exp not in producing_experiment_ids:
            return VALIDATED_FALSE, (
                f"classifier_operating_point.json at {d!r} records producing run {stamped_exp!r}, "
                f"not the run behind this delivery's counts ({sorted(producing_experiment_ids)}), "
                "so the stamp is not trusted here. Re-calibrate the classifier over the "
                "predictions this delivery counts.")
    return classifier_state, ""


# --- the delivery gate (one refuse-or-stamp check shared by every phenotype-delivery door) ---

@dataclass(frozen=True)
class Acknowledgement:
    """The breeder's own act of shipping an unvalidated delivery: who did it and why.

    In production, this is built only by the web results routes
    (``tcip_web.routes.results.export_csv`` and ``export_count_csv``) once a request names a real
    user; an agent acknowledging its own unvalidated output would be attesting to a breeder's
    judgment it never obtained. That is a convention at this Python seam, not a rail the type enforces: nothing here
    stops another caller from constructing one. Whoever builds an ``Acknowledgement`` is recorded
    by name and reason regardless, so a caller outside the web route still leaves an attributed
    trail rather than a silent bare number. ``acknowledged_by`` and ``reason`` are both required
    non-empty: ``acknowledged_by`` is the one thing the record carries that says who, ``reason``
    the one thing that says why.
    """

    acknowledged_by: str
    reason: str

    def __post_init__(self) -> None:
        if not self.acknowledged_by.strip():
            raise ValueError(
                "Acknowledgement.acknowledged_by is required non-empty: it is the one thing the "
                "record carries that says who."
            )
        if not self.reason.strip():
            raise ValueError(
                "Acknowledgement.reason is required non-empty: it is the one thing the record "
                "carries that says why."
            )


@dataclass(frozen=True)
class DeliveryGateResult:
    """Outcome of the delivery gate: whether the deliverable may be written, and how to stamp it."""

    ok: bool
    unvalidated: tuple[str, ...]  # dimensions whose validity is not a shippable reference
    stamp: dict[str, str]  # per-dimension validity to stamp onto the deliverable
    reason: str = ""  # generic refusal message when not ok
    acknowledged_by: str | None = None  # who acknowledged, when an Acknowledgement cleared this gate
    acknowledgement_reason: str | None = None  # why, from that same Acknowledgement

    def column_stamp(self, dimension: str, *, own_column: tuple[str, ...] = ()) -> str:
        """The value the deliverable's column for ``dimension`` carries.

        Not the same thing as ``stamp[dimension]``, which is only that one dimension's own cleared
        reference. A column stands for the trustworthiness of the number beside it, and an
        acknowledgement (or a staging escape) still lets an ungrounded dimension reach the writer,
        so stamping this dimension's own (possibly real) reference alone would report a partly
        acknowledged provisional delivery as fully validated. Every gated dimension without a
        column of its own therefore floors this one. Name in ``own_column`` the dimensions the
        deliverable does stamp into columns of their own; those report themselves and never floor
        this one.

        Owned here rather than re-derived per door so the doors cannot drift into disagreeing about
        what a validated column means.
        """
        if any(name not in own_column for name in self.unvalidated):
            return VALIDATED_FALSE
        return self.stamp[dimension]

    def owned_column_stamp(self) -> dict[str, str]:
        """Every dimension's ``column_stamp``, using :data:`_DIMENSION_TO_COLUMN`'s own membership
        as ``own_column``: exactly what a delivered tail carrying a column for every one of those
        dimensions would show per dimension. For a reader with no CSV columns of its own (the
        Results tab), this is the one way to show a per-dimension validity that cannot read
        stronger than the file the same gate result would produce.
        """
        owned = tuple(_DIMENSION_TO_COLUMN)
        return {dim: self.column_stamp(dim, own_column=owned) for dim in self.stamp}

    def effective_acknowledgement(self) -> Acknowledgement | None:
        """The acknowledgement this gate result actually applied, or ``None`` when every dimension
        validated and nothing needed one, regardless of what a caller passed in.

        Built from ``acknowledged_by``/``acknowledgement_reason`` (already ``None``/``None`` when
        the gate discarded a caller's acknowledgement as unneeded) rather than from a caller's own
        ``Acknowledgement`` object, so a writer recording the delivery event and a writer composing
        the CSV tail from the same gate result can never disagree about whether this delivery
        rested on one.
        """
        if self.acknowledged_by is None:
            return None
        return Acknowledgement(
            acknowledged_by=self.acknowledged_by, reason=self.acknowledgement_reason or "")

    def unvalidated_cell(self) -> str:
        """Every gated dimension that did not validate, in the platform's delivered-list-cell
        convention (``;``-joined, blank when none): the one rendering a delivered tail's own
        ``unvalidated_dimensions`` column and a door's refusal response share, so the two never
        spell the same join two different ways."""
        return ";".join(self.unvalidated)


class DeliveryRefused(ValueError):
    """A writer's delivery-gate refusal, carrying the ``DeliveryGateResult`` it refused on.

    A door composing its own counts-bearing refusal response needs to tell this refusal apart
    from the writer's other, count-free raises (a withdrawn operationalization): a bare
    ``ValueError`` cannot be told apart without a second classification of the same writer's own
    raises, drifting from it the moment either changes. ``str(self)`` is the gate's reason plus
    every gated dimension's own reconciler's binding notes, for a caller content to just log or
    propagate the message. ``facts`` starts empty and is set by a count-delivery core once it
    catches this raise, to the counts-bearing facts the core already had in hand.
    """

    def __init__(self, gate: DeliveryGateResult, notes: str = "") -> None:
        super().__init__(f"{gate.reason} {notes}".rstrip())
        self.gate = gate
        self.notes = notes
        # A count-delivery core sets this to the facts it already had in hand when it caught this.
        self.facts: dict[str, Any] = {}


class CountDeliveryRefused(ValueError):
    """A count-delivery core's own refusal for anything other than the delivery gate
    (:class:`DeliveryRefused`) or the meaning door
    (``operationalization.OperationalizationRefused``): a missing or malformed bucket, a mismatched
    stamp, an unreadable plant registry, a raster identity mismatch. Carries whatever counts-bearing
    facts the core already had in hand at the point it refused, named the same way the tool's own
    ``{"error": ...}`` response and the web route's own structured detail body both name them.
    """

    def __init__(self, message: str, **facts: Any) -> None:
        super().__init__(message)
        self.facts = facts


STAGING_DIMENSIONS: tuple[str, ...] = ("tile_size", "claim_scope")
"""The dimensions ``allow_unvalidated_staging`` may clear on their own, with no breeder
acknowledgement: the two ``run_inference`` (and the doors that share its publish bracket) gates
before an expensive tiled pass, never a dimension a phenotype's own delivered value rests on
directly. Persisting a raw, honestly-stamped bucket at an unproven tile scale or claim scope is a
different act from delivering a phenotype from one; the flag exists for the former and never
reaches ``export_detection_csv``/``export_aggregated_csv``/the phenology writer, whose own
dimensions clear only through a real reference or a breeder's ``Acknowledgement``."""

_DIMENSION_REFERENCES: dict[str, tuple[str, ...]] = {
    "operating_point": _ACCEPTED_REFERENCES["annotations"],
    "classifier": _ACCEPTED_REFERENCES["annotations"],
    "tile_size": _ACCEPTED_REFERENCES["geometry"],
    "scale": _ACCEPTED_REFERENCES["physical"],
    "claim_scope": CLAIM_SCOPE_REFERENCES,
}
"""Which references clear which delivery-gate dimension, every row read from the kind's own
acceptance table so no reference-to-kind pairing is stated a second time. The operating_point and
classifier dimensions are annotations-kind even for dimensional and ordinal/regression traits:
every reconciler that feeds them resolves through a ``_DOCUMENT_PARAM`` entry declared
``"annotations"``, and a physical scale is its own ``"scale"`` dimension, never folded into
``"operating_point"``. A dimension cleared by nothing (an empty tuple) states a missing
prerequisite rather than a reference of any kind, see ``check_delivery_gate``'s
cleared-by-nothing refusal arm; no production dimension reaches that arm today
(``deliver_per_image_counts``'s in-memory pass with no ``predictions_dir`` now floors through
``export_detection_csv``'s own no-``pred_dirs`` operating_point floor instead), so this stays
documentation of a mechanism a future floor-only dimension can use, not a live assertion."""


def check_delivery_gate(
    flags: dict[str, str | None], *,
    acknowledgement: Acknowledgement | None = None,
    allow_unvalidated_staging: bool = False,
) -> DeliveryGateResult:
    """Refuse-or-stamp a phenotype delivery against the validity of each dimension it rests on.

    ``flags`` maps each dimension the deliverable depends on (e.g. ``"operating_point"``, the
    sole dimension for a continuous/ordinal trait with no conf op-point, or ``"classifier"``) to
    its reconciled validity state. A dimension clears only on a reference
    ``_DIMENSION_REFERENCES`` accepts for that dimension; any other value (a wrong-kind reference
    included: a raster-scope identity says nothing about a count) is treated as unvalidated. A
    dimension name the mapping does not know raises rather than judging it against a vocabulary
    the gate does not have; a new door's dimension gets a mapping row stating what clears it.
    Read the on-disk state before calling; the gate does not trust a caller-asserted string on
    its own.

    Every dimension validated -> the gate passes. Any not -> two independent escapes, neither
    trusting the other's dimension:

    - ``acknowledgement``, a real :class:`Acknowledgement` naming who and why, clears every
      unvalidated dimension: the breeder's own act of shipping a clearly-flagged phenotype
      unvalidated, stamped ``false`` so the un-trustworthiness travels downstream, and recorded on
      the result's own ``acknowledged_by``/``acknowledgement_reason``.
    - ``allow_unvalidated_staging=True`` clears only :data:`STAGING_DIMENSIONS` (``tile_size``,
      ``claim_scope``): the pre-pass gate a raw prediction bucket is written under, never a
      phenotype's own delivered dimensions (``operating_point``, ``classifier``, ``scale``), which
      it cannot clear no matter what the caller states.

    Any dimension neither escape covers still refuses. The refusal targets a *silent bare number*,
    not an honestly-acknowledged provisional deliverable. ``stamp`` records, per dimension, the
    reference it cleared (or ``false``), regardless of which escape (if any) let the gate pass.
    """
    unknown = sorted(name for name in flags if name not in _DIMENSION_REFERENCES)
    if unknown:
        raise ValueError(
            f"check_delivery_gate has no reference vocabulary for dimension(s) {unknown}: add "
            "each to _DIMENSION_REFERENCES, stating which references clear it, before gating on "
            f"it (known dimensions: {sorted(_DIMENSION_REFERENCES)})."
        )
    stamp = {name: (st if st in _DIMENSION_REFERENCES[name] else VALIDATED_FALSE)
             for name, st in flags.items()}
    unvalidated = tuple(name for name, st in flags.items()
                        if st not in _DIMENSION_REFERENCES[name])
    covered: set[str] = set()
    if allow_unvalidated_staging:
        covered |= {name for name in unvalidated if name in STAGING_DIMENSIONS}
    if acknowledgement is not None:
        covered |= set(unvalidated)
    blocking = tuple(name for name in unvalidated if name not in covered)
    if blocking:
        clears = "; ".join(
            f"{name} is cleared by {list(_DIMENSION_REFERENCES[name])}"
            if _DIMENSION_REFERENCES[name]
            else f"{name} is cleared by nothing (it states a missing prerequisite)"
            for name in blocking)
        return DeliveryGateResult(
            ok=False, unvalidated=unvalidated, stamp=stamp,
            reason=(
                f"delivery refused: unvalidated dimension(s) {list(blocking)}. A "
                "phenotype deliverable requires each dimension validated against a reference of "
                f"its own kind ({clears})."
            ),
        )
    acknowledged_by = acknowledgement.acknowledged_by if (unvalidated and acknowledgement) else None
    acknowledgement_reason = acknowledgement.reason if (unvalidated and acknowledgement) else None
    return DeliveryGateResult(
        ok=True, unvalidated=unvalidated, stamp=stamp,
        acknowledged_by=acknowledged_by, acknowledgement_reason=acknowledgement_reason,
    )


# --- validation (returns list[str] of problems, empty = valid) ---

def validate_resolved_bundle(
    bundle: ResolvedBundle,
    *,
    probed_channels: int | None = None,
    inference_bundle: ResolvedBundle | None = None,
    target_dataset_hash: str | None = None,
    for_export: bool = False,
) -> list[str]:
    """Return human-readable issues for a resolved bundle (empty list = valid).

    Live checks (each guards a real failure mode):
    - ``in_chans`` must equal the probed raster band count (mismatch trains/infers channel-wrong).
    - a calibration operating point with ``validated=false`` must not feed an export/delivery.
    - the eval operating point must equal the inference operating point on the same dataset
      (select@0.25 / ship@0.5 divergence).
    - a dataset-scoped calibration must not be inherited across a different dataset hash.

    ``max_dets`` is an intentional, standing exemption from the ``inference_bundle`` equality
    check for one specific comparison: a block-calibrated bundle against its own whole-raster
    export bundle (``inference_tools._export_predictions_raster``). Those two bundles' ``max_dets``
    are deliberately different by design (the export pass commits to ``None``, uncapped, never the
    block bundle's own band-scoped density-derived value), so a caller comparing exactly that pair
    must exclude ``"max_dets"`` before calling this, not treat a mismatch there as the select/ship
    divergence this check exists to catch. No caller does that comparison today, so this check's
    own general logic stays untouched here rather than special-cased for a pairing nothing calls.
    """
    issues: list[str] = []

    # in_chans vs probed bands
    if probed_channels is not None and "in_chans" in bundle.params:
        ic = bundle.params["in_chans"]._raw
        if ic != probed_channels:
            issues.append(f"in_chans={ic} != probed raster bands={probed_channels}")

    # dataset-scoped inheritance across a hash mismatch + un-shippable calibration
    issues.extend(bundle.shippable_issues(target_dataset_hash=target_dataset_hash))

    # export must not ship an unvalidated operating point
    if for_export and not bundle.is_shippable:
        issues.append("export/delivery requires a validated (held-out) operating point; this bundle is not shippable")

    # eval op-point must match inference op-point on the same dataset, every param the two bundles
    # have in common, not a hardcoded list (a new param added to one bundle's construction site is
    # covered automatically rather than silently exempt from this check).
    if inference_bundle is not None:
        for key in sorted(set(bundle.params) & set(inference_bundle.params)):
            a = bundle.params[key]
            b = inference_bundle.params[key]
            if a._raw != b._raw:
                issues.append(
                    f"{key}: eval operating point {a._raw} != inference operating point {b._raw} "
                    f"on the same dataset (the select-point must equal the ship-point)"
                )

    return issues
