"""Runtime parameter resolution — the "derive, don't pin" currency.

Every result-affecting parameter that varies by dataset/model/trait is resolved at runtime into a
``ResolvedParam`` carrying not just a value but *how it was derived* and *whether it is trustworthy*.
The point (CLAUDE.md "Parameters: derive, don't pin"): the agent derives operating points from the
data in hand, per dataset, and the provenance travels with every result so a phenotype can always be
traced to the operating point that produced it.

The measurement-integrity firewall lives here: a **calibration** param (a precision/recall operating
point like a confidence threshold) is *structurally un-consumable as a bare number* unless it was
validated against held-out ground truth for the same dataset. ``.value`` raises; a caller that
genuinely means to ship an unvalidated operating point must go through ``unvalidated_value(...)`` and
say so explicitly. This makes an unvalidated measurement operating point physically un-shippable
rather than merely discouraged.

Pure stdlib — no torch, safe to import anywhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- vocabularies ---------------------------------------------------------
SOURCES = ("explicit", "derived", "default")
# How a parameter is (or should be) determined — see the scope doc / CLAUDE.md.
DERIVATION_CLASSES = ("deterministic", "distribution", "calibration", "semantic", "engineering")
# validated_vs_gt records WHICH reference confirmed a calibration operating point — the shared-reference
# principle (CLAUDE.md): a reference sized to the trait, not dense GT for every trait. Both shippable
# references pass the IDENTICAL disjoint-split + count-bias gate; the value keeps them distinct so
# provenance says which one validated.
#   None                = not applicable (facts, e.g. num_classes).
#   "validated_held_out" = passed on a disjoint held-out split of this dataset's GT annotations.
#   "review_confirmed"   = passed on a breeder-confirmed sample of the model's own outputs (review
#                          verdicts reconstructed into the same records the GT path sweeps).
#   "false"              = no such validation exists (not truth — every reference is bounded by its
#                          label/verdict quality).
VALIDATED_HELD_OUT = "validated_held_out"
VALIDATED_REVIEW_CONFIRMED = "review_confirmed"
VALIDATED_FALSE = "false"
# The references that make a calibration operating point shippable (each cleared the same gate).
VALIDATED_SHIPPABLE = (VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED)

# Shared inference operating-point defaults, referenced by both run_inference and the web route so the
# same model+images can't give a different count by entry point.
DEFAULT_CONF = 0.5
DEFAULT_NMS_IOU = 0.3
DEFAULT_TILED = True
DEFAULT_TILE_SIZE = 640
# A high full-frame detection cap so dense scenes (hundreds of catkins) aren't silently truncated
# at a framework default (torchvision 100 / ultralytics 300). Enforced after any tiled merge.
DEFAULT_MAX_DETS = 1000


class UnvalidatedOperatingPointError(RuntimeError):
    """Raised when an unvalidated calibration param is consumed as if it were trustworthy.

    This is the firewall: it means a measurement operating point (e.g. a confidence threshold that
    defines the object count, which *is* the phenotype) is about to flow into a result without having
    been validated against held-out ground truth for this dataset. Do not silence it by reaching for
    ``_raw``; either validate the operating point, or consume it via ``unvalidated_value(...)`` and
    stamp the result ``validated=false`` so the un-trustworthiness travels downstream.
    """


@dataclass(frozen=True)
class ResolvedParam:
    """One parameter, resolved from the data in hand, with provenance and a validation status."""

    name: str
    _raw: Any
    source: str  # one of SOURCES
    derivation_class: str  # one of DERIVATION_CLASSES
    derived_from: str = ""  # human-readable: what artifact/analysis produced it
    validated_vs_gt: str | None = None  # None | VALIDATED_HELD_OUT | VALIDATED_FALSE
    dataset_scoped: bool = False  # True => only valid for the dataset named by dataset_hash
    dataset_hash: str | None = None
    sweep: dict | None = None  # sensitivity data (e.g. count-vs-conf curve) for a calibration param

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.derivation_class not in DERIVATION_CLASSES:
            raise ValueError(f"derivation_class must be one of {DERIVATION_CLASSES}, got {self.derivation_class!r}")
        if self.validated_vs_gt not in (None, VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED, VALIDATED_FALSE):
            raise ValueError(f"validated_vs_gt invalid: {self.validated_vs_gt!r}")

    @property
    def is_shippable(self) -> bool:
        """Can this value flow into a delivered result as a trustworthy number?

        Non-calibration params (facts, structural choices, engineering knobs) are always shippable.
        A calibration param is shippable only when a reference sized to the trait (held-out GT or a
        breeder-confirmed output sample) cleared the disjoint-split + count-bias gate for its dataset.
        """
        if self.derivation_class != "calibration":
            return True
        return self.validated_vs_gt in VALIDATED_SHIPPABLE

    @property
    def value(self) -> Any:
        """The trustworthy value — raises the firewall for an unvalidated calibration param."""
        if not self.is_shippable:
            raise UnvalidatedOperatingPointError(
                f"{self.name!r} is a calibration operating point with validated_vs_gt="
                f"{self.validated_vs_gt!r} and cannot be consumed as a trustworthy value. "
                f"Calibrate it against a held-out split for this dataset, or consume it via "
                f"unvalidated_value(acknowledge_unvalidated=True) and stamp the result validated=false."
            )
        return self._raw

    def unvalidated_value(self, *, acknowledge_unvalidated: bool) -> Any:
        """Escape hatch: read the raw value while explicitly acknowledging it is unvalidated.

        The caller must pass ``acknowledge_unvalidated=True`` and is responsible for stamping the
        resulting output ``validated=false`` (carrying ``self.sweep``) so the uncertainty travels on.
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
            "derivation_class": self.derivation_class,
            "derived_from": self.derived_from,
            "validated_vs_gt": self.validated_vs_gt,
            "dataset_scoped": self.dataset_scoped,
            "dataset_hash": self.dataset_hash,
            # the full sweep can be large; keep a marker, callers attach it separately if wanted
            "has_sweep": self.sweep is not None,
        }


def derived(name: str, value: Any, *, derivation_class: str, derived_from: str,
            validated_vs_gt: str | None = None, dataset_scoped: bool = False,
            dataset_hash: str | None = None, sweep: dict | None = None) -> ResolvedParam:
    """Convenience constructor for a data/model-derived param (``source="derived"``)."""
    return ResolvedParam(
        name=name, _raw=value, source="derived", derivation_class=derivation_class,
        derived_from=derived_from, validated_vs_gt=validated_vs_gt,
        dataset_scoped=dataset_scoped, dataset_hash=dataset_hash, sweep=sweep,
    )


def default(name: str, value: Any, *, derivation_class: str = "engineering",
            derived_from: str = "documented default") -> ResolvedParam:
    """Convenience constructor for a documented default (``source="default"``)."""
    return ResolvedParam(
        name=name, _raw=value, source="default", derivation_class=derivation_class,
        derived_from=derived_from,
    )


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
    classifier_validated_vs_gt: str | None = None  # for the elongation classifier

    def get(self, name: str) -> ResolvedParam:
        return self.params[name]

    def value(self, name: str) -> Any:
        """Firewalled value accessor (raises for an unvalidated calibration param)."""
        return self.params[name].value

    @property
    def is_shippable(self) -> bool:
        """True only if every calibration param is validated_held_out for this dataset."""
        return all(p.is_shippable for p in self.params.values())

    def shippable_issues(self, *, target_dataset_hash: str | None = None) -> list[str]:
        """Reasons this bundle cannot ship a trustworthy phenotype (empty = shippable)."""
        issues: list[str] = []
        for p in self.params.values():
            if not p.is_shippable:
                issues.append(f"{p.name}: calibration operating point not validated (validated_vs_gt={p.validated_vs_gt})")
            if p.dataset_scoped and target_dataset_hash is not None and p.dataset_hash != target_dataset_hash:
                issues.append(
                    f"{p.name}: dataset-scoped value derived on {p.dataset_hash} inherited across a "
                    f"different dataset {target_dataset_hash} — re-resolve per dataset, never inherit"
                )
        return issues

    def to_provenance(self) -> dict:
        return {
            "trait": self.trait,
            "dataset_hash": self.dataset_hash,
            "classifier_validated_vs_gt": self.classifier_validated_vs_gt,
            "operating_point": {name: p.to_provenance() for name, p in self.params.items()},
        }


def raw_operating_point(
    *, conf: float, cross_tile_nms: float | None, tiled: bool, tile_size: int | None,
    max_dets: int, tile_size_source: str = "default",
) -> ResolvedBundle:
    """The operating point for RAW (uncalibrated) inference — the one both doors resolve through.

    ``conf`` is a documented default with no per-dataset GT behind it, so it is a calibration param
    stamped ``validated_vs_gt=false``: reading it requires ``unvalidated_value(...)`` and the caller
    must stamp its output ``validated=false``. This is what stops the MCP tool and the web job giving
    a different count (the phenotype) for the same model + images by entry point.

    ``tile_size_source`` records whether the tile edge was ``derived`` from the checkpoint's training
    geometry, ``explicit`` (caller override), or a ``default`` fallback (CV2) — so a 224-train /
    640-infer scale mismatch is visible in the provenance rather than silent.
    """
    if tiled and tile_size_source == "derived":
        tile_param = derived("tile_size", tile_size, derivation_class="deterministic",
                             derived_from="persisted training tile geometry")
    elif tiled and tile_size_source == "explicit":
        tile_param = ResolvedParam("tile_size", tile_size, source="explicit",
                                   derivation_class="deterministic", derived_from="caller override")
    else:
        tile_param = default("tile_size", tile_size if tiled else None)
    return ResolvedBundle(trait="", dataset_hash=None, params={
        "conf": ResolvedParam("conf", conf, source="default", derivation_class="calibration",
                              validated_vs_gt=VALIDATED_FALSE),
        "cross_tile_nms": default("cross_tile_nms", cross_tile_nms if tiled else None,
                                  derivation_class="distribution"),
        "tiled": default("tiled", tiled),
        "tile_size": tile_param,
        "max_dets": default("max_dets", max_dets, derivation_class="distribution"),
    })


# --- dataset identity -----------------------------------------------------

def dataset_hash(labels_dir: str | Path, stems: list[str] | None = None) -> str:
    """A content-addressed hash identifying a dataset's ground truth.

    Two datasets with the same labels hash equal (so a calibration is valid iff its hash matches the
    inference dataset's). Content-based (label bytes), so it is machine-independent — a path can move
    between machines but the GT identity does not. Missing labels are hashed as empty (they are valid
    negatives), so their presence/absence still contributes to identity.
    """
    labels_dir = Path(labels_dir)
    if stems is None:
        # Canonical labels are per-image JSON.
        stems = sorted(p.stem for p in labels_dir.glob("*.json"))
    else:
        stems = sorted(stems)
    h = hashlib.sha256()
    for stem in stems:
        h.update(stem.encode("utf-8"))
        h.update(b"\0")
        lp = labels_dir / f"{stem}.json"
        h.update(lp.read_bytes() if lp.is_file() else b"")
        h.update(b"\0")
    return h.hexdigest()[:16]


# --- on-disk operating-point reconciliation (the delivery gate reads the sidecar, not a caller string) ---

def read_operating_point_sidecar(pred_dir: str | Path) -> dict | None:
    """The bucket's ``operating_point.json`` stamp, or ``None`` if absent/unreadable (never raises)."""
    p = Path(pred_dir) / "operating_point.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _sidecar_reference(sidecar: dict | None) -> str:
    """Which reference the sidecar's conf operating point cleared — held_out / review_confirmed / false.

    Falls back to the top-level ``validated`` bool (a shippable stamp) when the per-param reference is
    missing, but never upgrades an unvalidated stamp.
    """
    if not sidecar or not sidecar.get("validated"):
        return VALIDATED_FALSE
    conf = (sidecar.get("operating_point") or {}).get("conf") or {}
    ref = conf.get("validated_vs_gt")
    return ref if ref in VALIDATED_SHIPPABLE else VALIDATED_HELD_OUT


def _validity_rank(state: str | None) -> int:
    """Floor ordering: unvalidated (0) < any shippable reference (1). ``None`` = no assertion (skip)."""
    if state is None:
        return 99
    return 1 if state in VALIDATED_SHIPPABLE else 0


def reconcile_operating_point_validity(
    pred_dirs: list[str] | tuple[str, ...], *, asserted: str | None = None,
) -> dict:
    """Floor the operating-point validity against every bucket's on-disk sidecar (T5-3 fix).

    The delivery gate must not trust a caller's asserted string: it reads each prediction bucket's
    ``operating_point.json`` and takes the FLOOR of asserted-vs-on-disk. A missing/unreadable sidecar,
    or any bucket stamped ``validated=false``, floors the whole curve to ``false`` — never a crash. An
    asserted string can only lower the on-disk result, never raise it (prefer the on-disk truth). The
    on-disk reference (held_out vs review_confirmed) is preserved so provenance still records which one.

    Returns ``{validated, on_disk_validated, missing_sidecars, unvalidated_buckets, conf, per_bucket}``.
    """
    per_bucket: dict[str, str] = {}
    missing: list[str] = []
    unvalidated: list[str] = []
    refs: set[str] = set()
    confs: list[float] = []
    all_validated = bool(pred_dirs)
    for d in pred_dirs:
        sc = read_operating_point_sidecar(d)
        if sc is None:
            missing.append(str(d))
            per_bucket[str(d)] = VALIDATED_FALSE
            all_validated = False
            continue
        ref = _sidecar_reference(sc)
        per_bucket[str(d)] = ref
        if ref in VALIDATED_SHIPPABLE:
            refs.add(ref)
            conf_val = ((sc.get("operating_point") or {}).get("conf") or {}).get("value")
            if isinstance(conf_val, (int, float)):
                confs.append(float(conf_val))
        else:
            unvalidated.append(str(d))
            all_validated = False

    if all_validated and refs:
        on_disk = VALIDATED_HELD_OUT if VALIDATED_HELD_OUT in refs else VALIDATED_REVIEW_CONFIRMED
    else:
        on_disk = VALIDATED_FALSE

    # Floor: the asserted string may only lower the on-disk result (prefer on-disk).
    validated = on_disk if _validity_rank(asserted) >= _validity_rank(on_disk) else VALIDATED_FALSE
    return {
        "validated": validated,
        "on_disk_validated": all_validated and bool(refs),
        "missing_sidecars": missing,
        "unvalidated_buckets": unvalidated,
        "conf": (confs[0] if len(set(confs)) == 1 else None),
        "per_bucket": per_bucket,
    }


# --- the delivery gate (one refuse-or-stamp check shared by every phenotype-delivery door) ---

@dataclass(frozen=True)
class DeliveryGateResult:
    """Outcome of the delivery gate: whether the deliverable may be written, and how to stamp it."""

    ok: bool
    unvalidated: tuple[str, ...]  # dimensions whose validity is not a shippable reference
    stamp: dict[str, str]  # per-dimension validity to stamp onto the deliverable
    reason: str = ""  # generic refusal message when not ok


def check_delivery_gate(
    flags: dict[str, str | None], *, acknowledge_unvalidated: bool = False,
) -> DeliveryGateResult:
    """Refuse-or-stamp a phenotype delivery against the validity of each dimension it rests on.

    ``flags`` maps each measurement dimension the deliverable depends on (e.g. ``"operating_point"``,
    ``"classifier"``, or a single ``"measurement"`` for a continuous/ordinal trait with no conf
    op-point) to its RECONCILED validity state — a shippable reference (``validated_held_out`` /
    ``review_confirmed``) or anything else (treated as unvalidated). Read the on-disk state before
    calling; the gate does not trust a caller-asserted string on its own.

    Every dimension validated -> the gate passes. Any not -> it refuses UNLESS
    ``acknowledge_unvalidated=True``, the escape hatch that ships a clearly-flagged provisional
    deliverable and stamps every unvalidated dimension ``false`` so the un-trustworthiness travels
    downstream. The refusal targets a *silent bare number*, not an honestly-acknowledged provisional
    CSV. ``stamp`` records, per dimension, the reference it cleared (or ``false``).
    """
    stamp = {name: (st if st in VALIDATED_SHIPPABLE else VALIDATED_FALSE)
             for name, st in flags.items()}
    unvalidated = tuple(name for name, st in flags.items() if st not in VALIDATED_SHIPPABLE)
    if unvalidated and not acknowledge_unvalidated:
        return DeliveryGateResult(
            ok=False, unvalidated=unvalidated, stamp=stamp,
            reason=(
                f"delivery refused: unvalidated measurement dimension(s) {list(unvalidated)}. A "
                "phenotype deliverable requires each dimension validated against a reference sized to "
                "the trait (held-out GT or a breeder-confirmed output sample); validate it, or pass "
                "acknowledge_unvalidated=True to write a clearly-flagged provisional result stamped "
                "validated=false."
            ),
        )
    return DeliveryGateResult(ok=True, unvalidated=unvalidated, stamp=stamp)


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

    Live checks (each guards a real failure the audit found):
    - ``in_chans`` must equal the probed raster band count (mismatch trains/infers channel-wrong).
    - a calibration operating point with ``validated=false`` must not feed an export/delivery.
    - the eval operating point must equal the inference operating point on the same dataset
      (select@0.25 / ship@0.5 divergence).
    - a dataset-scoped calibration must not be inherited across a different dataset hash.
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

    # eval op-point must match inference op-point on the same dataset
    if inference_bundle is not None:
        for key in ("conf", "cross_tile_nms", "tiled", "tile_size", "max_dets"):
            a = bundle.params.get(key)
            b = inference_bundle.params.get(key)
            if a is not None and b is not None and a._raw != b._raw:
                issues.append(
                    f"{key}: eval operating point {a._raw} != inference operating point {b._raw} "
                    f"on the same dataset (the select-point must equal the ship-point)"
                )

    return issues
