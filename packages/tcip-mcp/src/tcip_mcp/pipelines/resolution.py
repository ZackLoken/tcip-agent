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

No torch, safe to import anywhere; the storage seam (``tcip_store``) is the one dependency beyond
the standard library, since the prediction buckets' provenance stamps are declared here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import tcip_store
from tcip_store import Key, StoreDescriptor, StoreError, json_codec, register_store
from tcip_store.file_backend import RootedFileLocator

# --- vocabularies ---------------------------------------------------------
SOURCES = ("explicit", "derived", "default")

# validated_against records which reference confirmed a value that requires validation, the
# shared-reference principle (CLAUDE.md): a reference sized to the trait, not dense GT for every
# trait. "false" is a real, distinct value (not Python None) so a sidecar/provenance record can say
# "checked, and it wasn't" rather than leaving the field merely absent.
VALIDATED_HELD_OUT = "held_out_annotations"          # a disjoint held-out split of this dataset's GT
VALIDATED_REVIEW_CONFIRMED = "reviewer_confirmed_annotations"  # a breeder-confirmed output sample
VALIDATED_PHYSICAL_MEASUREMENT = "physical_measurement"  # checked against a known physical dimension
# A tile scale's own real basis for trust is the checkpoint's own persisted training geometry, or a
# caller's deliberate stated override (the same two bases run_full_frame_evaluation already accepts).
VALIDATED_PERSISTED_GEOMETRY = "persisted_training_geometry"
VALIDATED_EXPLICIT_GEOMETRY = "explicit_caller_stated_geometry"
# A raster export target's content identity matched the mosaic a block-calibrated bundle was
# validated against; check_delivery_gate's own "claim_scope" dimension, never a ResolvedParam.
VALIDATED_SAME_MOSAIC_IDENTITY = "same_mosaic_content_identity"
VALIDATED_FALSE = "false"

# Which validated_against values legitimately clear validation for which kind of thing being
# validated, an annotation-count operating point (conf, a mask-binarize threshold), a physical
# scale, and a tile geometry are checked against fundamentally different references, and none may
# satisfy another's requirement.
VALIDATION_KINDS = ("annotations", "physical", "geometry")
_GEOMETRY_REFERENCE_BY_SOURCE: dict[str, str] = {
    "derived": VALIDATED_PERSISTED_GEOMETRY,
    "explicit": VALIDATED_EXPLICIT_GEOMETRY,
}
"""Which ``tile_size_source`` earns which geometry reference, stated once for both directions:
:func:`resolve_tile_size_param` stamps a reference from a source and :func:`tile_size_source_of`
recovers the source behind a recorded one, so a door reading a persisted stamp back cannot disagree
with the door that wrote it about what counts as a real basis for a tile scale."""
_ACCEPTED_REFERENCES: dict[str, tuple[str, ...]] = {
    "annotations": (VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED),
    "physical": (VALIDATED_PHYSICAL_MEASUREMENT,),
    "geometry": tuple(_GEOMETRY_REFERENCE_BY_SOURCE.values()),
}


def accepted_references(validation_kind: str) -> tuple[str, ...]:
    """The validated_against values that legitimately clear validation for this kind."""
    return _ACCEPTED_REFERENCES[validation_kind]


def tile_size_source_of(reference: str | None, *, tile_size: int | None) -> str:
    """The ``tile_size_source`` a recorded geometry reference was earned by.

    The inverse of what :func:`resolve_tile_size_param` stamps, for a caller holding a persisted
    stamp rather than a live resolution (the review-promotion path, reading a bucket's sidecar back
    to re-resolve an operating point from it). Reading the reference, never the bare ``source``
    field a sidecar also carries, is what stops a native-size-ratio tile edge from being re-read as
    a real persisted geometry: that tier is a basis to tile at all but never an accepted geometry
    reference, so it comes back as ``"native_ratio"`` whenever a tile size is present with no
    accepted reference behind it, and ``"default"`` when there is no tile size at all.
    """
    for source, accepted in _GEOMETRY_REFERENCE_BY_SOURCE.items():
        if reference == accepted:
            return source
    return "native_ratio" if tile_size is not None else "default"


# The union of every real (non-"false") shippable reference, across every kind, used only by
# dimension-agnostic logic (check_delivery_gate, _validity_rank) whose callers have already
# resolved the right kind per-dimension upstream via accepted_references()/is_shippable. Never used
# to decide whether a specific param's reference is the right kind for it, that decision belongs to
# accepted_references(validation_kind), always.
VALIDATED_SHIPPABLE = (
    VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED, VALIDATED_PHYSICAL_MEASUREMENT,
    VALIDATED_PERSISTED_GEOMETRY, VALIDATED_EXPLICIT_GEOMETRY, VALIDATED_SAME_MOSAIC_IDENTITY,
)

# Shared inference operating-point defaults, referenced by both run_inference and the web route so the
# same model+images can't give a different count by entry point.
DEFAULT_CONF = 0.5
DEFAULT_NMS_IOU = 0.3
DEFAULT_OVERLAP = 0.2
# A high full-frame detection cap so dense scenes (hundreds of objects) aren't silently truncated
# at a framework default (torchvision 100 / ultralytics 300). Enforced after any tiled merge.
DEFAULT_MAX_DETS = 1000


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
    sweep: dict | None = None  # sensitivity data (e.g. count-vs-conf curve) for a validated param

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
            "derived_from": self.derived_from,
            "requires_validation": self.requires_validation,
            "validation_kind": self.validation_kind,
            "validated_against": self.validated_against,
            "dataset_scoped": self.dataset_scoped,
            "dataset_hash": self.dataset_hash,
            "capture_scoped": self.capture_scoped,
            "capture_id": self.capture_id,
            # the full sweep can be large; keep a marker, callers attach it separately if wanted
            "has_sweep": self.sweep is not None,
        }


def derived(name: str, value: Any, *, derived_from: str,
            requires_validation: bool = False, validation_kind: str | None = None,
            validated_against: str | None = None, dataset_scoped: bool = False,
            dataset_hash: str | None = None, capture_scoped: bool = False,
            capture_id: str | None = None, sweep: dict | None = None) -> ResolvedParam:
    """Convenience constructor for a data/model-derived param (``source="derived"``)."""
    return ResolvedParam(
        name=name, _raw=value, source="derived", derived_from=derived_from,
        requires_validation=requires_validation, validation_kind=validation_kind,
        validated_against=validated_against, dataset_scoped=dataset_scoped, dataset_hash=dataset_hash,
        capture_scoped=capture_scoped, capture_id=capture_id, sweep=sweep,
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
) -> ResolvedParam:
    """The ``tile_size`` dimension, gated the same shape ``conf`` already is, the shared
    construction site both :func:`raw_operating_point` (here) and
    :func:`tcip_mcp.pipelines.operating_point.resolve_operating_point` (the calibrated path) call,
    so the two doors can't drift into disagreeing about when a tile scale is trustworthy.

    Only meaningful when ``tiled``: an untiled run's count never depends on tile_size, so it stays a
    plain non-gating fact there (mirrors ``in_chans``), gating it anyway would refuse legitimate
    untiled work over a dimension that was never operative. When tiled, the value is shippable only
    when its source names a real basis for trusting the scale: the checkpoint's own persisted
    training geometry (``"derived"``), or a caller's deliberate explicit override (``"explicit"``,
    accepted on the same terms ``run_full_frame_evaluation`` already accepts an explicit value on:
    not cross-checked against the checkpoint's real training scale, but a stated decision, not a
    guess). ``"native_ratio"`` (a tile edge derived from a checkpoint's own uniform untiled training
    size) is a real basis to tile at all but never an accepted geometry reference on its own, so it
    floors to unvalidated exactly like the no-basis case. Any other source (``"unavailable"``, no
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
    if tile_size_source == "explicit" and tile_size is not None:
        return ResolvedParam(
            "tile_size", int(tile_size), source="explicit", derived_from="caller override",
            requires_validation=True, validation_kind="geometry",
            validated_against=_GEOMETRY_REFERENCE_BY_SOURCE["explicit"],
        )
    if tile_size_source == "native_ratio":
        return ResolvedParam(
            "tile_size", int(tile_size) if tile_size is not None else None, source="derived",
            derived_from="native-size ratio (not an independently validated geometry basis)",
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
    max_dets: int | None, tile_size_source: str = "default", tiled_source: str = "default",
) -> ResolvedBundle:
    """The operating point for raw (uncalibrated) inference, the one both doors resolve through.

    ``conf`` is a documented default with no per-dataset GT behind it, so it requires validation and
    is stamped ``validated_against=false``: reading it requires ``unvalidated_value(...)`` and the
    caller must stamp its output ``validated=false``. This is what stops the MCP tool and the web job
    giving a different count (the phenotype) for the same model + images by entry point.

    ``max_dets=None`` is a real, deliberate value (uncapped), not an unset caller: the block
    calibration export path (``inference_tools._export_predictions_raster``) commits to it on
    purpose, since a block-calibrated bundle's own density-derived cap is scoped to one reserved
    band and would truncate a whole-mosaic count if adopted wholesale.

    ``tile_size_source`` records whether the tile edge was ``derived`` from the checkpoint's training
    geometry, ``explicit`` (caller override), or has no real basis at all (``"unavailable"``), so a
    train/infer scale mismatch is visible in the provenance rather than silent, and whether tiled
    inference's tile_size has a real basis at all: see :func:`resolve_tile_size_param`, a tiled run
    with no persisted/explicit basis is a real, gating-firewalled unvalidated dimension (``tile_size``
    itself ``None``, never a fabricated number), not silently shippable engineering trivia.
    ``tiled_source`` is the same provenance vocabulary for the boolean itself: a caller that
    explicitly chose to tile (or not) stamps ``"explicit"``; a caller who passed nothing gets
    ``"default"``. Both callers of this function derive their own concrete ``tiled`` bool (no shared
    fallback constant) before reaching here, so ``tiled`` itself is always a real bool, never ``None``.
    """
    tile_param = resolve_tile_size_param(tile_size, tiled=tiled, tile_size_source=tile_size_source)
    if tiled_source == "explicit":
        tiled_param = ResolvedParam("tiled", tiled, source="explicit", derived_from="caller override")
    else:
        tiled_param = default("tiled", tiled)
    return ResolvedBundle(trait="", dataset_hash=None, params={
        "conf": ResolvedParam("conf", conf, source="default", requires_validation=True,
                              validation_kind="annotations", validated_against=VALIDATED_FALSE),
        "cross_tile_nms": default("cross_tile_nms", cross_tile_nms if tiled else None),
        "tiled": tiled_param,
        "tile_size": tile_param,
        "max_dets": default("max_dets", max_dets),
    })


def block_calibrated_export_operating_point(
    block_bundle: ResolvedBundle, *, trait: str, tile_size: int | None, tile_size_source: str,
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
            tile_size, tiled=True, tile_size_source=tile_size_source),
        "max_dets": default(
            "max_dets", None,
            derived_from="block calibration: not transferred, uncapped for the whole-mosaic pass"),
    })


# --- dataset identity -----------------------------------------------------

def dataset_hash(labels_dir: str | Path, stems: list[str] | None = None) -> str:
    """A content-addressed hash identifying a dataset's ground truth.

    Two datasets with the same labels hash equal (so a calibration is valid iff its hash matches the
    inference dataset's). Content-based (label bytes), so it is machine-independent, a path can move
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


_FINGERPRINT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _labels_term(annotations_root: Path) -> str | None:
    """Whole-dataset label identity, composed from :func:`dataset_hash` per label dir.

    Labels live at ``annotations/<date>/*.json`` (date-nested) or flat ``annotations/*.json``.
    ``dataset_hash`` is the single label-byte hasher (flat glob), so this *calls* it per dir and
    combines the per-dir digests keyed by dir name, never re-implementing label hashing. ``None``
    when no labels exist anywhere.
    """
    if not annotations_root.is_dir():
        return None
    subdirs = sorted(d for d in annotations_root.iterdir() if d.is_dir())
    flat = not subdirs
    label_dirs = subdirs if subdirs else [annotations_root]
    h = hashlib.sha256()
    any_labels = False
    for d in label_dirs:
        if not any(d.glob("*.json")):
            continue
        any_labels = True
        # A real subdir name can never be empty, so the flat root keys with "" rather than its own
        # name, otherwise a dated subdir named literally "annotations" would key identically to the
        # flat case and collide with it.
        key = "" if flat else d.name
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(dataset_hash(d).encode("utf-8"))  # reuse the label-byte hasher, per dir
        h.update(b"\0")
    return h.hexdigest()[:16] if any_labels else None


def _images_term(images_root: Path, cache_path: Path | None) -> str | None:
    """Whole-dataset image identity from each image's *pixel bytes* (content, not name/size), so a
    re-encode under the same filename changes identity (closes the labels-only/pixel-blind gap). Each
    file's sha is cached by ``(relpath, size, mtime_ns)`` so only changed files re-hash; a cache miss
    always hashes the bytes. ``None`` when there are no images (bespoke/imageless).
    """
    if not images_root.is_dir():
        return None
    files = sorted(p for p in images_root.rglob("*")
                   if p.is_file() and p.suffix.lower() in _FINGERPRINT_IMAGE_EXTS)
    if not files:
        return None
    old: dict[str, str] = {}
    if cache_path and cache_path.is_file():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            old = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            old = {}
    new: dict[str, str] = {}  # only current files -> the cache never grows unbounded
    manifest: dict[str, str] = {}
    for f in files:
        rel = f.relative_to(images_root).as_posix()
        st = f.stat()
        key = f"{rel}\0{st.st_size}\0{st.st_mtime_ns}"
        sha = old.get(key) or hashlib.sha256(f.read_bytes()).hexdigest()
        new[key] = sha
        manifest[rel] = sha
    if cache_path and new != old:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(new), encoding="utf-8")
        except OSError:
            pass
    h = hashlib.sha256()
    for rel in sorted(manifest):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(manifest[rel].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _registry_term(dataset_root: Path) -> str:
    """Digest over the canonical registry serialization in *declared order* (load-bearing in
    ``assign_class_ids``). Serialized via ``registry_to_dict`` rather than raw bytes, so a
    whitespace-only reformat of ``classes.json`` does not change identity but a value reorder/addition
    does. Empty string when the dataset has no registry."""
    from tcip_mcp.class_registry import RegistryError, read_registry, registry_to_dict
    from tcip_mcp.dataset_layout import classes_path

    cp = classes_path(dataset_root)
    if not cp.is_file():
        return ""
    try:
        canonical = json.dumps(registry_to_dict(read_registry(cp)), separators=(",", ":"))
    except (OSError, ValueError, RegistryError):
        return ""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _confirmations_term(dataset_root: Path) -> str:
    """Digest over the dataset-native confirmed-negative store's raw content (all buckets, sorted).

    Reads ``dataset_layout.image_status_path`` only, never a foreign store, since
    confirmations are dataset-native, the same way ``_registry_term`` reads only
    ``classes_path``. Hashes raw on-disk negative membership, not the quarantine-filtered view
    ``confirmed_negative_names`` returns: fingerprint is content identity (should two datasets be
    considered the same content), quarantine is a training-time trust decision, and conflating them
    would make the fingerprint *less* sensitive to a real on-disk difference than it should be, the
    unsafe direction. Empty string when there is no store or no negative entries, matching
    ``_registry_term``'s "optional, additive" convention so a dataset with zero confirmed negatives
    still gets a valid non-None fingerprint.
    """
    from tcip_mcp.dataset_layout import (
        image_status_path, is_confirmed_negative, normalize_status_store,
    )

    p = image_status_path(dataset_root)
    if not p.is_file():
        return ""
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    statuses = normalize_status_store(raw)
    h = hashlib.sha256()
    any_negatives = False
    for bucket in sorted(statuses):
        names = sorted(n for n, s in statuses[bucket].items() if is_confirmed_negative(s))
        if not names:
            continue
        any_negatives = True
        h.update(bucket.encode("utf-8"))
        h.update(b"\0")
        for n in names:
            h.update(n.encode("utf-8"))
            h.update(b"\0")
    return h.hexdigest()[:16] if any_negatives else ""


def dataset_fingerprint(dataset_root: str | Path) -> str | None:
    """Whole-dataset content identity: labels + image pixels + registry + confirmed negatives.

    A superset of :func:`dataset_hash` (which stays the per-split-subset firewall key): the label term
    *calls* ``dataset_hash``; the image term hashes pixel bytes; the registry term digests the canonical
    class registry; the confirmations term digests the dataset-native confirmed-negative store.
    Content-addressed, so it is machine-independent (a moved dataset keeps its fingerprint) and detects
    a change to any of the four (a re-encode, a relabel, a registry edit, confirming/un-confirming a
    negative). ``None`` for a dataset with no images or no labels (e.g. a bespoke ``dataset_source``),
    matching ``dataset_hash``'s honesty rather than fabricating identity. Authority is recompute-on-read;
    a stored fingerprint (``dataset.json``) is a cache.

    Adding the confirmations term was a one-time formula-version shift: recomputing against an
    existing ``dataset.json``/experiment ``lineage.json`` written under the 3-term formula reads as
    changed even with identical on-disk content. That is expected, not corruption, experiments are
    immutable, so old lineage records keep their old fingerprint value rather than being rewritten.
    """
    from tcip_mcp.dataset_layout import annotation_root, image_root

    root = Path(dataset_root)
    labels = _labels_term(annotation_root(root))
    images = _images_term(image_root(root), root / ".tcip" / "state" / "image_hash_cache.json")
    if labels is None or images is None:
        return None
    h = hashlib.sha256()
    h.update(b"labels:")
    h.update(labels.encode("utf-8"))
    h.update(b"\0images:")
    h.update(images.encode("utf-8"))
    h.update(b"\0classes:")
    h.update(_registry_term(root).encode("utf-8"))
    h.update(b"\0confirmations:")
    h.update(_confirmations_term(root).encode("utf-8"))
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
            codec=json_codec(),
            concurrency="cas",
            locator=_SIDECAR_LOCATOR,
        )
    ).name
    for document in (
        "operating_point",
        "classifier_operating_point",
        "ordinal_operating_point",
        "regression_operating_point",
        "resolve_scale",
    )
}
"""One store per measurement dimension, never one store holding every dimension's fields: the
dimensions are structurally independent (a physical scale is a fact about the imagery, a classifier
stamp is about a state call, the count operating point is about a threshold), and a single document
is exactly what would let a generic writer conflate them."""

SIDECAR_FILENAMES = frozenset(f"{document}.json" for document in _SIDECAR_STORES)
"""Every provenance stamp a prediction bucket carries beside its per-image records.

A stamp is not a per-image label: a reader enumerating a bucket's prediction files excludes these,
or it invents an image stem no image has and reads a stamp as if it were detections. Stated once
here, so a stamp added for a new dimension is excluded on every path that enumerates a bucket."""


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


def _read_sidecar(pred_dir: str | Path, document: str) -> dict | None:
    """One bucket's stamp for one dimension, or ``None`` when absent or unreadable.

    Never raises: an unreadable stamp floors the dimension it describes to unvalidated at every
    reconciler below, which is the safe direction, where a raised decode error would take down a
    delivery gate that has a well-defined answer for a stamp it cannot trust.
    """
    try:
        return tcip_store.read(sidecar_key(pred_dir, document), default=None)
    except StoreError:
        return None


def write_sidecar(pred_dir: str | Path, stamp: dict, document: str = "operating_point") -> None:
    """Write one bucket's stamp whole, under the stamp's own lock."""
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
        updated = updater(current if isinstance(current, dict) else {})
        if updated is None:
            return False
        txn.write(key, updated)
    return True


def operating_point_stamp(
    operating_point: dict | None,
    *,
    validated: bool,
    tile_size_validated: str | None,
    shippable_issues: list[str],
    id_map: dict | None,
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
    (a persisted sweep, a mask-binarize threshold, a block calibration's own record) travel through
    ``fields``.

    ``validated`` is the producing door's own verdict over the dimensions it resolved. The tile
    scale is floored in here rather than at each door: a bucket whose tile geometry has no real
    basis produced its counts at a scale nothing justifies, so it is not a validated bucket no
    matter what the conf dimension earned.
    """
    return {
        "trait": trait,
        "dataset_hash": dataset_hash,
        "operating_point": operating_point,
        "id_map": id_map,
        "validated": bool(validated) and tile_size_validated != VALIDATED_FALSE,
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


def read_operating_point_sidecar(pred_dir: str | Path) -> dict | None:
    """The bucket's ``operating_point.json`` stamp, or ``None`` if absent/unreadable (never raises)."""
    return _read_sidecar(pred_dir, "operating_point")


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
    ref = param.get("validated_against")
    accepted = accepted_references(validation_kind)
    return ref if ref in accepted else VALIDATED_FALSE


def _validity_rank(state: str | None) -> int:
    """Floor ordering: unvalidated (0) < any shippable reference, any kind (1). ``None`` = no
    assertion (skip). Comparing rank between two states already scoped to the same dimension/param,
    never used to decide whether a state is the right kind for a given param (that is
    ``accepted_references``'s job)."""
    if state is None:
        return 99
    return 1 if state in VALIDATED_SHIPPABLE else 0


def _reconcile_validity(
    pred_dirs: list[str] | tuple[str, ...], *, asserted: str | None,
    read_sidecar: Callable[[str | Path], dict | None], param_key: str,
    validation_kind: str = "annotations",
) -> dict:
    """Floor a validity dimension against every bucket's on-disk sidecar, generalized.

    Shared by :func:`reconcile_operating_point_validity` and :func:`reconcile_classifier_validity`,
    the flooring logic (read on-disk, never trust a caller string, an asserted value may only lower
    the result) is identical for both dimensions; only which sidecar file, which param key, and which
    validation kind is read differs, threaded in by the two thin public wrappers below.

    Returns ``{validated, on_disk_validated, missing_sidecars, unvalidated_buckets, conf, per_bucket}``.
    """
    per_bucket: dict[str, str] = {}
    missing: list[str] = []
    unvalidated: list[str] = []
    refs: set[str] = set()
    confs: list[float] = []
    all_validated = bool(pred_dirs)
    accepted = accepted_references(validation_kind)
    for d in pred_dirs:
        sc = read_sidecar(d)
        if sc is None:
            missing.append(str(d))
            per_bucket[str(d)] = VALIDATED_FALSE
            all_validated = False
            continue
        ref = _sidecar_reference(sc, param_key=param_key, validation_kind=validation_kind)
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
    validated = on_disk if _validity_rank(asserted) >= _validity_rank(on_disk) else VALIDATED_FALSE
    return {
        "validated": validated,
        "on_disk_validated": all_validated and bool(refs),
        "missing_sidecars": missing,
        "unvalidated_buckets": unvalidated,
        "conf": (confs[0] if len(set(confs)) == 1 else None),
        "per_bucket": per_bucket,
    }


def reconcile_operating_point_validity(
    pred_dirs: list[str] | tuple[str, ...], *, asserted: str | None = None,
) -> dict:
    """Floor the count operating-point validity against every bucket's ``operating_point.json``.

    The delivery gate must not trust a caller's asserted string: it reads each prediction bucket's
    on-disk sidecar and takes the floor of asserted-vs-on-disk. A missing/unreadable sidecar, or any
    bucket stamped ``validated=false``, floors the whole curve to ``false``, never a crash. See
    :func:`_reconcile_validity` for the shared mechanism.
    """
    return _reconcile_validity(
        pred_dirs, asserted=asserted,
        read_sidecar=read_operating_point_sidecar, param_key="conf", validation_kind="annotations",
    )


def reconcile_classifier_validity(
    pred_dirs: list[str] | tuple[str, ...], *, asserted: str | None = None,
) -> dict:
    """Floor the classifier validity against every bucket's ``classifier_operating_point.json``.

    Structurally the same reconciliation :func:`reconcile_operating_point_validity` performs for the
    count operating point, the same function, parameterized to a different sidecar file and param
    key, never a hand-written sibling. A bucket with no persisted classifier-calibration run floors
    to ``false``: there is no legitimate way to earn a classifier-validated stamp without one, so this
    never falls back to a caller-asserted string.
    """
    return _reconcile_validity(
        pred_dirs, asserted=asserted,
        read_sidecar=read_classifier_operating_point_sidecar, param_key="classifier",
        validation_kind="annotations",
    )


def reconcile_ordinal_validity(
    pred_dirs: list[str] | tuple[str, ...], *, asserted: str | None = None,
) -> dict:
    """Floor the ordinal compensating-error validity against every bucket's
    ``ordinal_operating_point.json``.

    Structurally the same reconciliation :func:`reconcile_classifier_validity` performs for the
    classifier dimension, the same shared mechanism, parameterized to a different sidecar file and
    param key. A bucket with no persisted ordinal-calibration run floors to ``false``: there is no
    legitimate way to earn an ordinal-validated stamp without one.
    """
    return _reconcile_validity(
        pred_dirs, asserted=asserted,
        read_sidecar=read_ordinal_operating_point_sidecar, param_key="ordinal",
        validation_kind="annotations",
    )


def reconcile_regression_validity(
    pred_dirs: list[str] | tuple[str, ...], *, asserted: str | None = None,
) -> dict:
    """Floor the regression compensating-error validity against every bucket's
    ``regression_operating_point.json``. Same shape as :func:`reconcile_ordinal_validity`, for the
    regression dimension's own sidecar/param key.
    """
    return _reconcile_validity(
        pred_dirs, asserted=asserted,
        read_sidecar=read_regression_operating_point_sidecar, param_key="regression",
        validation_kind="annotations",
    )


def tile_size_gate_flag(operating_point: dict | None) -> str | None:
    """The tile-geometry dimension's delivery-gate flag for one resolved operating point.

    ``operating_point`` is a bundle's ``to_provenance()["operating_point"]`` mapping, the same shape
    a run returns in-memory and ``export_predictions`` persists into ``operating_point.json``.
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


def reconcile_tile_size_validity(pred_dirs: list[str] | tuple[str, ...]) -> dict:
    """Floor the tile-geometry dimension across every prediction bucket's ``operating_point.json``.

    The sidecar-reading counterpart of :func:`tile_size_gate_flag`, for the delivery doors that
    assemble a phenotype from already-written prediction buckets rather than from a live run. A
    delivery spanning several buckets is only as grounded as its least-grounded tiled bucket, so any
    operative bucket whose tile scale has no real basis floors the whole dimension to
    ``VALIDATED_FALSE``; when the cleared references differ across buckets the weaker basis
    (a caller's stated override) is what travels, never the stronger one some other bucket earned.

    Returns ``{operative, validated, per_bucket, unvalidated_buckets}``. ``operative`` is False (and
    ``validated`` ``None``) when no bucket ran tiled, in which case the caller adds nothing to its
    gate. A bucket with no readable sidecar contributes nothing here; that bucket's missing stamp
    already floors the count operating point via :func:`reconcile_operating_point_validity`.
    """
    per_bucket: dict[str, str] = {}
    unvalidated: list[str] = []
    refs: set[str] = set()
    accepted = accepted_references("geometry")
    for d in pred_dirs:
        flag = tile_size_gate_flag((read_operating_point_sidecar(d) or {}).get("operating_point"))
        if flag is None:
            continue
        per_bucket[str(d)] = flag
        if flag in accepted:
            refs.add(flag)
        else:
            unvalidated.append(str(d))
    if not per_bucket:
        return {"operative": False, "validated": None, "per_bucket": {}, "unvalidated_buckets": []}
    if unvalidated:
        validated = VALIDATED_FALSE
    elif VALIDATED_EXPLICIT_GEOMETRY in refs:
        validated = VALIDATED_EXPLICIT_GEOMETRY
    else:
        validated = VALIDATED_PERSISTED_GEOMETRY
    return {"operative": True, "validated": validated, "per_bucket": per_bucket,
            "unvalidated_buckets": unvalidated}


def reconcile_scale_validity(
    pred_dirs: list[str] | tuple[str, ...], *, capture_id: str | None = None,
    asserted: str | None = None,
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

    ``asserted``, mirroring :func:`reconcile_operating_point_validity`, may only lower the on-disk
    result, never raise it: a caller string can never launder an ungrounded scale into a shippable one.

    Returns ``{operative, validated, per_bucket, unvalidated_buckets}``, the same shape
    :func:`reconcile_tile_size_validity` returns. ``operative`` is False (``validated`` ``None``) only
    when ``pred_dirs`` itself is empty, there is nothing to reconcile against.
    """
    if not pred_dirs:
        return {"operative": False, "validated": None, "per_bucket": {}, "unvalidated_buckets": []}
    accepted = accepted_references("physical")
    per_bucket: dict[str, str] = {}
    unvalidated: list[str] = []
    refs: set[str] = set()
    for d in pred_dirs:
        sc = read_scale_sidecar(d)
        ref = _sidecar_reference(sc, param_key="scale", validation_kind="physical")
        if ref in accepted and capture_id is not None:
            recorded = ((sc.get("operating_point") or {}).get("scale") or {}).get("capture_id")
            if recorded is not None and recorded != capture_id:
                ref = VALIDATED_FALSE
        per_bucket[str(d)] = ref
        if ref in accepted:
            refs.add(ref)
        else:
            unvalidated.append(str(d))
    on_disk = VALIDATED_FALSE if unvalidated or not refs else next(iter(refs))
    validated = on_disk if _validity_rank(asserted) >= _validity_rank(on_disk) else VALIDATED_FALSE
    return {"operative": True, "validated": validated, "per_bucket": per_bucket,
            "unvalidated_buckets": unvalidated}


def bind_classifier_validity(
    classifier_state: str | None,
    classifier_dirs: list[str] | tuple[str, ...] | None,
    producing_dirs: list[str] | tuple[str, ...],
    *,
    trait: str,
) -> tuple[str | None, str]:
    """Floor a reconciled classifier stamp to the delivery it is being used to validate.

    Unlike the count dimension (which reconciles from the same buckets it delivers),
    :func:`reconcile_classifier_validity` alone cannot see whether a genuinely-validated stamp was
    calibrated for an unrelated model or trait, it reads only the validity field. A sidecar's own
    recorded ``trait``/``experiment_id`` (written by ``calibrate_classifier_operating_point``) must
    agree with what is actually being delivered. A foreign/unregistered checkpoint calibration
    (``experiment_id=None``) is not rejected for lacking one to compare against; a ``trait`` mismatch
    always is, since the real writer always records one.

    Returns ``(state, note)``, ``state`` floored to ``VALIDATED_FALSE`` on a mismatch, and a
    breeder-readable ``note`` naming which sidecar failed and why (empty when nothing was floored).

    Every delivery door must call this after reconciling, not just the one that first needed it:
    ``compute_phenology`` and the web Results doors share it rather than each composing the flag,
    so the two surfaces cannot disagree about what a classifier stamp means.
    """
    if classifier_state in (None, VALIDATED_FALSE):
        return classifier_state, ""
    producing_experiment_ids = {
        sc["experiment_id"]
        for d in producing_dirs
        if (sc := read_operating_point_sidecar(d)) is not None and sc.get("experiment_id")
    }
    for d in (classifier_dirs or []):
        csc = read_classifier_operating_point_sidecar(d) or {}
        stamped_trait = csc.get("trait")
        stamped_exp = csc.get("experiment_id")
        if stamped_trait != trait:
            return VALIDATED_FALSE, (
                f"classifier_operating_point.json at {d!r} was calibrated for trait "
                f"{stamped_trait!r}, not {trait!r}, the stamp is not trusted for this delivery.")
        if stamped_exp is not None and producing_experiment_ids and stamped_exp not in producing_experiment_ids:
            return VALIDATED_FALSE, (
                f"classifier_operating_point.json at {d!r} was calibrated against experiment "
                f"{stamped_exp!r}, not the producing run ({sorted(producing_experiment_ids)}), "
                "the stamp is not trusted for this delivery.")
    return classifier_state, ""


# --- the delivery gate (one refuse-or-stamp check shared by every phenotype-delivery door) ---

@dataclass(frozen=True)
class DeliveryGateResult:
    """Outcome of the delivery gate: whether the deliverable may be written, and how to stamp it."""

    ok: bool
    unvalidated: tuple[str, ...]  # dimensions whose validity is not a shippable reference
    stamp: dict[str, str]  # per-dimension validity to stamp onto the deliverable
    reason: str = ""  # generic refusal message when not ok

    def column_stamp(self, dimension: str, *, own_column: tuple[str, ...] = ()) -> str:
        """The value the deliverable's column for ``dimension`` carries.

        Not the same thing as ``stamp[dimension]``, which is only that one dimension's own cleared
        reference. A column stands for the trustworthiness of the number beside it, and with
        ``acknowledge_unvalidated`` an ungrounded dimension still reaches the writer, so stamping
        this dimension's own (possibly real) reference alone would report a partly acknowledged
        provisional delivery as fully validated. Every gated dimension without a column of its own
        therefore floors this one. Name in ``own_column`` the dimensions the deliverable does stamp
        into columns of their own; those report themselves and never floor this one.

        Owned here rather than re-derived per door so the doors cannot drift into disagreeing about
        what a validated column means.
        """
        if any(name not in own_column for name in self.unvalidated):
            return VALIDATED_FALSE
        return self.stamp[dimension]


def check_delivery_gate(
    flags: dict[str, str | None], *, acknowledge_unvalidated: bool = False,
) -> DeliveryGateResult:
    """Refuse-or-stamp a phenotype delivery against the validity of each dimension it rests on.

    ``flags`` maps each measurement dimension the deliverable depends on (e.g. ``"operating_point"``,
    ``"classifier"``, or a single ``"measurement"`` for a continuous/ordinal trait with no conf
    op-point) to its reconciled validity state, a shippable reference (any member of
    ``VALIDATED_SHIPPABLE``) or anything else (treated as unvalidated). Read the on-disk state before
    calling; the gate does not trust a caller-asserted string on its own.

    Every dimension validated -> the gate passes. Any not -> it refuses unless
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
