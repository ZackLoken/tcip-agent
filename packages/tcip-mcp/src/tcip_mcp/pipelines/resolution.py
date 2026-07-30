"""Runtime parameter resolution — the "derive, don't pin" currency.

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

Pure stdlib — no torch, safe to import anywhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# --- vocabularies ---------------------------------------------------------
SOURCES = ("explicit", "derived", "default")

# validated_against records WHICH reference confirmed a value that requires validation — the
# shared-reference principle (CLAUDE.md): a reference sized to the trait, not dense GT for every
# trait. "false" is a real, distinct value (not Python None) so a sidecar/provenance record can say
# "checked, and it wasn't" rather than leaving the field merely absent.
VALIDATED_HELD_OUT = "held_out_annotations"          # a disjoint held-out split of this dataset's GT
VALIDATED_REVIEW_CONFIRMED = "reviewer_confirmed_annotations"  # a breeder-confirmed output sample
VALIDATED_PHYSICAL_MEASUREMENT = "physical_measurement"  # checked against a known physical dimension
VALIDATED_FALSE = "false"

# Which validated_against values legitimately clear validation for which KIND of thing being
# validated — an annotation-count operating point (conf, a mask-binarize threshold) and a physical
# scale are checked against fundamentally different references, and neither may satisfy the other's
# requirement (that cross-satisfaction was a real, closed hole: see resolution.py's own history).
VALIDATION_KINDS = ("annotations", "physical")
_ACCEPTED_REFERENCES: dict[str, tuple[str, ...]] = {
    "annotations": (VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED),
    "physical": (VALIDATED_PHYSICAL_MEASUREMENT,),
}


def accepted_references(validation_kind: str) -> tuple[str, ...]:
    """The validated_against values that legitimately clear validation for this kind."""
    return _ACCEPTED_REFERENCES[validation_kind]


# The union of every real (non-"false") shippable reference, across every kind — used only by
# dimension-agnostic logic (check_delivery_gate, _validity_rank) whose callers have already
# resolved the right kind per-dimension upstream via accepted_references()/is_shippable. Never used
# to decide whether a SPECIFIC param's reference is the right kind for it — that decision belongs to
# accepted_references(validation_kind), always.
VALIDATED_SHIPPABLE = (VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED, VALIDATED_PHYSICAL_MEASUREMENT)

# Shared inference operating-point defaults, referenced by both run_inference and the web route so the
# same model+images can't give a different count by entry point.
DEFAULT_CONF = 0.5
DEFAULT_NMS_IOU = 0.3
DEFAULT_TILED = True
DEFAULT_TILE_SIZE = 640
DEFAULT_OVERLAP = 0.2
# A high full-frame detection cap so dense scenes (hundreds of catkins) aren't silently truncated
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

    Most parameters never need validation — a fact read from the data (``in_chans``), a statistic
    computed from this dataset's own spread (``cross_tile_nms``), or a plain configuration default
    (``tiled``) are all trustworthy by construction (``requires_validation=False``, the default).
    ``derived_from`` (free text) is where *how it was produced* is described — there is no separate
    category label to learn on top of that.

    A parameter that DOES need validation (a confidence threshold, a physical scale) sets
    ``requires_validation=True`` and a real ``validation_kind`` (what KIND of reference can validate
    it — see ``VALIDATION_KINDS``); it is shippable only once ``validated_against`` names a reference
    ``accepted_references(validation_kind)`` recognizes for that kind.
    """

    name: str
    _raw: Any
    source: str  # one of SOURCES
    derived_from: str = ""  # human-readable: what artifact/analysis produced it
    requires_validation: bool = False
    validation_kind: str | None = None  # one of VALIDATION_KINDS — required iff requires_validation
    validated_against: str | None = None  # None | VALIDATED_HELD_OUT | VALIDATED_REVIEW_CONFIRMED |
                                           # VALIDATED_PHYSICAL_MEASUREMENT | VALIDATED_FALSE
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
                f"{VALIDATION_KINDS}, got {self.validation_kind!r} — a param that needs validation "
                f"must say what kind of reference can validate it, never left to a silent default."
            )
        if not self.requires_validation and self.validation_kind is not None:
            raise ValueError(
                f"{self.name!r}: validation_kind={self.validation_kind!r} set but "
                f"requires_validation=False — a kind with nothing to validate is a contradiction."
            )
        if self.validated_against is not None and self.validated_against not in (
                VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED, VALIDATED_PHYSICAL_MEASUREMENT, VALIDATED_FALSE):
            raise ValueError(f"validated_against invalid: {self.validated_against!r}")

    @property
    def is_shippable(self) -> bool:
        """Can this value flow into a delivered result as a trustworthy number?

        A param that doesn't require validation is always shippable. One that does is shippable only
        once ``validated_against`` is a reference ``accepted_references(validation_kind)`` recognizes
        for its own kind — a physical scale can never be satisfied by an annotation-based reference,
        or vice versa.
        """
        if not self.requires_validation:
            return True
        return self.validated_against in accepted_references(self.validation_kind)

    @property
    def value(self) -> Any:
        """The trustworthy value — raises the firewall for an unvalidated param that requires it."""
        if not self.is_shippable:
            raise UnvalidatedOperatingPointError(
                f"{self.name!r} requires validation ({self.validation_kind}) and has "
                f"validated_against={self.validated_against!r} — cannot be consumed as a trustworthy "
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
    """Convenience constructor for a documented default (``source="default"``) — never requires
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
    classifier_validated_vs_gt: str | None = None  # for the elongation classifier

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
                    f"different dataset {target_dataset_hash} — re-resolve per dataset, never inherit"
                )
            if p.capture_scoped and p.capture_id is None:
                # No deriver for capture_id exists yet — an honest "not comparable" marker rather
                # than silently passing a check that cannot actually run.
                issues.append(
                    f"{p.name}: capture-scoped but no capture_id was ever derived for it — "
                    "cross-capture reuse cannot be checked (not comparable, not confirmed safe)"
                )
            elif p.capture_scoped and target_capture_id is not None and p.capture_id != target_capture_id:
                issues.append(
                    f"{p.name}: capture-scoped value derived for capture {p.capture_id} inherited "
                    f"across a different capture {target_capture_id} — re-resolve per capture, "
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


def raw_operating_point(
    *, conf: float, cross_tile_nms: float | None, tiled: bool, tile_size: int | None,
    max_dets: int, tile_size_source: str = "default", tiled_source: str = "default",
) -> ResolvedBundle:
    """The operating point for RAW (uncalibrated) inference — the one both doors resolve through.

    ``conf`` is a documented default with no per-dataset GT behind it, so it requires validation and
    is stamped ``validated_against=false``: reading it requires ``unvalidated_value(...)`` and the
    caller must stamp its output ``validated=false``. This is what stops the MCP tool and the web job
    giving a different count (the phenotype) for the same model + images by entry point.

    ``tile_size_source`` records whether the tile edge was ``derived`` from the checkpoint's training
    geometry, ``explicit`` (caller override), or a ``default`` fallback (CV2) — so a 224-train /
    640-infer scale mismatch is visible in the provenance rather than silent. ``tiled_source`` is the
    same vocabulary for the boolean itself (K10 finding 3) — a caller that explicitly chose to tile
    (or not) stamps ``"explicit"``; a caller who passed nothing gets ``"default"``. Both callers of
    this function resolve their own bool once (``None`` sentinel -> ``DEFAULT_TILED`` internally)
    before reaching here, so ``tiled`` itself is always a concrete bool.
    """
    if tiled and tile_size_source == "derived":
        tile_param = derived("tile_size", tile_size, derived_from="persisted training tile geometry")
    elif tiled and tile_size_source == "explicit":
        tile_param = ResolvedParam("tile_size", tile_size, source="explicit",
                                   derived_from="caller override")
    else:
        tile_param = default("tile_size", tile_size if tiled else None)
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


_FINGERPRINT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _labels_term(annotations_root: Path) -> str | None:
    """Whole-dataset label identity, composed from :func:`dataset_hash` per label dir.

    Labels live at ``annotations/<date>/*.json`` (date-nested) or flat ``annotations/*.json``.
    ``dataset_hash`` is the single label-byte hasher (flat glob), so this *calls* it per dir and
    combines the per-dir digests keyed by dir name — never re-implementing label hashing. ``None``
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
        # name — otherwise a dated subdir named literally "annotations" would key identically to the
        # flat case and collide with it.
        key = "" if flat else d.name
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(dataset_hash(d).encode("utf-8"))  # reuse the label-byte hasher, per dir
        h.update(b"\0")
    return h.hexdigest()[:16] if any_labels else None


def _images_term(images_root: Path, cache_path: Path | None) -> str | None:
    """Whole-dataset image identity from each image's *pixel bytes* (content, not name/size) — so a
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

    Reads ``dataset_layout.image_status_path`` only — never a foreign/legacy store — since
    confirmations are now dataset-native (K13.5 slice 4), the same way ``_registry_term`` reads only
    ``classes_path``. Hashes RAW on-disk negative membership, not the quarantine-filtered view
    ``confirmed_negative_names`` returns: fingerprint is content identity (should two datasets be
    considered the same content), quarantine is a training-time trust decision, and conflating them
    would make the fingerprint *less* sensitive to a real on-disk difference than it should be — the
    unsafe direction. Empty string when there is no store or no negative entries, matching
    ``_registry_term``'s "optional, additive" convention so a dataset with zero confirmed negatives
    still gets a valid non-None fingerprint.
    """
    from tcip_mcp.dataset_layout import image_status_path

    p = image_status_path(dataset_root)
    if not p.is_file():
        return ""
    try:
        statuses = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(statuses, dict):
        return ""
    h = hashlib.sha256()
    any_negatives = False
    for bucket in sorted(k for k in statuses if isinstance(statuses[k], dict)):
        names = sorted(n for n, s in statuses[bucket].items() if s == "negative")
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
    negative). ``None`` for a dataset with no images or no labels (e.g. a bespoke ``dataset_source``) —
    matching ``dataset_hash``'s honesty rather than fabricating identity. Authority is recompute-on-read;
    a stored fingerprint (``dataset.json``) is a cache.

    Adding the confirmations term is a one-time formula-version shift (K13.5 slice 4): recomputing
    against an existing ``dataset.json``/experiment ``lineage.json`` written under the 3-term formula
    reads as CHANGED even with identical on-disk content. That is expected, not corruption — experiments
    are immutable, so old lineage records keep their old fingerprint value rather than being rewritten.
    """
    root = Path(dataset_root)
    labels = _labels_term(root / "annotations")
    images = _images_term(root / "images", root / ".tcip" / "state" / "image_hash_cache.json")
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


def read_classifier_operating_point_sidecar(pred_dir: str | Path) -> dict | None:
    """The bucket's ``classifier_operating_point.json`` stamp, or ``None`` if absent/unreadable.

    A file distinct from ``operating_point.json`` (K3) — the classifier-validity dimension is
    structurally independent from the count operating point's, so the two are never written to the
    same fields a generic writer could conflate (``_sidecar_reference`` reads exactly
    ``validated``/``operating_point.conf.validated_against``, which must stay the count dimension's
    alone).
    """
    p = Path(pred_dir) / "classifier_operating_point.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _sidecar_reference(
    sidecar: dict | None, *, param_key: str = "conf", validation_kind: str = "annotations",
) -> str:
    """Which reference the sidecar's named param cleared, for its ``validation_kind`` — or
    ``VALIDATED_FALSE``.

    Never upgrades a missing/unrecognized/wrong-kind value to a shippable reference — a param whose
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
    assertion (skip). Comparing rank between two states already scoped to the SAME dimension/param —
    never used to decide whether a state is the right KIND for a given param (that is
    ``accepted_references``'s job)."""
    if state is None:
        return 99
    return 1 if state in VALIDATED_SHIPPABLE else 0


def _reconcile_validity(
    pred_dirs: list[str] | tuple[str, ...], *, asserted: str | None,
    read_sidecar: Callable[[str | Path], dict | None], param_key: str,
    validation_kind: str = "annotations",
) -> dict:
    """Floor a validity dimension against every bucket's on-disk sidecar (T5-3 fix), generalized.

    Shared by :func:`reconcile_operating_point_validity` and :func:`reconcile_classifier_validity` —
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
    on-disk sidecar and takes the FLOOR of asserted-vs-on-disk. A missing/unreadable sidecar, or any
    bucket stamped ``validated=false``, floors the whole curve to ``false`` — never a crash. See
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
    count operating point — the same function, parameterized to a different sidecar file and param
    key (K3) — never a hand-written sibling. A bucket with no persisted classifier-calibration run
    floors to ``false``: there is no legitimate way to earn a classifier-validated stamp without one
    (TRAP 2), so this never falls back to a caller-asserted string.
    """
    return _reconcile_validity(
        pred_dirs, asserted=asserted,
        read_sidecar=read_classifier_operating_point_sidecar, param_key="classifier",
        validation_kind="annotations",
    )


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
    calibrated for an unrelated model or trait — it reads only the validity field. A sidecar's own
    recorded ``trait``/``experiment_id`` (written by ``calibrate_classifier_operating_point``) must
    agree with what is actually being delivered. A foreign/unregistered checkpoint calibration
    (``experiment_id=None``, the K3 owner decision) is not rejected for lacking one to compare
    against; a ``trait`` mismatch always is, since the real writer always records one.

    Returns ``(state, note)`` — ``state`` floored to ``VALIDATED_FALSE`` on a mismatch, and a
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
                f"{stamped_trait!r}, not {trait!r} — the stamp is not trusted for this delivery.")
        if stamped_exp is not None and producing_experiment_ids and stamped_exp not in producing_experiment_ids:
            return VALIDATED_FALSE, (
                f"classifier_operating_point.json at {d!r} was calibrated against experiment "
                f"{stamped_exp!r}, not the producing run ({sorted(producing_experiment_ids)}) — "
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


def check_delivery_gate(
    flags: dict[str, str | None], *, acknowledge_unvalidated: bool = False,
) -> DeliveryGateResult:
    """Refuse-or-stamp a phenotype delivery against the validity of each dimension it rests on.

    ``flags`` maps each measurement dimension the deliverable depends on (e.g. ``"operating_point"``,
    ``"classifier"``, or a single ``"measurement"`` for a continuous/ordinal trait with no conf
    op-point) to its RECONCILED validity state — a shippable reference (any member of
    ``VALIDATED_SHIPPABLE``) or anything else (treated as unvalidated). Read the on-disk state before
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

    # eval op-point must match inference op-point on the same dataset — every param the two bundles
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
