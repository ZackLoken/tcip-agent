"""Whole-dataset content identity: the ``dataset_fingerprint`` formula (labels + image files +
registry + confirmed negatives) and its formula-version stamp, recompute-on-read authority for
the cached value a dataset's own ``dataset.json`` carries.

No torch, safe to import anywhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _labels_term(annotations_root: Path) -> str | None:
    """Whole-dataset label identity, composed from :func:`~tcip_mcp.pipelines.resolution.dataset_hash`
    per label dir.

    Labels live at ``annotations/<date>/*.json`` (date-nested) or flat ``annotations/*.json``.
    ``dataset_hash`` is the single label-byte hasher (flat glob), so this *calls* it per dir and
    combines the per-dir digests keyed by dir name, never re-implementing label hashing. ``None``
    when no labels exist anywhere. Imported lazily, from the source module, at call time (not
    module top level): a caller that monkeypatches ``resolution.dataset_hash`` must reach this call.
    """
    if not annotations_root.is_dir():
        return None
    subdirs = sorted(d for d in annotations_root.iterdir() if d.is_dir())
    flat = not subdirs
    label_dirs = subdirs if subdirs else [annotations_root]
    h = hashlib.sha256()
    any_labels = False
    from tcip_annotation.json_io import prediction_documents
    from tcip_mcp.pipelines.resolution import dataset_hash

    for d in label_dirs:
        if not prediction_documents(d):
            continue
        any_labels = True
        # A real subdir name can never be empty, so the flat root keys with "" rather than its own name,
        # otherwise a dated subdir named literally "annotations" would key identically to the flat case and collide with it.
        key = "" if flat else d.name
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(dataset_hash(d).encode("utf-8"))  # reuse the label-byte hasher, per dir
        h.update(b"\0")
    return h.hexdigest()[:16] if any_labels else None


def _images_term(images_root: Path, cache_path: Path | None) -> str | None:
    """Whole-dataset image identity from each image's *raw file bytes* (content, not name/size), so a
    re-encode under the same filename changes identity (closes the labels-only/pixel-blind gap); a
    ``.bandgroup`` manifest is hashed as its own raw JSON bytes the same way, never decoded pixels.
    Each file's sha is cached by ``(relpath, size, mtime_ns)`` so only changed files re-hash; a
    cache miss always hashes the bytes. ``None`` when there are no images (bespoke/imageless).
    """
    if not images_root.is_dir():
        return None
    from tcip_mcp.pipelines.image_utils import IMAGE_EXTS

    files = sorted(p for p in images_root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
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
    does. Empty string when the dataset has no registry, or when ``read_registry`` refuses it as
    undecodable/malformed (``OSError``/``ValueError``/``RegistryError``).

    A ``schema_version`` this reader does not accept is a different fact from "no registry": it
    is a wrong content identity a delivered number could rest on, not an absent one, so
    ``read_registry``'s :class:`tcip_store.SchemaVersionRefused` is not in this except tuple and
    propagates uncaught, refusing the whole fingerprint computation (:func:`dataset_fingerprint`)
    rather than folding into the empty-string "no registry" answer.
    """
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

    Reads the dataset's own confirmed-negative store through
    ``dataset_layout.read_image_status_store``, the one read every other consumer of that store
    goes through, and never a foreign store, since confirmations are dataset-native the same way
    ``_registry_term`` reads only ``classes_path``. Hashes stored negative membership, not the
    quarantine-filtered view
    ``confirmed_negative_names`` returns: fingerprint is content identity (should two datasets be
    considered the same content), quarantine is a training-time trust decision, and conflating them
    would make the fingerprint *less* sensitive to a real on-disk difference than it should be, the
    unsafe direction. Empty string when there is no store or no negative entries, matching
    ``_registry_term``'s "optional, additive" convention so a dataset with zero confirmed negatives
    still gets a valid non-None fingerprint.
    """
    from tcip_mcp.dataset_layout import (
        is_confirmed_negative, normalize_status_store, read_image_status_store,
    )

    statuses = normalize_status_store(read_image_status_store(dataset_root))
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


FINGERPRINT_FORMULA_VERSION = 1
"""Bumped whenever the four terms :func:`dataset_fingerprint` hashes change (a new extension
walked, a term added or dropped): the value each formula version stamps must never compare as
same-or-different against a value another formula stamped, so the fingerprint carries this as
a prefix rather than the bare hex Part 14 Q2 found silently reused across two formula shifts."""

_FINGERPRINT_PATTERN = re.compile(r"^v(\d+):([0-9a-f]+)$")


def fingerprint_formula_version(value: object) -> int | None:
    """The formula version a stored fingerprint states, or ``None`` for anything that is not
    the ``v<n>:<hex>`` shape: a bare legacy value from before this prefix existed, or a value
    that is not a fingerprint at all. A non-``None`` result never says the formula matches the
    version this code computes under; only equal integers do."""
    if not isinstance(value, str):
        return None
    m = _FINGERPRINT_PATTERN.match(value)
    return int(m.group(1)) if m else None


def dataset_fingerprint(dataset_root: str | Path) -> str | None:
    """Whole-dataset content identity: labels + image files + registry + confirmed negatives.

    A superset of :func:`~tcip_mcp.pipelines.resolution.dataset_hash` (which stays the
    per-split-subset firewall key): the label term *calls* ``dataset_hash``; the image term
    hashes each file's raw bytes; the registry term digests the canonical class registry; the
    confirmations term digests the dataset-native confirmed-negative store.
    Content-addressed, so it is machine-independent (a moved dataset keeps its fingerprint) and detects
    a change to any of the four (a re-encode, a relabel, a registry edit, confirming/un-confirming a
    negative). ``None`` for a dataset with no images or no labels (e.g. a bespoke ``dataset_source``),
    matching ``dataset_hash``'s honesty rather than fabricating identity. Authority is recompute-on-read;
    a stored fingerprint (``dataset.json``) is a cache.

    Adding the confirmations term was a one-time formula-version shift: recomputing against an
    existing ``dataset.json``/experiment ``lineage.json`` written under the 3-term formula reads as
    changed even with identical on-disk content. That is expected, not corruption, experiments are
    immutable, so old lineage records keep their old fingerprint value rather than being rewritten.
    A second such shift: the image term now walks ``image_utils.IMAGE_EXTS`` (``.heic``/``.npy``/
    ``.npz``/``.bandgroup`` included, a ``.bandgroup`` manifest hashed as its own bytes like any
    other file) rather than a narrower photographic-only set, so a dataset holding one of those four
    extensions gets a new value and a multispectral-only dataset gets a real fingerprint instead of
    ``None``.

    The return value carries its formula version as a ``v<n>:<hex>`` prefix
    (:data:`FINGERPRINT_FORMULA_VERSION`), formula ``1`` being the four-term one this docstring
    describes, so two values computed under different formulas can never read as equal or
    unequal by accident: :func:`fingerprint_formula_version` names the formula a stored value
    states, and a bare legacy value (from before this prefix existed) names none.
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
    return f"v{FINGERPRINT_FORMULA_VERSION}:{h.hexdigest()[:16]}"
