"""Whole-dataset content identity: the ``dataset_fingerprint`` formula (labels + image files +
registry), recompute-on-read authority for the cached value a dataset's own ``dataset.json`` carries.

No torch, safe to import anywhere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _labels_term(dataset_root: Path) -> str | None:
    """Whole-dataset label identity, composed from each label document's capture, stem and
    :func:`~tcip_mcp.pipelines.data.selection.ground_truth_digest`. ``None`` when the dataset
    holds no label document."""
    import tcip_store

    from tcip_mcp.dataset_layout import LABEL_DOCUMENTS
    from tcip_mcp.pipelines.data.selection import ground_truth_digest

    keys = tcip_store.keys(LABEL_DOCUMENTS, str(dataset_root.resolve()))
    h = hashlib.sha256()
    for key in keys:
        h.update("\0".join((*key.parts, ground_truth_digest(key), "")).encode("utf-8"))
    return h.hexdigest()[:16] if keys else None


def _images_term(images_root: Path) -> str | None:
    """Whole-dataset image identity from each image file's byte digest
    (:func:`~tcip_store.read_blob_versioned`), read from the bytes on every call; a
    ``.bandgroup`` manifest digests as its own bytes. ``None`` when there are no images."""
    if not images_root.is_dir():
        return None
    from tcip_store import read_blob_versioned

    from tcip_mcp.pipelines.image_utils import IMAGE_EXTS

    files = sorted(p for p in images_root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(images_root).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(read_blob_versioned(f).version.token.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _registry_term(dataset_root: Path) -> str:
    """Digest over the canonical registry serialization (``registry_to_dict``) in declared order;
    empty when the dataset has no registry, and a registry ``read_registry`` refuses raises."""
    from tcip_annotation.json_io import canonical_digest

    from tcip_mcp.subject_registry import read_registry, registry_to_dict
    from tcip_mcp.dataset_layout import subjects_path

    if not subjects_path(dataset_root).is_file():
        return ""
    return canonical_digest(registry_to_dict(read_registry(dataset_root)))


def dataset_fingerprint(dataset_root: str | Path) -> str | None:
    """Whole-dataset content identity over :func:`_labels_term`, :func:`_images_term` and
    :func:`_registry_term`, so a moved dataset keeps its fingerprint and a change to any of the
    three changes it. ``None`` for a dataset with no images or no labels."""
    from tcip_mcp.dataset_layout import image_root

    root = Path(dataset_root)
    labels = _labels_term(root)
    images = _images_term(image_root(root))
    if labels is None or images is None:
        return None
    h = hashlib.sha256()
    h.update(b"labels:")
    h.update(labels.encode("utf-8"))
    h.update(b"\0images:")
    h.update(images.encode("utf-8"))
    h.update(b"\0classes:")
    h.update(_registry_term(root).encode("utf-8"))
    return h.hexdigest()[:16]
