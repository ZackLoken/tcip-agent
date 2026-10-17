"""Whole-dataset content identity over the label documents, image files and subject registry.

The identity is :func:`dataset_fingerprint`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from tcip_store import Key

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import Sample


def _labels_term(
    dataset_root: Path, versions: Mapping[Key, str]
) -> str | None:
    """Return the whole-dataset label identity, or ``None`` when it holds no label document.

    The identity is composed from each label document's capture, stem and version: the one
    ``versions`` holds for it, else its
    :func:`~tcip_mcp.pipelines.data.selection.ground_truth_digest`.
    """
    import tcip_store
    from tcip_mcp.dataset_layout import LABEL_DOCUMENTS
    from tcip_mcp.pipelines.data.selection import ground_truth_digest

    keys = tcip_store.keys(LABEL_DOCUMENTS, str(dataset_root))
    h = hashlib.sha256()
    for key in keys:
        version = (
            versions[key] if key in versions else ground_truth_digest(key)
        )
        h.update("\0".join((*key.parts, version, "")).encode("utf-8"))
    return h.hexdigest()[:16] if keys else None


def _images_term(images_root: Path) -> str | None:
    """Return the whole-dataset image identity, or ``None`` when there are no images.

    The identity is composed from each image file's byte digest
    (:func:`~tcip_store.read_blob_versioned`), read from the bytes on every call; a
    ``.bandgroup`` manifest digests as its own bytes.
    """
    if not images_root.is_dir():
        return None
    from tcip_mcp.pipelines.image_utils import IMAGE_EXTS
    from tcip_store import read_blob_versioned

    files = sorted(
        p
        for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(images_root).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(read_blob_versioned(f).version.token.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _registry_term(dataset_root: Path) -> str | None:
    """Return the digest of the dataset's canonical registry serialization, or ``None``.

    The serialization is ``registry_to_dict``'s, in declared order; ``None`` when the dataset has
    no registry, and a registry ``read_registry`` refuses raises.
    """
    from tcip_annotation.json_io import canonical_digest
    from tcip_mcp.subject_registry import read_registry, registry_to_dict

    try:
        registry = read_registry(dataset_root)
    except FileNotFoundError:
        return None
    return canonical_digest(registry_to_dict(registry))


def dataset_fingerprint(
    dataset_root: str | Path, admitted: Iterable[Sample] = ()
) -> str | None:
    """Return the whole-dataset content identity, or ``None`` with no images or no labels.

    The identity is over :func:`_labels_term`, :func:`_images_term` and :func:`_registry_term`,
    so a moved dataset keeps its fingerprint and a change to any of the three changes it; a
    label document among the ``admitted`` samples enters at the version its admission read
    rather than being read again.
    """
    from tcip_annotation.json_io import canonical_digest
    from tcip_mcp.dataset_layout import image_root

    root = Path(dataset_root)
    versions = {
        s.ground_truth: s.ground_truth_digest
        for s in admitted
        if isinstance(s.ground_truth, Key)
        and s.ground_truth_digest is not None
    }
    terms = {
        "labels": _labels_term(root, versions),
        "images": _images_term(image_root(root)),
        "classes": _registry_term(root),
    }
    if terms["labels"] is None or terms["images"] is None:
        return None
    return canonical_digest(terms)
