"""What a project bundle holds: the membership accounting ``archive_project``, ``import_project``
and ``tcip_mcp.stray_state`` compose from.

Roots are derived from the tree's own structure plus the anchored documents the platform's own
writers place (``selection.json``, ``curated_manifest.json``); an anchor found somewhere the
derivation constraints exclude (the tree root, under ``.tcip``, under a blob home, or under or
above another derived root) raises :class:`AnchorMisplaced` naming the file. One nesting is
admitted: a splits root sitting under a curated root. Classification of one file is by precedence:
bookkeeping first, then a record or log claimed by exactly one derived root's own layout (two
derived roots claiming the same file raises :class:`CrossRootCollision`), then a recognized blob
home, then everything else, unaccounted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from tcip_store.adoption import AdoptionPlan, plan_root
from tcip_store.file_backend import _is_bookkeeping
from tcip_store.layout_claims import CURATED, EXPERIMENTS, HPO_ROOT, ROOT, RUN, SPLITS, STATE, SWEEP

from tcip_mcp.registry_paths import is_at_or_under as _is_at_or_under

SELECTION_NAME = "selection.json"
CURATED_MANIFEST_NAME = "curated_manifest.json"


class AnchorMisplaced(ValueError):
    """A split or curated manifest sits somewhere the derivation constraints exclude."""


class CrossRootCollision(ValueError):
    """A file is claimed by two different derived roots at once."""


@dataclass(frozen=True)
class DerivedRoot:
    """One directory this tree serves as, and the layout naming its kind."""

    path: Path
    layout: str



@dataclass(frozen=True)
class BundleAccounting:
    """Every member of one project tree, classified.

    ``plans`` is one :class:`~tcip_store.adoption.AdoptionPlan` per derived root (entries already
    de-duplicated within their own root by :func:`~tcip_store.adoption.plan_root`); ``blobs`` is
    every file under a recognized blob home that no plan already adopts; ``bookkeeping`` and
    ``unaccounted`` are class 1 and class 4 respectively, over every other file the tree holds;
    ``collisions`` names any file two different derived roots both adopted.
    ``registered_checkpoints`` is the subset of ``blobs`` :func:`blob_home` calls
    ``BLOB_CHECKPOINTS`` on the strength of a model registry entry rather than sitting under
    ``.tcip/models``, computed while ``tree`` still holds its own registry index.
    """

    tree: Path
    derived: tuple[DerivedRoot, ...]
    plans: tuple[AdoptionPlan, ...]
    blobs: tuple[Path, ...]
    bookkeeping: tuple[Path, ...]
    unaccounted: tuple[Path, ...]
    collisions: tuple[Path, ...]
    registered_checkpoints: frozenset[Path]


def _validate_anchor(
    tree: Path, directory: Path, filename: str, others: list[Path],
    image_root: Path, annotation_root: Path,
    curated_dirs: frozenset[Path] = frozenset(), split_dirs: frozenset[Path] = frozenset(),
) -> None:
    member = directory / filename
    if directory == tree:
        raise AnchorMisplaced(f"{member} sits at the tree root, which no {filename} may claim")
    if _is_at_or_under(directory, tree / ".tcip"):
        raise AnchorMisplaced(f"{member} sits under .tcip, which no {filename} may claim")
    if _is_at_or_under(directory, image_root):
        raise AnchorMisplaced(f"{member} sits under the image tree, which no {filename} may claim")
    if _is_at_or_under(directory, annotation_root):
        raise AnchorMisplaced(f"{member} sits under the annotation tree, which no {filename} may claim")
    for other in others:
        # A curated dataset split in place: the one nesting the producer chain admits.
        if directory in split_dirs and other in curated_dirs and _is_at_or_under(directory, other):
            continue
        if other in split_dirs and directory in curated_dirs and _is_at_or_under(other, directory):
            continue
        if _is_at_or_under(directory, other) or _is_at_or_under(other, directory):
            raise AnchorMisplaced(f"{member} sits under or above another derived root, {other}")


def _anchored_dirs(tree: Path, filename: str) -> tuple[Path, ...]:
    """Every directory under ``tree`` holding a top-level file named ``filename``."""
    return tuple(sorted({found.parent for found in tree.rglob(filename) if found.is_file()}))


def derive_roots(tree: str | Path) -> tuple[DerivedRoot, ...]:
    """Every root ``tree`` is, or holds, per the platform's own writers and anchors. Raises
    :class:`AnchorMisplaced` naming the file when a split or curated manifest sits somewhere the
    constraints exclude.
    """
    from tcip_mcp.dataset_layout import annotation_root as _annotation_root
    from tcip_mcp.dataset_layout import image_root as _image_root

    root = Path(tree).resolve()
    derived: list[DerivedRoot] = [DerivedRoot(root, ROOT)]
    state, experiments, hpo = root / ".tcip" / "state", root / ".tcip" / "experiments", root / ".tcip" / "hpo"
    derived += [DerivedRoot(state, STATE), DerivedRoot(experiments, EXPERIMENTS), DerivedRoot(hpo, HPO_ROOT)]

    if experiments.is_dir():
        for child in sorted(experiments.iterdir()):
            if child.is_dir() and not _is_bookkeeping(child.name):
                derived.append(DerivedRoot(child, RUN))
    if hpo.is_dir():
        for child in sorted(hpo.iterdir()):
            if child.is_dir() and not _is_bookkeeping(child.name):
                derived.append(DerivedRoot(child, SWEEP))

    image_root, annotation_root = _image_root(root), _annotation_root(root)
    split_dirs = _anchored_dirs(root, SELECTION_NAME)
    curated_dirs = _anchored_dirs(root, CURATED_MANIFEST_NAME)
    every_anchor = [*split_dirs, *curated_dirs]
    curated_set, split_set = frozenset(curated_dirs), frozenset(split_dirs)
    for directory in split_dirs:
        _validate_anchor(root, directory, SELECTION_NAME,
                          [d for d in every_anchor if d != directory], image_root, annotation_root,
                          curated_dirs=curated_set, split_dirs=split_set)
        derived.append(DerivedRoot(directory, SPLITS))
    for directory in curated_dirs:
        _validate_anchor(root, directory, CURATED_MANIFEST_NAME,
                          [d for d in every_anchor if d != directory], image_root, annotation_root,
                          curated_dirs=curated_set, split_dirs=split_set)
        derived.append(DerivedRoot(directory, CURATED))
    return tuple(derived)


def _resolve_checkpoint_entry(tree: Path, raw: str) -> Path | None:
    """The absolute, existing-under-``tree`` path a registry's ``checkpoint_path`` entry (``raw``,
    exactly as stored) resolves to through :func:`~tcip_mcp.registry_paths.resolved_registry_path`,
    or ``None`` when it does not: an external entry naming a different tree entirely, a relative
    one with nothing at it any more, or a value this reader cannot resolve at all (empty, or
    carrying a ``..`` segment).
    """
    from tcip_mcp.registry_paths import RegistryPathEmpty, RegistryPathTraversal, resolved_registry_path

    try:
        resolved = resolved_registry_path(tree, raw).resolve()
    except (RegistryPathEmpty, RegistryPathTraversal):
        return None
    return resolved if resolved.is_file() and _is_at_or_under(resolved, tree) else None


def _stored_checkpoint_paths(tree: Path) -> list[str]:
    """Every registry entry's ``checkpoint_path`` under ``tree``, exactly as stored.

    Propagates :class:`~tcip_mcp.model_registry.RegistryVersionRefused` and
    :class:`tcip_store.SchemaVersionRefused`; every other :class:`tcip_store.StoreError` (an
    absent index, undecodable bytes, or an unadopted staging tree under the database backend)
    reads as none.
    """
    from tcip_store import SchemaVersionRefused, StoreError

    from tcip_mcp.model_registry import read_registry_index

    try:
        entries = read_registry_index(tree)
    except SchemaVersionRefused:
        raise
    except StoreError:
        return []
    return [entry["checkpoint_path"] for entry in entries]


def _registered_checkpoint_paths(tree: Path) -> frozenset[Path]:
    """Every checkpoint a model registry entry under ``tree`` points at, resolved, restricted to
    ones inside ``tree``; refusals as :func:`_stored_checkpoint_paths`."""
    found = (_resolve_checkpoint_entry(tree, raw) for raw in _stored_checkpoint_paths(tree))
    return frozenset(path for path in found if path is not None)


def unresolved_registered_checkpoints(tree: Path) -> tuple[str, ...]:
    """Every registry ``checkpoint_path`` under ``tree`` that is not a designed-external claim
    (:func:`~tcip_mcp.registry_paths.is_external_form`) and names no file under it, spelled as
    stored; refusals as :func:`_stored_checkpoint_paths`."""
    from tcip_mcp.registry_paths import is_external_form

    return tuple(sorted(
        raw for raw in _stored_checkpoint_paths(tree)
        if not is_external_form(raw) and _resolve_checkpoint_entry(tree, raw) is None
    ))


def external_registered_checkpoints(tree: Path) -> tuple[dict, ...]:
    """Every registry ``checkpoint_path`` under ``tree`` that is a designed-external claim, each
    as ``{"checkpoint_path", "exists"}``; refusals as :func:`_stored_checkpoint_paths`."""
    from tcip_mcp.registry_paths import is_external_form

    found = [
        {"checkpoint_path": raw, "exists": Path(raw).is_file()}
        for raw in _stored_checkpoint_paths(tree) if is_external_form(raw)
    ]
    return tuple(sorted(found, key=lambda d: d["checkpoint_path"]))


def _blob_files(
    tree: Path, claimed: frozenset[str], registered_checkpoints: frozenset[Path],
) -> tuple[Path, ...]:
    """Every file under a recognized blob home that no record or log plan already adopts, each
    named once even when more than one recognized home would otherwise find the same file.

    A ``.pt`` file anywhere under ``.tcip/experiments/`` is found by shape alone, in addition to a
    registry-named path and the ``.tcip/models`` location.
    """
    from tcip_mcp.dataset_layout import LABEL_SUFFIX
    from tcip_mcp.dataset_layout import annotation_root as _annotation_root
    from tcip_mcp.dataset_layout import dataset_identity_path, subjects_path
    from tcip_mcp.dataset_layout import image_root as _image_root
    from tcip_mcp.pipelines.image_utils import IMAGE_EXTS

    found: list[Path] = []
    seen: set[str] = set()

    def _add(candidate: Path) -> None:
        marker = os.path.normcase(str(candidate))
        if candidate.is_file() and not _is_bookkeeping(candidate.name) \
                and marker not in claimed and marker not in seen:
            seen.add(marker)
            found.append(candidate)

    image_dir = _image_root(tree)
    if image_dir.is_dir():
        for f in image_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                _add(f)
    ann_dir = _annotation_root(tree)
    if ann_dir.is_dir():
        for f in ann_dir.rglob(f"*{LABEL_SUFFIX}"):
            _add(f)
    _add(subjects_path(tree))
    _add(dataset_identity_path(tree))

    experiments = tree / ".tcip" / "experiments"
    if experiments.is_dir():
        for run_dir in experiments.iterdir():
            model_src = run_dir / "model_src"
            if model_src.is_dir():
                for f in model_src.rglob("*"):
                    _add(f)
        for f in experiments.rglob("*.pt"):
            _add(f)
    models_dir = tree / ".tcip" / "models"
    if models_dir.is_dir():
        for f in models_dir.glob("*.pt"):
            _add(f)
    for f in registered_checkpoints:
        _add(f)
    return tuple(found)


BLOB_IMAGERY = "imagery"
BLOB_LABELS = "labels"
BLOB_SUBJECT_REGISTRY = "subject_registry"
BLOB_DATASET_IDENTITY = "dataset_identity"
BLOB_MODEL_SRC = "model_src"
BLOB_CHECKPOINTS = "checkpoints"
BLOB_OTHER = "other"

BLOB_HOMES = (
    BLOB_IMAGERY, BLOB_LABELS, BLOB_SUBJECT_REGISTRY, BLOB_DATASET_IDENTITY, BLOB_MODEL_SRC,
    BLOB_CHECKPOINTS, BLOB_OTHER,
)
"""Every home a blob :func:`account_for` finds can belong to, in the same terms
:func:`_blob_files` finds them by, so a caller disclosing what it bundled or dropped names the
same homes rather than re-deriving its own notion of what a blob is."""


def blob_home(
    tree: Path, path: Path, registered_checkpoints: frozenset[Path] = frozenset(),
) -> str:
    """Which recognized blob home ``path`` (already known to be one of ``account_for``'s blobs)
    belongs to, in the same terms :func:`_blob_files` found it by.

    ``registered_checkpoints`` (``BundleAccounting.registered_checkpoints``) names every checkpoint
    a registry entry points at outside ``.tcip/models``; pass it back in for a caller classifying
    blobs after the tree has moved. A ``.pt`` file under ``.tcip/experiments/`` is recognized as a
    checkpoint by shape alone. Omitted, this still recognizes every checkpoint physically under
    ``.tcip/models`` or shaped as one under ``.tcip/experiments/``.

    A ``model_src`` snapshot is classified before either checkpoint clause is consulted, whatever
    ``include_models`` says.
    """
    from tcip_mcp.dataset_layout import annotation_root as _annotation_root
    from tcip_mcp.dataset_layout import dataset_identity_path, subjects_path
    from tcip_mcp.dataset_layout import image_root as _image_root

    if path == subjects_path(tree):
        return BLOB_SUBJECT_REGISTRY
    if path == dataset_identity_path(tree):
        return BLOB_DATASET_IDENTITY
    if _is_at_or_under(path, _image_root(tree)):
        return BLOB_IMAGERY
    if _is_at_or_under(path, _annotation_root(tree)):
        return BLOB_LABELS
    experiments = tree / ".tcip" / "experiments"
    if _is_at_or_under(path, experiments):
        rel = path.relative_to(experiments).parts
        if len(rel) >= 2 and rel[1] == "model_src":
            return BLOB_MODEL_SRC
    if path.parent == tree / ".tcip" / "models" or path in registered_checkpoints:
        return BLOB_CHECKPOINTS
    if _is_at_or_under(path, experiments) and path.suffix == ".pt":
        return BLOB_CHECKPOINTS
    return BLOB_OTHER


def _cross_root_collisions(plans: tuple[AdoptionPlan, ...]) -> tuple[Path, ...]:
    """Files adopted by more than one derived root's plan, refused unconditionally by the door
    that finds them: a claim collision is never a specificity tie to break across roots."""
    counted: dict[str, int] = {}
    paths: dict[str, Path] = {}
    for plan in plans:
        for entry in plan.entries:
            marker = os.path.normcase(str(entry.path))
            counted[marker] = counted.get(marker, 0) + 1
            paths[marker] = entry.path
    return tuple(sorted(paths[marker] for marker, count in counted.items() if count > 1))


def _walk_files(tree: Path) -> tuple[Path, ...]:
    return tuple(p for p in tree.rglob("*") if p.is_file())


def account_for(tree: str | Path) -> BundleAccounting:
    """Classify every file under ``tree`` into the four membership classes.

    Raises only :class:`AnchorMisplaced`; bookkeeping members and cross-root collisions are
    reported on the result. Imports ``tcip_mcp.store_catalog`` first, the list of every module
    that registers a store.
    """
    import tcip_mcp.store_catalog  # noqa: F401

    root = Path(tree).resolve()
    derived = derive_roots(root)
    plans = tuple(plan_root(str(d.path), d.layout) for d in derived)
    collisions = _cross_root_collisions(plans)

    claimed = frozenset(os.path.normcase(str(entry.path)) for plan in plans for entry in plan.entries)
    registered_checkpoints = _registered_checkpoint_paths(root)
    blobs = _blob_files(root, claimed, registered_checkpoints)
    blob_paths = frozenset(os.path.normcase(str(p)) for p in blobs)

    bookkeeping: list[Path] = []
    unaccounted: list[Path] = []
    for path in _walk_files(root):
        marker = os.path.normcase(str(path))
        if _is_bookkeeping(path.name):
            bookkeeping.append(path)
        elif marker not in claimed and marker not in blob_paths:
            unaccounted.append(path)

    return BundleAccounting(
        tree=root, derived=derived, plans=plans, blobs=blobs,
        bookkeeping=tuple(sorted(bookkeeping)), unaccounted=tuple(sorted(unaccounted)),
        collisions=collisions, registered_checkpoints=registered_checkpoints,
    )


__all__ = [
    "AnchorMisplaced",
    "BLOB_CHECKPOINTS",
    "BLOB_DATASET_IDENTITY",
    "BLOB_HOMES",
    "BLOB_IMAGERY",
    "BLOB_LABELS",
    "BLOB_MODEL_SRC",
    "BLOB_OTHER",
    "BLOB_SUBJECT_REGISTRY",
    "BundleAccounting",
    "CrossRootCollision",
    "DerivedRoot",
    "account_for",
    "blob_home",
    "derive_roots",
    "external_registered_checkpoints",
    "unresolved_registered_checkpoints",
]
