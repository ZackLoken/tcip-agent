"""What a project bundle holds: the one membership accounting both doors compose from.

``archive_project`` composes its bundle from this accounting's record/log and blob classes;
``import_project`` classifies every extracted member by it and refuses on anything bookkeeping,
cross-root-collided or unaccounted. One implementation, so the two doors cannot silently drift
onto two different notions of "what a bundle holds".

Roots are derived from the tree's own structure plus the anchored documents the platform's own
writers place (``split_manifest.json``, ``curated_manifest.json``); an anchor found somewhere
the derivation constraints exclude (the tree root, under ``.tcip``, under a blob home, or under
or above another derived root) raises :class:`AnchorMisplaced` naming the file, since a
mislabelled anchor would recruit a directory that is something else. One nesting is admitted
rather than excluded: a splits root sitting under a curated root, the shape ``make_splits``
produces when it partitions a ``materialize_review_dataset`` output in place. Classification of
one file is by precedence, not disjointness: bookkeeping first, then a record or log claimed by
exactly one derived root's own layout (two derived roots claiming the same file raises
:class:`CrossRootCollision`), then a recognized blob home, then everything else, unaccounted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from tcip_store.adoption import AdoptionPlan, plan_root
from tcip_store.file_backend import _is_bookkeeping
from tcip_store.layout_claims import CURATED, EXPERIMENTS, HPO_ROOT, ROOT, RUN, SPLITS, STATE, SWEEP

SPLIT_MANIFEST_NAME = "split_manifest.json"
CURATED_MANIFEST_NAME = "curated_manifest.json"

LABEL_EXTS = frozenset({".txt", ".xml", ".json"})
"""Ground-truth label document suffixes archive_project has always bundled."""


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

    ``plans`` is one :class:`~tcip_store.adoption.AdoptionPlan` per derived root (the record and
    log accounting, entries already de-duplicated within their own root by
    :func:`~tcip_store.adoption.plan_root`); ``blobs`` is every file under a recognized blob
    home that no plan already adopts; ``bookkeeping`` and ``unaccounted`` are class 1 and class 4
    respectively, over every other file the tree holds; ``collisions`` names any file two
    different derived roots both adopted, which no shipped claim table can produce on its own.
    ``registered_checkpoints`` is the subset of ``blobs`` :func:`blob_home` calls
    ``BLOB_CHECKPOINTS`` on the strength of a model registry entry rather than sitting under
    ``.tcip/models``; computed once here, while ``tree`` still holds its own registry index, so
    a caller classifying blobs after moving or deleting the tree (``import_project``, once it has
    renamed staging onto the destination) still has it to pass back in.
    """

    tree: Path
    derived: tuple[DerivedRoot, ...]
    plans: tuple[AdoptionPlan, ...]
    blobs: tuple[Path, ...]
    bookkeeping: tuple[Path, ...]
    unaccounted: tuple[Path, ...]
    collisions: tuple[Path, ...]
    registered_checkpoints: frozenset[Path]


def _is_at_or_under(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


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
    """Every root ``tree`` is, or holds, per the platform's own writers and anchors.

    Raises :class:`AnchorMisplaced` naming the file when a split or curated manifest sits
    somewhere the constraints exclude, rather than silently recruiting a directory that is
    something else.
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
    split_dirs = _anchored_dirs(root, SPLIT_MANIFEST_NAME)
    curated_dirs = _anchored_dirs(root, CURATED_MANIFEST_NAME)
    every_anchor = [*split_dirs, *curated_dirs]
    curated_set, split_set = frozenset(curated_dirs), frozenset(split_dirs)
    for directory in split_dirs:
        _validate_anchor(root, directory, SPLIT_MANIFEST_NAME,
                          [d for d in every_anchor if d != directory], image_root, annotation_root,
                          curated_dirs=curated_set, split_dirs=split_set)
        derived.append(DerivedRoot(directory, SPLITS))
    for directory in curated_dirs:
        _validate_anchor(root, directory, CURATED_MANIFEST_NAME,
                          [d for d in every_anchor if d != directory], image_root, annotation_root,
                          curated_dirs=curated_set, split_dirs=split_set)
        derived.append(DerivedRoot(directory, CURATED))
    return tuple(derived)


def _registered_checkpoint_paths(tree: Path) -> frozenset[Path]:
    """Every checkpoint a model registry entry under ``tree`` points at, resolved, restricted to
    ones actually inside ``tree``.

    ``register_model`` admits any checkpoint path an explicit-mode caller names, and
    ``register_model_from_experiment`` registers wherever a run's own ``output_dir`` wrote its
    weights (``.tcip/experiments/<experiment_id>/`` by ``launch_training``'s own default), so a
    registered checkpoint is not confined to the ``.tcip/models/`` convenience location
    :func:`_blob_files`'s own glob covers. A registry the index will not decode is a concern for
    the checks that already read it (``doctor.py``'s ``check_registry``), not a reason to refuse
    a bundle: this reads best-effort and answers with no checkpoints found rather than raising.
    """
    from tcip_store import StoreError

    from tcip_mcp.model_registry import read_registry_index

    try:
        entries = read_registry_index(tree)
    except StoreError:
        return frozenset()
    found: set[Path] = set()
    for entry in entries if isinstance(entries, list) else []:
        raw = entry.get("checkpoint_path") if isinstance(entry, dict) else None
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = tree / candidate
        resolved = candidate.resolve()
        if resolved.is_file() and _is_at_or_under(resolved, tree):
            found.add(resolved)
    return frozenset(found)


def _blob_files(
    tree: Path, claimed: frozenset[str], registered_checkpoints: frozenset[Path],
) -> tuple[Path, ...]:
    """Every file under a recognized blob home that no record or log plan already adopts, each
    named once even when more than one recognized home would otherwise find the same file (a
    checkpoint sitting under ``.tcip/models`` and also registered, or an experiment-dir ``.pt``
    that is also part of a ``model_src`` snapshot walk).

    A ``.pt`` file anywhere under ``.tcip/experiments/`` is found by shape alone, in addition to
    a registry-named path and the ``.tcip/models`` convenience location: a run's own checkpoint
    (``launch_training``'s default ``output_dir``) is not confined to either, and a tree that has
    moved away from where it was registered (a staged import) still has to find it.
    """
    from tcip_mcp.dataset_layout import annotation_root as _annotation_root
    from tcip_mcp.dataset_layout import classes_path, dataset_identity_path
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
        for f in ann_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in LABEL_EXTS:
                _add(f)
    _add(classes_path(tree))
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
BLOB_CLASS_REGISTRY = "class_registry"
BLOB_DATASET_IDENTITY = "dataset_identity"
BLOB_MODEL_SRC = "model_src"
BLOB_CHECKPOINTS = "checkpoints"
BLOB_OTHER = "other"

BLOB_HOMES = (
    BLOB_IMAGERY, BLOB_LABELS, BLOB_CLASS_REGISTRY, BLOB_DATASET_IDENTITY, BLOB_MODEL_SRC,
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

    ``registered_checkpoints`` (``BundleAccounting.registered_checkpoints``, computed once by
    :func:`account_for` while ``tree`` still holds its own registry index) names every checkpoint
    a registry entry points at outside ``.tcip/models``; pass it back in for a caller classifying
    blobs after the tree has moved (``import_project``, past its own rename), when a fresh
    registry read would find nothing there any more, since the moved tree's own registry still
    names the exporting root's absolute paths. A ``.pt`` file under ``.tcip/experiments/`` is
    recognized as a checkpoint by shape alone, whether or not any registry names it, so the same
    file classifies the same way on both sides of that move. Omitted, this still recognizes every
    checkpoint physically under ``.tcip/models`` or shaped as one under ``.tcip/experiments/``.
    """
    from tcip_mcp.dataset_layout import annotation_root as _annotation_root
    from tcip_mcp.dataset_layout import classes_path, dataset_identity_path
    from tcip_mcp.dataset_layout import image_root as _image_root

    if path == classes_path(tree):
        return BLOB_CLASS_REGISTRY
    if path == dataset_identity_path(tree):
        return BLOB_DATASET_IDENTITY
    if _is_at_or_under(path, _image_root(tree)):
        return BLOB_IMAGERY
    if _is_at_or_under(path, _annotation_root(tree)):
        return BLOB_LABELS
    if path.parent == tree / ".tcip" / "models" or path in registered_checkpoints:
        return BLOB_CHECKPOINTS
    experiments = tree / ".tcip" / "experiments"
    if _is_at_or_under(path, experiments):
        if path.suffix == ".pt":
            return BLOB_CHECKPOINTS
        rel = path.relative_to(experiments).parts
        if len(rel) >= 2 and rel[1] == "model_src":
            return BLOB_MODEL_SRC
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

    Passive: this raises only on :class:`AnchorMisplaced` (a structural fact about the tree
    itself, true for either door); bookkeeping members and cross-root collisions are reported
    on the result rather than raised, since ``archive_project`` treats a live tree's transient
    bookkeeping (a lock file mid-write) as simply not bundled, while ``import_project`` is the
    door that refuses on either.

    Attributing a file to a store needs that store's descriptor, so this imports
    ``tcip_mcp.store_catalogue`` first, the platform's one list of every module that registers
    one: a running MCP server already has them all from its own tool imports, but a door called
    on its own (a script, a focused test) must not silently see fewer stores than the server
    does. Package-only, so this reaches the catalogue with no need for the repository's
    ``scripts`` package, which exists only with the repo root on ``sys.path``.
    """
    import tcip_mcp.store_catalogue  # noqa: F401

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
    "BLOB_CLASS_REGISTRY",
    "BLOB_DATASET_IDENTITY",
    "BLOB_HOMES",
    "BLOB_IMAGERY",
    "BLOB_LABELS",
    "BLOB_MODEL_SRC",
    "BLOB_OTHER",
    "BundleAccounting",
    "CrossRootCollision",
    "DerivedRoot",
    "account_for",
    "blob_home",
    "derive_roots",
]
