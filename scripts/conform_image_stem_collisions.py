"""Census, and on request conform, a project's already-collided image buckets: a bucket already
holding two logical identities under one case-folded stem key (``image_utils.stem_collision_key``)
before the ingest door started refusing them.

For each named project root, and each dataset root its own registry names (plus the project root
itself, its own tree being the common case), this walks the flat ``images/`` root and every
directory under it (the union ``resolve_images_dir`` and ``plant_mapping.build_mapping`` reach,
never ``list_dates`` alone) and reads ``image_utils.bucket_logical_identities`` of each: the same
function the ingest door's pre-scan and ``list_logical_images`` check, so this census and every
reader agree on exactly which keys are ambiguous.

    python scripts/conform_image_stem_collisions.py [--plan] <project_root> [<project_root> ...]
    python scripts/conform_image_stem_collisions.py --apply <project_root> --keep <path> [--keep <path> ...]

``--plan`` is the default (an explicit ``--plan`` is accepted and behaves the same as omitting
both flags); ``--apply`` is the opt-in, the inverse of every other ``conform_*.py`` script here,
because this one moves a breeder's raw bytes rather than rewriting a record. The two flags are
exclusive; passing both refuses before either mode runs. ``--plan`` prints, for
each ambiguous key, every one of its files with size, modification time and content digest, which
file (if any) today's un-corrected enumeration would still serve (the first by sorted filename
among files sharing one exact stem; ``None`` when a raw file's exact stem already matches a
``.bandgroup`` manifest's own, since every reader already refuses that directory outright), and
whether a label document, an image-status entry or a prediction document exists for each of the
key's own exact stems, since those records were made against the served file's pixels. It never
writes anything and creates no parking directory.

``--apply`` takes one ``--keep <absolute path>`` per ambiguous key, matched against the key's own
files by path, not by name or position. A key with no matching ``--keep``, a ``--keep`` naming a
file that is not one of the key's own files, a ``--keep`` naming a file other than the one served
today while a label document, an image-status entry or a prediction document exists for the served
file's own stem (the operator resolves the records first; re-attaching a record to different pixels
is the fabrication this family exists to prevent), or a parking destination that already exists
(never an overwrite of already-parked bytes), each refuse that key by name; every other key's other
files move to ``<dataset_root>/.tcip/collisions/<bucket>/<filename>`` (outside ``images/``, so no
walk over it and no store's locator claims the path; the flat root's own parking subdirectory is
named ``.flat``, a name no real bucket can hold), and one ``record_event_or_raise`` entry per moved
file is written under the dataset root's own log, naming the stem key, the bucket, the kept file,
the moved file, and both files' sizes and digests.

Exit codes: 0 when every named root exists and, in ``--plan`` mode, holds no ambiguous key, or, in
``--apply`` mode, every ambiguous key was resolved; 2 when a named root is not a project root, a
``--plan`` finds any ambiguous key, or an ``--apply`` leaves any key refused.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

from tcip_mcp.pipelines.data.band_groups import MANIFEST_EXT, read_band_group_manifest  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402

FLAT_BUCKET_SENTINEL = ".flat"
"""The flat images/ root's own parking subdirectory name: a leading dot no real bucket can hold
(``dataset_layout.is_bucket_name`` refuses one), so it never collides with an actual bucket."""

PARK_DIR = "collisions"


@dataclass
class Collision:
    """One ambiguous stem key: every one of its own identities' paths, in the bucket it was
    found in, under the dataset root that bucket belongs to."""

    project_root: Path
    dataset_root: Path
    bucket: str | None
    key: str
    paths: list[Path]


def _bucket_label(bucket: str | None) -> str:
    return bucket if bucket is not None else "(the flat images/ root)"


def _dataset_roots_for(project_root: Path) -> list[Path]:
    """``project_root`` itself, plus every dataset root its own registry names, resolved and
    de-duplicated: the project's own tree is the common case even before it is registered."""
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets_raw

    roots = {project_root.resolve()}
    for entry in read_datasets_raw(project_root):
        try:
            roots.add(dataset_entry_path(project_root, entry).resolve())
        except ValueError:
            continue
    return sorted(roots, key=str)


def _bucket_dirs_for(dataset_root: Path) -> list[tuple[str | None, Path]]:
    """``(bucket, directory)`` for the flat ``images/`` root itself and every directory under it,
    the union :func:`~tcip_mcp.dataset_layout.resolve_images_dir` and
    :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.build_mapping` reach, not
    :func:`~tcip_mcp.dataset_layout.list_dates` alone (which excludes a dot-prefixed directory)."""
    from tcip_mcp.dataset_layout import image_dir, image_root

    dirs: list[tuple[str | None, Path]] = [(None, image_dir(dataset_root, None))]
    root = image_root(dataset_root)
    if root.is_dir():
        for date_dir in sorted(root.iterdir()):
            if date_dir.is_dir():
                dirs.append((date_dir.name, date_dir))
    return dirs


def collect_collisions(project_roots: list[Path]) -> list[Collision]:
    """Every ambiguous key across ``project_roots``' own datasets, a dataset root visited at
    most once even when more than one named project registers it."""
    from tcip_mcp.pipelines.image_utils import bucket_logical_identities

    seen: set[Path] = set()
    collisions: list[Collision] = []
    for project_root in project_roots:
        for dataset_root in _dataset_roots_for(project_root):
            if dataset_root in seen:
                continue
            seen.add(dataset_root)
            for bucket, bucket_dir in _bucket_dirs_for(dataset_root):
                identities = bucket_logical_identities(bucket_dir)
                for key in sorted(identities):
                    paths = identities[key]
                    if len(paths) > 1:
                        collisions.append(Collision(project_root, dataset_root, bucket, key, paths))
    return collisions


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_info(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "digest": _file_digest(path),
    }


def served_today(paths: list[Path]) -> Path | None:
    """The file today's un-corrected ``list_logical_images`` (the reader before this family's
    fold) would still serve among ``paths``: the first by sorted filename among a same-exact-stem
    raw group. ``None`` when every path carries its own distinct exact stem (no prior duplicate:
    a case-variant pair was never ambiguous before the fold) or a raw file's exact stem already
    matches a ``.bandgroup`` manifest's own (every reader already refuses that directory today)."""
    manifests = [p for p in paths if p.suffix.lower() == MANIFEST_EXT]
    for m in manifests:
        if any(p.suffix.lower() != MANIFEST_EXT and p.stem == m.stem for p in paths):
            return None
    by_stem: dict[str, list[Path]] = {}
    for p in paths:
        if p.suffix.lower() != MANIFEST_EXT:
            by_stem.setdefault(p.stem, []).append(p)
    duplicated = [group for group in by_stem.values() if len(group) > 1]
    if len(duplicated) != 1:
        return None
    return sorted(duplicated[0])[0]


def records_for_stem(dataset_root: Path, bucket: str | None, stem: str) -> list[str]:
    """Every record kind made against ``stem``'s pixels: a label document, an image-status entry
    (any subject, matched by the recorded name's own stem), a prediction document (any model)."""
    from tcip_mcp.dataset_layout import (
        annotation_path, bucket_subject_date, label_filename, normalize_status_store,
        prediction_root, read_image_status_store,
    )
    from tcip_store import StoreError

    kinds: list[str] = []
    if annotation_path(dataset_root, bucket, stem).is_file():
        kinds.append("a label document")

    try:
        by_bucket = normalize_status_store(read_image_status_store(dataset_root))
    except StoreError:
        by_bucket = {}
    for status_key, statuses in by_bucket.items():
        _subject, date = bucket_subject_date(status_key)
        if date == bucket and any(Path(name).stem == stem for name in statuses):
            kinds.append("an image-status entry")
            break

    pred_root = prediction_root(dataset_root)
    if pred_root.is_dir():
        for model_dir in sorted(pred_root.iterdir()):
            if not model_dir.is_dir():
                continue
            candidate = (model_dir / bucket / label_filename(stem)) if bucket \
                else (model_dir / label_filename(stem))
            if candidate.is_file():
                kinds.append("a prediction document")
                break
    return kinds


def _key_stems(paths: list[Path]) -> list[str]:
    return sorted({p.stem for p in paths})


def report_plan(collisions: list[Collision]) -> int:
    """Print every collision's census; return 2 if any exists, else 0. Writes nothing."""
    if not collisions:
        print("no stem collisions found")
        return 0
    for c in collisions:
        print(f"{c.dataset_root}: bucket {_bucket_label(c.bucket)}, stem {c.key!r}:")
        served = served_today(c.paths)
        for p in sorted(c.paths, key=str):
            info = _identity_info(p)
            marker = " (served today)" if served is not None and p == served else ""
            print(f"  {info['path']}: size={info['size']} mtime={info['mtime']} "
                  f"digest={info['digest']}{marker}")
        if served is None:
            print("  served today: none; every reader already refuses this directory")
        for stem in _key_stems(c.paths):
            records = records_for_stem(c.dataset_root, c.bucket, stem)
            print(f"  records for {stem!r}: {', '.join(records) if records else 'none'}")
    return 2


def apply_collisions(collisions: list[Collision], keep_paths: list[Path]) -> int:
    """Resolve every collision whose own files include a matching ``--keep``; refuse the rest by
    name. Returns 2 if any key (or any ``--keep`` value) was refused, else 0."""
    from tcip_mcp.audit import dataset_scope_of, record_event_or_raise

    resolved_keeps = {p.resolve() for p in keep_paths}
    matched: set[Path] = set()
    refused = False

    for c in collisions:
        own_paths = {p.resolve(): p for p in c.paths}
        candidates = [rp for rp in resolved_keeps if rp in own_paths]
        where = f"{c.dataset_root} bucket {_bucket_label(c.bucket)} stem {c.key!r}"
        if not candidates:
            print(f"refused: {where}: no --keep given for this key")
            refused = True
            continue
        if len(candidates) > 1:
            print(f"refused: {where}: more than one --keep matches this key's own files "
                  f"{sorted(str(own_paths[rp]) for rp in candidates)}")
            refused = True
            continue
        keep_resolved = candidates[0]
        matched.add(keep_resolved)
        keep = own_paths[keep_resolved]
        others = [p for p in c.paths if p.resolve() != keep_resolved]

        parked_manifests = [p for p in others if p.suffix.lower() == MANIFEST_EXT]
        if parked_manifests:
            for m in parked_manifests:
                ref = read_band_group_manifest(m)
                members = sorted(str(p) for p in ref.bands.values())
                print(f"refused: {where}: --keep {keep} would park manifest {m}, leaving its "
                      f"claimed band files {members} behind as standalone images; keep the "
                      "manifest instead")
            refused = True
            continue

        served = served_today(c.paths)
        if served is not None:
            if served.resolve() != keep_resolved:
                records = records_for_stem(c.dataset_root, c.bucket, served.stem)
                if records:
                    print(f"refused: {where}: --keep {keep} is not the file served today "
                          f"({served}), and {', '.join(records)} exist for {served.stem!r}; "
                          "resolve the records first")
                    refused = True
                    continue
        else:
            by_stem_records = {
                stem: records_for_stem(c.dataset_root, c.bucket, stem) for stem in _key_stems(c.paths)
            }
            present = {stem: recs for stem, recs in by_stem_records.items() if recs}
            if present:
                details = "; ".join(
                    f"{stem!r}: {', '.join(recs)}" for stem, recs in sorted(present.items())
                )
                print(f"refused: {where}: no single file is served today for this key, and "
                      f"records exist ({details}); resolve the records first")
                refused = True
                continue

        park_bucket = c.bucket if c.bucket is not None else FLAT_BUCKET_SENTINEL
        moves: list[tuple[Path, Path]] = []
        conflict = False
        for other in others:
            dest = c.dataset_root / ".tcip" / PARK_DIR / park_bucket / other.name
            if dest.exists():
                print(f"refused: {where}: parking destination {dest} already exists; never "
                      "overwriting parked bytes")
                refused = True
                conflict = True
                break
            moves.append((other, dest))
        if conflict:
            continue

        keep_info = _identity_info(keep)
        for other, dest in moves:
            other_info = _identity_info(other)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(other), str(dest))
            print(f"moved: {other} -> {dest} (kept {keep})")
            record_event_or_raise(
                "conform_image_stem_collisions",
                {"dataset_root": str(c.dataset_root)},
                scope=dataset_scope_of(c.dataset_root) or c.dataset_root,
                stem_key=c.key, bucket=c.bucket, kept_file=str(keep), moved_file=str(other),
                kept_size=keep_info["size"], kept_digest=keep_info["digest"],
                moved_size=other_info["size"], moved_digest=other_info["digest"],
            )

    for p in sorted(resolved_keeps - matched, key=str):
        print(f"refused: --keep {p} is not one of any ambiguous key's own files")
        refused = True

    return 2 if refused else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument(
        "--plan", action="store_true",
        help="preview only; the default when neither flag is given")
    ap.add_argument(
        "--apply", action="store_true",
        help="move every colliding file but the one named by --keep")
    ap.add_argument(
        "--keep", action="append", default=[], type=Path,
        help="the file to keep for one ambiguous key; pass once per key")
    args = ap.parse_args()

    if args.plan and args.apply:
        print("refused: --plan and --apply are exclusive; pass one")
        return 2

    bind_default()

    valid_roots: list[Path] = []
    any_bad_root = False
    for root in args.roots:
        if not (root / ".tcip").is_dir():
            print(f"refused: {root}: no .tcip directory found; not a project root")
            any_bad_root = True
            continue
        valid_roots.append(root.resolve())

    collisions = collect_collisions(valid_roots)
    code = apply_collisions(collisions, args.keep) if args.apply else report_plan(collisions)
    return 2 if (any_bad_root or code) else 0


if __name__ == "__main__":
    sys.exit(main())
