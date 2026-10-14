"""Image ingestion: turn a raw folder of photos into a structured TCIP project.

``ingest_images`` copies (or moves) raw images into the canonical layout
(``images/<YYYY-MM-DD>/<stem><ext>``) under the project, bucketing by the capture date each file
states. It does not annotate, split, choose a task, or write ``subjects.json`` (see
``packages/tcip-mcp/src/tcip_mcp/knowledge/project-setup.md``).
"""

from __future__ import annotations

import datetime
import errno
import logging
from glob import glob
from pathlib import Path

import tcip_store
from tcip_mcp import dataset_layout
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.data.band_groups import (
    MANIFEST_EXT,
    detect_and_write_band_groups,
)
from tcip_mcp.pipelines.image_utils import (
    IMAGE_EXTS,
    bucket_logical_identities,
    stem_collision_key,
)
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool

logger = logging.getLogger(__name__)

_EXIF_DATE = "EXIF_DateTimeOriginal"
"""A file's EXIF capture time, as GDAL's EXIF metadata domain names it and as the manifest names
the date source it answered from."""
_TIFF_DATE = "TIFFTAG_DATETIME"
"""A TIFF's ``DateTime`` tag, as GDAL's default metadata domain names it."""


def _iso_date(raw: object) -> str | None:
    """A capture-date value in any spelling
    :func:`~tcip_mcp.pipelines.image_utils.parse_capture_time` reads, as ISO ``YYYY-MM-DD``;
    ``None`` if none fits.
    """
    from tcip_mcp.pipelines.image_utils import parse_capture_time

    parsed = parse_capture_time(raw)
    return parsed.strftime("%Y-%m-%d") if parsed is not None else None


def _photographic_capture_date(path: Path) -> tuple[str | None, str | None]:
    """EXIF ``DateTimeOriginal`` from a photographic container
    (:func:`~tcip_mcp.pipelines.image_utils.exif_capture_time`), and why it could not be read.
    ``Image.open`` decodes the header only and never the pixels.
    """
    from PIL import Image

    from tcip_mcp.pipelines.image_utils import exif_capture_time

    try:
        with Image.open(path) as im:
            exif = im.getexif()
            try:
                raw = exif_capture_time(exif)
            except Exception as exc:
                return None, f"EXIF block could not be read: {exc}"
    except Exception as exc:
        return None, f"image header could not be read: {exc}"
    if raw is None:
        return None, None
    iso = _iso_date(raw)
    if iso is None:
        return (
            None,
            f"EXIF DateTimeOriginal {str(raw)!r} is not a date this reads",
        )
    return iso, None


def _gdal_capture_date(
    path: Path, date_item: str | None
) -> tuple[str | None, str | None, str | None]:
    """The capture date a GDAL-readable raster's metadata states, why it could not be read, and
    the metadata item that stated it.

    Asks the TIFF ``DateTime`` tag first, then an EXIF IFD if the container exposes one, then the
    stated ``date_item`` in the default metadata domain (matched case-insensitively), where a
    stitching engine may write the day an orthomosaic was flown. Metadata only: no pixels are
    read.
    """
    import rasterio

    try:
        with rasterio.open(str(path)) as ds:
            tags = ds.tags()
            stated = {
                _TIFF_DATE: tags.get(_TIFF_DATE),
                _EXIF_DATE: ds.tags(ns="EXIF").get(_EXIF_DATE),
            }
            if date_item:
                stated[date_item] = next(
                    (
                        v
                        for k, v in tags.items()
                        if k.lower() == date_item.lower()
                    ),
                    None,
                )
    except Exception as exc:  # noqa: BLE001, rasterio raises driver-specific errors
        return None, f"raster metadata could not be read: {exc}", None
    item, raw = next(((k, v) for k, v in stated.items() if v), (None, None))
    if not raw:
        return None, None, None
    iso = _iso_date(raw)
    if iso is None:
        return (
            None,
            f"capture date {str(raw)!r} is not a date this reads",
            None,
        )
    return iso, None, item


def _capture_iso_date(
    path: Path, date_item: str | None
) -> tuple[str | None, str | None, str | None]:
    """``(ISO YYYY-MM-DD or None, why it could not be read or None, the item that stated it or
    None)`` for one file, a raster's ``date_item`` asked after its own tags
    (:func:`_gdal_capture_date`).

    No date and no reason is the undated fact: a photo with no EXIF date, a raster stating none,
    or an array file or band group, which has nowhere to put one and is not read. A reason is the
    different fact that the container itself could not be read this far.
    Neither outcome stops the file being ingested. The container family is the file's
    :attr:`~tcip_mcp.pipelines.raster_source.SourceHeader.kind`: EXIF in a photograph, raster
    metadata in a TIFF, nothing in any other.
    """
    from tcip_mcp.pipelines.raster_source import SourceHeader

    kind = SourceHeader(path).kind
    if kind == "photo":
        iso, unreadable = _photographic_capture_date(path)
        return iso, unreadable, _EXIF_DATE if iso else None
    if kind == "tif":
        return _gdal_capture_date(path, date_item)
    return None, None, None


def _iter_source_images(source: str, recursive: bool):
    """Yield image files under ``source``: a directory or a glob pattern."""
    src = Path(source)
    if src.is_dir():
        it = src.rglob("*") if recursive else src.iterdir()
        for p in sorted(it):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p
    else:
        for s in sorted(glob(source, recursive=recursive)):
            p = Path(s)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def _collision_refusal(
    dest_root: Path,
    resolved_sources: list[tuple[Path, str, str | None, str | None]],
) -> str | None:
    """The whole-call refusal naming every source that would create a second logical image under a
    stem key another source, an existing raw file, or an existing band-group manifest already holds
    in its bucket; ``None`` when every source either names a new stem or names the same destination
    filename an existing raw identity already holds at its key.

    The carve-out tests the destination filename only, not the source bytes: such a source is left
    out of this refusal and out of ``identities``, whether or not its own bytes match that file,
    and is skipped by the copy loop's own ``dest.exists()`` check.
    """
    buckets = sorted({bucket for _src, bucket, *_ in resolved_sources})
    collisions: dict[str, dict[str, list[Path]]] = {}
    for bucket in buckets:
        # disk_identities is a fixed pre-call snapshot for the re-ingest carve-out below, never
        # mutated: a batch-mate sharing a key must never read as one already on disk.
        disk_identities = bucket_logical_identities(
            dataset_layout.image_dir(dest_root, bucket)
        )
        identities = {
            key: list(paths) for key, paths in disk_identities.items()
        }
        for src_path, src_bucket, *_ in resolved_sources:
            if src_bucket != bucket:
                continue
            key = stem_collision_key(src_path.stem)
            dest = dataset_layout.image_path(
                dest_root, bucket, src_path.stem, src_path.suffix
            )
            disk_existing = disk_identities.get(key, [])
            if (
                len(disk_existing) == 1
                and disk_existing[0].suffix.lower() != MANIFEST_EXT
                and disk_existing[0].name == dest.name
            ):
                continue  # the exact re-ingest of a file already on disk, not a second identity
            identities.setdefault(key, []).append(src_path)
            if len(identities[key]) > 1:
                collisions.setdefault(bucket, {})[key] = identities[key]

    if not collisions:
        return None
    lines = [
        f"bucket {bucket!r}, stem {key!r}: "
        + ", ".join(str(p) for p in collisions[bucket][key])
        for bucket in sorted(collisions)
        for key in sorted(collisions[bucket])
    ]
    return (
        "stem collision, refusing the whole call before any byte is copied: "
        + "; ".join(lines)
    )


@tool()
@audited
def ingest_images(
    project: Path,
    source: str,
    copy: bool = True,
    recursive: bool = True,
    raster_date_item: str | None = None,
) -> dict:
    """Copy raw images into the project, bucketed by the capture date each file states.

    Turns a raw folder (or glob) of photos into the canonical layout
    (``images/<YYYY-MM-DD>/<stem><ext>``) under the project. Copies by default; originals are left
    byte-identical; pass ``copy=False`` to move. Refuses the whole call, before anything is copied,
    when a source's destination stem would create a second logical image under a key another
    source, an existing raw file, or an existing band-group manifest already holds in that bucket;
    re-ingesting the exact same destination file is skipped instead, recorded in
    ``skipped_collisions``. Does not annotate, split, choose a task, or write ``subjects.json``.

    Each file's bucket is the capture date it states (a photo's EXIF ``DateTimeOriginal``, a
    raster's own date metadata), read from its header only. The capture date never gates
    ingestion: a file that states none, or whose date cannot be read, is copied like any other and
    lands in ``undated/``, its entry in ``files`` carrying why; ``state_capture_date`` moves it
    to the date known for it.

    After copying, the band-group correlation strategies (``pipelines.data.band_groups``) run over
    each touched bucket, writing a ``.bandgroup`` manifest for every sibling-single-band-file
    group found (a multispectral rig that writes one file per band instead of one multi-band file
    per capture).

    Args:
        source: Folder (or glob) of raw images, anywhere on disk.
        copy: Copy (True, default) or move (False) the source images.
        recursive: Recurse into source subfolders.
        raster_date_item: The default-domain metadata item a raster's writer states its capture
            date in (a stitching engine's own item name, matched case-insensitively), asked
            after the raster's TIFF ``DateTime`` tag and EXIF date. Omitted, a raster stating
            neither lands in ``undated/``.

    Returns a manifest: ``{image_root, found, copied, moved, files, skipped_collisions, errors,
    move, band_groups}``, where ``files`` holds one entry per ingested file, ``{source, dest,
    bucket, date_source, date_unreadable}``: ``date_source`` is the metadata item that dated it
    (``TIFFTAG_DATETIME``, ``EXIF_DateTimeOriginal`` or ``raster_date_item``), ``None`` for a file
    no source dated, which lands in ``undated/``, and ``date_unreadable`` why its container could
    not be read that far, else ``None``. ``skipped_collisions`` names only an exact re-ingest.
    ``band_groups`` is ``{formed, refused, manifests}`` over the touched buckets.

    """
    source = str(located(source, project))
    sources = list(_iter_source_images(source, recursive))
    if not sources:
        return {"error": f"No images found under {source!r}"}

    resolved_sources = []
    for src_path in sources:
        iso, unreadable, item = _capture_iso_date(src_path, raster_date_item)
        resolved_sources.append(
            (src_path, iso or dataset_layout.UNDATED_BUCKET, unreadable, item)
        )

    collision_error = _collision_refusal(project, resolved_sources)
    if collision_error is not None:
        return {"error": collision_error}

    copied = 0
    moved = 0
    files: list[dict] = []
    skipped_collisions: list[dict] = []
    errors: list[dict] = []
    touched_buckets: set[str] = set()

    for src_path, bucket, date_unreadable, date_source in resolved_sources:
        dest = dataset_layout.image_path(
            project, bucket, src_path.stem, src_path.suffix
        )
        if dest.exists():
            # No-overwrite: the exact re-ingest of a file already placed; every other stem
            # collision was already refused above, whole-call.
            skipped_collisions.append(
                {
                    "stem": src_path.stem,
                    "source": str(src_path),
                    "existing": str(dest),
                    "bucket": bucket,
                }
            )
            continue

        # A bad file (locked, vanished, unreadable) must not abort the batch; the blob becomes
        # the image only once whole, so a crash mid-copy can't leave a torn image.
        try:
            tcip_store.put_blob(dest, src_path.read_bytes())
        except OSError as exc:
            errors.append({"source": str(src_path), "error": str(exc)})
            continue

        if copy:
            copied += 1
        else:
            # Move = copy-then-unlink (works across filesystems, unlike rename). The
            # source is removed only after the destination is fully written.
            try:
                src_path.unlink()
                moved += 1
            except OSError:
                logger.warning(
                    "ingest: could not remove source after move: %s", src_path
                )
                copied += 1

        touched_buckets.add(bucket)
        files.append(
            {
                "source": str(src_path),
                "dest": str(dest),
                "bucket": bucket,
                "date_source": date_source,
                "date_unreadable": date_unreadable,
            }
        )

    band_groups_result: dict = {"formed": [], "refused": [], "manifests": []}
    for bucket in sorted(touched_buckets):
        bucket_dir = dataset_layout.image_dir(project, bucket)
        result = detect_and_write_band_groups(bucket_dir)
        for g in result["formed"]:
            band_groups_result["formed"].append({**g, "bucket": bucket})
        for g in result["refused"]:
            band_groups_result["refused"].append({**g, "bucket": bucket})
        band_groups_result["manifests"].extend(result["manifests"])

    return {
        "image_root": str(dataset_layout.image_root(project)),
        "found": len(sources),
        "copied": copied,
        "moved": moved,
        "files": files,
        "skipped_collisions": skipped_collisions,
        "errors": errors,
        "move": not copy,
        "band_groups": band_groups_result,
    }


@tool()
def state_capture_date(
    project: Path, images: list[str], date: datetime.date, rationale: str
) -> dict:
    """Move images into the bucket of a capture date stated for them: one their files do not
    state, or one correcting the date they state.

    Each of ``images`` names a logical image under one dataset's ``images/`` tree, a plain image's
    file or a grouped capture's ``.bandgroup`` manifest
    (:func:`~tcip_mcp.pipelines.image_utils.resolve_image_paths`), and moves whole into
    ``images/<date>/`` in three phases: its files copied there (a grouped capture's band files
    beside its manifest), the call making the bucket directory through the store's own route
    (:func:`tcip_store.ensure_parent`) when it is absent; then every record keyed by its capture
    and stem (its label document, its flags and its staged proposals) re-keyed unchanged to the
    new capture in one transaction with the act's audit line, in the dataset root's log, naming
    the images, the date and the rationale; then the original files deleted. A copy or the
    transaction that raises removes what the call made (each copy with its lock file, then the
    directory it made, which stays when another writer's file sits in it) and propagates the
    cause. An image already in that bucket stays.

    Args:
        images: The images to move, each as ``ingest_images`` answers its ``dest``.
        date: The capture date, which names the bucket in ISO ``YYYY-MM-DD``.
        rationale: How the date is known.

    Refuses with nothing moved: images under more than one dataset root; an image a published
    prediction bucket names, naming the bucket, whose record states the capture the image was
    predicted under; a stem the date's bucket already holds, as ``ingest_images`` refuses it; a
    file or record already at an image's new place; and each refusal of the resolver.

    Returns ``{date, moved}``, ``moved`` holding ``{source, dest}`` per file moved.
    """
    from tcip_annotation.flags import flag_key
    from tcip_mcp.audit import audit_entry, audit_log_key
    from tcip_mcp.buckets import buckets_under, source_root
    from tcip_mcp.pipelines.image_utils import (
        BandGroupRef,
        resolve_image_paths,
        stem_of,
    )
    from tcip_mcp.tools.proposal_tools import proposal_staging_key
    from tcip_store.file_backend import lock_file_for

    record_keys = (
        dataset_layout.label_key,
        lambda *at: flag_key(dataset_layout.label_key(*at)),
        proposal_staging_key,
    )
    bucket = date.isoformat()
    try:
        named = [located(image, project) for image in images]
        root = source_root(named)
        logical = resolve_image_paths(named)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}
    files: list[Path] = []
    rekeyed: list[tuple[tcip_store.Key, tcip_store.Key]] = []
    bound = {
        (b.date, stem): b.name
        for b in buckets_under(root)
        for stem in b.documents
    }
    held = []
    for path, image in zip(named, logical):
        capture, stem = (
            dataset_layout.parse_image_path(path)[1],
            stem_of(image),
        )
        if capture == bucket:
            continue
        if (capture, stem) in bound:
            held.append(f"{capture}/{stem} (bucket {bound[capture, stem]!r})")
        files += (
            [image.manifest_path, *image.bands.values()]
            if isinstance(image, BandGroupRef)
            else [path]
        )
        rekeyed += [
            (key_of(root, capture, stem), key_of(root, bucket, stem))
            for key_of in record_keys
        ]
    if held:
        return {
            "error": f"published prediction buckets name {held}: each bucket's record states "
            "the capture its images were predicted under, and a published record is never "
            "rewritten. Publish the predictions again after the date is stated, under a new "
            "bucket."
        }
    bucket_dir = dataset_layout.image_dir(root, bucket)
    dests = [bucket_dir / f.name for f in files]
    taken = [str(d) for d in dests if d.exists()] + [
        str(new.parts) for _old, new in rekeyed if tcip_store.exists(new)
    ]
    collision = _collision_refusal(
        root, [(f, bucket, None, None) for f in files]
    )
    if collision or taken:
        return {
            "error": collision
            or f"{taken} already exist under {root}, and a move never overwrites"
        }

    if not files:
        return {"date": bucket, "moved": []}
    audit = audit_log_key(root)
    minted = bucket_dir in tcip_store.ensure_parent(dests[0])
    copied: list[Path] = []
    try:
        for path, dest in zip(files, dests):
            tcip_store.put_blob(
                dest, path.read_bytes(), expect=tcip_store.Version.ABSENT
            )
            copied.append(dest)
        with tcip_store.transaction(
            audit, *(k for pair in rekeyed for k in pair)
        ) as txn:
            for old, new in rekeyed:
                record = txn.read(old, default=None)
                if record is not None:
                    txn.write(new, record)
                    txn.delete(old)
            txn.append(
                audit,
                audit_entry(
                    "state_capture_date",
                    {"images": images, "date": bucket, "rationale": rationale},
                    None,
                    "ok",
                ),
            )
    except BaseException:
        for dest in copied:
            tcip_store.delete_blob(dest)
            lock_file_for(dest).unlink(missing_ok=True)
        if minted:
            try:
                bucket_dir.rmdir()
            except OSError as exc:
                if exc.errno != errno.ENOTEMPTY:
                    raise
        raise
    for path in files:
        tcip_store.delete_blob(path)
    return {
        "date": bucket,
        "moved": [
            {"source": str(path), "dest": str(dest)}
            for path, dest in zip(files, dests)
        ],
    }


@tool()
def import_coco(
    project: Path,
    document: str,
    dataset_root: str,
    date: str,
    epsilon_frac: float,
    rationale: str,
) -> dict:
    """Convert an external dataset-level COCO document into the dataset's per-image label
    documents.

    Run it after ``ingest_images`` has placed the images it names. Every declared category must be
    a subject the dataset's registry declares, and each image is the capture's image of that exact
    file name (a ``.bandgroup`` capture by stem). Every fault found before writing is named
    together and refuses the import with nothing written: a malformed or unregistered category, a
    malformed record, a duplicate id or image, an image not in the capture, or a frame that
    disagrees with the image. The documents and the import's audit line then commit as one store
    transaction, each document checked absent inside it, so a label already there refuses the
    whole import with nothing written; an import that writes no document leaves no event. A crowd
    region keeps its flag and a run-length mask becomes rings. An image with no annotations writes
    nothing.

    Args:
        document: Path to the COCO ``.json`` (an ``images``/``categories`` key).
        dataset_root: The dataset the images were ingested into.
        date: The capture under ``images/`` the images sit in, which the documents are keyed
            under; one ``list_dates`` does not list refuses. A COCO document states no capture,
            so it is named.
        epsilon_frac: The fraction of each ring's own perimeter a run-length mask's rings are
            simplified at (``cv2.approxPolyDP``), recorded on the import's audit line.
        rationale: Why the import simplifies at ``epsilon_frac``, recorded beside it.

    Returns ``{document, capture, epsilon_frac, rationale, written}``, ``written`` the stem of
    every label document written.

    """
    from tcip_mcp.pipelines.data.coco_import import import_coco_document

    try:
        return import_coco_document(
            document,
            dataset_root,
            date=date,
            epsilon_frac=epsilon_frac,
            rationale=rationale,
            project=project,
        )
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}
