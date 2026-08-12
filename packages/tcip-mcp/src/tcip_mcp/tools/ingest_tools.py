"""Image ingestion: turn a raw folder of photos into a structured TCIP project.

``ingest_images`` is the missing structuring primitive. It copies (or moves) raw
images into the canonical layout (``images/<YYYY-MM-DD>/<stem><ext>``) under a
workspace project, bucketing by the capture date each file states. It does not annotate, split,
choose a task, or write ``classes.json``; those are later steps in the project-setup
arc (see ``.github/skills/project-setup``). Keeping it thin is deliberate: one
auditable primitive the agent composes, instead of improvising file ops per project.
"""

from __future__ import annotations

import logging
from glob import glob
from pathlib import Path

from tcip_mcp import dataset_layout, workspace
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.image_utils import IMAGE_EXTS
from tcip_mcp.server import mcp
from tcip_mcp.utils.atomic_io import atomic_write_bytes

logger = logging.getLogger(__name__)

UNDATED_BUCKET = "undated"


_DT_ORIGINAL = 0x9003  # EXIF DateTimeOriginal tag id
_EXIF_IFD = 0x8769  # Exif sub-IFD offset tag

# The container families a capture date can be asked of, by extension: EXIF in a photographic file,
# raster metadata in a GDAL-readable one. Every other ingestible extension is neither.
_PHOTOGRAPHIC_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}
_GDAL_EXTS = {".tif", ".tiff"}

# The spellings a capture-date value arrives in: the colon form EXIF's DateTimeOriginal and TIFF's
# DateTime tag are specified to use, and the ISO forms a stitching engine's own item is written in.
_DATE_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")

# Default-domain raster metadata items observed to name a capture date, matched case-insensitively
# in this order: a Sentera-stitched orthomosaic writes ``capture_date``.
_GDAL_DATE_ITEMS = ("capture_date",)


def _iso_date(raw: object) -> str | None:
    """A capture-date value in any spelling this reads → ISO ``YYYY-MM-DD``; ``None`` if none fit."""
    from datetime import datetime

    text = str(raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _photographic_capture_date(path: Path) -> tuple[str | None, str | None]:
    """EXIF ``DateTimeOriginal`` from a photographic container, and why it could not be read.

    Reads via the public ``Image.getexif()`` + Exif sub-IFD so it works across formats (JPEG, PNG,
    HEIC). ``Image.open`` decodes the header only and never the pixels, so a file whose header and
    EXIF block read is reported as stating a date or as stating none, whatever its pixel data later
    turns out to be. PIL is lazy-imported so tool startup stays fast.
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            exif = im.getexif()
            try:
                sub = exif.get_ifd(_EXIF_IFD)
            except Exception as exc:
                return None, f"EXIF block could not be read: {exc}"
    except Exception as exc:
        return None, f"image header could not be read: {exc}"
    raw = sub.get(_DT_ORIGINAL)
    if raw is None:
        return None, None
    iso = _iso_date(raw)
    if iso is None:
        return None, f"EXIF DateTimeOriginal {str(raw)!r} is not a date this reads"
    return iso, None


def _gdal_capture_date(path: Path) -> tuple[str | None, str | None]:
    """The capture date a GDAL-readable raster's metadata states, and why it could not be read.

    Asks the TIFF ``DateTime`` tag first, then an EXIF IFD if the container exposes one, then the
    stitching-engine items of :data:`_GDAL_DATE_ITEMS` in the default metadata domain, which is
    where an orthomosaic states the day it was flown. Metadata only: no pixels are read. rasterio
    is lazy-imported so tool startup stays fast.
    """
    import rasterio

    try:
        with rasterio.open(str(path)) as ds:
            tags = ds.tags()
            raw = tags.get("TIFFTAG_DATETIME")
            if not raw:
                raw = ds.tags(ns="EXIF").get("EXIF_DateTimeOriginal")
            if not raw:
                items = {k.lower(): v for k, v in tags.items()}
                raw = next((items[name] for name in _GDAL_DATE_ITEMS if items.get(name)), None)
    except Exception as exc:  # noqa: BLE001, rasterio raises driver-specific errors
        return None, f"raster metadata could not be read: {exc}"
    if not raw:
        return None, None
    iso = _iso_date(raw)
    if iso is None:
        return None, f"capture date {str(raw)!r} is not a date this reads"
    return iso, None


def _capture_iso_date(path: Path) -> tuple[str | None, str | None]:
    """``(ISO YYYY-MM-DD or None, why it could not be read or None)`` for one file.

    Both ``None`` is the readable-but-undated fact: the container was read and states no capture
    date (a photo with no EXIF date, a raster with no date item, an array file with nowhere to put
    one). A reason is the different fact that the container itself could not be read this far, so
    whether it states a date is unknown. Neither outcome stops the file being ingested; the reason
    is what ``ingest_images`` reports so the difference stays visible.
    """
    ext = path.suffix.lower()
    if ext in _PHOTOGRAPHIC_EXTS:
        return _photographic_capture_date(path)
    if ext in _GDAL_EXTS:
        return _gdal_capture_date(path)
    return None, None


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


def _validate_bucket_literal(date_from: str) -> None:
    """A literal ``date_from`` becomes a path segment: reject separators/traversal."""
    if date_from in ("exif", "none"):
        return
    if not workspace.is_valid_name(date_from):
        raise ValueError(f"invalid date_from literal bucket: {date_from!r}")


def _bucket_for(path: Path, date_from: str) -> tuple[str, str | None]:
    """The bucket ``path`` lands in, and why its own capture date could not be read.

    Only ``"exif"`` asks the file anything at all; the other two modes name the bucket from the
    caller's own argument and open nothing, so they never have a reason to report.
    """
    if date_from == "exif":
        iso, unreadable = _capture_iso_date(path)
        return iso or UNDATED_BUCKET, unreadable
    if date_from == "none":
        return UNDATED_BUCKET, None
    return date_from, None  # validated literal bucket


@mcp.tool()
@audited
def ingest_images(
    source: str,
    name: str,
    project_path: str = "",
    copy: bool = True,
    date_from: str = "exif",
    recursive: bool = True,
    detect_band_groups: bool = False,
) -> dict:
    """Copy raw images into a structured project, bucketed by the capture date each file states.

    Turns a raw folder (or glob) of photos into the canonical layout
    (``images/<YYYY-MM-DD>/<stem><ext>``) under a workspace project. Copies by
    default; originals are left byte-identical; pass ``copy=False`` to move.
    Refuses to overwrite an existing image (records the collision and skips it).
    Does not annotate, split, choose a task, or write ``classes.json``.

    The capture date never gates ingestion: a file whose date cannot be read is copied and counted
    like any other, lands in ``undated/``, and is listed in ``unreadable_dates`` so the difference
    between a file that states no date and one that could not be asked stays visible.

    Args:
        source: Folder (or glob) of raw images, anywhere on disk.
        name: Project slug (``{crop}_{trait}_{site}``); the destination folder is
            ``<TCIP_WORKSPACE>/<name>/`` unless ``project_path`` overrides it.
        project_path: Absolute destination path instead of ``workspace/<name>``.
        copy: Copy (True, default) or move (False) the source images.
        date_from: ``"exif"`` (each file's own capture date → ISO date, missing →
            ``undated/``; a photo's EXIF ``DateTimeOriginal``, a raster's own date metadata),
            ``"none"`` (all → ``undated/``), or a literal bucket name
            (all → ``images/<literal>/``, e.g. a known ISO capture date). Only ``"exif"``
            opens a file at all, and only its header.
        recursive: Recurse into source subfolders.
        detect_band_groups: After copying, run the band-group correlation strategies
            (``pipelines.data.band_groups``) over each touched bucket, writing a ``.bandgroup``
            manifest for every sibling-single-band-file group found (e.g. a multispectral rig
            that writes one file per band instead of one multi-band file per capture). Default
            ``False``; a project with no such capture pays nothing for this pass.

    Returns a manifest: ``{project_path, name, image_root, total, found, copied,
    moved, buckets, undated, skipped_collisions, errors, unreadable_dates, move, band_groups}``,
    where ``unreadable_dates`` names each ingested file whose capture date could not be read and
    the reason.
    """
    try:
        _validate_bucket_literal(date_from)
        if project_path:
            # Resolve so a relative override is explicit/absolute, not silently CWD-based.
            dest_root: Path = Path(project_path).expanduser().resolve()
        else:
            dest_root = workspace.project_path(name)
    except ValueError as exc:
        return {"error": str(exc)}

    sources = list(_iter_source_images(source, recursive))
    if not sources:
        return {"error": f"No images found under {source!r}"}

    # Lazy import avoids a module-load import cycle (server → ingest_tools → project_tools).
    from tcip_mcp.tools.project_tools import _scaffold_project

    scaffold = _scaffold_project(str(dest_root))

    buckets: dict[str, int] = {}
    undated = 0
    copied = 0
    moved = 0
    skipped_collisions: list[dict] = []
    errors: list[dict] = []
    unreadable_dates: list[dict] = []
    touched_buckets: set[str] = set()
    # Collisions are keyed by stem within a bucket (case-insensitively): labels and
    # predictions pair to an image by stem alone (see dataset_layout), so two sources with
    # the same stem but different extensions would otherwise silently share one label file.
    placed: set[tuple[str, str]] = set()

    for src_path in sources:
        bucket, date_unreadable = _bucket_for(src_path, date_from)
        stem_key = (bucket, src_path.stem.lower())
        dest = dataset_layout.image_path(dest_root, bucket, src_path.stem, src_path.suffix)
        if stem_key in placed or dest.exists():
            # No-overwrite: a stem collision (two sources → same bucket/stem) or a
            # re-ingest. Report it; never clobber.
            skipped_collisions.append(
                {
                    "stem": src_path.stem,
                    "source": str(src_path),
                    "existing": str(dest),
                    "bucket": bucket,
                }
            )
            continue

        # One bad file (locked by antivirus, vanished, unreadable) must not abort the whole
        # batch; record it and keep going. Reading bytes never mutates the original;
        # atomic_write_bytes writes a temp file + fsync + os.replace, so a crash mid-copy
        # can't leave a torn image.
        try:
            data = src_path.read_bytes()
            atomic_write_bytes(dest, data)
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
                logger.warning("ingest: could not remove source after move: %s", src_path)
                copied += 1

        placed.add(stem_key)
        touched_buckets.add(bucket)
        if date_unreadable:
            unreadable_dates.append({"source": str(src_path), "dest": str(dest),
                                     "bucket": bucket, "reason": date_unreadable})
        if bucket == UNDATED_BUCKET:
            undated += 1
        else:
            buckets[bucket] = buckets.get(bucket, 0) + 1

    band_groups_result: dict = {"formed": [], "refused": [], "manifests": []}
    if detect_band_groups:
        from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

        for bucket in sorted(touched_buckets):
            bucket_dir = dataset_layout.image_dir(dest_root, bucket)
            result = detect_and_write_band_groups(bucket_dir)
            for g in result["formed"]:
                band_groups_result["formed"].append({**g, "bucket": bucket})
            band_groups_result["refused"].extend(result["refused"])
            band_groups_result["manifests"].extend(result["manifests"])

    return {
        "project_path": str(dest_root),
        "name": name,
        "image_root": str(dataset_layout.image_dir(dest_root, None)),
        "total": copied + moved,
        "found": len(sources),
        "copied": copied,
        "moved": moved,
        "buckets": dict(sorted(buckets.items())),
        "undated": undated,
        "skipped_collisions": skipped_collisions,
        "errors": errors,
        "unreadable_dates": unreadable_dates,
        "move": not copy,
        "tcip_dir": scaffold["tcip_dir"],
        "band_groups": band_groups_result,
    }
