"""Image ingestion — turn a raw folder of photos into a structured TCIP project.

``ingest_images`` is the missing structuring primitive. It copies (or moves) raw
images into the canonical layout (``images/<YYYY-MM-DD>/<stem><ext>``) under a
workspace project, bucketing by EXIF capture date. It does **not** annotate, split,
choose a task, or write ``classes.json`` — those are later steps in the project-setup
arc (see ``.github/skills/project-setup``). Keeping it thin is deliberate: one
auditable primitive the agent composes, instead of improvising file ops per project.
"""

from __future__ import annotations

import logging
from glob import glob
from pathlib import Path

from tcip_mcp import dataset_layout, workspace
from tcip_mcp.audit import audited
from tcip_mcp.server import mcp
from tcip_mcp.utils.atomic_io import atomic_write_bytes

logger = logging.getLogger(__name__)

# Mirrors the extensions the GUI/plant-mapping already accept.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp")
UNDATED_BUCKET = "undated"


_DT_ORIGINAL = 0x9003  # EXIF DateTimeOriginal tag id
_EXIF_IFD = 0x8769  # Exif sub-IFD offset tag


def _exif_iso_date(path: Path) -> str | None:
    """EXIF ``DateTimeOriginal`` → ISO ``YYYY-MM-DD``, or ``None``.

    EXIF stores the colon-date ``YYYY:MM:DD HH:MM:SS`` form. Reads via the public
    ``Image.getexif()`` + Exif sub-IFD so it works across formats (JPEG, TIFF, HEIC). PIL is
    lazy-imported so tool startup stays fast; any read failure degrades to ``None`` (→ the
    ``undated/`` bucket) rather than raising.
    """
    from datetime import datetime

    from PIL import Image

    raw = None
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            try:
                sub = exif.get_ifd(_EXIF_IFD)
            except Exception:
                sub = {}
            raw = sub.get(_DT_ORIGINAL)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _iter_source_images(source: str, recursive: bool):
    """Yield image files under ``source`` — a directory or a glob pattern."""
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
    """A literal ``date_from`` becomes a path segment — reject separators/traversal."""
    if date_from in ("exif", "none"):
        return
    if not workspace.is_valid_name(date_from):
        raise ValueError(f"invalid date_from literal bucket: {date_from!r}")


def _bucket_for(path: Path, date_from: str) -> str:
    if date_from == "exif":
        return _exif_iso_date(path) or UNDATED_BUCKET
    if date_from == "none":
        return UNDATED_BUCKET
    return date_from  # validated literal bucket


@mcp.tool()
@audited
def ingest_images(
    source: str,
    name: str,
    project_path: str = "",
    copy: bool = True,
    date_from: str = "exif",
    recursive: bool = True,
) -> dict:
    """Copy raw images into a structured project, bucketed by EXIF capture date.

    Turns a raw folder (or glob) of photos into the canonical layout
    (``images/<YYYY-MM-DD>/<stem><ext>``) under a workspace project. Copies by
    default — originals are left byte-identical; pass ``copy=False`` to move.
    Refuses to overwrite an existing image (records the collision and skips it).
    Does not annotate, split, choose a task, or write ``classes.json``.

    Args:
        source: Folder (or glob) of raw images, anywhere on disk.
        name: Project slug (``{crop}_{trait}_{site}``); the destination folder is
            ``<TCIP_WORKSPACE>/<name>/`` unless ``project_path`` overrides it.
        project_path: Absolute destination path instead of ``workspace/<name>``.
        copy: Copy (True, default) or move (False) the source images.
        date_from: ``"exif"`` (per-image ``DateTimeOriginal`` → ISO date, missing →
            ``undated/``), ``"none"`` (all → ``undated/``), or a literal bucket name
            (all → ``images/<literal>/``, e.g. a known ISO capture date).
        recursive: Recurse into source subfolders.

    Returns a manifest: ``{project_path, name, image_root, total, found, copied,
    moved, buckets, undated, skipped_collisions, errors, move}``.
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
    # Collisions are keyed by stem within a bucket (case-insensitively): labels and
    # predictions pair to an image by stem alone (see dataset_layout), so two sources with
    # the same stem but different extensions would otherwise silently share one label file.
    placed: set[tuple[str, str]] = set()

    for src_path in sources:
        bucket = _bucket_for(src_path, date_from)
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
        # batch — record it and keep going. Reading bytes never mutates the original;
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
        if bucket == UNDATED_BUCKET:
            undated += 1
        else:
            buckets[bucket] = buckets.get(bucket, 0) + 1

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
        "move": not copy,
        "tcip_dir": scaffold["tcip_dir"],
    }
