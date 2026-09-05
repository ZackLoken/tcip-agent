"""Shared image utilities for the composable ML pipeline (channel-aware)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

import tcip_store
from tcip_store import Key, StoreDescriptor, Version, VersionConflict, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.data.band_groups import (
    BandGroupIncomplete,
    BandGroupRef,
    MANIFEST_EXT,
    band_group_manifest_key,
    read_band_group_manifest,
)
from tcip_mcp.pipelines.raster_source import Rect

if TYPE_CHECKING:
    import torch

__all__ = [
    "AmbiguousImageStem", "BandGroupIncomplete", "BandGroupRef", "IMAGE_EXTS",
    "bucket_logical_identities", "capture_kind", "crop_pad_tile", "display_source_path",
    "flat_image_key", "image_dimensions", "list_logical_images", "load_image", "load_multiband",
    "logical_image_name", "pad_tile", "pil_to_tensor", "place_logical_image",
    "resolve_image_source", "stem_collision_key", "stem_of", "to_pil_if_faithful",
]


class AmbiguousImageStem(ValueError):
    """A directory holds more than one logical identity under one case-folded stem
    (:func:`stem_collision_key`): two raw files (``foo.jpg``, ``foo.png``, or a same-key case
    variant such as ``Foo.jpg``), or a raw file and a ``.bandgroup`` manifest recorded under a
    different exact stem than its own.

    Raised from :func:`list_logical_images` rather than silently keeping only one identity (which
    would make the other vanish from every listing, including training, splits, review, and
    gallery, with no error): CLAUDE.md's "no silent fallback on ambiguous identity" rule.
    """

# ``.npy``/``.npz`` are a multi-band raster; ``.bandgroup`` a manifest standing in for the image it
# names.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".tif", ".tiff", ".npy", ".npz", MANIFEST_EXT}

def stem_collision_key(name: str) -> str:
    """The fold that decides whether two stems name one logical identity: ``str.lower()``.

    A portability decision, not a guess at any one filesystem's own table: the label file every
    reader resolves by stem is one file for ``Foo`` and ``foo`` on a case-insensitive filesystem,
    so a dataset holding both is unreadable there whatever filesystem it was built on, and this
    fold refuses the pair everywhere rather than only where it happens to break. Exact for ASCII
    names; not ``casefold()`` and not a filesystem's own table, so a pair differing only by
    Unicode normalization, or by a casefold-but-not-``lower()`` difference, is two keys here.
    """
    return name.lower()


def _scan_identities(d: Path) -> dict[str, list[tuple[Path, "BandGroupRef | None"]]]:
    """One manifest glob and one ``d.iterdir()`` walk with one extension test, the shared
    enumeration :func:`list_logical_images` and :func:`bucket_logical_identities` both build on,
    so directory enumeration happens once per call rather than the two independent walks each
    used to make. Keyed by :func:`stem_collision_key`; a key's list holds more than one entry
    exactly when it is ambiguous.

    A readable manifest is one identity under its own exact stem, paired with the parsed
    :class:`BandGroupRef` so a caller building :func:`list_logical_images`'s result never re-reads
    it; its claimed band files are that identity's members and no identity of their own. An
    unreadable or corrupt manifest claims nothing and is no identity. A manifest whose
    ``schema_version`` this reader does not accept raises :class:`tcip_store.SchemaVersionRefused`,
    uncaught, before anything else. Every other file with a suffix in :data:`IMAGE_EXTS` that no
    readable manifest claims is one raw identity under its own exact stem, paired with ``None``.
    """
    identities: dict[str, list[tuple[Path, BandGroupRef | None]]] = {}
    claimed: set[str] = set()
    for mp in sorted(d.glob(f"*{MANIFEST_EXT}")):
        try:
            ref = read_band_group_manifest(mp)
        except (OSError, ValueError):
            continue
        identities.setdefault(stem_collision_key(ref.stem), []).append((ref.manifest_path, ref))
        claimed.update(p.name for p in ref.bands.values())
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == MANIFEST_EXT or p.name in claimed:
            continue
        if ext not in IMAGE_EXTS:
            continue
        identities.setdefault(stem_collision_key(p.stem), []).append((p, None))
    return identities


def bucket_logical_identities(images_dir: str | Path) -> dict[str, list[Path]]:
    """The bucket's logical identities, grouped by :func:`stem_collision_key`: a key held by more
    than one identity is ambiguous, the one rule :func:`list_logical_images`, the ingest door's
    pre-scan and the conform script's census all check.

    Each list entry is one identity's own defining path (a ``.bandgroup`` manifest's path, or a
    raw file's own path); an absent directory answers empty. Never raises on an ambiguous key
    itself: that is each caller's own decision (a refusal here, a reported collision there).
    """
    d = Path(images_dir)
    if not d.is_dir():
        return {}
    return {key: [path for path, _ref in entries] for key, entries in _scan_identities(d).items()}


def list_logical_images(images_dir: str | Path) -> dict[str, "Path | BandGroupRef"]:
    """Every logical image in ``images_dir``, by exact stem.

    Built from :func:`bucket_logical_identities`' own scan (:func:`_scan_identities`): a
    :class:`BandGroupRef` for a manifest identity, that file's own path for a raw one. Raises
    :class:`AmbiguousImageStem` naming every ambiguous key's own paths; folded uniqueness never
    means case-insensitive lookup, so the returned mapping is keyed by the exact stem, exactly as
    it was before the fold, since the refusal above makes a key's divergent identities unreachable.

    A manifest whose ``schema_version`` this reader does not accept propagates as
    :class:`tcip_store.SchemaVersionRefused`, uncaught, rather than as this refusal: a newer-written
    grouped capture refuses the enumeration outright rather than silently dissolving into its
    individual band files.
    """
    d = Path(images_dir)
    if not d.is_dir():
        return {}
    scanned = _scan_identities(d)
    ambiguous = {key: entries for key, entries in scanned.items() if len(entries) > 1}
    if ambiguous:
        names = sorted(str(path) for entries in ambiguous.values() for path, _ref in entries)
        raise AmbiguousImageStem(
            f"{d}: {names} name more than one logical image under one case-folded stem, "
            "refusing to silently keep one. Rename so each logical image has its own stem."
        )
    result: dict[str, Path | BandGroupRef] = {}
    for entries in scanned.values():
        path, ref = entries[0]
        if ref is not None:
            result[ref.stem] = ref
        else:
            result[path.stem] = path
    return result


def capture_kind(source: "Path | BandGroupRef") -> str:
    """The kind of capture ``list_logical_images`` enumerated a stem under: ``"band_group"`` for
    a :class:`BandGroupRef`, ``"raster"`` for the suffixes ``load_multiband`` treats as an array
    container (``raster_source.ARRAY_CONTAINER_EXTS``: ``.npy``/``.npz``/``.tif``/``.tiff``),
    ``"image"`` for the rest (``.jpg``/``.jpeg``/``.png``/``.bmp``/``.heic``).
    """
    if isinstance(source, BandGroupRef):
        return "band_group"
    if Path(source).suffix.lower() in raster_source.ARRAY_CONTAINER_EXTS:
        return "raster"
    return "image"


def resolve_image_source(images_dir: str | Path, stem: str) -> "Path | BandGroupRef":
    """``list_logical_images(images_dir)[stem]``: never a second implementation of "what images
    live here."

    Raises ``FileNotFoundError`` for an unknown stem, and ``AmbiguousImageStem`` (propagated from
    ``list_logical_images``, uncaught here) when ``images_dir`` holds a collision, whether or not
    ``stem`` is one of the colliding keys: the directory is unlistable, so no single stem in it
    resolves cleanly. For a ``BandGroupRef`` whose manifest references a sibling file that no
    longer exists, raises ``BandGroupIncomplete`` here (the resolver) rather than letting a bare
    decode error surface later inside a stacking loop.
    """
    src = list_logical_images(images_dir).get(stem)
    if src is None:
        raise FileNotFoundError(f"No image for stem: {stem}")
    if isinstance(src, BandGroupRef):
        missing = [name for name, p in src.bands.items() if not p.is_file()]
        if missing:
            raise BandGroupIncomplete(
                f"band group {stem!r} ({src.manifest_path}) references missing band(s) "
                f"{sorted(missing)}: delete the manifest to let a later detection pass re-group "
                "the surviving siblings, or restore the missing file(s)."
            )
    return src


def logical_image_name(source: "Path | BandGroupRef") -> str:
    """The name a by-name reader (an image-status bucket, a COCO ``file_name``) resolves this
    logical image under: a :class:`BandGroupRef`'s own ``.bandgroup`` manifest name, since that
    file stands in for the whole grouped capture everywhere a name is matched against a store;
    a plain path's own name otherwise.
    """
    return source.manifest_path.name if isinstance(source, BandGroupRef) else source.name


FLAT_IMAGE_STORE = "flat_image"
_FLAT_IMAGE_LOCATOR = RootedFileLocator()
register_store(
    StoreDescriptor(
        name=FLAT_IMAGE_STORE,
        kind="blob",
        key_fields=("filename",),
        frozen=True,
        cannot_carry_field="raw placed-image bytes, the same raw-bytes nature as imagery",
        locator=_FLAT_IMAGE_LOCATOR,
    )
)


def flat_image_key(images_dir: str | Path, filename: str) -> Key:
    """One placed image's bytes, addressed by the flat directory it was materialized into.

    A directory-rooted, one-part blob key sibling to ``annotation_record_key`` and
    ``band_group_manifest_key``'s own shape, for a curated dataset's or a materialized split's
    ``images/`` tree, which holds every placed band and plain image directly with no further
    layout claim. This is a different tree from ``image_key``'s dated ingest layout, not a
    second address for the same one: an ingested capture and a placed copy of it are two
    distinct stores that may legally coexist over one file, since the file backend locks the
    canonical path, not the store name.
    """
    return Key(FLAT_IMAGE_STORE, str(Path(images_dir).absolute()), (filename,))


def place_logical_image(
    source: "Path | BandGroupRef",
    dest_dir: str | Path,
    *,
    copy_files: bool,
    dest_key: "Callable[[str], Key]",
) -> str:
    """Copies (or symlinks) one logical image into ``dest_dir`` and returns the name
    :func:`logical_image_name` gives it there.

    A :class:`BandGroupRef` places every sibling band it names plus its own ``.bandgroup``
    manifest: the manifest alone resolves to nothing once its siblings are absent, so a caller
    materializing a band-grouped capture (a curated review tree, a train/val split) must place
    the whole group, never the manifest by itself. A plain path places just that one file.

    ``dest_dir`` is absolutized once, here, at entry: the bands and plain images route through
    the caller's own ``dest_key`` (already absolute, since ``flat_image_key`` absolutizes its own
    directory argument), but the destination manifest's key is derived from ``dest_dir`` itself
    below, and a relative caller-supplied one would otherwise reach that derivation unabsolutized
    and refuse ``BadKey`` after the bands had already landed, half-placing the group.

    ``copy_files=True`` routes each band and plain image through the store under
    ``dest_key(filename)`` -- the caller's own key, since deriving one from ``dest_dir`` here
    would nest ``images/images/`` -- and the destination manifest through its own
    ``band_group_manifest_key(dest_dir, stem)``, so that store earns its row rather than
    borrowing the imagery one. A cheap ``dst.exists()`` check in front of each store write skips
    a destination already placed without reading or hashing its existing bytes; it is an
    optimization only; the store's own create-only write (``expect=Version.ABSENT``) and a
    caught ``VersionConflict`` are the leave-alone answer's actual correctness, atomic under the
    key's lock, for the race the cheap check cannot see: a concurrent placer of the same file. On
    POSIX a store-routed write also leaves a ``.lock`` sidecar beside its destination (the
    store's own documented convention), so a placed tree carries the same sidecars an ingested
    tree already does there.

    ``copy_files=False`` symlinks instead, a filesystem operation the store takes no part in: a
    destination is checked for presence with ``os.path.lexists`` (a dangling link reads as
    present here, where the store-routed copy above would overwrite one) before ``os.symlink``,
    whose ``FileExistsError`` is also caught, for idempotency against a concurrent placement of
    the same link.
    """
    dest_dir = Path(dest_dir).absolute()

    def _place_copy(src_path: Path, key: Key, dst: Path) -> None:
        if dst.exists():
            return  # already placed: a shared band group or a re-run over an existing tree
        try:
            tcip_store.put_blob_from_path(key, src_path, expect=Version.ABSENT)
        except VersionConflict:
            pass  # a concurrent placer won the race; the file is there either way

    def _place_symlink(src_path: Path, dst: Path) -> None:
        if os.path.lexists(dst):
            return
        try:
            os.symlink(str(src_path), str(dst))
        except FileExistsError:
            pass  # a concurrent placement won the race; the link is there either way

    if isinstance(source, BandGroupRef):
        for band_path in source.bands.values():
            if copy_files:
                _place_copy(band_path, dest_key(band_path.name), dest_dir / band_path.name)
            else:
                _place_symlink(band_path, dest_dir / band_path.name)
        if copy_files:
            _place_copy(
                source.manifest_path, band_group_manifest_key(dest_dir, source.stem),
                dest_dir / source.manifest_path.name,
            )
        else:
            _place_symlink(source.manifest_path, dest_dir / source.manifest_path.name)
        return source.manifest_path.name

    if copy_files:
        _place_copy(source, dest_key(source.name), dest_dir / source.name)
    else:
        _place_symlink(source, dest_dir / source.name)
    return source.name


def stem_of(source: "str | Path | BandGroupRef") -> str:
    """The logical image's own stem, whatever concrete type ``source`` is.

    A :class:`BandGroupRef` carries its canonical stem directly (there is no single sibling
    filename to re-derive it from); a plain path/string uses ``Path(...).stem``. The one place a
    caller holding a list that mixes raw paths and band-grouped captures (e.g. every element
    ``list_logical_images`` returned) gets each one's stem without a per-call type check of its own.
    """
    if isinstance(source, BandGroupRef):
        return source.stem
    return Path(source).stem


def display_source_path(source: "str | Path | BandGroupRef") -> str:
    """A JSON-safe, human-meaningful identity string for a predict result's ``image`` field.

    A :class:`BandGroupRef` has no single sibling file that names the logical image, its own
    ``.bandgroup`` manifest path is the closest thing (stable, on disk, unique per capture); a plain
    path/string is returned as-is. A caller decoding the source itself keeps passing the original
    source object (never this string) to ``load_image``/``load_multiband``, so a band-grouped
    capture still decodes through the channel-aware loader instead of a stringified dataclass repr
    that no reader can open.
    """
    if isinstance(source, BandGroupRef):
        return str(source.manifest_path)
    return str(source)


def _channels_from_shape(shape: tuple[int, ...]) -> int:
    """1 for a 2-D ``(H, W)`` shape; otherwise the channel axis of a channel-first-or-last 3-D
    shape, by "the smaller of the two non-spatial-looking axes is the channel axis", used only
    where no expected channel count is already known to compare against (``derivations.probe_channels``'s
    TIFF branch); a caller that already knows the expected count (``image_dimensions``,
    ``load_multiband``) compares against it directly instead, which is the more precise formula."""
    if len(shape) == 2:
        return 1
    return int(shape[0]) if shape[0] < shape[-1] else int(shape[-1])


def image_dimensions(path: "str | Path | BandGroupRef", num_channels: int = 3) -> tuple[int, int]:
    """``(width, height)`` as ``load_image`` will decode it, without decoding pixels where possible.

    ``tcip_annotation.get_image_dimensions`` reads through PIL, which is right for the photographic
    formats it was written for and wrong for a multi-band raster: PIL reports a 5-band 40x24
    GeoTIFF as 5x40. Labels clipped against one frame and tiles cropped from another displace every
    box silently, so anything that measures a frame it will later decode must route the same way
    ``load_image`` does.

    A :class:`BandGroupRef` reads its dims from one sibling band file (a group's members share one
    spatial frame by construction, which is what makes stacking them into ``[H, W, C]`` valid).

    Which decode a source routes to is ``raster_source.photographic_container``'s decision, the
    same one :func:`load_image` delegates to, never a second predicate here that could drift.
    """
    if isinstance(path, BandGroupRef):
        one_band = next(iter(path.bands.values()))
        return image_dimensions(one_band, 1)
    path = Path(path)
    ext = path.suffix.lower()
    if raster_source.photographic_container(path, num_channels):
        from tcip_annotation.utils import get_image_dimensions

        return get_image_dimensions(str(path))  # header-only, EXIF-aware
    if ext in (".tif", ".tiff"):
        # The frame the TIFF dispatch's own backend will serve, from one shared set of rules.
        frame = raster_source.tiff_frame(path, num_channels)
        if frame is not None:
            return int(frame[1]), int(frame[0])
    arr = load_multiband(path, num_channels)
    return int(arr.shape[1]), int(arr.shape[0])


def pad_tile(crop, tile_size: int):
    """Zero-pad an already-cropped tile up to ``tile_size`` x ``tile_size``.

    Channel-generic: PIL for 1/3/4-channel images, numpy ``[H, W, C]`` for multi-band rasters. The
    padding half of :func:`crop_pad_tile`, split out so a caller that sources a tile from
    somewhere other than a fully decoded in-memory image (e.g. a windowed raster reader, which
    hands back an already-cropped, possibly short edge tile) pads it identically instead of a
    second padding implementation drifting from this one.
    """
    if isinstance(crop, Image.Image):
        if crop.size != (tile_size, tile_size):
            padded = Image.new(crop.mode, (tile_size, tile_size))  # 0-fill for the image's mode
            padded.paste(crop, (0, 0))
            crop = padded
        return crop
    ph, pw = tile_size - crop.shape[0], tile_size - crop.shape[1]
    if ph or pw:
        pad_width = [(0, ph), (0, pw)] + ([(0, 0)] if crop.ndim == 3 else [])
        crop = np.pad(crop, pad_width, mode="constant")
    return crop


def crop_pad_tile(img, x: int, y: int, tile_size: int, w: int, h: int):
    """Crop a ``tile_size`` window at (x, y) and zero-pad short (edge) tiles.

    Channel-generic: PIL for 1/3/4-channel images, numpy ``[H, W, C]`` for multi-band rasters
    (which have no ``.crop``). Shared by the training tiler and the inference tiler so the two ends
    of the reproduce-a-number chain cannot crop differently. Slices the window, then delegates the
    padding to :func:`pad_tile`.
    """
    x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
    crop = img.crop((x, y, x2, y2)) if isinstance(img, Image.Image) else img[y:y2, x:x2]
    return pad_tile(crop, tile_size)


def to_pil_if_faithful(arr, *, band_interpretations: "tuple[str, ...] | None" = None):
    """An ``[H, W, C]`` uint8 ndarray with 1 or 3 channels as a PIL image (mode L or RGB, a
    single channel squeezed); a 4-channel array only when ``band_interpretations`` names its
    4th band ``"alpha"``; anything else is returned unchanged.

    The one conversion the loaders share, so uint8 pixels reach the PIL-only augmentation chain
    whatever container they decoded from, while dtypes and band counts PIL has no faithful mode
    for stay ndarray. RGBA is a faithful *pixel* round-trip for any 4-channel uint8 array, but
    PIL's augmentation chain (brightness/contrast/saturation) treats an alpha channel differently
    than a color one, silently leaving it unperturbed -- correct when the 4th channel really is
    transparency, wrong when it is a genuine spectral band (RGB+NIR) that should be treated like
    any other channel. ``band_interpretations`` is the caller's own source's GDAL color
    interpretations (``getattr(src, "band_interpretations", None)``, the convention
    ``raster_source`` already uses elsewhere), the only backend-carried fact that distinguishes
    the two; with no such signal (an ``.npy``/``.npz`` stack, a band group, or a GDAL file whose
    4th band is untagged) a 4-channel array is never guessed into RGBA, and stays ndarray on the
    unaugmented path uint16/5-band inputs already take.
    """
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        return arr
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    if arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    if band_interpretations is not None and len(band_interpretations) == 4 \
            and band_interpretations[3] == "alpha":
        return Image.fromarray(arr, mode="RGBA")
    return arr


def pil_to_tensor(img) -> torch.Tensor:
    """Convert a PIL Image or H×W[×C] array to a float32 ``[C, H, W]`` tensor in ``[0, 1]``.

    Channel-aware: a 2-D grayscale array becomes ``[1, H, W]``; any channel count is
    supported. Integer inputs are scaled by their dtype max (uint8→/255, uint16→/65535);
    float inputs are assumed already normalized.
    """
    import torch

    arr = np.asarray(img)
    if arr.ndim == 2:  # grayscale [H, W] -> [H, W, 1]
        arr = arr[:, :, None]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_image(path: "str | Path | BandGroupRef", num_channels: int = 3):
    """Open an image honoring ``num_channels``.

    Returns a ``PIL.Image`` wherever PIL has a faithful mode for the decoded pixels: photographic
    formats at 1/3/4 channels, and any array container (``.npy`` / ``.npz`` / GeoTIFF / a
    :class:`BandGroupRef`) whose pixels come back uint8 with 1 or 3 channels, or 4 channels whose
    4th band the source itself declares alpha (:func:`to_pil_if_faithful`), so the PIL-only
    augmentation chain applies to those regardless of container. Everything else (uint16/float
    rasters, other band counts, an undeclared or genuinely spectral 4th band) stays an
    ``[H, W, C]`` ndarray. An RGB file requested as 1 channel is converted to grayscale; as 3,
    kept RGB.

    Reads through ``raster_source``: which backend decodes a source is that module's own dispatch,
    so a frame this returns and one ``image_dimensions`` measures can never come from two
    different decisions.
    """
    with raster_source.open_raster(path, num_channels) as src:
        if isinstance(src, raster_source.PhotographicSource):
            return src.image
        pixels = src.read_region(Rect(0, 0, src.width, src.height))[0]
        return to_pil_if_faithful(pixels, band_interpretations=getattr(src, "band_interpretations", None))


def load_multiband(path: "str | Path | BandGroupRef", num_channels: int) -> np.ndarray:
    """Load a multi-band image as ``[H, W, C]`` (NPY/NPZ natively; GeoTIFF via tifffile).

    A :class:`BandGroupRef` decodes each sibling file (each already a supported single-band
    source) and stacks them into one ``[H, W, C]`` array in the manifest's declared band order:
    the one place virtual (in-memory) stacking happens; never written back to disk.

    A photographic container (anything outside ``.npy`` / ``.npz`` / ``.tif`` / ``.tiff``) raises
    ``ValueError`` at any channel count: band data is what this returns, and a PIL frame is
    :func:`load_image`'s business.
    """
    with raster_source.open_array_source(path, num_channels) as src:
        return src.read_region(Rect(0, 0, src.width, src.height))[0]
