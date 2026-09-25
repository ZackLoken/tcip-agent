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
    "refuse_incomplete_band_group", "resolve_image_source", "stem_collision_key", "stem_of",
    "to_pil_if_faithful",
]


class AmbiguousImageStem(ValueError):
    """A directory holds more than one logical identity under one case-folded stem
    (:func:`stem_collision_key`): two raw files (``foo.jpg``, ``foo.png``, or a same-key case
    variant such as ``Foo.jpg``), or a raw file and a ``.bandgroup`` manifest recorded under a
    different exact stem than its own. Raised from :func:`list_logical_images`.
    """

# ``.npy``/``.npz`` are a multi-band raster; ``.bandgroup`` a manifest standing in for the image it
# names.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".tif", ".tiff", ".npy", ".npz", MANIFEST_EXT}

def stem_collision_key(name: str) -> str:
    """The fold that decides whether two stems name one logical identity: ``str.lower()``.

    Exact for ASCII names; not ``casefold()`` and not a filesystem's own table, so a pair differing
    only by Unicode normalization, or by a casefold-but-not-``lower()`` difference, is two keys
    here.
    """
    return name.lower()


def _scan_identities(d: Path) -> dict[str, list[tuple[Path, "BandGroupRef | None"]]]:
    """One manifest glob and one ``d.iterdir()`` walk with one extension test, keyed by
    :func:`stem_collision_key`; a key's list holds more than one entry exactly when it is
    ambiguous.

    A readable manifest is one identity under its own exact stem, paired with the parsed
    :class:`BandGroupRef`; its claimed band files are that identity's members and no identity of
    their own. An unreadable or corrupt manifest claims nothing and is no identity. A manifest
    whose ``schema_version`` this reader does not accept raises
    :class:`tcip_store.SchemaVersionRefused`, uncaught, before anything else. Every other file with
    a suffix in :data:`IMAGE_EXTS` that no readable manifest claims is one raw identity under its
    own exact stem, paired with ``None``.
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
    than one identity is ambiguous.

    Each list entry is one identity's own defining path (a ``.bandgroup`` manifest's path, or a raw
    file's own path); an absent directory answers empty. Never raises on an ambiguous key.
    """
    d = Path(images_dir)
    if not d.is_dir():
        return {}
    return {key: [path for path, _ref in entries] for key, entries in _scan_identities(d).items()}


def list_logical_images(images_dir: str | Path) -> dict[str, "Path | BandGroupRef"]:
    """Every logical image in ``images_dir``, by exact stem.

    Built from :func:`_scan_identities`: a :class:`BandGroupRef` for a manifest identity, that
    file's own path for a raw one. Raises :class:`AmbiguousImageStem` naming every ambiguous key's
    own paths; the returned mapping is keyed by the exact stem. A manifest whose ``schema_version``
    this reader does not accept propagates as :class:`tcip_store.SchemaVersionRefused`.
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


def refuse_incomplete_band_group(source: "Path | BandGroupRef") -> "Path | BandGroupRef":
    """``source`` itself, refusing a grouped capture whose manifest names a band that is gone."""
    if isinstance(source, BandGroupRef):
        missing = [name for name, p in source.bands.items() if not p.is_file()]
        if missing:
            raise BandGroupIncomplete(
                f"band group {source.stem!r} ({source.manifest_path}) references missing band(s) "
                f"{sorted(missing)}: delete the manifest to let a later detection pass re-group "
                "the surviving siblings, or restore the missing file(s)."
            )
    return source


def resolve_image_source(images_dir: str | Path, stem: str) -> "Path | BandGroupRef":
    """``list_logical_images(images_dir)[stem]``.

    Raises ``FileNotFoundError`` for an unknown stem, and ``AmbiguousImageStem`` (propagated from
    ``list_logical_images``) when ``images_dir`` holds a collision, whether or not ``stem`` is one
    of the colliding keys. A grouped capture missing a band refuses through
    :func:`refuse_incomplete_band_group`.
    """
    src = list_logical_images(images_dir).get(stem)
    if src is None:
        raise FileNotFoundError(f"No image for stem: {stem}")
    return refuse_incomplete_band_group(src)


def resolve_source_path(source: str | Path) -> "Path | BandGroupRef":
    """The logical image one recorded source path names, for a reader holding the path itself
    rather than a directory and a stem.

    A ``.bandgroup`` path resolves to the grouped capture it stands for, checked through
    :func:`refuse_incomplete_band_group`. Any other path is the image itself, refused by name when
    it is not on disk.
    """
    path = Path(source)
    if path.suffix.lower() != MANIFEST_EXT:
        if not path.exists():
            raise FileNotFoundError(f"No image at the recorded source path: {path}")
        return path
    if not path.is_file():
        raise FileNotFoundError(f"No band group manifest at the recorded source path: {path}")
    return refuse_incomplete_band_group(read_band_group_manifest(path))


def source_path_of(source: "Path | BandGroupRef") -> str:
    """The path a selection records this logical image under: a grouped capture's own
    ``.bandgroup`` manifest, a plain image's own file. The inverse
    :func:`resolve_source_path` reads back."""
    return str(source.manifest_path if isinstance(source, BandGroupRef) else source)


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
    """One placed image's bytes, addressed by the flat directory it was materialized into: a
    curated dataset's ``images/`` tree, distinct from ``image_key``'s dated ingest layout.
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
    manifest; a plain path places just that one file. ``dest_dir`` is absolutized at entry.

    ``copy_files=True`` routes each band and plain image through the store under
    ``dest_key(filename)`` and the destination manifest under ``band_group_manifest_key(dest_dir,
    stem)``, each a create-only write (``expect=Version.ABSENT``) that leaves an already-placed
    destination alone. ``copy_files=False`` symlinks instead, skipping a destination
    ``os.path.lexists`` reports present and tolerating a concurrent placement's
    ``FileExistsError``.
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
    """The logical image's own stem, whatever concrete type ``source`` is: a
    :class:`BandGroupRef`'s canonical stem, else ``Path(...).stem``.
    """
    if isinstance(source, BandGroupRef):
        return source.stem
    return Path(source).stem


def display_source_path(source: "str | Path | BandGroupRef") -> str:
    """A JSON-safe, human-meaningful identity string for a predict result's ``image`` field: a
    :class:`BandGroupRef`'s ``.bandgroup`` manifest path, else the path/string as-is. Never a
    decodable source.
    """
    if isinstance(source, BandGroupRef):
        return str(source.manifest_path)
    return str(source)


def _channels_from_shape(shape: tuple[int, ...]) -> int:
    """1 for a 2-D ``(H, W)`` shape; otherwise the channel axis of a channel-first-or-last 3-D
    shape, by "the smaller of the two non-spatial-looking axes is the channel axis", for a caller
    with no expected channel count to compare against.
    """
    if len(shape) == 2:
        return 1
    return int(shape[0]) if shape[0] < shape[-1] else int(shape[-1])


def image_dimensions(path: "str | Path | BandGroupRef", num_channels: int = 3) -> tuple[int, int]:
    """``(width, height)`` as ``load_image`` will decode it, without decoding pixels where
    possible.

    A :class:`BandGroupRef` reads its dims from one sibling band file (a group's members share one
    spatial frame). Which decode a source routes to is ``raster_source.photographic_container``'s
    decision, as for :func:`load_image`.
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
    """Zero-pad an already-cropped tile up to ``tile_size`` x ``tile_size``. Channel-generic: PIL
    for 1/3/4-channel images, numpy ``[H, W, C]`` for multi-band rasters.
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
    """Crop a ``tile_size`` window at (x, y) and zero-pad short (edge) tiles through
    :func:`pad_tile`.

    Channel-generic: PIL for 1/3/4-channel images, numpy ``[H, W, C]`` for multi-band rasters
    (which have no ``.crop``).
    """
    x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
    crop = img.crop((x, y, x2, y2)) if isinstance(img, Image.Image) else img[y:y2, x:x2]
    return pad_tile(crop, tile_size)


def to_pil_if_faithful(arr, *, band_interpretations: "tuple[str, ...] | None" = None):
    """An ``[H, W, C]`` uint8 ndarray with 1 or 3 channels as a PIL image (mode L or RGB, a single
    channel squeezed); a 4-channel array only when ``band_interpretations`` (the source's GDAL
    color interpretations) names its 4th band ``"alpha"``; anything else is returned unchanged.
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


def load_image(path: "str | Path | BandGroupRef", num_channels: int):
    """Open an image at ``num_channels``, which every caller states.

    Returns a ``PIL.Image`` wherever PIL has a faithful mode for the decoded pixels: photographic
    formats at 1/3/4 channels, and any array container (``.npy`` / ``.npz`` / GeoTIFF / a
    :class:`BandGroupRef`) whose pixels come back uint8 with 1 or 3 channels, or 4 channels whose
    4th band the source itself declares alpha (:func:`to_pil_if_faithful`). Everything else
    (uint16/float rasters, other band counts, an undeclared or genuinely spectral 4th band) stays
    an ``[H, W, C]`` ndarray. An RGB file requested as 1 channel is converted to grayscale; as 3,
    kept RGB. Reads through ``raster_source``.
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
