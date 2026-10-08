"""Shared image utilities for the composable ML pipeline (channel-aware)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.data.band_groups import (
    BandGroupIncompleteError,
    BandGroupRef,
    MANIFEST_EXT,
    read_band_group_manifest,
)
from tcip_mcp.pipelines.raster_source import Rect

if TYPE_CHECKING:
    import torch

__all__ = [
    "AmbiguousImageStemError", "BandGroupIncompleteError", "BandGroupRef", "IMAGE_EXTS",
    "bucket_logical_identities", "capture_kind", "display_frame",
    "image_dimensions", "list_logical_images", "load_image", "load_multiband",
    "logical_image_name", "pil_to_tensor", "pixel_array",
    "refuse_incomplete_band_group", "resolve_image_path", "resolve_image_paths",
    "source_path_of", "stem_collision_key", "stem_of",
    "to_pil_if_faithful",
]


class AmbiguousImageStemError(ValueError):
    """A directory holds more than one logical identity under one case-folded stem
    (:func:`stem_collision_key`): two raw files (``foo.jpg``, ``foo.png``, or a same-key case
    variant such as ``Foo.jpg``), or a raw file and a ``.bandgroup`` manifest recorded under a
    different exact stem than its own.
    """


# ``.npy``/``.npz`` are a multi-band raster; ``.bandgroup`` a manifest standing in for the image it
# names.
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".heic", ".tif", ".tiff", ".npy", ".npz", MANIFEST_EXT
}


def stem_collision_key(name: str) -> str:
    """The fold that decides whether two stems name one logical identity: ``str.lower()``, so a
    pair differing only by Unicode normalization, or by a casefold-but-not-``lower()`` difference,
    is two keys."""
    return name.lower()


def _scan_identities(d: Path) -> dict[str, list[tuple[Path, "BandGroupRef | None"]]]:
    """One manifest glob and one ``d.iterdir()`` walk with one extension test, keyed by
    :func:`stem_collision_key`; a key's list holds more than one entry exactly when it is
    ambiguous.

    A readable manifest is one identity under its own exact stem, paired with the parsed
    :class:`BandGroupRef`; its claimed band files are that identity's members and no identity of
    their own. An unreadable or corrupt manifest claims nothing and is no identity. Every other
    file with a suffix in :data:`IMAGE_EXTS` that no readable manifest claims is one raw identity
    under its own exact stem, paired with ``None``.
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
    file's own path for a raw one. Raises :class:`AmbiguousImageStemError` naming every ambiguous
    key's own paths; the returned mapping is keyed by the exact stem.
    """
    d = Path(images_dir)
    if not d.is_dir():
        return {}
    scanned = _scan_identities(d)
    _refuse_ambiguous(d, scanned.values())
    return {stem_of(source): source for source in map(_identity, scanned.values())}


def _identity(entries: list[tuple[Path, "BandGroupRef | None"]]) -> "Path | BandGroupRef":
    """The logical image one unambiguous :func:`_scan_identities` key holds."""
    path, ref = entries[0]
    return ref if ref is not None else path


def _refuse_ambiguous(d: Path, keys: "Iterable[list[tuple[Path, BandGroupRef | None]]]") -> None:
    """Refuse (:class:`AmbiguousImageStemError`) the :func:`_scan_identities` keys of ``d`` holding
    more than one identity, naming their paths."""
    names = sorted(str(path) for entries in keys if len(entries) > 1 for path, _ref in entries)
    if names:
        raise AmbiguousImageStemError(
            f"{d}: {names} name more than one logical image under one case-folded stem, "
            "refusing to silently keep one. Rename so each logical image has its own stem."
        )


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
            raise BandGroupIncompleteError(
                f"band group {source.stem!r} ({source.manifest_path}) references missing band(s) "
                f"{sorted(missing)}: delete the manifest to let a later detection pass re-group "
                "the surviving siblings, or restore the missing file(s)."
            )
    return source


def resolve_image_paths(paths: "Iterable[str | Path]") -> "list[Path | BandGroupRef]":
    """The logical image each of ``paths`` names, each directory scanned once
    (:func:`_scan_identities`): a ``.bandgroup`` manifest's grouped capture, a raw image its own
    file.

    Refuses ``FileNotFoundError`` for a path naming no logical image, naming the manifest that
    claims it when it is a band of a grouped capture; :class:`AmbiguousImageStemError` when its own
    stem names more than one; ``BandGroupIncompleteError`` for a grouped capture missing a band.
    """
    named_paths = [Path(p) for p in paths]
    scanned = {d: _scan_identities(d) if d.is_dir() else {}
               for d in {p.parent for p in named_paths}}

    def named(path: Path) -> "Path | BandGroupRef":
        entries = scanned[path.parent].get(stem_collision_key(path.stem), [])
        _refuse_ambiguous(path.parent, [entries])
        if not entries or entries[0][0].name != path.name:
            owners = [str(ref.manifest_path) for (_p, ref), *_ in scanned[path.parent].values()
                      if ref is not None and path.name in {b.name for b in ref.bands.values()}]
            raise FileNotFoundError(
                f"{path} names no logical image" + (
                    f": it is a band of the grouped capture {owners[0]}; name the manifest"
                    if owners else ""))
        return refuse_incomplete_band_group(_identity(entries))

    return [named(p) for p in named_paths]


def resolve_image_path(image_path: str | Path) -> "Path | BandGroupRef":
    """The logical image ``image_path`` names (:func:`resolve_image_paths`); its refusals
    propagate."""
    return resolve_image_paths([image_path])[0]


def image_path_dimensions(image_path: str | Path) -> tuple[int, int]:
    """:func:`display_frame` of the logical image ``image_path`` names
    (:func:`resolve_image_path`, a grouped capture folded into its one frame). Its refusals
    propagate."""
    return display_frame(resolve_image_path(image_path))


def display_frame(source: "str | Path | BandGroupRef") -> tuple[int, int]:
    """``(width, height)`` of ``source`` as the image route's plain read opens it, at
    :func:`~tcip_mcp.pipelines.raster_source.image_route_channel_count`: the frame a viewer draws
    the raster in and annotation coordinates are measured in."""
    return image_dimensions(source, raster_source.image_route_channel_count(source))


def source_path_of(source: "str | Path | BandGroupRef") -> str:
    """The path this logical image is named by, recorded and displayed: a grouped capture's own
    ``.bandgroup`` manifest, a plain image's own file. The inverse :func:`resolve_image_path`
    reads back."""
    return str(source.manifest_path if isinstance(source, BandGroupRef) else source)


def logical_image_name(source: "Path | BandGroupRef") -> str:
    """The name a by-name reader (a verdict shard, a COCO ``file_name``) resolves this
    logical image under: a :class:`BandGroupRef`'s own ``.bandgroup`` manifest name, since that
    file stands in for the whole grouped capture everywhere a name is matched against a store;
    a plain path's own name otherwise.
    """
    return source.manifest_path.name if isinstance(source, BandGroupRef) else source.name


def logical_images_by_name(images_dir: str | Path) -> dict[str, str]:
    """Every logical image in ``images_dir`` (:func:`list_logical_images`), its
    :func:`logical_image_name` to its exact stem, in name order. Raises what
    :func:`list_logical_images` raises."""
    named = {logical_image_name(src): stem for stem, src in list_logical_images(images_dir).items()}
    return {name: named[name] for name in sorted(named)}


def stem_of(source: "str | Path | BandGroupRef") -> str:
    """The logical image's own stem, whatever concrete type ``source`` is: a
    :class:`BandGroupRef`'s canonical stem, else ``Path(...).stem``.
    """
    if isinstance(source, BandGroupRef):
        return source.stem
    return Path(source).stem


CAPTURE_TIME_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d")
"""The spellings a capture time arrives in: the colon form EXIF's DateTimeOriginal and TIFF's
DateTime tag are specified to use, and the ISO forms a stitching engine writes its own in."""

_EXIF_IFD_TAG = 0x8769
_DT_ORIGINAL_TAG = 0x9003


def exif_capture_time(exif) -> object | None:
    """The raw DateTimeOriginal an image's ``exif`` (``Image.getexif()``) carries: from the Exif
    sub-IFD where a camera nests it, else from the top level where a fresh ``PIL.Image.Exif``
    writes it; ``None`` when it carries none."""
    raw = exif.get_ifd(_EXIF_IFD_TAG).get(_DT_ORIGINAL_TAG)
    return exif.get(_DT_ORIGINAL_TAG) if raw is None else raw


def parse_capture_time(raw: object):
    """A capture-time value in any of :data:`CAPTURE_TIME_FORMATS` as a ``datetime``; ``None``
    when none fits."""
    from datetime import datetime

    text = str(raw).strip()
    for fmt in CAPTURE_TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def image_dimensions(path: "str | Path | BandGroupRef", num_channels: int) -> tuple[int, int]:
    """``(width, height)`` as ``load_image`` will decode it at ``num_channels``, without decoding
    pixels where possible. :func:`display_frame` is the frame a viewer and annotations use.

    A :class:`BandGroupRef` reads its dims from one sibling band file (a group's members share one
    spatial frame).
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
        frame = raster_source.tiff_frame(path, num_channels)
        return int(frame[1]), int(frame[0])
    return frame_size(load_multiband(path, num_channels))


def frame_size(img) -> tuple[int, int]:
    """``(width, height)`` of a decoded image: a PIL image, or an ``[H, W, ...]`` array."""
    if isinstance(img, Image.Image):
        return img.size
    return int(img.shape[1]), int(img.shape[0])


def pixel_array(img) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """A decoded image (:func:`load_image`'s return) as the ``[H, W, C]`` array slices are cut
    from, with the band interpretations that let :func:`to_pil_if_faithful` rebuild a slice's PIL
    mode: a PIL ``RGBA`` image names its 4th band alpha, anything else names none."""
    interpretations = (("red", "green", "blue", "alpha")
                       if isinstance(img, Image.Image) and img.mode == "RGBA" else None)
    return raster_source.hwc_array(img), interpretations


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

    arr = raster_source.hwc_array(img)
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
    an ``[H, W, C]`` ndarray. A photographic RGB file requested as 1 channel is converted to
    grayscale; as 3, kept RGB. Reads through ``raster_source``.
    """
    with raster_source.open_raster(path, num_channels) as src:
        if isinstance(src, raster_source.PhotographicSource):
            return src.image
        pixels = src.read_region(Rect(0, 0, src.width, src.height))[0]
        return to_pil_if_faithful(
            pixels, band_interpretations=getattr(src, "band_interpretations", None)
        )


def load_multiband(path: "str | Path | BandGroupRef", num_channels: int) -> np.ndarray:
    """Load a multi-band image as ``[H, W, C]`` through ``raster_source``'s array backends.

    A :class:`BandGroupRef` decodes each sibling file (each already a supported single-band
    source) and stacks them into one ``[H, W, C]`` array in the manifest's declared band order.

    A photographic container (anything outside ``.npy`` / ``.npz`` / ``.tif`` / ``.tiff``) raises
    ``ValueError`` at any channel count.
    """
    with raster_source.open_array_source(path, num_channels) as src:
        return src.read_region(Rect(0, 0, src.width, src.height))[0]
