"""Shared image utilities for the composable ML pipeline (channel-aware)."""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.data.band_groups import (
    MANIFEST_EXT,
    BandGroupIncompleteError,
    BandGroupRef,
    read_band_group_manifest,
)
from tcip_mcp.pipelines.raster_source import Rect

if TYPE_CHECKING:
    from datetime import datetime

    import torch

__all__ = [
    "AmbiguousImageStemError",
    "BandGroupIncompleteError",
    "BandGroupRef",
    "IMAGE_EXTS",
    "bucket_logical_identities",
    "image_dimensions",
    "list_logical_images",
    "load_image",
    "logical_image_name",
    "pil_to_tensor",
    "pixel_array",
    "refuse_incomplete_band_group",
    "resolve_image_path",
    "resolve_image_paths",
    "scan_identities",
    "source_path_of",
    "stem_collision_key",
    "stem_of",
    "to_pil_if_faithful",
]


class AmbiguousImageStemError(ValueError):
    """A directory holds more than one logical identity under one case-folded stem.

    The fold is :func:`stem_collision_key`'s. The identities are two raw files (``foo.jpg``,
    ``foo.png``, or a same-key case variant such as ``Foo.jpg``), or a raw file and a
    ``.bandgroup`` manifest recorded under a different exact stem than its own.
    """


# ``.npy``/``.npz`` are a multi-band raster; ``.bandgroup`` a manifest standing in for the image it
# names.
IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".heic",
    ".tif",
    ".tiff",
    ".npy",
    ".npz",
    MANIFEST_EXT,
}


def stem_collision_key(name: str) -> str:
    """Fold ``name`` into the key that decides whether two stems name one logical identity.

    The fold is ``str.lower()``, so a pair differing only by Unicode normalization, or by a
    casefold-but-not-``lower()`` difference, is two keys.
    """
    return name.lower()


def scan_identities(d: Path) -> dict[str, list[Path | BandGroupRef]]:
    """Scan ``d`` for its logical identities, keyed by :func:`stem_collision_key`.

    The scan is one manifest glob and one ``d.iterdir()`` walk with one extension test; a key's
    list holds more than one entry exactly when it is ambiguous.

    A readable manifest is one identity under its own exact stem, its parsed
    :class:`BandGroupRef`; its band files are that identity's members and no identity of their
    own. An unreadable or corrupt manifest claims nothing and is no identity. Every other file
    with a suffix in :data:`IMAGE_EXTS` that no readable manifest claims is one raw identity under
    its own exact stem, its own path.
    """
    identities: dict[str, list[Path | BandGroupRef]] = {}
    claimed: set[Path] = set()
    for mp in sorted(d.glob(f"*{MANIFEST_EXT}")):
        try:
            ref = read_band_group_manifest(mp)
        except (OSError, ValueError):
            continue
        identities.setdefault(stem_collision_key(ref.stem), []).append(ref)
        claimed.update(ref.bands.values())
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == MANIFEST_EXT or p in claimed:
            continue
        if ext not in IMAGE_EXTS:
            continue
        identities.setdefault(stem_collision_key(p.stem), []).append(p)
    return identities


def bucket_logical_identities(images_dir: str | Path) -> dict[str, list[Path]]:
    """Return the bucket's logical identities, grouped by :func:`stem_collision_key`.

    A key held by more than one identity is ambiguous. Each list entry is one identity's own
    defining path (a ``.bandgroup`` manifest's path, or a raw file's own path); an absent
    directory answers empty. Never raises on an ambiguous key.
    """
    d = Path(images_dir)
    if not d.is_dir():
        return {}
    return {
        key: [Path(source_path_of(entry)) for entry in entries]
        for key, entries in scan_identities(d).items()
    }


def scan_captures[T](
    root: str | Path,
    scan: Callable[[Path], T],
    captures: Container[str] | None = None,
) -> dict[str, T]:
    """Return ``scan`` of each capture directory under the dataset root ``root``, by capture.

    The captures are :func:`~tcip_mcp.dataset_layout.list_dates`', in its order, or those of them
    ``captures`` holds when it is given; a capture it does not hold is never scanned. ``scan``'s
    refusals propagate.
    """
    from tcip_mcp import dataset_layout

    return {
        capture: scan(dataset_layout.image_dir(root, capture))
        for capture in dataset_layout.list_dates(root)
        if captures is None or capture in captures
    }


def logical_image_counts(root: str | Path) -> dict[str, int]:
    """Count the logical images each capture under the dataset root ``root`` holds, by capture.

    The captures are :func:`scan_captures`'; each is counted by :func:`bucket_logical_identities`,
    each identity of an ambiguous stem counted.
    """
    return {
        capture: sum(len(ids) for ids in found.values())
        for capture, found in scan_captures(
            root, bucket_logical_identities
        ).items()
    }


def list_logical_images(
    images_dir: str | Path,
) -> dict[str, Path | BandGroupRef]:
    """Return every logical image in ``images_dir``, by exact stem.

    Built from :func:`scan_identities`: a :class:`BandGroupRef` for a manifest identity, that
    file's own path for a raw one. Raises :class:`AmbiguousImageStemError` naming every ambiguous
    key's own paths; the returned mapping is keyed by the exact stem.
    """
    d = Path(images_dir)
    if not d.is_dir():
        return {}
    scanned = scan_identities(d)
    _refuse_ambiguous(d, scanned.values())
    return {stem_of(first): first for first, *_ in scanned.values()}


def _refuse_ambiguous(
    d: Path, keys: Iterable[list[Path | BandGroupRef]]
) -> None:
    """Refuse the :func:`scan_identities` keys of ``d`` holding more than one identity.

    The :class:`AmbiguousImageStemError` names their paths.
    """
    names = sorted(
        source_path_of(entry)
        for entries in keys
        if len(entries) > 1
        for entry in entries
    )
    if names:
        raise AmbiguousImageStemError(
            f"{d}: {names} name more than one logical image under one case-folded stem, "
            "refusing to silently keep one. Rename so each logical image has its own stem."
        )


def refuse_incomplete_band_group(
    source: Path | BandGroupRef,
) -> Path | BandGroupRef:
    """Return ``source``, refusing a grouped capture whose manifest names a band that is gone."""
    if isinstance(source, BandGroupRef):
        missing = [name for name, p in source.bands.items() if not p.is_file()]
        if missing:
            raise BandGroupIncompleteError(
                f"band group {source.stem!r} ({source.manifest_path}) references missing band(s) "
                f"{sorted(missing)}: delete the manifest to let a later detection pass re-group "
                "the surviving siblings, or restore the missing file(s)."
            )
    return source


def resolve_image_paths(
    paths: Iterable[str | Path],
) -> list[Path | BandGroupRef]:
    """Resolve the logical image each of ``paths`` names, each directory scanned once.

    The scan is :func:`scan_identities`: a ``.bandgroup`` manifest's grouped capture, a raw image
    its own file. Refuses ``FileNotFoundError`` for a path naming no logical image, naming the
    manifest that claims it when it is a band of a grouped capture;
    :class:`AmbiguousImageStemError` when its own stem names more than one;
    ``BandGroupIncompleteError`` for a grouped capture missing a band.
    """
    named_paths = [Path(p) for p in paths]
    scanned = {
        d: scan_identities(d) if d.is_dir() else {}
        for d in {p.parent for p in named_paths}
    }

    def named(path: Path) -> Path | BandGroupRef:
        entries = scanned[path.parent].get(stem_collision_key(path.stem), [])
        _refuse_ambiguous(path.parent, [entries])
        if not entries or logical_image_name(entries[0]) != path.name:
            owners = [
                source_path_of(first)
                for first, *_ in scanned[path.parent].values()
                if isinstance(first, BandGroupRef)
                and path in first.bands.values()
            ]
            raise FileNotFoundError(
                f"{path} names no logical image"
                + (
                    f": it is a band of the grouped capture {owners[0]}; name the manifest"
                    if owners
                    else ""
                )
            )
        return refuse_incomplete_band_group(entries[0])

    return [named(p) for p in named_paths]


def resolve_image_path(image_path: str | Path) -> Path | BandGroupRef:
    """Resolve the logical image ``image_path`` names (:func:`resolve_image_paths`).

    Its refusals propagate.
    """
    return resolve_image_paths([image_path])[0]


def image_path_dimensions(image_path: str | Path) -> tuple[int, int]:
    """Return the display frame of the logical image ``image_path`` names.

    The frame is :attr:`~tcip_mcp.pipelines.raster_source.SourceHeader.display_frame`, the image
    resolved by :func:`resolve_image_path`, a grouped capture folded into its one frame. Its
    refusals propagate.
    """
    return raster_source.SourceHeader(
        resolve_image_path(image_path)
    ).display_frame


def source_path_of(source: str | Path | BandGroupRef) -> str:
    """Return the path this logical image is named by, recorded and displayed.

    That is a grouped capture's own ``.bandgroup`` manifest, a plain image's own file.
    """
    return str(
        source.manifest_path if isinstance(source, BandGroupRef) else source
    )


def logical_image_name(source: str | Path | BandGroupRef) -> str:
    """Return this logical image's file name: the basename of :func:`source_path_of`."""
    return Path(source_path_of(source)).name


def logical_images_by_name(
    images_dir: str | Path,
) -> dict[str, Path | BandGroupRef]:
    """Map every logical image in ``images_dir`` from its name to itself, in name order.

    The images are :func:`list_logical_images`'; the name is :func:`logical_image_name`. Raises
    what :func:`list_logical_images` raises.
    """
    named = {
        logical_image_name(src): src
        for src in list_logical_images(images_dir).values()
    }
    return {name: named[name] for name in sorted(named)}


def stem_of(source: str | Path | BandGroupRef) -> str:
    """Return this logical image's stem: the stem of :func:`source_path_of`."""
    return Path(source_path_of(source)).stem


CAPTURE_TIME_FORMATS = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)
"""The spellings a capture time arrives in: the colon form EXIF's DateTimeOriginal and TIFF's
DateTime tag are specified to use, and the ISO forms a stitching engine writes its own in."""

_EXIF_IFD_TAG = 0x8769
_DT_ORIGINAL_TAG = 0x9003


def exif_capture_time(exif: Image.Exif) -> object | None:
    """Return the raw DateTimeOriginal an image's ``exif`` (``Image.getexif()``) carries.

    It is read from the Exif sub-IFD, else from the top level; ``None`` when it carries none.
    """
    raw = exif.get_ifd(_EXIF_IFD_TAG).get(_DT_ORIGINAL_TAG)
    return exif.get(_DT_ORIGINAL_TAG) if raw is None else raw


def parse_capture_time(raw: object) -> datetime | None:
    """Parse a capture-time value in any of :data:`CAPTURE_TIME_FORMATS` as a ``datetime``.

    ``None`` when none fits.
    """
    from datetime import datetime

    text = str(raw).strip()
    for fmt in CAPTURE_TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def image_dimensions(
    path: str | Path | BandGroupRef, num_channels: int
) -> tuple[int, int]:
    """Return ``(width, height)`` as ``load_image`` will decode it at ``num_channels``."""
    return raster_source.SourceHeader(path).frame_at(num_channels)


def frame_size(img: Image.Image | np.ndarray) -> tuple[int, int]:
    """Return ``(width, height)`` of a decoded image: a PIL image, or an ``[H, W, ...]`` array."""
    if isinstance(img, Image.Image):
        return img.size
    return int(img.shape[1]), int(img.shape[0])


def pixel_array(
    img: Image.Image | np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """Return a decoded image as an ``[H, W, C]`` array.

    The image is :func:`load_image`'s return. The array comes with its band interpretations: a PIL
    ``RGBA`` image names its 4th band alpha, anything else names none.
    """
    interpretations = (
        ("red", "green", "blue", "alpha")
        if isinstance(img, Image.Image) and img.mode == "RGBA"
        else None
    )
    return raster_source.hwc_array(img), interpretations


def to_pil_if_faithful(
    arr: np.ndarray, *, band_interpretations: tuple[str, ...] | None = None
) -> Image.Image | np.ndarray:
    """Return an ``[H, W, C]`` ndarray as a PIL image when a faithful mode exists.

    The mode is the one :func:`~tcip_mcp.pipelines.raster_source.faithful_pil_mode` gives its
    dtype, channel count and ``band_interpretations`` (a single channel squeezed); anything else
    is returned unchanged.
    """
    if arr.ndim != 3:
        return arr
    mode = raster_source.faithful_pil_mode(
        arr.dtype, arr.shape[2], band_interpretations
    )
    if mode is None:
        return arr
    return Image.fromarray(arr[:, :, 0] if mode == "L" else arr, mode=mode)


def pil_to_tensor(img: Image.Image | np.ndarray) -> torch.Tensor:
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


def load_image(
    path: str | Path | BandGroupRef, num_channels: int
) -> Image.Image | np.ndarray:
    """Open an image at ``num_channels``.

    Returns a photograph as the ``PIL.Image`` of its count's mode, and an array container's
    pixels (``.npy`` / ``.npz`` / GeoTIFF / a :class:`BandGroupRef`) as a ``PIL.Image`` where
    :func:`~tcip_mcp.pipelines.raster_source.faithful_pil_mode` gives them a mode, else as an
    ``[H, W, C]`` ndarray. Reads through
    ``raster_source``.
    """
    with raster_source.open_raster(path, num_channels) as src:
        if isinstance(src, raster_source.PhotographicSource):
            return src.image
        pixels = src.read_region(Rect(0, 0, src.width, src.height))[0]
        return to_pil_if_faithful(
            pixels,
            band_interpretations=getattr(src, "band_interpretations", None),
        )
