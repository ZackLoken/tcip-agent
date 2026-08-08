"""Shared image utilities for the composable ML pipeline (channel-aware)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.data.band_groups import (
    BandGroupIncomplete,
    BandGroupRef,
    MANIFEST_EXT,
    read_band_group_manifest,
)
from tcip_mcp.pipelines.raster_source import Rect

__all__ = [
    "AmbiguousImageStem", "BandGroupIncomplete", "BandGroupRef", "IMAGE_EXTS",
    "crop_pad_tile", "image_dimensions", "list_logical_images", "load_image",
    "load_multiband", "pad_tile", "pil_to_tensor", "resolve_image_source", "stem_of",
]


class AmbiguousImageStem(ValueError):
    """A directory listing found a raw standalone file whose own stem collides with a
    ``.bandgroup`` manifest's canonical stem: two unrelated identities claiming the same name.

    Raised from :func:`list_logical_images` rather than silently keeping only the manifest's entry
    (which would make the standalone file vanish from every listing, including training, splits,
    review, and gallery, with no error): CLAUDE.md's "no silent fallback on ambiguous identity" rule.
    """

# The recognized "this is an image" extensions shared by every enumeration/resolution call site
# this consolidates onto list_logical_images/resolve_image_source. ``.npy``/``.npz`` are a
# multi-band raster (nothing enumerated them before); ``.bandgroup`` is a manifest file that
# stands in for the logical image it names.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".tif", ".tiff", ".npy", ".npz", MANIFEST_EXT}


def list_logical_images(images_dir: str | Path) -> dict[str, "Path | BandGroupRef"]:
    """Every logical image in ``images_dir``, by canonical stem.

    Scans for ``.bandgroup`` manifests first: each becomes a :class:`BandGroupRef`, and its
    referenced sibling filenames are excluded from the raw-file listing below, so a stem with an
    existing manifest never also appears as its own single-band entry. An unreadable/corrupt
    manifest is skipped (claims nothing), not raised: a directory listing must not fail whole
    because one manifest is bad; ``resolve_image_source`` is where a caller pays for its own stem.

    Raises :class:`AmbiguousImageStem` when a raw standalone file's stem collides with a
    manifest's own canonical stem (and that file isn't one of the manifest's own sibling bands):
    two unrelated identities can't share one name silently; the caller (agent or human) renames one.
    """
    d = Path(images_dir)
    if not d.is_dir():
        return {}
    result: dict[str, Path | BandGroupRef] = {}
    manifest_stems: set[str] = set()
    claimed: set[str] = set()
    for mp in sorted(d.glob(f"*{MANIFEST_EXT}")):
        try:
            ref = read_band_group_manifest(mp)
        except (OSError, ValueError):
            continue
        result[ref.stem] = ref
        manifest_stems.add(ref.stem)
        claimed.update(p.name for p in ref.bands.values())
    ambiguous: set[str] = set()
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == MANIFEST_EXT or p.name in claimed:
            continue
        if ext not in IMAGE_EXTS:
            continue
        if p.stem in manifest_stems:
            ambiguous.add(p.stem)
            continue
        if p.stem not in result:
            result[p.stem] = p
    if ambiguous:
        raise AmbiguousImageStem(
            f"{d}: stem(s) {sorted(ambiguous)} name both a .bandgroup manifest's canonical stem "
            "and an unrelated standalone file, refusing to silently keep only one. Rename the "
            "standalone file(s), or the manifest, so each logical image has one unambiguous stem."
        )
    return result


def resolve_image_source(images_dir: str | Path, stem: str) -> "Path | BandGroupRef":
    """``list_logical_images(images_dir)[stem]``: never a second implementation of "what images
    live here."

    Raises ``FileNotFoundError`` for an unknown stem. For a ``BandGroupRef`` whose manifest
    references a sibling file that no longer exists, raises ``BandGroupIncomplete`` here (the
    resolver) rather than letting a bare decode error surface later inside a stacking loop.
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
        shape = raster_source.tiff_series_shape(path)
        if shape is not None:
            if len(shape) == 2:
                return int(shape[1]), int(shape[0])
            if len(shape) == 3:
                # Same channel-first heuristic load_multiband applies, so both agree.
                if shape[0] == num_channels and shape[2] != num_channels:
                    return int(shape[2]), int(shape[1])
                return int(shape[1]), int(shape[0])
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


def pil_to_tensor(img) -> torch.Tensor:
    """Convert a PIL Image or H×W[×C] array to a float32 ``[C, H, W]`` tensor in ``[0, 1]``.

    Channel-aware: a 2-D grayscale array becomes ``[1, H, W]``; any channel count is
    supported. Integer inputs are scaled by their dtype max (uint8→/255, uint16→/65535);
    float inputs are assumed already normalized.
    """
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

    Returns a ``PIL.Image`` for 1/3/4-channel raster images (so the PIL augmentation
    pipeline keeps working), or an ``[H, W, C]`` ndarray for multi-band inputs
    (``.npy`` / ``.npz`` / multi-band GeoTIFF, or a :class:`BandGroupRef`). An RGB file requested
    as 1 channel is converted to grayscale; as 3, kept RGB.

    Reads through ``raster_source``: which backend decodes a source is that module's own dispatch,
    so a frame this returns and one ``image_dimensions`` measures can never come from two
    different decisions.
    """
    with raster_source.open_raster(path, num_channels) as src:
        if isinstance(src, raster_source.PhotographicSource):
            return src.image
        return src.read_region(Rect(0, 0, src.width, src.height))[0]


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
