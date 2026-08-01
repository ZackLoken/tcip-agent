"""Shared image utilities for the composable ML pipeline (channel-aware)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tcip_mcp.pipelines.data.band_groups import (
    BandGroupIncomplete,
    BandGroupRef,
    MANIFEST_EXT,
    read_band_group_manifest,
)

__all__ = [
    "AmbiguousImageStem", "BandGroupIncomplete", "BandGroupRef", "IMAGE_EXTS",
    "crop_pad_tile", "image_dimensions", "list_logical_images", "load_image",
    "load_multiband", "pil_to_tensor", "resolve_image_source", "stem_of",
]


class AmbiguousImageStem(ValueError):
    """A directory listing found a raw standalone file whose own stem collides with a
    ``.bandgroup`` manifest's canonical stem — two unrelated identities claiming the same name.

    Raised from :func:`list_logical_images` rather than silently keeping only the manifest's entry
    (which would make the standalone file vanish from every listing — training, splits, review,
    gallery — with no error): CLAUDE.md's "no silent fallback on ambiguous identity" rule.
    """

# The recognized "this is an image" extensions shared by every enumeration/resolution call site
# this consolidates onto list_logical_images/resolve_image_source. ``.npy``/``.npz`` are a
# multi-band raster (nothing enumerated them before); ``.bandgroup`` is a manifest file that
# stands in for the logical image it names.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".tif", ".tiff", ".npy", ".npz", MANIFEST_EXT}


def list_logical_images(images_dir: str | Path) -> dict[str, "Path | BandGroupRef"]:
    """Every logical image in ``images_dir``, by canonical stem.

    Scans for ``.bandgroup`` manifests first — each becomes a :class:`BandGroupRef`, and its
    referenced sibling filenames are excluded from the raw-file listing below, so a stem with an
    existing manifest never also appears as its own single-band entry. An unreadable/corrupt
    manifest is skipped (claims nothing), not raised — a directory listing must not fail whole
    because one manifest is bad; ``resolve_image_source`` is where a caller pays for its own stem.

    Raises :class:`AmbiguousImageStem` when a raw standalone file's stem collides with a
    manifest's own canonical stem (and that file isn't one of the manifest's own sibling bands) —
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
            "and an unrelated standalone file — refusing to silently keep only one. Rename the "
            "standalone file(s), or the manifest, so each logical image has one unambiguous stem."
        )
    return result


def resolve_image_source(images_dir: str | Path, stem: str) -> "Path | BandGroupRef":
    """``list_logical_images(images_dir)[stem]`` — never a second implementation of "what images
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
    shape, by "the smaller of the two non-spatial-looking axes is the channel axis" — used only
    where no expected channel count is already known to compare against (``derivations.probe_channels``'s
    TIFF branch); a caller that already knows the expected count (``image_dimensions``,
    ``load_multiband``) compares against it directly instead, which is the more precise formula."""
    if len(shape) == 2:
        return 1
    return int(shape[0]) if shape[0] < shape[-1] else int(shape[-1])


def _tiff_series_shape(path: Path) -> tuple[int, ...] | None:
    """Header-only TIFF series shape (no pixel decode) — ``None`` if it can't be read this way.

    ``tif.series[0].shape``, not ``pages[0]``: a channel-last TIFF stores each row-block as its
    own page, so ``pages[0]`` of a 24x40x5 raster is ``(40, 5)``. Shared by ``image_dimensions``
    and ``derivations.probe_channels`` so a full pixel read is never paid just to learn the shape.
    """
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            return tuple(int(x) for x in tif.series[0].shape)
    except Exception:  # noqa: BLE001 — fall through to a full read rather than guess
        return None


def image_dimensions(path: "str | Path | BandGroupRef", num_channels: int = 3) -> tuple[int, int]:
    """``(width, height)`` as ``load_image`` will decode it, without decoding pixels where possible.

    ``tcip_annotation.get_image_dimensions`` reads through PIL, which is right for the photographic
    formats it was written for and wrong for a multi-band raster — PIL reports a 5-band 40x24
    GeoTIFF as 5x40. Labels clipped against one frame and tiles cropped from another displace every
    box silently, so anything that measures a frame it will later decode must route the same way
    ``load_image`` does.

    A :class:`BandGroupRef` reads its dims from one sibling band file (a group's members share one
    spatial frame by construction — that's what makes stacking them into ``[H, W, C]`` valid).
    """
    if isinstance(path, BandGroupRef):
        one_band = next(iter(path.bands.values()))
        return image_dimensions(one_band, 1)
    path = Path(path)
    ext = path.suffix.lower()
    if num_channels in (1, 3, 4) and ext not in (".npy", ".npz", ".tif", ".tiff"):
        from tcip_annotation.utils import get_image_dimensions

        return get_image_dimensions(str(path))  # header-only, EXIF-aware
    if ext in (".tif", ".tiff"):
        shape = _tiff_series_shape(path)
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


def crop_pad_tile(img, x: int, y: int, tile_size: int, w: int, h: int):
    """Crop a ``tile_size`` window at (x, y) and zero-pad short (edge) tiles.

    Channel-generic: PIL for 1/3/4-channel images, numpy ``[H, W, C]`` for multi-band rasters
    (which have no ``.crop``). Shared by the training tiler and the inference tiler so the two ends
    of the reproduce-a-number chain cannot crop differently.
    """
    x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
    if isinstance(img, Image.Image):
        crop = img.crop((x, y, x2, y2))
        if crop.size != (tile_size, tile_size):
            padded = Image.new(img.mode, (tile_size, tile_size))  # 0-fill for the image's mode
            padded.paste(crop, (0, 0))
            crop = padded
        return crop
    crop = img[y:y2, x:x2]
    ph, pw = tile_size - crop.shape[0], tile_size - crop.shape[1]
    if ph or pw:
        pad_width = [(0, ph), (0, pw)] + ([(0, 0)] if crop.ndim == 3 else [])
        crop = np.pad(crop, pad_width, mode="constant")
    return crop


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
    """
    if isinstance(path, BandGroupRef):
        return load_multiband(path, num_channels)
    path = Path(path)
    ext = path.suffix.lower()
    if num_channels in (1, 3, 4) and ext not in (".npy", ".npz", ".tif", ".tiff"):
        mode = {1: "L", 3: "RGB", 4: "RGBA"}[num_channels]
        # EXIF-orient before convert so the returned frame matches get_image_dimensions()
        # (both apply auto_orient_image). Labels are authored in this upright frame; without
        # this the loader would denormalize upright coords against the raw sensor frame and
        # scatter every box (Orientation-6 JPEGs differ 5712×4284 ↔ 4284×5712).
        from tcip_annotation.utils import auto_orient_image

        return auto_orient_image(Image.open(path)).convert(mode)
    # >4 channels, or a numpy/GeoTIFF container -> multi-band array.
    return load_multiband(path, num_channels)


def load_multiband(path: "str | Path | BandGroupRef", num_channels: int) -> np.ndarray:
    """Load a multi-band image as ``[H, W, C]`` (NPY/NPZ natively; GeoTIFF via tifffile).

    A :class:`BandGroupRef` decodes each sibling file (each already a supported single-band
    source) and stacks them into one ``[H, W, C]`` array in the manifest's declared band order —
    the one place virtual (in-memory) stacking happens; never written back to disk.
    """
    if isinstance(path, BandGroupRef):
        bands = [load_multiband(p, 1) for p in path.bands.values()]
        return np.concatenate(bands, axis=-1)
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path))
    elif ext == ".npz":
        npz = np.load(str(path))
        arr = npz[npz.files[0]]
    elif ext in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        raise ValueError(
            f"Cannot load a {num_channels}-channel image from '{ext}'. "
            "Use .npy/.npz or a multi-band GeoTIFF (.tif/.tiff)."
        )
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape[0] == num_channels and arr.shape[2] != num_channels:
        arr = np.transpose(arr, (1, 2, 0))  # channel-first -> channel-last
    return arr
