"""uint8 pixels reach the PIL augmentation chain whatever container they decoded from.

to_pil_if_faithful is the single conversion; load_image applies it on the array-backend return
and the tiled dataset applies it on windowed tiles, so an augmentation configured for a run
covers a uint8 GeoTIFF the same as a JPEG, while dtypes PIL has no faithful mode for stay
ndarray, and a run stating an augmentation over them refuses when its loader is built. A
4-channel array only converts to RGBA when the source's own band_interpretations names the 4th
band alpha; with no such signal, or a signal naming something else (a genuine spectral band), it
stays an ndarray rather than guess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor
from tests._producer_fixtures import dataset_over  # noqa: E402
from tests._training_values import augmentation, drawn_split, tiling_block

TILING = tiling_block(tile_size=64, overlap=0.2, sliver_frac=0.5)
"""The tiling every windowed case states; its sliver cutoff stated, since one box derives none."""


def _grid(
    height: int, width: int, channels: int, dtype=np.uint8
) -> np.ndarray:
    arr = np.zeros((height, width, channels), dtype=dtype)
    for c in range(channels):
        arr[:, :, c] = (
            np.add.outer(np.arange(height) * (c + 2), np.arange(width))
        ) % 199
    return arr


# ── The helper itself ────────────────────────────────────────────────────


def test_uint8_channel_counts_convert_to_their_pil_mode():
    from tcip_mcp.pipelines.image_utils import to_pil_if_faithful

    for channels, mode in ((1, "L"), (3, "RGB")):
        arr = _grid(6, 5, channels)
        img = to_pil_if_faithful(arr)
        assert isinstance(img, Image.Image) and img.mode == mode
        back = np.asarray(img)
        assert np.array_equal(back, arr[:, :, 0] if channels == 1 else arr)


def test_unfaithful_inputs_return_unchanged():
    from tcip_mcp.pipelines.image_utils import to_pil_if_faithful

    for arr in (
        _grid(6, 5, 3, dtype=np.uint16),
        _grid(6, 5, 2),
        _grid(6, 5, 5),
        _grid(6, 5, 3).astype(np.float32),
        np.zeros((6, 5), dtype=np.uint8),
    ):
        assert to_pil_if_faithful(arr) is arr
    pil = Image.new("RGB", (5, 6))
    assert to_pil_if_faithful(pil) is pil


# ── The 4-channel alpha-vs-spectral-band ambiguity ───────────────────────


def test_four_channel_converts_to_rgba_only_when_the_source_declares_alpha():
    """RGBA is a faithful *pixel* round-trip for any 4-channel uint8 array, but PIL's
    augmentation chain treats an alpha channel differently than a color one: correct only when
    the 4th channel really is transparency. band_interpretations (the same GDAL-color-
    interpretation fact raster_source's other consumers already read) is the only signal that
    settles it; with no signal, or a signal naming something other than alpha, the array must
    stay ndarray rather than guess.
    """
    from tcip_mcp.pipelines.image_utils import to_pil_if_faithful

    arr = _grid(6, 5, 4)

    img = to_pil_if_faithful(
        arr, band_interpretations=("red", "green", "blue", "alpha")
    )
    assert isinstance(img, Image.Image) and img.mode == "RGBA"
    assert np.array_equal(np.asarray(img), arr)

    assert (
        to_pil_if_faithful(arr) is arr
    )  # no signal at all (.npy/.npz, a band group)
    assert to_pil_if_faithful(arr, band_interpretations=None) is arr
    # a real signal that names the 4th band something other than alpha (a genuinely spectral one,
    # or a file whose color interpretation is simply undeclared)
    assert (
        to_pil_if_faithful(
            arr, band_interpretations=("red", "green", "blue", "undefined")
        )
        is arr
    )
    assert (
        to_pil_if_faithful(
            arr,
            band_interpretations=(
                "gray",
                "undefined",
                "undefined",
                "undefined",
            ),
        )
        is arr
    )


# ── load_image's array-backend return ────────────────────────────────────


def test_load_image_returns_pil_for_uint8_tiffs(tmp_path: Path):
    import tifffile

    rgb = _grid(20, 16, 3)
    tifffile.imwrite(
        str(tmp_path / "rgb.tif"), rgb, photometric="rgb", rowsperstrip=8
    )
    got = load_image(tmp_path / "rgb.tif", 3)
    assert isinstance(got, Image.Image) and got.mode == "RGB"
    assert np.array_equal(np.asarray(got), rgb)

    gray = _grid(20, 16, 1)[:, :, 0]
    tifffile.imwrite(str(tmp_path / "gray.tif"), gray, rowsperstrip=8)
    got = load_image(tmp_path / "gray.tif", 1)
    assert isinstance(got, Image.Image) and got.mode == "L"
    assert np.array_equal(np.asarray(got), gray)


def test_load_image_converts_a_real_alpha_tagged_geotiff_to_rgba(
    tmp_path: Path,
):
    """A GDAL-served 4-band GeoTIFF whose 4th band the file itself declares alpha (TIFF
    ExtraSamples=2, unassociated alpha) is the real signal load_image now reads through
    raster_source.GdalSource.band_interpretations, not a guess from the channel count.
    """
    import tifffile

    rgba = _grid(20, 16, 4)
    tifffile.imwrite(
        str(tmp_path / "rgba.tif"),
        rgba,
        photometric="rgb",
        rowsperstrip=8,
        extrasamples=["unassalpha"],
    )
    got = load_image(tmp_path / "rgba.tif", 4)
    assert isinstance(got, Image.Image) and got.mode == "RGBA"
    assert np.array_equal(np.asarray(got), rgba)


def test_load_image_keeps_a_declared_spectral_fourth_band_as_ndarray(
    tmp_path: Path,
):
    """A 4-band GeoTIFF whose file tags the 4th band ExtraSamples=0 (unspecified, GDAL's
    'undefined' color interpretation) is a genuine 4th channel, never alpha; load_image must not
    guess it into RGBA, where PIL's augmentation chain would silently leave it untouched.
    """
    import tifffile

    ms = _grid(20, 16, 4)
    tifffile.imwrite(
        str(tmp_path / "ms.tif"),
        ms,
        photometric="rgb",
        rowsperstrip=8,
        extrasamples=["unspecified"],
    )
    got = load_image(tmp_path / "ms.tif", 4)
    assert isinstance(got, np.ndarray)
    assert np.array_equal(got, ms)


def test_load_image_keeps_an_undeclared_fourth_band_npy_as_ndarray(
    tmp_path: Path,
):
    """A .npy stack carries no color-interpretation metadata at all: no signal ever exists to
    call its 4th channel alpha, so load_image must never guess it into RGBA.
    """
    stack = _grid(20, 16, 4)
    np.save(str(tmp_path / "stack.npy"), stack)
    got = load_image(tmp_path / "stack.npy", 4)
    assert isinstance(got, np.ndarray)
    assert np.array_equal(got, stack)


def test_load_image_keeps_ndarray_where_pil_has_no_faithful_mode(
    tmp_path: Path,
):
    import tifffile

    u16 = _grid(20, 16, 3, dtype=np.uint16)
    tifffile.imwrite(
        str(tmp_path / "deep.tif"), u16, photometric="rgb", rowsperstrip=8
    )
    got = load_image(tmp_path / "deep.tif", 3)
    assert isinstance(got, np.ndarray) and got.dtype == np.uint16

    five = _grid(20, 16, 5)
    tifffile.imwrite(str(tmp_path / "five.tif"), five)
    got = load_image(tmp_path / "five.tif", 5)
    assert isinstance(got, np.ndarray) and got.shape == (20, 16, 5)


@pytest.mark.parametrize(
    "name, channels, dtype, tags",
    [
        ("rgb.tif", 3, np.uint8, {"photometric": "rgb"}),
        ("gray.tif", 1, np.uint8, {}),
        (
            "alpha.tif",
            4,
            np.uint8,
            {"photometric": "rgb", "extrasamples": ["unassalpha"]},
        ),
        (
            "spectral.tif",
            4,
            np.uint8,
            {"photometric": "rgb", "extrasamples": ["unspecified"]},
        ),
        ("deep.tif", 3, np.uint16, {"photometric": "rgb"}),
        ("five.tif", 5, np.uint8, {}),
        ("stack.npy", 4, np.uint8, None),
        ("photo.png", 3, np.uint8, None),
    ],
)
def test_the_header_and_the_decode_agree_on_whether_a_source_is_served_as_pil(
    tmp_path: Path, name, channels, dtype, tags
):
    """The mode a source's header says it is served in (what a loader built under the chain
    admits by) is the mode ``load_image`` decodes it to, ``None`` for an array.
    """
    import tifffile

    from tcip_mcp.pipelines.raster_source import SourceHeader

    path = tmp_path / name
    arr = _grid(20, 16, channels, dtype=dtype)
    if path.suffix == ".npy":
        np.save(path, arr)
    elif path.suffix == ".png":
        Image.fromarray(arr, mode="RGB").save(path)
    else:
        tifffile.imwrite(
            str(path), arr[:, :, 0] if channels == 1 else arr, **tags
        )

    decoded = load_image(path, channels)
    assert SourceHeader(path).pil_mode_at(channels)[2] == getattr(
        decoded, "mode", None
    )


# ── The augmentation chain applies end to end ────────────────────────────


def _flip_transform():
    def transform(img, target):
        return pil_to_tensor(img.transpose(Image.FLIP_LEFT_RIGHT)), target

    return transform


def _detection_project(
    tmp_path: Path, arr: np.ndarray, *, extrasamples: list[str] | None = None
):
    """One TIFF holding ``arr`` and its label document; its images directory."""
    import tifffile

    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    kwargs = {"photometric": "rgb", "rowsperstrip": 8}
    if extrasamples:
        kwargs["extrasamples"] = extrasamples
    tifffile.imwrite(str(images_dir / "img0.tif"), arr, **kwargs)
    h, w = arr.shape[:2]
    label_image(
        images_dir / "img0.tif",
        [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))],
        w,
        h,
        keep_empty=True,
    )
    return images_dir


def test_a_uint8_tiff_trains_augmented_through_detection(tmp_path: Path):
    pytest.importorskip("torch")
    import torch

    arr = _grid(40, 32, 3)
    images_dir = _detection_project(tmp_path, arr)
    ds = dataset_over(
        "detection",
        str(images_dir),
        subject="bud",
        transforms=_flip_transform(),
    )
    got, _target = ds[0]
    flipped = torch.from_numpy(
        arr[:, ::-1].astype(np.float32) / 255.0
    ).permute(2, 0, 1)
    assert torch.equal(got, flipped)


def test_a_uint8_windowed_tile_trains_augmented_through_tiling(tmp_path: Path):
    pytest.importorskip("torch")
    import torch

    from tcip_mcp.pipelines import raster_source

    raster_source.close_source_pool()
    try:
        arr = _grid(96, 96, 3)
        images_dir = _detection_project(tmp_path, arr)
        ds = dataset_over(
            "detection",
            str(images_dir),
            subject="bud",
            transforms=_flip_transform(),
            tiling=TILING,
        )
        got, _target = ds[0]  # tile at (0, 0)
        tile = arr[0:64, 0:64]
        flipped = torch.from_numpy(
            tile[:, ::-1].astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        assert torch.equal(got, flipped)
        assert ds.source_frames[ds.stems[0]]["windowed"] is True
    finally:
        raster_source.close_source_pool()


def test_a_declared_alpha_windowed_tile_trains_augmented(tmp_path: Path):
    """The windowed tile path reads the same real band_interpretations signal load_image does:
    a 4-band GeoTIFF whose file declares its 4th band alpha still gets the PIL augmentation
    chain, byte-identical to the whole-decode case above.
    """
    pytest.importorskip("torch")
    import torch

    from tcip_mcp.pipelines import raster_source

    raster_source.close_source_pool()
    try:
        arr = _grid(96, 96, 4)
        images_dir = _detection_project(
            tmp_path, arr, extrasamples=["unassalpha"]
        )
        ds = dataset_over(
            "detection",
            str(images_dir),
            subject="bud",
            stated={"num_channels": 4},
            transforms=_flip_transform(),
            tiling=TILING,
        )
        got, _target = ds[0]  # tile at (0, 0)
        tile = arr[0:64, 0:64]
        flipped = torch.from_numpy(
            tile[:, ::-1].astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        assert torch.equal(got, flipped)
    finally:
        raster_source.close_source_pool()


def test_a_declared_spectral_fourth_band_windowed_tile_refuses_the_chain(
    tmp_path: Path,
):
    """A 4-band GeoTIFF whose file declares its 4th band something other than alpha (a genuine
    spectral band) is never guessed into RGBA, so a tiled loader built under the chain refuses
    by name rather than serve the tile unaugmented.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines import raster_source

    raster_source.close_source_pool()
    try:
        images_dir = _detection_project(
            tmp_path, _grid(96, 96, 4), extrasamples=["unspecified"]
        )
        with pytest.raises(ValueError, match="4-band uint8 pixels"):
            dataset_over(
                "detection",
                str(images_dir),
                subject="bud",
                stated={"num_channels": 4},
                transforms=_flip_transform(),
                tiling=TILING,
            )
    finally:
        raster_source.close_source_pool()


def _augmented_run(
    tmp_path: Path, arr: np.ndarray, *, bespoke: bool = False
) -> dict:
    """A one-epoch run over one frame holding ``arr`` (:func:`_detection_project`), no validation
    side, stating an augmentation that flips every frame: a detection run whose loaders the
    platform builds, or with ``bespoke`` a run of a task no built-in loader reads whose loaders a
    ``dataset_source`` builder builds.
    """
    from tests._chain_fixtures import training_config
    from tests._training_values import evaluation_block
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR
    from tests.test_dataset_source_seam import DATASET_SOURCE

    images_dir = _detection_project(tmp_path / "data", arr)
    return training_config(
        {**BUILT_DETECTOR, "task": "grape_bunch_count"}
        if bespoke
        else BUILT_DETECTOR,
        {
            "images_dir": str(images_dir),
            "num_channels": 3,
            "scope": {"subject": "bud"},
            "split": drawn_split(val_ratio=0.0),
            **({"dataset_source": DATASET_SOURCE} if bespoke else {}),
        },
        augmentation=augmentation(horizontal_flip={"p": 1.0}),
        evaluation=evaluation_block(selection_metric="loss"),
    )


@pytest.mark.parametrize("bespoke", [False, True], ids=["platform", "bespoke"])
def test_a_run_stating_augmentation_over_uint16_frames_refuses_at_its_resolution(
    tmp_path: Path, bespoke: bool
):
    """Whoever builds the run's loaders, the platform's factory or its ``dataset_source``
    builder, the run refuses at the one door where its samples meet its augmentation.
    """
    pytest.importorskip("torch")
    from tests._verified_checkpoint_fixtures import opened_run

    config = _augmented_run(
        tmp_path, _grid(40, 32, 3, dtype=np.uint16), bespoke=bespoke
    )
    with pytest.raises(
        ValueError, match=r"(?s)RandomHorizontalFlip.*3-band uint16 pixels"
    ):
        opened_run(tmp_path, config)


def test_a_bodys_tiler_refuses_the_chain_over_uint16_frames(tmp_path: Path):
    """``ctx.tiled_dataset`` wraps a dataset a body already holds, so it admits the transforms
    it is handed over that dataset's samples the way a run's loaders do, and tiles a uint8 one.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines import raster_source
    from tests.tiny_trainer_fixtures import run_context

    raster_source.close_source_pool()
    try:
        ctx = run_context(tmp_path, _augmented_run(tmp_path, _grid(40, 32, 3)))
        lattice = {
            "tile_size": 16,
            "overlap": 0.0,
            "sliver_frac": 0.5,
            "skip_empty": False,
        }
        deep = dataset_over(
            "detection",
            str(
                _detection_project(
                    tmp_path / "deep", _grid(40, 32, 3, dtype=np.uint16)
                )
            ),
            subject="bud",
        )
        with pytest.raises(ValueError, match="3-band uint16 pixels"):
            ctx.tiled_dataset(deep, transforms=_flip_transform(), **lattice)
        tiled = ctx.tiled_dataset(
            ctx.train_loader.dataset, transforms=_flip_transform(), **lattice
        )
        assert len(tiled) > 0
    finally:
        raster_source.close_source_pool()


def test_a_dataset_built_around_the_door_refuses_the_chain_at_its_first_read(
    tmp_path: Path,
):
    """A loader constructed directly with transforms over a uint16 frame reaches no door, so its
    first read refuses naming the frame rather than serve it unaugmented.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.datasets import DetectionDataset
    from tests._producer_fixtures import admit_over

    admitted = admit_over(
        _detection_project(tmp_path, _grid(40, 32, 3, dtype=np.uint16)),
        subject="bud",
    )
    ds = DetectionDataset(
        admitted.every_sample(),
        transforms=_flip_transform(),
        scope=admitted.scope,
    )
    ds.expected_channels = 3
    with pytest.raises(
        ValueError, match=r"(?s)over \S*img0.*3-band uint16 pixels"
    ):
        ds[0]


def test_a_bespoke_run_over_uint8_frames_resolves_with_its_augmentation(
    tmp_path: Path,
):
    pytest.importorskip("torch")
    from tests._verified_checkpoint_fixtures import opened_run

    run_dir = opened_run(
        tmp_path, _augmented_run(tmp_path, _grid(40, 32, 3), bespoke=True)
    )
    assert run_dir.is_dir()


def test_the_same_run_over_uint8_frames_trains_augmented(tmp_path: Path):
    pytest.importorskip("torch")
    import torch

    from tests.tiny_trainer_fixtures import run_context

    arr = _grid(40, 32, 3)
    ctx = run_context(tmp_path, _augmented_run(tmp_path, arr))
    got, _target = ctx.train_loader.dataset[0]
    flipped = torch.from_numpy(
        arr[:, ::-1].astype(np.float32) / 255.0
    ).permute(2, 0, 1)
    assert torch.equal(got, flipped)
