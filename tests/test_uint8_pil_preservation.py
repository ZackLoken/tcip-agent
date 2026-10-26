"""uint8 pixels from a TIFF, whole or windowed, reach the PIL augmentation chain.

``to_pil_if_faithful`` converts a uint8 array to PIL; ``load_image`` applies it on the
array-backend return and the tiled dataset on windowed tiles. Dtypes PIL has no faithful mode
for stay ndarray, and a run stating an augmentation over them refuses when its loader is built.
A 4-channel array converts to RGBA when the source's own band_interpretations names the 4th
band alpha; with no such signal, or a signal naming something else, it stays an ndarray.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor
from tests._producer_fixtures import dataset_over  # noqa: E402
from tests._training_values import augmentation, drawn_split, tiling_block

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

TILING = tiling_block(tile_size=64, overlap=0.2, sliver_frac=0.5)
"""The tiling every windowed case states; its sliver cutoff stated, since one box derives none."""


def _grid(
    height: int,
    width: int,
    channels: int,
    dtype: type[np.uint8] | type[np.uint16] = np.uint8,
) -> np.ndarray:
    arr = np.zeros((height, width, channels), dtype=dtype)
    for c in range(channels):
        arr[:, :, c] = (
            np.add.outer(np.arange(height) * (c + 2), np.arange(width))
        ) % 199
    return arr


# ── The helper itself ────────────────────────────────────────────────────


def test_uint8_channel_counts_convert_to_their_pil_mode() -> None:
    from tcip_mcp.pipelines.image_utils import to_pil_if_faithful

    for channels, mode in ((1, "L"), (3, "RGB")):
        arr = _grid(6, 5, channels)
        img = to_pil_if_faithful(arr)
        assert isinstance(img, Image.Image) and img.mode == mode
        back = np.asarray(img)
        assert np.array_equal(back, arr[:, :, 0] if channels == 1 else arr)


def test_unfaithful_inputs_return_unchanged() -> None:
    from tcip_mcp.pipelines.image_utils import to_pil_if_faithful

    for arr in (
        _grid(6, 5, 3, dtype=np.uint16),
        _grid(6, 5, 2),
        _grid(6, 5, 5),
        _grid(6, 5, 3).astype(np.float32),
        np.zeros((6, 5), dtype=np.uint8),
    ):
        assert to_pil_if_faithful(arr) is arr


# ── The 4-channel alpha-vs-spectral-band ambiguity ───────────────────────


def test_four_channel_converts_to_rgba_only_when_the_source_declares_alpha() -> (
    None
):
    """A 4-channel array converts to RGBA only when its source declares alpha.

    With band_interpretations naming the 4th band alpha, the array converts to an RGBA image
    with its pixels unchanged; with no signal, or a signal naming something other than alpha,
    the array comes back as the same ndarray object.
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


def test_load_image_returns_pil_for_uint8_tiffs(tmp_path: Path) -> None:
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
) -> None:
    """``load_image`` converts a TIFF whose file declares its 4th band alpha to RGBA.

    The file tags its 4th band ExtraSamples=2, unassociated alpha, and ``load_image`` returns an
    RGBA image of its pixels.
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
) -> None:
    """``load_image`` keeps a declared spectral fourth band as an ndarray.

    A 4-band TIFF whose file tags the 4th band ExtraSamples=0 (unspecified, GDAL's
    'undefined' color interpretation) loads as an ndarray of its pixels, not RGBA.
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
) -> None:
    """``load_image`` keeps a four-band .npy stack as an ndarray.

    A .npy stack carries no color-interpretation metadata; load_image returns its pixels as an
    ndarray.
    """
    stack = _grid(20, 16, 4)
    np.save(str(tmp_path / "stack.npy"), stack)
    got = load_image(tmp_path / "stack.npy", 4)
    assert isinstance(got, np.ndarray)
    assert np.array_equal(got, stack)


def test_load_image_keeps_ndarray_where_pil_has_no_faithful_mode(
    tmp_path: Path,
) -> None:
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
    tmp_path: Path,
    name: str,
    channels: int,
    dtype: type[np.uint8] | type[np.uint16],
    tags: dict | None,
) -> None:
    """The header and the decode agree on whether a source is served as PIL.

    The mode a source's header says it is served in (what a loader built under the chain
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


def _flip_transform() -> Callable[
    [Image.Image, dict], tuple[torch.Tensor, dict]
]:
    def transform(img: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
        return pil_to_tensor(img.transpose(Image.FLIP_LEFT_RIGHT)), target

    return transform


def _detection_project(
    tmp_path: Path, arr: np.ndarray, *, extrasamples: list[str] | None = None
) -> Path:
    """Write ``img0.tif`` holding ``arr`` and its one-box label document; return their directory.

    The TIFF sits under ``images/<undated bucket>`` and its ``bud`` box is (2, 2, 10, 10).
    """
    import tifffile

    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
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


def test_a_uint8_tiff_trains_augmented_through_detection(
    tmp_path: Path,
) -> None:
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


def test_a_uint8_windowed_tile_trains_augmented_through_tiling(
    tmp_path: Path,
) -> None:
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


def test_a_declared_alpha_windowed_tile_trains_augmented(
    tmp_path: Path,
) -> None:
    """A windowed tile of a TIFF declaring alpha is read through the flip augmentation.

    For a 96x96 4-band TIFF whose file declares its 4th band alpha, the dataset's first tile
    equals the horizontally flipped top-left 64x64 pixels exactly.
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
) -> None:
    """A tiled loader under the chain refuses a TIFF declaring a spectral fourth band.

    A 4-band TIFF tagging its 4th band unspecified, read under a flip, is refused naming its
    4-band uint8 pixels.
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
    """Return a one-epoch run config over one frame holding ``arr``, flipped with probability 1.

    The frame comes from :func:`_detection_project`, under a val_ratio of zero. The run is a
    detection run of ``BUILT_DETECTOR``, or with ``bespoke`` one of task ``grape_bunch_count``
    naming a ``dataset_source`` builder.
    """
    from tests._chain_fixtures import training_config
    from tests._training_values import evaluation_block
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR
    from tests.test_dataset_source_seam import DATASET_SOURCE

    images_dir = _detection_project(tmp_path / "data", arr)
    return training_config(
        tmp_path,
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
) -> None:
    """A run stating augmentation over uint16 frames refuses at its resolution.

    Whoever builds the run's loaders, the platform's factory or its ``dataset_source``
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


def test_a_bodys_tiler_refuses_the_chain_over_uint16_frames(
    tmp_path: Path,
) -> None:
    """A body's tiler refuses the augmentation chain over uint16 frames.

    ``ctx.tiled_dataset`` under a flip refuses a uint16 dataset, naming its 3-band uint16
    pixels, and tiles the run's uint8 training dataset.
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
) -> None:
    """A dataset built around the door refuses the chain at its first read.

    A loader constructed directly with transforms over a uint16 frame refuses at its first
    read, naming the frame and its 3-band uint16 pixels.
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
) -> None:
    pytest.importorskip("torch")
    from tests._verified_checkpoint_fixtures import opened_run

    run_dir = opened_run(
        tmp_path, _augmented_run(tmp_path, _grid(40, 32, 3), bespoke=True)
    )
    assert run_dir.is_dir()


def test_the_same_run_over_uint8_frames_trains_augmented(
    tmp_path: Path,
) -> None:
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
