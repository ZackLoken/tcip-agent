"""Tests for the native-size ratio tile tier.

A checkpoint that trained untiled on frames that all shared one square size justifies tiling at
that frame's own size, with each tile run through the resize that run's recorded augmentation
config applied to a training frame.

A detector's own ``GeneralizedRCNNTransform`` itself resizes the tensor it is handed and maps its
boxes back to that tensor's coordinate space, so the tier's rescale undoes its own resize and
nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._producer_fixtures import checkpoint_admission

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.execution import Execution
    from tcip_mcp.pipelines.inference.generic_predictor import (
        WindowedRasterReader,
    )
    from tcip_mcp.pipelines.schemas import AugmentationSpec

pytest.importorskip("torch")

from tests._chain_fixtures import (  # noqa: E402
    BESPOKE_DETECTION,
    BESPOKE_MODELS,
    FRAME_STATISTICS,
)
from tests._producer_fixtures import gray_frame  # noqa: E402
from tests._training_values import (  # noqa: E402
    IOU_THRESHOLD,
    augmentation,
    tiling_block,
)

SCORED = {"iou_threshold": IOU_THRESHOLD, "trait": None}
"""What the full-frame runner scores these references under: the sample IoU and no trait."""

pytest.importorskip("torchvision")
import torch  # noqa: E402

from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor  # noqa: E402

TILE = 64
IMAGE = 128
# The band count and class space a three-band, one-subject run states on its data section.
_RUN_DATA = {"num_channels": 3, "scope": {"subject": "bud"}}


class _GeometryStub:
    """A predictor stub holding the four ``train_*`` geometry attributes it is given."""

    def __init__(
        self,
        *,
        train_tile_size: int | None = None,
        train_overlap: float | None = None,
        train_native_size: tuple[int, int] | None = None,
        train_augmentation: AugmentationSpec | None = None,
    ) -> None:
        self.train_tile_size = train_tile_size
        self.train_overlap = train_overlap
        self.train_native_size = train_native_size
        self.train_augmentation = train_augmentation


class _MiddleHalfDetector(torch.nn.Module):
    """A detector stub returning one box per image, scored 0.9, via ``GeneralizedRCNNTransform``.

    The box is the middle half of each image as the transform at ``min_size`` and ``max_size``
    resized it, mapped back by the transform's own ``postprocess``.
    """

    def __init__(
        self, min_size: int, max_size: int, channels: int = 3
    ) -> None:
        super().__init__()
        from torchvision.models.detection.transform import (
            GeneralizedRCNNTransform,
        )

        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, [0.0] * channels, [1.0] * channels
        )
        # One box scored 0.9 per input: within any cap and above any threshold a test states.
        self.score_thresh = 0.0
        self.detections_per_img = 100

    def forward(
        self, images: list[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        original_sizes = [
            (int(im.shape[-2]), int(im.shape[-1])) for im in images
        ]
        image_list, _ = self.transform(images)
        results = [
            {
                "boxes": torch.tensor(
                    [[w * 0.25, h * 0.25, w * 0.75, h * 0.75]],
                    dtype=torch.float32,
                ),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
            for h, w in image_list.image_sizes
        ]
        return self.transform.postprocess(
            results, image_list.image_sizes, original_sizes
        )


def _geometry(
    stub: _GeometryStub,
    *,
    tile_size: int | None = None,
    overlap: float | None = None,
    tiled: bool = True,
) -> tuple:
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    g = resolve_tile_geometry(
        stub, tiled=tiled, tile_size=tile_size, overlap=overlap
    )
    return g.tile_size, g.tile_size_source


def _stub_predictor(
    model: torch.nn.Module, *, task: str = "detection"
) -> GenericPredictor:
    p = GenericPredictor.__new__(GenericPredictor)
    p.task = task
    p.in_chans = 3
    p.attribute_sizes = []
    p.device = torch.device("cpu")
    p.model = model.eval()
    return p


def _sliced(
    pred: GenericPredictor,
    source: str | WindowedRasterReader,
    *,
    tile_resize: tuple[int, int] | None,
    **kwargs: str | dict | Callable[[int, int, dict], None] | None,
) -> dict:
    """Run ``predict_sliced`` at this module's lattice.

    The lattice is a ``TILE`` edge, no overlap, and NMS at the sample merge threshold
    (``tiled_record``'s).
    """
    from tests._verified_checkpoint_fixtures import tiled_record

    return pred.predict_sliced(
        source,
        execution=tiled_record(
            tile_size=TILE, overlap=0.0, conf=0.0, tile_resize=tile_resize
        ),
        tile_batch_size=8,
        **kwargs,
    )


def _expected_middle_half_boxes() -> set[tuple[float, float, float, float]]:
    """Return the middle half of every tile on a gapless 2x2 lattice, in image pixel space."""
    return {
        (x + TILE * 0.25, y + TILE * 0.25, x + TILE * 0.75, y + TILE * 0.75)
        for x in (0, TILE)
        for y in (0, TILE)
    }


# --- which tier the geometry comes from ---------------------------------


def test_square_untiled_training_frame_yields_a_tile_edge() -> None:
    """A run trained untiled on one square frame size yields that size as its tile edge."""
    assert _geometry(_GeometryStub(train_native_size=[512, 512])) == (
        512,
        "native_ratio",
    )


def test_rectangular_untiled_training_frame_yields_no_tile_edge() -> None:
    """A rectangular untiled training frame yields no tile edge.

    Tile geometry is a single square edge, and no square edge reproduces a rectangular frame's
    scale on both axes, so the geometry is unavailable.
    """
    assert _geometry(_GeometryStub(train_native_size=[1024, 768])) == (
        None,
        "unavailable",
    )


def test_persisted_tile_geometry_outranks_the_native_frame() -> None:
    stub = _GeometryStub(train_tile_size=224, train_native_size=[512, 512])

    assert _geometry(stub) == (224, "derived")


def test_explicit_tile_size_outranks_the_native_frame() -> None:
    """On an untiled pass the resolver records a stated edge as stated over the native frame."""
    stub = _GeometryStub(train_native_size=[512, 512])

    assert _geometry(stub, tile_size=320, tiled=False) == (320, "explicit")


def test_no_native_frame_yields_no_tile_edge() -> None:
    assert _geometry(_GeometryStub(train_native_size=None)) == (
        None,
        "unavailable",
    )


@pytest.mark.parametrize(
    "native_size", [[512], [0, 0], ["wide", "tall"], 512, [-4, -4]]
)
def test_a_geometry_record_refuses_a_native_size_that_is_no_frame(
    native_size: list[int] | list[str] | int,
) -> None:
    """``TrainGeometry`` refuses a native size that is not two positive pixel counts, by name."""
    from pydantic import ValidationError

    from tcip_mcp.pipelines.schemas import TrainGeometry

    with pytest.raises(ValidationError, match="native_size"):
        TrainGeometry.model_validate(
            {"native_size": native_size, "object_density": None}
        )


# --- the recorded train-time resize -------------------------------------


def _admitted(config: dict | str) -> AugmentationSpec:
    """Return ``config``, an ``augmentation`` block, as the schema admits it."""
    from tcip_mcp.pipelines.schemas import AugmentationSpec

    return AugmentationSpec.model_validate(config)


def _resize(config: dict | str) -> tuple[int, int] | None:
    """Return the recorded resize of the ``augmentation`` block ``config`` as admitted."""
    from tcip_mcp.pipelines.data.augmentations import recorded_resize

    return recorded_resize(_admitted(config))


def test_a_config_with_no_resize_records_no_resize() -> None:
    from tcip_mcp.pipelines.data.augmentations import recorded_resize

    assert recorded_resize(None) is None
    assert _resize(augmentation()) is None
    assert _resize(augmentation(horizontal_flip={"p": 0.5})) is None


def test_a_recorded_resize_is_read_through_the_builders_own_chain() -> None:
    """The recorded resize is the dict-form resize's size, alone or beside a flip."""
    assert _resize(augmentation(resize={"size": [320, 240]})) == (320, 240)
    assert _resize(
        augmentation(resize={"size": [800, 600]}, horizontal_flip={"p": 0.5})
    ) == (800, 600)


@pytest.mark.parametrize(
    "config, refusal",
    [
        (augmentation(not_a_transform={"p": 0.5}), "Unknown augmentation"),
        (augmentation(resize=[640, 640]), "'resize'"),
        (augmentation(resize=True), "'resize'"),
        ("nadir_rotation", "'nadir_rotation'"),
        ({"resize": {"size": [640, 640]}}, "rationale"),
    ],
)
def test_an_unbuildable_recorded_config_raises_rather_than_reading_as_no_resize(
    config: dict | str, refusal: str
) -> None:
    """A recorded config the schema or the builder refuses raises when its resize is read.

    An unknown transform, a shorthand, a preset name and a missing reason each raise their
    refusal.
    """
    with pytest.raises(ValueError, match=refusal):
        _resize(config)


def test_the_recorded_resize_travels_only_with_a_native_frame_tile_edge() -> (
    None
):
    """The recorded resize travels only with a native-frame tile edge.

    An untiled pass, an explicit tile edge and a persisted-geometry tile edge each carry no
    resize: those edges mean the tile as it stands.
    """
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    chain = _admitted(augmentation(resize={"size": [640, 640]}))
    native = _GeometryStub(
        train_native_size=[512, 512], train_augmentation=chain
    )
    persisted = _GeometryStub(train_tile_size=512, train_augmentation=chain)

    def resize(
        stub: _GeometryStub, **kw: bool | int | None
    ) -> tuple[int, int] | None:
        return resolve_tile_geometry(stub, overlap=None, **kw).tile_resize

    assert resize(native, tiled=True, tile_size=None) == (640, 640)
    assert resize(native, tiled=False, tile_size=None) is None
    assert resize(persisted, tiled=True, tile_size=None) is None
    assert resize(native, tiled=True, tile_size=512) is None
    assert (
        resize(
            _GeometryStub(train_augmentation=chain),
            tiled=True,
            tile_size=None,
        )
        is None
    )


def test_a_checkpoint_carries_its_untiled_training_geometry_to_the_predictor(
    tmp_path: Path,
) -> None:
    from tcip_mcp.model_registry import load_registered_checkpoint

    checkpoint = load_registered_checkpoint(
        _native_frame_checkpoint(
            tmp_path, augmentation(resize={"size": [32, 32]})
        ),
        project=tmp_path,
    )

    pred = GenericPredictor(checkpoint, device="cpu")

    assert pred.train_tile_size is None
    assert pred.train_native_size == (TILE, TILE)
    assert pred.train_augmentation == _admitted(
        augmentation(resize={"size": [32, 32]})
    )


# --- the tile pass itself -----------------------------------------------


def test_tiles_at_native_size_with_no_recorded_resize(tmp_path: Path) -> None:
    """A recorded chain that pins no size tiles at the native frame size and nothing else.

    Boxes land in image pixel space untouched by any rescale.
    """
    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    r = _sliced(pred, gray_frame(tmp_path, IMAGE), tile_resize=None)

    assert {tuple(b) for b in r["boxes"]} == _expected_middle_half_boxes()


def test_a_recorded_resize_is_undone_per_axis_and_not_confused_with_the_detectors_own(
    tmp_path: Path,
) -> None:
    """A recorded resize is undone per axis, apart from the detector's own internal resize.

    ``min_size`` (800) differs from both the tile edge (64) and the resize target (128x96), and
    the target is not square. Rounded to four decimals, the boxes equal
    ``_expected_middle_half_boxes()``.
    """
    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    r = _sliced(pred, gray_frame(tmp_path, IMAGE), tile_resize=(128, 96))

    boxes = sorted(tuple(round(v, 4) for v in b) for b in r["boxes"])
    assert boxes == sorted(_expected_middle_half_boxes())


def test_a_windowed_raster_source_is_resized_and_undone_the_same_way() -> None:
    """A windowed raster source's boxes come back in full-raster pixel space under a resize.

    The windowed path hands each tile over as an array, not a PIL image, and
    ``to_pil_if_faithful`` decides whether the recorded resize applies to it; a uint8 three-band
    window is faithfully PIL, so the resize applies and is undone.
    """
    import numpy as np

    class _Reader:
        height = IMAGE
        width = IMAGE
        num_channels = 3

        def read_window(
            self, y0: int, y1: int, x0: int, x1: int
        ) -> np.ndarray:
            return np.full((y1 - y0, x1 - x0, 3), 120, dtype=np.uint8)

    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    r = _sliced(pred, _Reader(), tile_resize=(128, 96), source_label="raster")

    assert {
        tuple(round(v, 4) for v in b) for b in r["boxes"]
    } == _expected_middle_half_boxes()


def test_a_windowed_alpha_tagged_source_is_resized_and_undone_the_same_way() -> (
    None
):
    """A windowed 4-band reader declaring its 4th band alpha gets the recorded resize and undo.

    The reader's ``band_interpretations`` declares the 4th band alpha, so its boxes come back in
    full-raster pixel space at the middle half of every tile.
    """
    import numpy as np

    class _Reader:
        height = IMAGE
        width = IMAGE
        num_channels = 4
        band_interpretations = ("red", "green", "blue", "alpha")

        def read_window(
            self, y0: int, y1: int, x0: int, x1: int
        ) -> np.ndarray:
            return np.full((y1 - y0, x1 - x0, 4), 120, dtype=np.uint8)

    pred = _stub_predictor(
        _MiddleHalfDetector(min_size=800, max_size=1333, channels=4)
    )
    pred.in_chans = 4

    r = _sliced(pred, _Reader(), tile_resize=(128, 96), source_label="raster")

    assert {
        tuple(round(v, 4) for v in b) for b in r["boxes"]
    } == _expected_middle_half_boxes()


def test_a_windowed_undeclared_fourth_band_source_refuses_the_recorded_resize() -> (
    None
):
    """A windowed 4-band reader with no band interpretations refuses the recorded resize by name.

    Such a reader (an .npy-backed reader, or a GDAL file whose 4th band carries no alpha tag) is
    not guessed into RGBA, so the resize refuses.
    """
    import numpy as np

    class _Reader:
        height = IMAGE
        width = IMAGE
        num_channels = 4

        def read_window(
            self, y0: int, y1: int, x0: int, x1: int
        ) -> np.ndarray:
            return np.full((y1 - y0, x1 - x0, 4), 120, dtype=np.uint8)

    pred = _stub_predictor(
        _MiddleHalfDetector(min_size=800, max_size=1333, channels=4)
    )
    pred.in_chans = 4

    with pytest.raises(ValueError, match="4-band uint8 pixels"):
        _sliced(pred, _Reader(), tile_resize=(128, 96), source_label="raster")


def test_a_tile_no_pil_mode_represents_refuses_the_recorded_resize(
    tmp_path: Path,
) -> None:
    """A recorded resize over a tile no PIL mode takes refuses by name.

    A 3-band uint16 tile has no PIL mode to resize through, so the pass refuses naming its pixels.
    """
    import numpy as np

    arr = np.zeros((IMAGE, IMAGE, 3), dtype=np.uint16)
    path = tmp_path / "sixteen_bit.npy"
    np.save(path, arr)
    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    with pytest.raises(ValueError, match="3-band uint16 pixels"):
        _sliced(pred, str(path), tile_resize=(128, 96))


# --- the doors on either side of the tier -------------------------------


def _trained_checkpoint(
    tmp_path: Path, tiling: dict | None, *, size: int = TILE, **config: object
) -> str:
    """Return the checkpoint ``registered_checkpoint`` completes over two ``size`` px frames.

    ``seed_bud_images`` writes the frames under ``tmp_path``; the run is of ``BESPOKE_DETECTION``
    at ``FRAME_STATISTICS``, under ``tiling``, with ``config``'s keys passed through.
    """
    from tests._producer_fixtures import seed_bud_images
    from tests._verified_checkpoint_fixtures import (
        fixture_data_dir,
        registered_checkpoint,
    )

    images = seed_bud_images(
        fixture_data_dir(tmp_path, f"frames-{size}")
        / "images"
        / UNDATED_BUCKET,
        n=2,
        size=size,
    )
    return registered_checkpoint(
        tmp_path,
        model_source={
            "builder": BESPOKE_DETECTION,
            "task": "detection",
            "source_files": [BESPOKE_MODELS],
            "normalization": FRAME_STATISTICS,
        },
        data={"images_dir": str(images), "tiling": tiling, **_RUN_DATA},
        **config,
    )


def _native_frame_checkpoint(
    tmp_path: Path, augmentation: dict | None = None
) -> str:
    """Return :func:`_trained_checkpoint` of an untiled run, under ``augmentation`` when given."""
    return _trained_checkpoint(
        tmp_path,
        None,
        **({} if augmentation is None else {"augmentation": augmentation}),
    )


def test_a_tiled_pass_over_a_native_frame_checkpoint_says_what_it_rests_on(
    tmp_path: Path,
) -> None:
    """A tiled pass over a native-frame checkpoint runs and records the basis it rests on.

    A tiled pass at a stated overlap, over a checkpoint whose only geometry is its untiled
    training frame, runs at that frame's edge, its record naming that basis and the recorded
    resize.
    """
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_OVERLAP,
        predicted_over,
    )

    ckpt = _native_frame_checkpoint(
        tmp_path, augmentation(resize={"size": [32, 32]})
    )

    p, results = predicted_over(
        tmp_path,
        ckpt,
        str(Path(gray_frame(tmp_path, IMAGE)).parent),
        device="cpu",
        tile=True,
        overlap=SAMPLE_OVERLAP,
        conf=0.0,
    )

    assert len(results) == 1
    assert (p.execution.tile_size, p.execution.sources["tile_size"]) == (
        TILE,
        "native_ratio",
    )
    assert p.execution.tile_resize == (32, 32)


def test_a_native_frame_checkpoint_stays_untiled_unless_asked(
    tmp_path: Path,
) -> None:
    """A native-frame checkpoint runs untiled when ``tile`` is unset.

    ``tile`` unset follows the checkpoint's own regime, and an untiled-trained checkpoint's regime
    is untiled.
    """
    from tests._verified_checkpoint_fixtures import predicted_over

    ckpt = _native_frame_checkpoint(tmp_path)

    p, _results = predicted_over(
        tmp_path,
        ckpt,
        str(Path(gray_frame(tmp_path, IMAGE)).parent),
        device="cpu",
        conf=0.0,
    )

    assert p.execution.tile_size is None


def _native_frame_gt(images_dir: Path) -> None:
    """Write a single 128x128 image labeled at the middle half of every tile of a 2x2 lattice.

    The lattice is gapless at the TILE edge, and the labels are ``_expected_middle_half_boxes()``.
    """
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    Image.new("RGB", (IMAGE, IMAGE), (120, 120, 120)).save(
        images_dir / "a.png"
    )
    label_image(
        images_dir / "a.png",
        [
            Annotation(subject="bud", geometry=BBox(*b))
            for b in sorted(_expected_middle_half_boxes())
        ],
        IMAGE,
        IMAGE,
    )


def _persisted_regime_predictor() -> GenericPredictor:
    p = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))
    p.train_tile_size, p.train_overlap = TILE, 0.0
    p.train_native_size, p.train_augmentation = None, None
    return p


def _native_frame_regime_predictor() -> GenericPredictor:
    """Return a predictor whose only tiling basis is its uniform untiled training frame.

    Its recorded chain pins a resize to twice the tile edge.
    """
    p = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))
    p.train_tile_size, p.train_overlap = None, 0.0
    p.train_native_size = [TILE, TILE]
    p.train_augmentation = _admitted(
        augmentation(resize={"size": [TILE * 2, TILE * 2]})
    )
    return p


def test_delivery_grade_evaluation_admits_a_native_frame_basis_and_reproduces_the_persisted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native-frame basis passes delivery-grade evaluation and matches a persisted regime.

    A checkpoint whose only tiling basis is its own uniform untiled training frame reaches the
    delivery-grade gate and produces the identical counts, metrics and box coordinates a persisted
    tiled regime produces, while its recorded augmentation chain pins a resize to twice the tile
    edge that the native-frame regime carries.
    """
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    from tcip_mcp.pipelines.execution import Stated, execution_record
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation,
    )
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_DETECTOR_PASS,
        verified_checkpoint,
    )

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _native_frame_gt(images_dir)
    checkpoint = verified_checkpoint(tmp_path)
    stated = Stated(**SAMPLE_DETECTOR_PASS)

    monkeypatch.setattr(
        predictor_mod,
        "GenericPredictor",
        lambda *a, **kw: _persisted_regime_predictor(),
    )
    persisted = run_full_frame_evaluation(
        checkpoint,
        checkpoint_admission(checkpoint, images_dir),
        stated=stated,
        **SCORED,
    )
    monkeypatch.setattr(
        predictor_mod,
        "GenericPredictor",
        lambda *a, **kw: _native_frame_regime_predictor(),
    )
    native = run_full_frame_evaluation(
        checkpoint,
        checkpoint_admission(checkpoint, images_dir),
        stated=stated,
        **SCORED,
    )
    monkeypatch.undo()

    persisted_execution, native_execution = (
        persisted["execution"],
        native["execution"],
    )
    assert persisted_execution["sources"]["tile_size"] == "derived"
    assert native_execution["sources"]["tile_size"] == "native_ratio"
    assert (
        native_execution["tile_size"]
        == persisted_execution["tile_size"]
        == TILE
    )
    assert (
        persisted["tp"] == 4 and persisted["fp"] == 0 and persisted["fn"] == 0
    )
    for key in (
        "tp",
        "fp",
        "fn",
        "n_gt",
        "n_pred",
        "precision",
        "recall",
        "f1",
        "map",
        "map50",
    ):
        assert native[key] == persisted[key], key

    # The metrics alone cannot distinguish "byte-identical boxes" from "close enough to still
    # match": run the exact geometry each regime resolved directly and compare coordinates.
    persisted_predictor, native_predictor = (
        _persisted_regime_predictor(),
        _native_frame_regime_predictor(),
    )
    p_geo = resolve_tile_geometry(
        persisted_predictor, tiled=True, tile_size=None, overlap=None
    )
    n_geo = resolve_tile_geometry(
        native_predictor, tiled=True, tile_size=None, overlap=None
    )
    assert n_geo.tile_resize == (TILE * 2, TILE * 2)
    unfiltered = stated.model_copy(update={"conf": 0.0})
    r_p, r_n = (
        predictor.predict_sliced(
            str(images_dir / "a.png"),
            execution=execution_record(checkpoint, unfiltered, geo, None),
            tile_batch_size=8,
        )
        for predictor, geo in (
            (persisted_predictor, p_geo),
            (native_predictor, n_geo),
        )
    )
    assert (
        {tuple(b) for b in r_p["boxes"]}
        == {tuple(b) for b in r_n["boxes"]}
        == _expected_middle_half_boxes()
    )


def test_delivery_grade_evaluation_forwards_the_native_frame_resize_into_predict_sliced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delivery-grade evaluation hands ``predict_sliced`` a native-frame checkpoint's resize."""
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation,
    )
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_DETECTOR_PASS,
        verified_checkpoint,
    )

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _native_frame_gt(images_dir)

    captured: dict = {}

    def _spy_predictor(*a: object, **kw: object) -> GenericPredictor:
        p = _native_frame_regime_predictor()
        real_predict_sliced = p.predict_sliced

        def _spy(
            *a: str | Path | BandGroupRef | WindowedRasterReader,
            **kwargs: Execution
            | int
            | str
            | dict
            | Callable[[int, int, dict], None]
            | None,
        ) -> dict:
            captured.update(kwargs)
            return real_predict_sliced(*a, **kwargs)

        p.predict_sliced = _spy
        return p

    checkpoint = verified_checkpoint(tmp_path)
    monkeypatch.setattr(predictor_mod, "GenericPredictor", _spy_predictor)
    r = run_full_frame_evaluation(
        checkpoint,
        checkpoint_admission(checkpoint, images_dir),
        stated=Stated(**SAMPLE_DETECTOR_PASS),
        **SCORED,
    )

    assert "error" not in r
    assert captured["execution"].tile_resize == (TILE * 2, TILE * 2)


# --- the contradiction refusal -------------------------------------------


def test_an_explicit_edge_contradicting_persisted_geometry_refuses() -> None:
    from tcip_mcp.pipelines.slicing import (
        TileEdgeContradictionError,
        resolve_tile_geometry,
    )

    stub = _GeometryStub(train_tile_size=128)

    with pytest.raises(TileEdgeContradictionError) as exc_info:
        resolve_tile_geometry(stub, tiled=True, tile_size=64, overlap=None)
    assert str(exc_info.value) == (
        "stated tile_size 64 contradicts this checkpoint's own persisted training tile geometry "
        "of 128. Pass tile_size 128 to match the checkpoint, or leave tile_size unset to derive "
        "it from the checkpoint."
    )


def test_an_explicit_edge_contradicting_the_native_frame_refuses() -> None:
    from tcip_mcp.pipelines.slicing import (
        TileEdgeContradictionError,
        resolve_tile_geometry,
    )

    stub = _GeometryStub(train_native_size=[512, 512])

    with pytest.raises(TileEdgeContradictionError) as exc_info:
        resolve_tile_geometry(stub, tiled=True, tile_size=64, overlap=None)
    assert str(exc_info.value) == (
        "stated tile_size 64 contradicts this checkpoint's own recorded untiled training frame "
        "of 512. Pass tile_size 512 to match the checkpoint, or leave tile_size unset to derive "
        "it from the checkpoint."
    )


def test_an_untiled_call_with_a_contradicting_stated_edge_is_inert() -> None:
    """An untiled call keeps a stated edge that contradicts the persisted geometry, as explicit."""
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    stub = _GeometryStub(train_tile_size=128)

    g = resolve_tile_geometry(stub, tiled=False, tile_size=64, overlap=None)

    assert (g.tile_size, g.tile_size_source) == (64, "explicit")


def test_an_explicit_edge_equal_to_persisted_geometry_clears() -> None:
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    stub = _GeometryStub(train_tile_size=128)

    g = resolve_tile_geometry(stub, tiled=True, tile_size=128, overlap=None)

    assert (g.tile_size, g.tile_size_source) == (128, "explicit")


def test_an_explicit_edge_equal_to_the_native_frame_clears() -> None:
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    stub = _GeometryStub(train_native_size=[512, 512])

    g = resolve_tile_geometry(stub, tiled=True, tile_size=512, overlap=None)

    assert (g.tile_size, g.tile_size_source) == (512, "explicit")


def test_an_explicit_edge_on_a_checkpoint_recording_no_geometry_clears() -> (
    None
):
    """A stated edge on a checkpoint recording no geometry stands as explicit on a tiled call."""
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    stub = _GeometryStub()

    g = resolve_tile_geometry(stub, tiled=True, tile_size=64, overlap=None)

    assert (g.tile_size, g.tile_size_source) == (64, "explicit")


def _tiled_checkpoint(tmp_path: Path, tile_size: int) -> str:
    """Return :func:`_trained_checkpoint` of a run tiled at ``tile_size`` over frames that size."""
    return _trained_checkpoint(  # sliver_frac stated: one box per frame derives no spread
        tmp_path,
        tiling_block(tile_size=tile_size, overlap=0.2, sliver_frac=0.5),
        size=tile_size,
    )


@pytest.mark.parametrize(
    "make, recorded",
    [
        (lambda tmp_path: _tiled_checkpoint(tmp_path, 128), "128"),
        (
            lambda tmp_path: _trained_checkpoint(tmp_path, None, size=512),
            "512",
        ),
    ],
    ids=["persisted-geometry", "native-frame"],
)
def test_a_stated_edge_contradicting_the_checkpoints_geometry_refuses_the_pass(
    tmp_path: Path, make: Callable[[Path], str], recorded: str
) -> None:
    """A stated tile edge contradicting the checkpoint's geometry refuses the pass.

    A caller-typed edge that differs from the checkpoint's own persisted training geometry, or
    from its recorded untiled training frame when it persists none, refuses naming both edges.
    """
    from tcip_mcp.pipelines.execution import ExecutionRefusedError
    from tests._verified_checkpoint_fixtures import predicted_over

    ckpt = make(tmp_path)

    with pytest.raises(ExecutionRefusedError) as exc_info:
        predicted_over(
            tmp_path,
            ckpt,
            str(Path(gray_frame(tmp_path, IMAGE)).parent),
            device="cpu",
            tile=True,
            tile_size=64,
            conf=0.0,
        )
    assert "64" in str(exc_info.value) and recorded in str(exc_info.value)


def test_a_stated_edge_matching_persisted_geometry_is_admitted_as_stated(
    tmp_path: Path,
) -> None:
    """A stated tile edge agreeing with the persisted geometry is admitted and recorded as stated.

    An explicit edge equal to the checkpoint's persisted geometry is no contradiction.
    """
    from tests._verified_checkpoint_fixtures import predicted_over

    ckpt = _tiled_checkpoint(tmp_path, TILE)

    p, results = predicted_over(
        tmp_path,
        ckpt,
        str(Path(gray_frame(tmp_path, IMAGE)).parent),
        device="cpu",
        tile=True,
        tile_size=TILE,
        conf=0.0,
    )

    assert len(results) == 1
    assert (p.execution.tile_size, p.execution.sources["tile_size"]) == (
        TILE,
        "explicit",
    )
