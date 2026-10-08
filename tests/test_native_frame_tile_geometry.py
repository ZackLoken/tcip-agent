"""The native-size ratio tile tier: a checkpoint that trained untiled on frames that all shared
one square size justifies tiling at that frame's own size, with each tile run through the resize
that run's recorded augmentation config applied to a training frame.

The coordinate hazard these tests exist for: a detector's own ``GeneralizedRCNNTransform`` already
resizes the tensor it is handed and maps its boxes back to that tensor's coordinate space, so the
tier's rescale must undo its own resize and nothing else. A test whose model has no internal
transform, or whose ``min_size`` happens to equal the tile edge, cannot tell a correct rescale from
a doubled one.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._producer_fixtures import checkpoint_admission

from pathlib import Path

import pytest

pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import CONFIG_KEY, STATE_DICT_KEY  # noqa: E402
from tests._chain_fixtures import BESPOKE_DETECTION  # noqa: E402
from tests._producer_fixtures import gray_frame  # noqa: E402
pytest.importorskip("torchvision")
import torch  # noqa: E402

from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor  # noqa: E402

TILE = 64
IMAGE = 128
# The band count and class space a three-band, one-subject run records on its data section.
_RUN_DATA = {"num_channels": 3, "scope": {"subject": "bud", "attributes": []}}


class _GeometryStub:
    """Only the geometry facts ``resolve_tile_geometry`` reads off a predictor."""

    def __init__(self, *, train_tile_size=None, train_overlap=None, train_native_size=None,
                 train_augmentation=None) -> None:
        self.train_tile_size = train_tile_size
        self.train_overlap = train_overlap
        self.train_native_size = train_native_size
        self.train_augmentation = train_augmentation


class _MiddleHalfDetector(torch.nn.Module):
    """Proposes one box over the middle half of whatever tensor it is handed, through torchvision's
    own ``GeneralizedRCNNTransform`` so the box makes the same internal-resize round trip a real
    detector's boxes make. Its output is therefore exactly the middle half of the *input* tile
    tensor, whatever ``min_size`` is, which is what makes a doubled correction visible.
    """

    def __init__(self, min_size: int, max_size: int, channels: int = 3) -> None:
        super().__init__()
        from torchvision.models.detection.transform import GeneralizedRCNNTransform

        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, [0.0] * channels, [1.0] * channels)

    def forward(self, images):
        original_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
        image_list, _ = self.transform(images)
        results = [
            {"boxes": torch.tensor([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=torch.float32),
             "scores": torch.tensor([0.9]),
             "labels": torch.tensor([1])}
            for h, w in image_list.image_sizes
        ]
        return self.transform.postprocess(results, image_list.image_sizes, original_sizes)


def _geometry(stub, *, tile_size=None, overlap=None, tiled=True) -> tuple:
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    g = resolve_tile_geometry(stub, tiled=tiled, tile_size=tile_size, overlap=overlap)
    return g.tile_size, g.tile_size_source


def _stub_predictor(model, *, task: str = "detection") -> GenericPredictor:
    p = GenericPredictor.__new__(GenericPredictor)
    p.task = task
    p.in_chans = 3
    p.attribute_sizes = []
    p.device = torch.device("cpu")
    p.model_source = {}
    p.model = model.eval()
    return p


def _sliced(pred, source, *, tile_resize, **kwargs) -> dict:
    """``predict_sliced`` at this module's lattice: ``TILE`` edge, no overlap, NMS at 0.3."""
    from tests._verified_checkpoint_fixtures import tiled_record

    return pred.predict_sliced(
        source, execution=tiled_record(tile_size=TILE, overlap=0.0, conf=0.0,
                                       tile_resize=tile_resize),
        tile_batch_size=8, require_masks=True, **kwargs)


def _expected_middle_half_boxes() -> set[tuple[float, float, float, float]]:
    """The middle half of every tile on a gapless 2x2 lattice, in image pixel space."""
    return {(x + TILE * 0.25, y + TILE * 0.25, x + TILE * 0.75, y + TILE * 0.75)
            for x in (0, TILE) for y in (0, TILE)}


# --- which tier the geometry comes from ---------------------------------


def test_square_untiled_training_frame_yields_a_tile_edge():
    """A run that trained untiled on frames that all shared one square size justifies tiling at
    that frame: an object in such a tile reaches the model at the scale a whole training frame
    presented it at."""
    assert _geometry(_GeometryStub(train_native_size=[512, 512])) == (512, "native_ratio")


def test_rectangular_untiled_training_frame_yields_no_tile_edge():
    """Tile geometry is a single square edge everywhere it travels, and no square edge reproduces a
    rectangular frame's scale on both axes. Refuse, rather than pick an edge that silently
    mis-scales one axis."""
    assert _geometry(_GeometryStub(train_native_size=[1024, 768])) == (None, "unavailable")


def test_persisted_tile_geometry_outranks_the_native_frame():
    stub = _GeometryStub(train_tile_size=224, train_native_size=[512, 512])

    assert _geometry(stub) == (224, "derived")


def test_explicit_tile_size_outranks_the_native_frame():
    """On an untiled pass a stated edge is a fact the resolver records as stated; a tiled pass
    checks it against the native frame instead (the contradiction tests below)."""
    stub = _GeometryStub(train_native_size=[512, 512])

    assert _geometry(stub, tile_size=320, tiled=False) == (320, "explicit")


@pytest.mark.parametrize("stamp", [None, [512], [0, 0], ["wide", "tall"], 512, [-4, -4]])
def test_an_unusable_native_frame_stamp_yields_no_tile_edge_and_never_raises(stamp):
    """A checkpoint this platform did not write can carry anything under that key, and
    ``resolve_tile_geometry`` is a fact-return every caller relies on not to raise."""
    assert _geometry(_GeometryStub(train_native_size=stamp)) == (None, "unavailable")


# --- the recorded train-time resize -------------------------------------


def test_a_config_with_no_resize_records_no_resize():
    from tcip_mcp.pipelines.data.augmentations import recorded_resize

    assert recorded_resize(None) is None
    assert recorded_resize({}) is None
    assert recorded_resize({"horizontal_flip": 0.5}) is None


def test_a_recorded_resize_is_read_through_the_builders_own_conventions():
    from tcip_mcp.pipelines.data.augmentations import recorded_resize

    assert recorded_resize({"resize": [800, 600]}) == (800, 600)
    assert recorded_resize({"resize": {"size": [320, 240]}}) == (320, 240)
    assert recorded_resize({"resize": True}) == (640, 640)


def test_a_preset_name_resolves_through_the_same_preset_the_run_built():
    """A preset string is not a ``[w, h]`` pair; it resolves through
    ``get_augmentation_preset``, at the same default size every production caller builds it with.

    That default is pinned here rather than read back off the preset, so the size a preset name
    reproduces is a stated fact and not whatever the two sides happen to agree on today. A caller
    that names its own size gets that one instead.
    """
    from tcip_mcp.pipelines.data.augmentations import get_augmentation_preset, recorded_resize

    assert recorded_resize("nadir_rotation") == (640, 640)
    assert recorded_resize("nadir_rotation") == tuple(
        get_augmentation_preset("nadir_rotation")["resize"])
    assert tuple(get_augmentation_preset("nadir_rotation", (512, 384))["resize"]) == (512, 384)


def test_an_unbuildable_recorded_config_raises_rather_than_reading_as_no_resize():
    from tcip_mcp.pipelines.data.augmentations import recorded_resize

    with pytest.raises(ValueError, match="Unknown augmentation"):
        recorded_resize({"not_a_transform": 0.5})


def test_the_recorded_resize_travels_only_with_a_native_frame_tile_edge():
    """An explicit or persisted-geometry tile edge means the tile as it stands, which is what every
    count already produced at those tiers was measured at."""
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    augmentation = {"resize": [640, 640]}
    native = _GeometryStub(train_native_size=[512, 512], train_augmentation=augmentation)
    persisted = _GeometryStub(train_tile_size=512, train_augmentation=augmentation)

    def resize(stub, **kw):
        return resolve_tile_geometry(stub, overlap=None, **kw).tile_resize

    assert resize(native, tiled=True, tile_size=None) == (640, 640)
    assert resize(native, tiled=False, tile_size=None) is None
    assert resize(persisted, tiled=True, tile_size=None) is None
    assert resize(native, tiled=True, tile_size=512) is None
    assert resize(
        _GeometryStub(train_augmentation=augmentation), tiled=True, tile_size=None) is None


def test_a_checkpoint_carries_its_untiled_training_geometry_to_the_predictor(tmp_path):
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims
    from tcip_mcp.tools.model_tools import register_model

    model_source = {"builder": BESPOKE_DETECTION,
                    "builder_kwargs": {"min_size": TILE, "max_size": TILE * 2},
                    "task": "detection"}
    ckpt = tmp_path / "model_best.pt"
    config = {"model_source": model_source,
              "data": {"tiling": {"enabled": False}, "train_native_size": [TILE, TILE],
                       **_RUN_DATA},
              "augmentation": {"resize": [32, 32]}}
    torch.save({STATE_DICT_KEY: build_model(config, recorded_model_dims(config)).state_dict(),
                CONFIG_KEY: config}, str(ckpt))
    result = register_model(name="native-frame-carry", checkpoint_path=str(ckpt), config={},
                            project=tmp_path)
    assert "error" not in result, result
    checkpoint = load_registered_checkpoint(str(ckpt), project=tmp_path)

    pred = GenericPredictor(checkpoint, device="cpu")

    assert pred.train_tile_size is None
    assert pred.train_native_size == [TILE, TILE]
    assert pred.train_augmentation == {"resize": [32, 32]}


# --- the tile pass itself -----------------------------------------------


def test_tiles_at_native_size_with_no_recorded_resize(tmp_path):
    """The common case: the recorded chain pins no size, so the tier is tiling at the native frame
    size and nothing else. Boxes land in image pixel space untouched by any rescale."""
    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    r = _sliced(pred, gray_frame(tmp_path, IMAGE), tile_resize=None)

    assert {tuple(b) for b in r["boxes"]} == _expected_middle_half_boxes()


def test_a_recorded_resize_is_undone_per_axis_and_not_confused_with_the_detectors_own(tmp_path):
    """``min_size`` (800) is deliberately unlike both the tile edge (64) and the resize target
    (128x96), and the target is deliberately not square. Boxes must land exactly where the
    no-resize pass puts them: the detector's internal transform already maps its own boxes back to
    the tensor it was handed, so only the tier's own stretch is left to undo, and undoing it with
    one scalar factor instead of two would displace every y coordinate."""
    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    r = _sliced(pred, gray_frame(tmp_path, IMAGE), tile_resize=(128, 96))

    boxes = sorted(tuple(round(v, 4) for v in b) for b in r["boxes"])
    assert boxes == sorted(_expected_middle_half_boxes())


def test_a_windowed_raster_source_is_resized_and_undone_the_same_way():
    """The windowed path hands each tile over as an array, not a PIL image, and the same
    ``to_pil_if_faithful`` rule the training loader uses decides whether the recorded resize applies
    to it; a uint8 three-band window is faithfully PIL, so it does, and the boxes come back in
    full-raster pixel space all the same."""
    import numpy as np

    class _Reader:
        height = IMAGE
        width = IMAGE
        num_channels = 3

        def read_window(self, y0, y1, x0, x1):
            return np.full((y1 - y0, x1 - x0, 3), 120, dtype=np.uint8)

    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    r = _sliced(pred, _Reader(), tile_resize=(128, 96), source_label="raster")

    assert {tuple(round(v, 4) for v in b) for b in r["boxes"]} == _expected_middle_half_boxes()


def test_a_windowed_alpha_tagged_source_is_resized_and_undone_the_same_way(caplog):
    """A windowed 4-band reader whose ``band_interpretations`` declares the 4th band alpha gets
    the recorded resize applied and undone as the 3-band case does, logging no skip warning."""
    import logging

    import numpy as np

    class _Reader:
        height = IMAGE
        width = IMAGE
        num_channels = 4
        band_interpretations = ("red", "green", "blue", "alpha")

        def read_window(self, y0, y1, x0, x1):
            return np.full((y1 - y0, x1 - x0, 4), 120, dtype=np.uint8)

    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333, channels=4))
    pred.in_chans = 4

    with caplog.at_level(logging.WARNING):
        r = _sliced(pred, _Reader(), tile_resize=(128, 96), source_label="raster")

    assert {tuple(round(v, 4) for v in b) for b in r["boxes"]} == _expected_middle_half_boxes()
    assert not any("recorded train-time resize" in m for m in caplog.messages)


def test_a_windowed_undeclared_fourth_band_source_keeps_its_own_pixels(caplog):
    """A windowed 4-band reader with no band_interpretations fact (an .npy-backed reader, or a
    GDAL file whose 4th band carries no alpha tag) must not be guessed into RGBA: the recorded
    resize is skipped, reported the same way the uint16/5-band case already is, not silently."""
    import logging

    import numpy as np

    class _Reader:
        height = IMAGE
        width = IMAGE
        num_channels = 4

        def read_window(self, y0, y1, x0, x1):
            return np.full((y1 - y0, x1 - x0, 4), 120, dtype=np.uint8)

    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333, channels=4))
    pred.in_chans = 4

    with caplog.at_level(logging.WARNING):
        r = _sliced(pred, _Reader(), tile_resize=(128, 96), source_label="raster")

    assert {tuple(round(v, 4) for v in b) for b in r["boxes"]} == _expected_middle_half_boxes()
    assert any("recorded train-time resize" in m for m in caplog.messages)


def test_a_tile_no_pil_mode_represents_keeps_its_own_pixels(tmp_path, caplog):
    """The training loader's transform chain is PIL-only and skipped such a sample too, so applying
    the recorded resize here would introduce a geometry training never applied. The skip is
    reported, not silent."""
    import logging

    import numpy as np

    arr = np.zeros((IMAGE, IMAGE, 3), dtype=np.uint16)
    path = tmp_path / "sixteen_bit.npy"
    np.save(path, arr)
    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    with caplog.at_level(logging.WARNING):
        r = _sliced(pred, str(path), tile_resize=(128, 96))

    assert {tuple(b) for b in r["boxes"]} == _expected_middle_half_boxes()
    assert any("recorded train-time resize" in m for m in caplog.messages)


# --- the doors on either side of the tier -------------------------------


def _native_frame_checkpoint(tmp_path: Path, augmentation: dict | str | None = None) -> str:
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims

    model_source = {"builder": BESPOKE_DETECTION,
                    "builder_kwargs": {"min_size": TILE, "max_size": TILE * 2},
                    "task": "detection"}
    config: dict = {"model_source": model_source,
                    "data": {"tiling": {"enabled": False}, "train_native_size": [TILE, TILE],
                             **_RUN_DATA}}
    if augmentation is not None:
        config["augmentation"] = augmentation
    ckpt = tmp_path / "model_best.pt"
    torch.save({STATE_DICT_KEY: build_model(config, recorded_model_dims(config)).state_dict(),
                CONFIG_KEY: config}, str(ckpt))
    return str(ckpt)


def _registered(tmp_path: Path, ckpt: str, name: str) -> str:
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(name=name, checkpoint_path=ckpt, config={}, project=tmp_path)
    assert "error" not in result, result
    return ckpt


def test_a_tiled_pass_over_a_native_frame_checkpoint_says_what_it_rests_on(tmp_path):
    """The rail admits the work: a caller who asks to tile a checkpoint whose only geometry is its
    untiled training frame gets a real pass at that frame's edge, its record naming that basis and
    the recorded resize each tile runs through."""
    from tests._verified_checkpoint_fixtures import predicted_over

    ckpt = _registered(tmp_path, _native_frame_checkpoint(tmp_path, {"resize": [32, 32]}),
                       "native-frame-tiles")

    p, results = predicted_over(tmp_path, ckpt, str(Path(gray_frame(tmp_path, IMAGE)).parent),
                                device="cpu", tile=True, conf=0.0)

    assert len(results) == 1
    assert (p.execution.tile_size, p.execution.sources["tile_size"]) == (TILE, "native_ratio")
    assert p.execution.tile_resize == (32, 32)


def test_a_native_frame_checkpoint_stays_untiled_unless_asked(tmp_path):
    """``tile`` unset follows the checkpoint's own regime, and an untiled-trained checkpoint's
    regime is untiled: the tier is a capability a caller opts into, never a silent upgrade."""
    from tests._verified_checkpoint_fixtures import predicted_over

    ckpt = _registered(tmp_path, _native_frame_checkpoint(tmp_path), "native-frame-untiled")

    p, _results = predicted_over(tmp_path, ckpt, str(Path(gray_frame(tmp_path, IMAGE)).parent),
                                 device="cpu", conf=0.0)

    assert p.execution.tile_size is None


def test_an_unreadable_recorded_augmentation_config_does_not_sink_an_untiled_run(tmp_path):
    """The recorded config is only consulted to reproduce a training input geometry, which an
    untiled run never does; a run that reads no tile geometry is not refused over it."""
    from tests._verified_checkpoint_fixtures import predicted_over

    ckpt = _registered(tmp_path, _native_frame_checkpoint(tmp_path, {"not_a_transform": 0.5}),
                       "native-frame-unreadable-aug")

    p, results = predicted_over(tmp_path, ckpt, str(Path(gray_frame(tmp_path, IMAGE)).parent),
                                device="cpu", conf=0.0)

    assert p.execution.tile_size is None and len(results) == 1


def _native_frame_gt(images_dir: Path) -> None:
    """A single 128x128 image, ground truth at exactly the middle half of every tile on a gapless
    2x2 TILE-edge lattice: what ``_MiddleHalfDetector`` reports whatever intermediate resize a
    tile is run through, so a perfect-match reference isolates the geometry reproduction this
    admits-valid-work proof is about from any unrelated matching noise.
    """
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import label_image

    Image.new("RGB", (IMAGE, IMAGE), (120, 120, 120)).save(images_dir / "a.png")
    label_image(
        images_dir / "a.png",
        [Annotation(subject="bud", geometry=BBox(*b))
         for b in sorted(_expected_middle_half_boxes())],
        IMAGE, IMAGE)


def _persisted_regime_predictor():
    p = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))
    p.train_tile_size, p.train_overlap = TILE, 0.0
    p.train_native_size, p.train_augmentation = None, None
    return p


def _native_frame_regime_predictor():
    """Only the checkpoint's own uniform untiled training frame, with a recorded chain that pins a
    resize to a different edge than the tile itself, so the reproduction below exercises the
    resize/rescale round trip rather than comparing two identical no-op calls."""
    p = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))
    p.train_tile_size, p.train_overlap = None, 0.0
    p.train_native_size = [TILE, TILE]
    p.train_augmentation = {"resize": [TILE * 2, TILE * 2]}
    return p


def test_delivery_grade_evaluation_admits_a_native_frame_basis_and_reproduces_the_persisted_one(
        tmp_path, monkeypatch):
    """A checkpoint whose only tiling basis is its own
    uniform untiled training frame reaches the delivery-grade gate and produces the identical
    counts, metrics and box coordinates a persisted tiled regime already trusted would, even though
    its recorded augmentation chain pins a real resize the native-frame regime alone must run each
    tile through and undo, so the two runs are not merely two identical no-resize calls."""
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    from tcip_mcp.pipelines.execution import Stated, tiled_execution, untiled_execution
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _native_frame_gt(images_dir)
    checkpoint = verified_checkpoint(tmp_path)

    monkeypatch.setattr(predictor_mod, "GenericPredictor",
                        lambda *a, **kw: _persisted_regime_predictor())
    persisted = run_full_frame_evaluation(
        checkpoint, checkpoint_admission(checkpoint, images_dir), stated=Stated())
    monkeypatch.setattr(predictor_mod, "GenericPredictor",
                        lambda *a, **kw: _native_frame_regime_predictor())
    native = run_full_frame_evaluation(
        checkpoint, checkpoint_admission(checkpoint, images_dir), stated=Stated())
    monkeypatch.undo()

    persisted_execution, native_execution = persisted["execution"], native["execution"]
    assert persisted_execution["sources"]["tile_size"] == "derived"
    assert native_execution["sources"]["tile_size"] == "native_ratio"
    assert native_execution["tile_size"] == persisted_execution["tile_size"] == TILE
    assert persisted["tp"] == 4 and persisted["fp"] == 0 and persisted["fn"] == 0
    for key in ("tp", "fp", "fn", "n_gt", "n_pred", "precision", "recall", "f1", "map", "map50"):
        assert native[key] == persisted[key], key

    # The metrics alone cannot distinguish "byte-identical boxes" from "close enough to still
    # match": run the exact geometry each regime resolved directly and compare coordinates.
    persisted_predictor, native_predictor = (
        _persisted_regime_predictor(), _native_frame_regime_predictor())
    p_geo = resolve_tile_geometry(persisted_predictor, tiled=True, tile_size=None, overlap=None)
    n_geo = resolve_tile_geometry(native_predictor, tiled=True, tile_size=None, overlap=None)
    assert n_geo.tile_resize == (TILE * 2, TILE * 2)
    base = untiled_execution(checkpoint, conf=0.0, max_dets=None)
    r_p, r_n = (predictor.predict_sliced(
        str(images_dir / "a.png"),
        execution=tiled_execution(base, geo, postprocess=None, cross_tile_nms=None),
        tile_batch_size=8, require_masks=False)
        for predictor, geo in ((persisted_predictor, p_geo), (native_predictor, n_geo)))
    assert ({tuple(b) for b in r_p["boxes"]} == {tuple(b) for b in r_n["boxes"]}
            == _expected_middle_half_boxes())


def test_delivery_grade_evaluation_forwards_the_native_frame_resize_into_predict_sliced(
        tmp_path, monkeypatch):
    """A native-frame checkpoint whose recorded chain pins a resize reaches ``predict_sliced``
    with it, so the evaluation door never silently runs each tile at its own native size."""
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _native_frame_gt(images_dir)

    captured: dict = {}

    def _spy_predictor(*a, **kw):
        p = _native_frame_regime_predictor()
        real_predict_sliced = p.predict_sliced

        def _spy(*a, **kwargs):
            captured.update(kwargs)
            return real_predict_sliced(*a, **kwargs)

        p.predict_sliced = _spy
        return p

    checkpoint = verified_checkpoint(tmp_path)
    monkeypatch.setattr(predictor_mod, "GenericPredictor", _spy_predictor)
    r = run_full_frame_evaluation(
        checkpoint, checkpoint_admission(checkpoint, images_dir), stated=Stated())

    assert "error" not in r
    assert captured["execution"].tile_resize == (TILE * 2, TILE * 2)


# --- the contradiction refusal -------------------------------------------


def test_an_explicit_edge_contradicting_persisted_geometry_refuses():
    from tcip_mcp.pipelines.slicing import TileEdgeContradictionError, resolve_tile_geometry

    stub = _GeometryStub(train_tile_size=128)

    with pytest.raises(TileEdgeContradictionError) as exc_info:
        resolve_tile_geometry(stub, tiled=True, tile_size=64, overlap=None)
    assert str(exc_info.value) == (
        "stated tile_size 64 contradicts this checkpoint's own persisted training tile geometry "
        "of 128. Pass tile_size 128 to match the checkpoint, or leave tile_size unset to derive "
        "it from the checkpoint."
    )


def test_an_explicit_edge_contradicting_the_native_frame_refuses():
    from tcip_mcp.pipelines.slicing import TileEdgeContradictionError, resolve_tile_geometry

    stub = _GeometryStub(train_native_size=[512, 512])

    with pytest.raises(TileEdgeContradictionError) as exc_info:
        resolve_tile_geometry(stub, tiled=True, tile_size=64, overlap=None)
    assert str(exc_info.value) == (
        "stated tile_size 64 contradicts this checkpoint's own recorded untiled training frame "
        "of 512. Pass tile_size 512 to match the checkpoint, or leave tile_size unset to derive "
        "it from the checkpoint."
    )


def test_an_untiled_call_with_a_contradicting_stated_edge_is_inert():
    """The edge never governs a count when the run doesn't tile, so it is never checked."""
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    stub = _GeometryStub(train_tile_size=128)

    g = resolve_tile_geometry(stub, tiled=False, tile_size=64, overlap=None)

    assert (g.tile_size, g.tile_size_source, g.tile_size_derived_from) == (64, "explicit", None)


def test_an_explicit_edge_equal_to_persisted_geometry_clears():
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    stub = _GeometryStub(train_tile_size=128)

    g = resolve_tile_geometry(stub, tiled=True, tile_size=128, overlap=None)

    assert (g.tile_size, g.tile_size_source) == (128, "explicit")
    assert g.tile_size_derived_from == "equal to the checkpoint's persisted training tile geometry"


def test_an_explicit_edge_equal_to_the_native_frame_clears():
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    stub = _GeometryStub(train_native_size=[512, 512])

    g = resolve_tile_geometry(stub, tiled=True, tile_size=512, overlap=None)

    assert (g.tile_size, g.tile_size_source) == (512, "explicit")
    assert "recorded untiled training frame" in g.tile_size_derived_from


def test_an_explicit_edge_on_a_checkpoint_recording_no_geometry_clears():
    """The foreign-checkpoint case: nothing to contradict, so any stated edge stands."""
    from tcip_mcp.pipelines.slicing import resolve_tile_geometry

    stub = _GeometryStub()

    g = resolve_tile_geometry(stub, tiled=True, tile_size=64, overlap=None)

    assert (g.tile_size, g.tile_size_source) == (64, "explicit")
    assert g.tile_size_derived_from == "stated on a checkpoint that records no tile geometry"


def _tiled_checkpoint(tmp_path: Path, tile_size: int) -> str:
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims

    model_source = {"builder": BESPOKE_DETECTION,
                    "builder_kwargs": {"min_size": tile_size, "max_size": tile_size * 2},
                    "task": "detection"}
    config = {"model_source": model_source,
              "data": {"tiling": {"tile_size": tile_size, "overlap": 0.2}, **_RUN_DATA}}
    ckpt = tmp_path / "model_tiled.pt"
    torch.save({STATE_DICT_KEY: build_model(config, recorded_model_dims(config)).state_dict(),
                CONFIG_KEY: config}, str(ckpt))
    return str(ckpt)


def _native_frame_checkpoint_of_size(tmp_path: Path, size: int) -> str:
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims

    model_source = {"builder": BESPOKE_DETECTION,
                    "builder_kwargs": {"min_size": size, "max_size": size * 2},
                    "task": "detection"}
    config = {"model_source": model_source,
              "data": {"tiling": {"enabled": False}, "train_native_size": [size, size],
                       **_RUN_DATA}}
    ckpt = tmp_path / "model_native.pt"
    torch.save({STATE_DICT_KEY: build_model(config, recorded_model_dims(config)).state_dict(),
                CONFIG_KEY: config}, str(ckpt))
    return str(ckpt)


@pytest.mark.parametrize("make, recorded", [
    (lambda tmp_path: _tiled_checkpoint(tmp_path, 128), "128"),
    (lambda tmp_path: _native_frame_checkpoint_of_size(tmp_path, 512), "512"),
], ids=["persisted-geometry", "native-frame"])
def test_a_stated_edge_contradicting_the_checkpoints_geometry_refuses_the_pass(
        tmp_path, make, recorded):
    """A caller-typed tile edge that differs from the checkpoint's own persisted training geometry,
    or from its recorded untiled training frame when it persists none, is a real contradiction,
    never a caller override to trust blindly."""
    from tcip_mcp.pipelines.execution import ExecutionRefusedError
    from tests._verified_checkpoint_fixtures import predicted_over

    ckpt = _registered(tmp_path, make(tmp_path), f"contradiction-{recorded}")

    with pytest.raises(ExecutionRefusedError) as exc_info:
        predicted_over(tmp_path, ckpt, str(Path(gray_frame(tmp_path, IMAGE)).parent), device="cpu",
                       tile=True, tile_size=64, conf=0.0)
    assert "64" in str(exc_info.value) and recorded in str(exc_info.value)


def test_a_stated_edge_matching_persisted_geometry_is_admitted_as_stated(tmp_path):
    """The rail refuses a contradiction, not an explicit edge that simply agrees."""
    from tests._verified_checkpoint_fixtures import predicted_over

    ckpt = _registered(tmp_path, _tiled_checkpoint(tmp_path, TILE), "tiled-native-edge-match")

    p, results = predicted_over(tmp_path, ckpt, str(Path(gray_frame(tmp_path, IMAGE)).parent),
                                device="cpu", tile=True, tile_size=TILE, conf=0.0)

    assert len(results) == 1
    assert (p.execution.tile_size, p.execution.sources["tile_size"]) == (TILE, "explicit")
