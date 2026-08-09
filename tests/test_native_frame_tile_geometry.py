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

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor  # noqa: E402

TILE = 64
IMAGE = 128


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

    def __init__(self, min_size: int, max_size: int) -> None:
        super().__init__()
        from torchvision.models.detection.transform import GeneralizedRCNNTransform

        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

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


class _MiddleHalfMaskDetector(_MiddleHalfDetector):
    """The same, plus one soft mask per detection at the size of the tensor handed in, the shape a
    torchvision Mask R-CNN's own ``masks`` come back at."""

    def forward(self, images):
        results = super().forward(images)
        for im, res in zip(images, results):
            h, w = int(im.shape[-2]), int(im.shape[-1])
            res["masks"] = torch.full((1, 1, h, w), 0.75)
        return results


def _geometry(stub, *, tile_size=None, overlap=None) -> tuple:
    from tcip_mcp.pipelines.inference.predictor import resolve_tile_geometry

    edge, source, _, _ = resolve_tile_geometry(stub, tile_size=tile_size, overlap=overlap)
    return edge, source


def _stub_predictor(model, *, task: str = "detection") -> GenericPredictor:
    p = GenericPredictor.__new__(GenericPredictor)
    p.task = task
    p.score_threshold = 0.0
    p.max_dets = None
    p.in_chans = 3
    p.device = torch.device("cpu")
    p.model_source = {}
    p.model = model.eval()
    return p


def _image(tmp_path: Path, size: int = IMAGE) -> str:
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (size, size), (120, 120, 120)).save(p)
    return str(p)


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
    stub = _GeometryStub(train_native_size=[512, 512])

    assert _geometry(stub, tile_size=320) == (320, "explicit")


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
    ``get_augmentation_preset``, at the same default size every production caller builds it with."""
    from tcip_mcp.pipelines.data.augmentations import get_augmentation_preset, recorded_resize

    assert recorded_resize("nadir_rotation") == tuple(
        get_augmentation_preset("nadir_rotation")["resize"])


def test_an_unbuildable_recorded_config_raises_rather_than_reading_as_no_resize():
    from tcip_mcp.pipelines.data.augmentations import recorded_resize

    with pytest.raises(ValueError, match="Unknown augmentation"):
        recorded_resize({"not_a_transform": 0.5})


def test_the_recorded_resize_travels_only_with_a_native_frame_tile_edge():
    """An explicit or persisted-geometry tile edge means the tile as it stands, which is what every
    count already produced at those tiers was measured at."""
    from tcip_mcp.pipelines.inference.predictor import native_ratio_tile_resize

    stub = _GeometryStub(train_native_size=[512, 512], train_augmentation={"resize": [640, 640]})

    assert native_ratio_tile_resize(stub, "native_ratio") == (640, 640)
    assert native_ratio_tile_resize(stub, "derived") is None
    assert native_ratio_tile_resize(stub, "explicit") is None
    assert native_ratio_tile_resize(stub, "unavailable") is None


def test_a_checkpoint_carries_its_untiled_training_geometry_to_the_predictor(tmp_path):
    from tcip_mcp.pipelines.model_build import build_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "min_size": TILE, "max_size": TILE * 2},
                    "task": "detection"}
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source,
                "model_state_dict": build_model({"model_source": model_source}).state_dict(),
                "config": {"data": {"tiling": {"enabled": False}, "train_native_size": [TILE, TILE]},
                           "augmentation": {"resize": [32, 32]}}}, str(ckpt))

    pred = GenericPredictor(str(ckpt), device="cpu", score_threshold=0.0)

    assert pred.train_tile_size is None
    assert pred.train_native_size == [TILE, TILE]
    assert pred.train_augmentation == {"resize": [32, 32]}


# --- the tile pass itself -----------------------------------------------


def test_tiles_at_native_size_with_no_recorded_resize(tmp_path):
    """The common case: the recorded chain pins no size, so the tier is tiling at the native frame
    size and nothing else. Boxes land in image pixel space untouched by any rescale."""
    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    r = pred.predict_tiled(_image(tmp_path), tile_size=TILE, overlap=0.0, tile_resize=None)

    assert {tuple(b) for b in r["boxes"]} == _expected_middle_half_boxes()


def test_a_recorded_resize_is_undone_per_axis_and_not_confused_with_the_detectors_own(tmp_path):
    """``min_size`` (800) is deliberately unlike both the tile edge (64) and the resize target
    (128x96), and the target is deliberately not square. Boxes must land exactly where the
    no-resize pass puts them: the detector's internal transform already maps its own boxes back to
    the tensor it was handed, so only the tier's own stretch is left to undo, and undoing it with
    one scalar factor instead of two would displace every y coordinate."""
    pred = _stub_predictor(_MiddleHalfDetector(min_size=800, max_size=1333))

    r = pred.predict_tiled(_image(tmp_path), tile_size=TILE, overlap=0.0, tile_resize=(128, 96))

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

    r = pred.predict_tiled(_Reader(), tile_size=TILE, overlap=0.0, tile_resize=(128, 96),
                           source_label="raster")

    assert {tuple(round(v, 4) for v in b) for b in r["boxes"]} == _expected_middle_half_boxes()


def test_mask_patches_come_back_at_the_native_tile_size(tmp_path):
    """A mask arrives at the resized tile's size; ``reconstruct_core`` places patches by the tile's
    own origin in native pixels, so a patch left at the resized size would be offset into the wrong
    pixels."""
    pred = _stub_predictor(_MiddleHalfMaskDetector(min_size=800, max_size=1333),
                           task="instance_seg")

    r = pred.predict_tiled(_image(tmp_path), tile_size=TILE, overlap=0.0, tile_resize=(128, 96))

    assert r["masks"], "instance_seg tiled inference carries mask patches by default"
    for m in r["masks"]:
        patch = m["mask_patch"]
        assert (len(patch), len(patch[0])) == (TILE, TILE)


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
        r = pred.predict_tiled(str(path), tile_size=TILE, overlap=0.0, tile_resize=(128, 96))

    assert {tuple(b) for b in r["boxes"]} == _expected_middle_half_boxes()
    assert any("recorded train-time resize" in m for m in caplog.messages)


# --- the doors on either side of the tier -------------------------------


def _native_frame_checkpoint(tmp_path: Path, augmentation: dict | str | None = None) -> str:
    from tcip_mcp.pipelines.model_build import build_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "min_size": TILE, "max_size": TILE * 2},
                    "task": "detection"}
    config: dict = {"data": {"tiling": {"enabled": False}, "train_native_size": [TILE, TILE]}}
    if augmentation is not None:
        config["augmentation"] = augmentation
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source,
                "model_state_dict": build_model({"model_source": model_source}).state_dict(),
                "config": config}, str(ckpt))
    return str(ckpt)


def test_run_inference_tiles_a_native_frame_checkpoint_and_says_what_it_rests_on(tmp_path):
    """The rail admits the work: a caller who asks to tile a checkpoint whose only geometry is its
    untiled training frame gets a real pass at that frame's edge, with the basis named in the
    warning and floored to unvalidated in the provenance."""
    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(_native_frame_checkpoint(tmp_path, {"resize": [32, 32]}),
                      image_paths=[_image(tmp_path)], device="cpu", tile=True, conf_threshold=0.0)

    assert "error" not in r
    assert r["tiled"] is True and len(r["results"]) == 1
    tile_param = r["operating_point"]["tile_size"]
    assert tile_param["value"] == TILE
    assert tile_param["validated_against"] == "false"
    assert "untiled training frame" in r["warning"]
    assert "(32, 32)" in r["warning"]


def test_run_inference_leaves_a_native_frame_checkpoint_untiled_unless_asked(tmp_path):
    """``tile`` unset still derives the checkpoint's own regime, and an untiled-trained checkpoint's
    regime is untiled: the tier is a capability a caller opts into, never a silent upgrade."""
    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(_native_frame_checkpoint(tmp_path), image_paths=[_image(tmp_path)],
                      device="cpu", conf_threshold=0.0)

    assert r["tiled"] is False
    assert r["operating_point"]["tile_size"]["value"] is None


def test_an_unreadable_recorded_augmentation_config_does_not_sink_an_untiled_run(tmp_path):
    """The recorded config is only consulted to reproduce a training input geometry, which an
    untiled run never does; a run that reads no tile geometry must not be refused over it."""
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = _native_frame_checkpoint(tmp_path, {"not_a_transform": 0.5})

    r = run_inference(ckpt, image_paths=[_image(tmp_path)], device="cpu", conf_threshold=0.0)

    assert "error" not in r and r["tiled"] is False and len(r["results"]) == 1


def test_delivery_grade_evaluation_refuses_a_native_frame_basis(tmp_path):
    """The tier is a real basis to tile at, never an independently validated one, so the path whose
    number gates a phenotype delivery refuses it and names the untiled regime as the right gate."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tcip_mcp.pipelines.training.evaluation import run_full_frame_evaluation

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    build = predictor_mod.build_predictor
    try:
        predictor_mod.build_predictor = lambda **kw: _GeometryStub(train_native_size=[TILE, TILE])
        with pytest.raises(ValueError, match="use_tiled_inference=False"):
            run_full_frame_evaluation("ckpt.pt", str(images_dir), str(labels_dir),
                                      str(tmp_path / "out"))
    finally:
        predictor_mod.build_predictor = build
