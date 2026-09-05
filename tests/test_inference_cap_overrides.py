"""The NMS/detection-cap parameters of the public inference tools: unset means derive.

``run_inference``/``deliver_per_image_counts`` decide whether the caller stated a cap
by the ``None`` sentinel, so a stated value is honored as an override at every value it can take,
including the one the platform would otherwise have fallen back to. The resolver stays the only
thing that derives an unstated cap.
"""

from __future__ import annotations

import inspect

import pytest

OVERLAPPING_GT = [(0.0, 0.0, 20.0, 20.0), (10.0, 0.0, 30.0, 20.0), (60.0, 60.0, 80.0, 80.0)]


class _StubPredictor:
    """Stands in for a built predictor: carries no training tile geometry and returns no boxes."""

    def __init__(self) -> None:
        self.train_tile_size = None
        self.train_overlap = None
        self.train_native_size = None
        self.in_chans = 3
        self.max_dets = None
        self.score_threshold = 0.5
        self.model = object()
        self.config = {"data": {"subject": "bud", "id_map": {"bud": 1}}}

    def predict_batch(self, image_paths, **kwargs):
        return [{"boxes": [], "scores": [], "labels": [], "count": 0, "width": 100, "height": 100}
                for _ in image_paths]


def _calibration_records() -> list[dict]:
    """Two images of overlapping ground truth, enough for a real cross-tile NMS derivation."""
    from tcip_mcp.pipelines.training.evaluation import build_coco_image_record

    records = []
    for stem in ("a", "b"):
        gt = [{"category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0}
              for (x1, y1, x2, y2) in OVERLAPPING_GT]
        records.append(build_coco_image_record(100, 100, gt, [], image_id=stem))
    return records


@pytest.fixture
def inference_call(tmp_path, monkeypatch):
    """Call the verified pass behind ``run_inference`` with the model build stubbed, the real
    resolver behind calibration."""
    from tcip_mcp import model_registry as model_registry_module
    from tcip_mcp.pipelines import calibration as calibration_pipeline
    from tcip_mcp.pipelines.inference import predictor as predictor_module

    from tests._verified_checkpoint_fixtures import run_inference_verified, stub_verified_checkpoint

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "a.png"
    image_path.write_bytes(b"")
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    built: dict = {}
    forwarded: dict = {}

    def _stub_build_predictor(ckpt, **kwargs):
        built.update(kwargs)
        return _StubPredictor()

    def _spy_calibrate(predictor, trait, labels, images, **kwargs):
        from tcip_mcp.pipelines.operating_point import resolve_operating_point

        forwarded.update(kwargs)
        inputs = {
            "dataset_hash": "d", "calibration_records": _calibration_records(),
            "tiled": kwargs["tile"], "tile_size": kwargs["tile_size"],
            "cross_tile_nms": kwargs["cross_tile_nms"], "max_dets": kwargs["max_dets"],
        }
        bundle = resolve_operating_point(trait, **inputs)
        evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                    "reference_inputs": {"label_dirs": {"calibration": labels}}}
        return bundle, "d", 0, evidence

    monkeypatch.setattr(predictor_module, "build_predictor", _stub_build_predictor)
    monkeypatch.setattr(
        model_registry_module, "load_registered_checkpoint",
        lambda *a, **kw: stub_verified_checkpoint(str(checkpoint)))
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point", _spy_calibrate)

    def _call(**kwargs):
        result = run_inference_verified(
            str(checkpoint), image_paths=[str(image_path)], device="cpu",
            experiment_id="run-1", **kwargs)
        assert "error" not in result, result
        return result

    _call.built = built
    _call.forwarded = forwarded
    _call.labels_dir = str(labels_dir)
    return _call


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_a_cap_stated_at_the_platform_default_resolves_as_an_explicit_override(inference_call):
    """The value a caller states is the value that ships, whatever it happens to equal: reading a
    stated cap back as unset would hand the caller a derived operating point they did not ask for,
    and therefore a different count."""
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS, DEFAULT_NMS_IOU

    result = inference_call(trait="bud_opening", calibration_labels_dir=inference_call.labels_dir,
                            global_nms_iou=DEFAULT_NMS_IOU, max_dets=DEFAULT_MAX_DETS)

    op = result["operating_point"]
    assert inference_call.forwarded["cross_tile_nms"] == DEFAULT_NMS_IOU
    assert inference_call.forwarded["max_dets"] == DEFAULT_MAX_DETS
    assert op["cross_tile_nms"]["source"] == "explicit"
    assert op["cross_tile_nms"]["value"] == DEFAULT_NMS_IOU
    assert op["max_dets"]["source"] == "explicit"
    assert op["max_dets"]["value"] == DEFAULT_MAX_DETS


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_an_unstated_cap_is_still_derived_from_the_calibration_data(inference_call):
    """The derivation an unstated cap gets is the whole point of leaving it unstated."""
    result = inference_call(trait="bud_opening", calibration_labels_dir=inference_call.labels_dir)

    op = result["operating_point"]
    assert inference_call.forwarded["cross_tile_nms"] is None
    assert inference_call.forwarded["max_dets"] is None
    assert op["cross_tile_nms"]["source"] == "derived"
    assert "neighbor-IoU" in op["cross_tile_nms"]["derived_from"]
    assert op["max_dets"]["source"] == "derived"
    assert "p99 GT objects/image" in op["max_dets"]["derived_from"]


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_an_ordinary_stated_cap_still_lands_as_the_caller_stated_it(inference_call):
    """The common override path must keep working, not only the value that equals the default."""
    result = inference_call(trait="bud_opening", calibration_labels_dir=inference_call.labels_dir,
                            global_nms_iou=0.42, max_dets=250)

    op = result["operating_point"]
    assert op["cross_tile_nms"]["source"] == "explicit"
    assert op["cross_tile_nms"]["value"] == 0.42
    assert op["max_dets"]["source"] == "explicit"
    assert op["max_dets"]["value"] == 250


def test_an_unstated_cap_runs_the_pass_at_the_shared_platform_default(inference_call):
    """Unstated is not uncapped: the pass itself still runs at the shared default, so an
    uncalibrated run's count is unchanged by the sentinel."""
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS, DEFAULT_NMS_IOU

    result = inference_call()

    assert inference_call.built["nms_iou"] == DEFAULT_NMS_IOU
    assert inference_call.built["max_dets"] == DEFAULT_MAX_DETS
    op = result["operating_point"]
    assert op["max_dets"]["value"] == DEFAULT_MAX_DETS
    assert op["cross_tile_nms"]["value"] is None  # untiled: cross-tile NMS is not operative


def test_a_stated_cap_governs_the_pass_the_predictor_actually_runs(inference_call):
    """A stated cap has to reach the model, not only the provenance record."""
    inference_call(global_nms_iou=0.55, max_dets=77)

    assert inference_call.built["nms_iou"] == 0.55
    assert inference_call.built["max_dets"] == 77


def test_an_unstated_cap_on_the_raw_path_stamps_the_default_source(inference_call):
    """The raw (uncalibrated) path never resolves through resolve_operating_point, so it has its
    own source-stamping to get right: an unstated max_dets/conf is stamped 'default', and the pass
    itself still runs at the platform default conf."""
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF

    result = inference_call()

    op = result["operating_point"]
    assert op["max_dets"]["source"] == "default"
    assert op["conf"]["source"] == "default"
    assert inference_call.built["score_threshold"] == DEFAULT_CONF


def test_a_stated_cap_on_the_raw_path_stamps_explicit_even_at_the_platform_default(
    inference_call,
):
    """The rail this row exists for: a caller-stated cap that happens to equal the platform
    default is stamped 'explicit' on the raw path too, never laundered into 'default'."""
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_MAX_DETS

    result = inference_call(max_dets=DEFAULT_MAX_DETS, conf_threshold=DEFAULT_CONF)

    op = result["operating_point"]
    assert op["max_dets"]["source"] == "explicit"
    assert op["max_dets"]["value"] == DEFAULT_MAX_DETS
    assert op["conf"]["source"] == "explicit"
    assert op["conf"]["value"] == DEFAULT_CONF


def test_the_public_inference_tools_agree_that_an_unstated_cap_is_none():
    """One sentinel across the door: a concrete default on any one of them would erase the stated
    versus unstated distinction for every caller of that door."""
    from tcip_mcp.tools.inference_tools import run_inference, deliver_per_image_counts

    for tool in (run_inference, deliver_per_image_counts):
        params = inspect.signature(tool).parameters
        assert params["global_nms_iou"].default is None, tool.__name__
        assert params["max_dets"].default is None, tool.__name__


def test_the_dry_run_report_shows_the_applied_conf_never_the_raw_none(tmp_path):
    """The dry-run report has to show the value the pass will actually run at: an omitted conf is
    None on the wire, and reporting that bare None would tell a caller nothing about what their
    run would do."""
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF
    from tcip_mcp.tools.inference_tools import run_inference

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")

    result = run_inference(
        checkpoint_path=str(checkpoint), output_dir=str(tmp_path / "out"), dry_run=True)

    assert result["operating_point"]["conf"] == DEFAULT_CONF


def test_the_raster_export_path_receives_an_unstated_cap_unstated(tmp_path, monkeypatch):
    """The raster regime is reached through the same door, so the sentinel has to survive the hop
    rather than being resolved to a number on the way."""
    from tcip_mcp import model_registry as model_registry_module
    from tcip_mcp.tools import inference_tools

    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    raster = tmp_path / "mosaic.tif"
    raster.write_bytes(b"")

    forwarded: dict = {}

    def _spy_raster(**kwargs):
        forwarded.update(kwargs)
        return {"image_count": 1}

    monkeypatch.setattr(inference_tools, "_export_predictions_raster", _spy_raster)
    monkeypatch.setattr(
        model_registry_module, "load_registered_checkpoint",
        lambda *a, **kw: stub_verified_checkpoint(str(checkpoint)))

    inference_tools.run_inference(
        checkpoint_path=str(checkpoint), raster_path=str(raster),
        output_dir=str(tmp_path / "out"))

    assert forwarded["global_nms_iou"] is None
    assert forwarded["max_dets"] is None
