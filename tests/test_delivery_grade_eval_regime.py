"""``evaluate_model`` and the full-frame runner: the density each regime caps its frames at, a
saturated cap reported, the execution record each run records, and the gate's refusals."""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._producer_fixtures import checkpoint_admission, seed_bud_images

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tests._predictor_fixtures import StubPredictor, install  # noqa: E402
from tests._verified_checkpoint_fixtures import SAMPLE_CONF, SAMPLE_CROSS_TILE_NMS  # noqa: E402

CONF_AND_MERGE = {"conf": SAMPLE_CONF, "cross_tile_nms": SAMPLE_CROSS_TILE_NMS}
"""The conf and merge threshold a tiled pass over these single-object references states."""

# seed_bud_trait_spec (conftest.py) confirms bud_opening in this test's project, so the
# trait/subject="bud" call sites resolve.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def test_gating_path_runs_at_the_density_the_evaluated_reference_derives(tmp_path):
    """The door's own full-frame pass, run to the runner's own answer: its density is the one the
    evaluated reference's frames derive, never one the caller states or the checkpoint's."""
    from tcip_mcp.pipelines.derivations import OBJECT_DENSITY_DERIVATION
    from tcip_mcp.tools.training_tools import evaluate_model

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    r = evaluate_model(tmp_path, str(ckpt), str(images_dir), use_tiled_inference=True,
                       stated=Stated(tile_size=128, overlap=0.0, **CONF_AND_MERGE))
    assert "error" not in r, r
    # One bud in each 128 px reference frame; the checkpoint trained on one in each 64 x 48.
    assert r["execution"]["sources"]["density"] == OBJECT_DENSITY_DERIVATION
    assert r["execution"]["density"] == pytest.approx(1 / (128 * 128))
    assert r["execution"]["density"] != pytest.approx(1 / (64 * 48))


def _detector(monkeypatch, *, in_chans: int = 3, **answer) -> StubPredictor:
    """Make every predictor the evaluation builds one three-band (or ``in_chans``-band) detector
    trained at 100px tiles answering ``answer`` (:class:`StubPredictor`'s), on a 128px frame
    holding nothing unless ``answer`` says otherwise; the detector."""
    return install(monkeypatch, StubPredictor(
        task="detection", in_chans=in_chans, train_tile_size=100, train_overlap=0.2,
        **{"width": 128, "height": 128, "boxes": (), "scores": (), **answer}))


def _captured_execution(monkeypatch) -> dict:
    """Stand in for the diagnostic runner, recording under ``"execution"`` the execution record
    of the pass it is handed."""
    import tcip_mcp.pipelines.training.eval_runners as runners

    captured: dict = {}

    def _fake(pass_, loader, device, **kw):
        captured["execution"] = pass_.execution
        return {"eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)
    return captured


def test_diagnostic_path_runs_at_the_checkpoints_own_density(tmp_path, monkeypatch):
    """The diagnostic regime holds no reference pass: its record carries the density the
    checkpoint recorded of its training frames."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.tools.training_tools import evaluate_model

    captured = _captured_execution(monkeypatch)
    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    evaluate_model(tmp_path, str(ckpt), str(images_dir), stated=Stated(conf=SAMPLE_CONF))
    density = load_registered_checkpoint(ckpt, project=tmp_path).spec.data.train_object_density
    assert (captured["execution"].density, captured["execution"].sources["density"]) == (
        density, "derived")


def test_bare_checkpoint_path_reuses_its_own_stamped_tiling_and_subject(tmp_path, monkeypatch):
    """A checkpoint path (not a run id) carries its own stamped config["data"]: evaluate_model
    reads its tiling and its class space from it, the one record either comes from."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    captured: dict = {}

    def _fake(ckpt, images_dir, **kw):
        captured.update(kw, checkpoint=ckpt)
        return {"eval_regime": "full-frame-tiled-inference"}

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _fake)
    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    from tests._verified_checkpoint_fixtures import SCOPED_DATA, registered_checkpoint

    ckpt = registered_checkpoint(
        tmp_path, data={**SCOPED_DATA, "tiling": {"tile_size": 384, "overlap": 0.15,
                                                  "sliver_frac": 0.5}})

    evaluate_model(tmp_path, str(ckpt), str(images_dir), use_tiled_inference=True)
    # The measurement is handed the checkpoint whose own record states its class space and its
    # tiling, and the door states no geometry of its own beside it.
    assert captured["checkpoint"].spec.data.recorded_scope.subject == "bud"
    assert captured["stated"].tile_size is None and captured["stated"].overlap is None

    from tcip_mcp.pipelines.execution import prepare

    record = prepare(captured["checkpoint"], Stated(
        tile=True, **CONF_AND_MERGE)).runnable().execution
    assert (record.tile_size, record.overlap) == (384, 0.15)


def test_gate_translates_geometry_refusal_to_error_dict(tmp_path, monkeypatch):
    """evaluate_model is an @mcp.tool() surface that returns {"error": ...} for every other
    failure: a bare raise from run_full_frame_evaluation would surface as an MCP exception
    instead, inconsistent with the rest of this tool's contract."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.tools.training_tools import evaluate_model

    def _refuse(*a, **kw):
        raise ValueError("Cannot resolve a trustworthy tile_size for ckpt.pt: ... tiling=")

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _refuse)
    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    r = evaluate_model(tmp_path, str(ckpt), str(images_dir), use_tiled_inference=True)
    assert "error" in r
    assert "tiling=" in r["error"]


def test_gate_translates_unreadable_label_to_error_dict(tmp_path, monkeypatch):
    """A present, unreadable label document raised out of run_full_frame_evaluation is this
    tool's own {"error": ...} shape too, not a raise through the MCP boundary."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_annotation.json_io import UnreadableLabelDocumentError
    from tcip_mcp.tools.training_tools import evaluate_model

    def _refuse(*a, **kw):
        raise UnreadableLabelDocumentError("label document 2026-03-02/IMG_0001 does not decode")

    monkeypatch.setattr(runners, "run_full_frame_evaluation", _refuse)
    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    r = evaluate_model(tmp_path, str(ckpt), str(images_dir), use_tiled_inference=True)
    assert "error" in r
    assert "2026-03-02/IMG_0001" in r["error"]


@pytest.mark.parametrize("use_tiled_inference", [True, False])
def test_both_regimes_refuse_a_stray_labels_dir_by_name(tmp_path, use_tiled_inference):
    """A labels_dir holding no document for the images is refused naming it on either regime,
    never dropped by one of them in favor of the images' own documents."""
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)
    stray = tmp_path / "stray_labels"
    stray.mkdir()
    ckpt = registered_checkpoint(tmp_path)

    r = evaluate_model(tmp_path, str(ckpt), str(images_dir), labels_dir=str(stray),
                       use_tiled_inference=use_tiled_inference)

    assert "stray_labels" in r["error"] and "admits nothing to evaluate" in r["error"], r


def test_a_frame_that_hits_its_cap_is_reported_saturated(tmp_path, monkeypatch):
    """A cap that actually binds on real detections must be visible in the result, not silently
    assumed safe: a frame whose count reaches the cap the pass gave it counts toward the
    saturated fraction."""
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET, n=1, size=200,
                                 box=(10, 10, 30, 30))

    checkpoint = verified_checkpoint(tmp_path)
    # Five detections kept at a cap of five, as predict_sliced writes them.
    _detector(monkeypatch, width=200, height=200, cap=5,
              boxes=((10, 10, 30, 30), (50, 50, 70, 70), (90, 90, 110, 110),
                     (130, 130, 150, 150), (170, 170, 190, 190)),
              scores=(0.9, 0.8, 0.7, 0.6, 0.5))
    r = run_full_frame_evaluation(checkpoint, checkpoint_admission(checkpoint, images_dir),
                                  stated=Stated(**CONF_AND_MERGE))
    assert r["cap_saturated_frac"] == 1.0  # the one image hit the cap, now visible


def test_the_gate_reads_its_references_at_the_predictors_own_width(tmp_path, monkeypatch):
    """The delivery gate builds its loader at the width the predictor reads at, and never derives
    one: this gate scores source paths through the predictor and reads only targets off the
    loader, so references of differing band counts are a measurement it can take."""
    import numpy as np
    import tifffile
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._producer_fixtures import label_image
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET)  # three-band sources
    array = np.zeros((128, 128, 5), dtype=np.uint8)
    tifffile.imwrite(str(images_dir / "five_band.tif"), array)  # and one of five bands
    label_image(images_dir / "five_band.tif",
                [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)

    _detector(monkeypatch, in_chans=1, boxes=((10, 10, 40, 40),), scores=(0.9,))
    checkpoint = verified_checkpoint(tmp_path)
    r = run_full_frame_evaluation(checkpoint, checkpoint_admission(checkpoint, images_dir),
                                  stated=Stated(**CONF_AND_MERGE))

    assert r["scored_images"] == 4
    assert r["tp"] == 4


def test_run_full_frame_evaluation_records_merge_and_execution(tmp_path, monkeypatch):
    """The runner's record carries the execution record the pass ran under: its conf and merge
    threshold each the stated one, recorded ``explicit``, the merge threshold the one the
    predictor merged at."""
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET, n=1, size=128,
                                 box=(10, 10, 30, 30))
    checkpoint = verified_checkpoint(tmp_path)
    detector = _detector(monkeypatch)
    r = run_full_frame_evaluation(checkpoint, checkpoint_admission(checkpoint, images_dir),
                                  stated=Stated(**CONF_AND_MERGE))

    assert [execution.cross_tile_nms for execution in detector.executions] == [
        SAMPLE_CROSS_TILE_NMS]
    assert r["execution"]["postprocess"] == "nms"
    assert (r["execution"]["conf"], r["execution"]["cross_tile_nms"]) == (
        SAMPLE_CONF, SAMPLE_CROSS_TILE_NMS)
    assert {r["execution"]["sources"][name] for name in ("conf", "cross_tile_nms")} == {
        "explicit"}


def test_the_gate_refuses_documents_whose_geometry_a_detector_cannot_read(tmp_path, monkeypatch):
    """The delivery gate measures over the loader a run would build, so ground truth carrying
    the subject only as points refuses in that loader's own words rather than scoring every
    image against an empty reference and reporting the number as a delivery metric."""
    from tcip_annotation.state import Annotation, Point

    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._producer_fixtures import seed_labeled_images
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = seed_labeled_images(
        tmp_path / "images" / UNDATED_BUCKET,
        [Annotation(subject="bud", geometry=Point(20.0, 30.0))], n=8, width=128, height=128)
    checkpoint = verified_checkpoint(tmp_path)
    _detector(monkeypatch)
    with pytest.raises(ValueError, match="only in geometries a detection loader"):
        run_full_frame_evaluation(checkpoint, checkpoint_admission(checkpoint, images_dir),
                                  stated=Stated())

    # Admits valid work: the same eight images, their documents carrying boxes, score.
    seed_bud_images(images_dir, n=8, size=128, box=(10, 10, 30, 30))
    scored = run_full_frame_evaluation(checkpoint, checkpoint_admission(checkpoint, images_dir),
                                       stated=Stated(**CONF_AND_MERGE))
    assert scored["scored_images"] == 8


def test_the_gate_refuses_an_images_tree_with_no_ground_truth(tmp_path, monkeypatch):
    """A measurement is against a reference: with no label document there is nothing to score
    against, so the gate refuses by name rather than scoring every image against empty ground
    truth and reporting a perfect-looking miss rate."""
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation
    from tests._producer_fixtures import blank_image
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    images_dir = blank_image(tmp_path, "a.png", (128, 128)).parent
    checkpoint = verified_checkpoint(tmp_path)
    _detector(monkeypatch)
    with pytest.raises(ValueError, match="no trainable samples"):
        run_full_frame_evaluation(checkpoint, checkpoint_admission(checkpoint, images_dir),
                                  stated=Stated())
