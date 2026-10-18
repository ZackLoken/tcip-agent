"""The merge-threshold, density and conf values of a pass, stated or derived.

The conf and merge threshold are each the stated one, else the one a handed reference derives,
else a refusal naming it; the density is the one the reference's counted regions
derive, else the checkpoint's own. A stated value is recorded as stated, and the record's conf
and density reach the predictor.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

OVERLAPPING_GT = [
    [0.0, 0.0, 20.0, 20.0],
    [10.0, 0.0, 30.0, 20.0],
    [60.0, 60.0, 80.0, 80.0],
]
"""One image's overlapping ground truth (xyxy), enough for a real cross-tile merge
derivation."""


def _checkpoint(tmp_path: Path):
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import project_checkpoint

    return load_registered_checkpoint(
        project_checkpoint(tmp_path), project=tmp_path
    )


def _pass(tmp_path: Path, reference=None, **stated):
    """Prepare a registered checkpoint of ``tmp_path`` under ``stated`` and return the pass.

    The statement is the sample rationale with ``stated`` merged over it, and the prepared pass
    is made runnable with ``reference``.
    """
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tests._training_values import RATIONALE

    return prepare(
        _checkpoint(tmp_path),
        Stated(**{"rationale": RATIONALE, **stated}),
        device="cpu",
    ).runnable(reference)


def test_stated_values_are_recorded_as_explicit(tmp_path):
    """The execution record keeps each stated value and marks its source explicit."""
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_DETECTOR_PASS,
        SAMPLE_OVERLAP,
    )

    p = _pass(
        tmp_path,
        tile=True,
        tile_size=64,
        overlap=SAMPLE_OVERLAP,
        **SAMPLE_DETECTOR_PASS,
    )

    record = p.execution.record()
    assert {k: record[k] for k in SAMPLE_DETECTOR_PASS} == SAMPLE_DETECTOR_PASS
    assert {record["sources"][k] for k in ("conf", "cross_tile_nms")} == {
        "explicit"
    }


def test_a_reference_derives_the_unstated_merge_threshold_and_the_density_and_keeps_stated_ones(
    tmp_path,
):
    """A reference derives an unstated merge threshold and the density and keeps stated ones.

    The execution record carries the merge threshold the reference's boxes derive and the
    density its counted regions derive in place of the checkpoint's; a stated merge threshold
    stays as stated beside that reference.
    """
    from tcip_mcp.pipelines.derivations import (
        CROSS_TILE_NMS_DERIVATION,
        OBJECT_DENSITY_DERIVATION,
    )
    from tcip_mcp.pipelines.execution import Reference
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF,
        SAMPLE_OVERLAP,
        objects_over,
    )

    frame = 100.0 * 100.0
    reference = Reference(regions=[objects_over(OVERLAPPING_GT, frame)] * 2)
    unstated = _pass(
        tmp_path,
        reference,
        tile=True,
        tile_size=64,
        overlap=SAMPLE_OVERLAP,
        conf=SAMPLE_CONF,
        postprocess="nms",
    )
    stated = _pass(
        tmp_path,
        reference,
        tile=True,
        tile_size=64,
        overlap=SAMPLE_OVERLAP,
        conf=SAMPLE_CONF,
        postprocess="nms",
        cross_tile_nms=0.42,
    )

    assert (
        unstated.execution.sources["cross_tile_nms"]
        == CROSS_TILE_NMS_DERIVATION
    )
    assert (
        unstated.execution.density,
        unstated.execution.sources["density"],
    ) == (pytest.approx(3 / frame), OBJECT_DENSITY_DERIVATION)
    assert (
        unstated.execution.density
        != _checkpoint(tmp_path).geometry.object_density
    )
    assert stated.execution.cross_tile_nms == 0.42
    assert stated.execution.sources["cross_tile_nms"] == "explicit"


def test_a_detector_pass_refuses_an_unstated_conf_by_name(tmp_path):
    """A detector pass refuses an unstated conf by name.

    No default stands behind a detector's conf: a pass with no reference states it, and one
    left unstated refuses naming it.
    """
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    with pytest.raises(ExecutionRefusedError, match="conf"):
        _pass(tmp_path, tile=False)


def _in_model_point(p, tmp_path: Path) -> dict:
    """Return the operating point the pass's model holds after predicting one 64 px frame."""
    from PIL import Image

    from tcip_mcp.pipelines.operating_point import (
        OPERATING_POINT_ATTRS,
        detector_operating_point_holder,
    )

    Image.new("RGB", (64, 64)).save(tmp_path / "frame.png")
    p.predict([str(tmp_path / "frame.png")])
    holder, _path = detector_operating_point_holder(p.predictor.model)
    return {attr: getattr(holder, attr) for attr in OPERATING_POINT_ATTRS}


def test_the_record_a_prediction_is_handed_governs_the_model_it_runs(tmp_path):
    """The execution record a prediction is handed governs the model it runs.

    The pass's own record sets the model's conf and the cap its density gives a 64 px frame,
    and a record handed to one prediction sets them for that prediction.
    """
    from tcip_mcp.pipelines.operating_point import (
        DETECTIONS_PER_IMG,
        SCORE_THRESH,
        detector_operating_point_holder,
    )
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF

    p = _pass(tmp_path, tile=False, conf=SAMPLE_CONF)

    assert _in_model_point(p, tmp_path) == {
        SCORE_THRESH: SAMPLE_CONF,
        DETECTIONS_PER_IMG: math.ceil(p.execution.density * 64 * 64),
    }
    assert p.execution.cross_tile_nms is None

    staged = p.execution.with_value("conf", 0.01, "staged").with_value(
        "density", 5 / (64 * 64), "fit"
    )
    p.predict([str(tmp_path / "frame.png")], execution=staged)

    holder, _path = detector_operating_point_holder(p.predictor.model)
    assert (holder.score_thresh, holder.detections_per_img) == (0.01, 5)


def test_every_execution_door_takes_its_statement_with_no_default():
    """The four execution doors each take their statement as a required ``Stated``.

    ``run_inference``, ``assess_checkpoint``, ``assess_reserved_regions`` and
    ``evaluate_model`` give ``stated`` no default, and the empty statement holds no value.
    """
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.calibration_tools import (
        assess_checkpoint,
        assess_reserved_regions,
    )
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_mcp.tools.training_tools import evaluate_model

    for tool in (
        run_inference,
        assess_checkpoint,
        assess_reserved_regions,
        evaluate_model,
    ):
        assert (
            inspect.signature(tool).parameters["stated"].default
            is inspect.Parameter.empty
        ), tool.__name__
    assert set(Stated.model_validate({}).model_dump().values()) == {None}


def test_a_statement_carries_its_reason_and_nothing_stated_carries_none():
    """A statement carries its reason, and an empty statement carries none.

    A statement of conf or tile without a reason refuses naming the field, a reason with
    nothing stated refuses, and a statement with a reason is admitted.
    """
    from pydantic import ValidationError

    from tcip_mcp.pipelines.execution import Stated
    from tests._training_values import RATIONALE
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF

    with pytest.raises(ValidationError, match="'conf'.* with rationale None"):
        Stated(conf=SAMPLE_CONF)
    with pytest.raises(ValidationError, match="'tile'.* with rationale None"):
        Stated(tile=False)
    with pytest.raises(ValidationError, match="states nothing with rationale"):
        Stated(rationale=RATIONALE)
    assert Stated().rationale is None
    assert Stated(conf=SAMPLE_CONF, rationale=RATIONALE).rationale == RATIONALE


@pytest.mark.parametrize("value", [-1.0, 1.5])
@pytest.mark.parametrize("name", ["conf", "cross_tile_nms"])
def test_a_stated_threshold_outside_the_unit_interval_is_refused_by_the_type(
    name, value
):
    """A stated threshold outside the unit interval is refused at the statement by name.

    A stated confidence or cross-tile merge threshold below 0 or above 1 is refused at the
    statement, naming the field.
    """
    from pydantic import ValidationError

    from tcip_mcp.pipelines.execution import Stated
    from tests._training_values import RATIONALE

    with pytest.raises(ValidationError, match=name):
        Stated.model_validate({name: value, "rationale": RATIONALE})


def test_the_dry_run_reports_the_stated_values_and_refuses_an_unstated_conf(
    tmp_path,
):
    """The dry run reports the stated values and refuses an unstated conf.

    Over images and over a raster alike, the dry-run report carries the stated conf and the
    checkpoint's density; a dry run stating no conf refuses, and no bucket is published.
    """
    from PIL import Image

    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._training_values import RATIONALE
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF,
        SAMPLE_DETECTOR_PASS,
        SAMPLE_OVERLAP,
        project_checkpoint,
    )

    checkpoint = project_checkpoint(tmp_path)
    density = _checkpoint(tmp_path).geometry.object_density
    images_dir = tmp_path / "images" / "undated"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(images_dir / "a.png")
    raster = tmp_path / "ortho" / "images" / "undated" / "mosaic.tif"
    raster.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(raster)

    over_images = run_inference(
        tmp_path,
        checkpoint_path=checkpoint,
        images_dir=str(images_dir),
        bucket="out/2026-01-01",
        dry_run=True,
        stated=Stated(tile=False, conf=SAMPLE_CONF, rationale=RATIONALE),
    )
    over_raster = run_inference(
        tmp_path,
        checkpoint_path=checkpoint,
        raster_path=str(raster),
        bucket="out/2026-01-01",
        dry_run=True,
        stated=Stated(
            tile_size=32, overlap=SAMPLE_OVERLAP, **SAMPLE_DETECTOR_PASS
        ),
    )
    unstated = run_inference(
        tmp_path,
        checkpoint_path=checkpoint,
        images_dir=str(images_dir),
        bucket="out/2026-01-01",
        dry_run=True,
        stated=Stated(tile=False, rationale=RATIONALE),
    )

    import tcip_store
    from tcip_mcp.dataset_layout import bucket_key

    for result in (over_images, over_raster):
        assert "error" not in result, result
        assert (
            result["execution"]["conf"],
            result["execution"]["density"],
        ) == (SAMPLE_CONF, density)
    assert "conf" in unstated.get("error", ""), unstated
    for root in (tmp_path, tmp_path / "ortho"):
        assert not tcip_store.exists(bucket_key(root, "out/2026-01-01"))
