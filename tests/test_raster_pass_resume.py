"""run_inference's raster_path regime resumes an interrupted tiled pass.

A pass over a small (64x64, tile 32, no overlap: exactly four tiles) geo-referenced raster
records its own identity under the project and one batch record per flushed tile
(tile_batch_size=1 here, so each tile is its own batch). Interruption is simulated by making the
store's own replace raise after the pass has recorded one batch, leaving the identity record and
that one batch record.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.pipelines.raster_source import RasterSource, SourceHeader

torch = pytest.importorskip("torch")

pytest.importorskip("torchvision")

import tcip_store  # noqa: E402
from tcip_mcp.pipelines.execution import EXECUTION_RECORD, MaskCut  # noqa: E402
from tests._training_values import MASK_CUT  # noqa: E402
from tests.test_orthomosaic_tools import (  # noqa: E402
    TILE,
    _bespoke_detection_checkpoint,
    _raster,
    _write_geo_raster,
)


def _instance_seg_checkpoint(tmp_path: Path) -> str:
    """Return the checkpoint a fixed-mask instance_seg run under ``tmp_path`` completed.

    The run completed over frames of its own, registering the checkpoint. Its builder,
    ``build_fixed_mask_instance_seg``, is given one box and one mask region, both at
    ``[TILE // 4, TILE // 4, 3 * TILE // 4, 3 * TILE // 4]``.
    """
    from tests._chain_fixtures import BESPOKE_MODELS, FRAME_STATISTICS
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    central = [TILE // 4, TILE // 4, 3 * TILE // 4, 3 * TILE // 4]
    return registered_checkpoint(
        tmp_path,
        model_source={
            "builder": "bespoke_models:build_fixed_mask_instance_seg",
            "builder_kwargs": {
                "regions": [central],
                "value": 1.0,
                "box": central,
            },
            "source_files": [BESPOKE_MODELS],
            "task": "instance_seg",
            "normalization": FRAME_STATISTICS,
        },
    )


def _setup(tmp_path: Path) -> tuple[str, Path]:
    """Return a registered bespoke detection checkpoint and a 64x64 geo raster.

    The raster is written in the undated capture of the dataset ``ds``, in the project
    ``tmp_path``.
    """
    raster_path = _raster(tmp_path)
    _write_geo_raster(raster_path, height=64, width=64)
    return _bespoke_detection_checkpoint(tmp_path), raster_path


def _run(
    project: Path,
    ckpt: str,
    raster_path: Path,
    bucket: str,
    *,
    conf: float = 0.0,
    masks: bool | None = None,
    mask_cut: MaskCut | None = None,
    **kwargs: str | int | bool | Stated | None,
) -> dict:
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import SAMPLE_CROSS_TILE_NMS

    call_kwargs = {
        "raster_path": str(raster_path),
        "bucket": bucket,
        "stated": Stated(
            conf=conf,
            masks=masks,
            mask_cut=mask_cut,
            tile_size=TILE,
            overlap=0.0,
            postprocess="nms",
            cross_tile_nms=SAMPLE_CROSS_TILE_NMS,
            rationale="a sample reason the test states",
        ),
        "tile_batch_size": 1,
        "device": "cpu",
    }
    call_kwargs.update(kwargs)
    return run_inference(project, ckpt, **call_kwargs, actor=None)


def _progress(project: Path, raster_path: Path, bucket: str) -> list[str]:
    """Return the segments of every progress record a pass toward ``bucket`` left in ``project``."""
    from tcip_mcp.dataset_layout import image_capture
    from tcip_mcp.tools.inference_tools import _progress_keys

    root, _capture = image_capture(raster_path)
    assert root is not None
    return sorted(
        key.parts[-1] for key in _progress_keys(project, root, bucket)
    )


def _published(raster_path: Path, bucket: str) -> bool:
    from tcip_mcp.dataset_layout import bucket_key, image_capture

    root, _capture = image_capture(raster_path)
    assert root is not None
    return tcip_store.exists(bucket_key(root, bucket))


def _interrupt_after_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the store's replace raise on the second raster-pass batch record it is handed."""
    import tcip_store.store as store_mod

    real_replace = store_mod.replace
    seen = {"batches": 0}

    def _flaky_replace(
        key: tcip_store.Key,
        value: JsonValue,
        **kw: tcip_store.Version | None,
    ) -> tcip_store.Version:
        if key.store == "raster_pass_progress" and key.parts[-1].startswith(
            "batch-"
        ):
            seen["batches"] += 1
            if seen["batches"] == 2:
                raise RuntimeError("simulated crash mid-pass")
        return real_replace(key, value, **kw)

    monkeypatch.setattr(store_mod, "replace", _flaky_replace)


def _interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bucket: str,
    ckpt: str,
    raster_path: Path,
    **kwargs: str | int | float | bool | MaskCut | Stated | None,
) -> None:
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(tmp_path, ckpt, raster_path, bucket, **kwargs)
    monkeypatch.undo()


def _document(raster_path: Path, bucket: str) -> dict:
    """Return the stored document of ``raster_path``'s stem in the bucket ``bucket``."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import image_capture

    root, _capture = image_capture(raster_path)
    assert root is not None
    key = read_bucket(root, bucket).document_key(raster_path.stem)
    assert key is not None
    return tcip_store.read(key)


def _as_produced(raster_path: Path, bucket: str) -> list[tuple]:
    return [
        (tuple(a["bbox"]), a["score"], a["subject"])
        for a in _document(raster_path, bucket)["annotations"]
    ]


def test_resume_refuses_with_images_dir(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    result = run_inference(
        tmp_path,
        str(tmp_path / "m.pt"),
        stated=Stated(),
        images_dir=str(images_dir),
        bucket="out/2026-01-01",
        resume=True,
        actor=None,
    )

    assert "resume applies only to a raster_path pass" in result["error"]


def test_resume_refuses_with_no_recorded_progress(tmp_path: Path) -> None:
    ckpt, raster_path = _setup(tmp_path)

    result = _run(tmp_path, ckpt, raster_path, "preds/2026-01-01", resume=True)

    assert "no raster-pass progress toward" in result["error"]
    assert not _published(raster_path, "preds/2026-01-01")


def test_an_interrupted_instance_segmentation_pass_resumes_to_the_uninterrupted_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted instance segmentation pass resumes to the uninterrupted pass's masks.

    The resumed pass's annotations, segmentations included, equal the uninterrupted pass's.
    """
    raster_path = _raster(tmp_path)
    _write_geo_raster(raster_path, height=64, width=64)
    ckpt = _instance_seg_checkpoint(tmp_path)
    interrupted, uninterrupted = (
        "interrupted/2026-01-01",
        "uninterrupted/2026-01-01",
    )

    baseline = _run(
        tmp_path,
        ckpt,
        raster_path,
        uninterrupted,
        masks=True,
        mask_cut=MASK_CUT,
    )
    assert "error" not in baseline, baseline
    _interrupted(
        tmp_path,
        monkeypatch,
        interrupted,
        ckpt,
        raster_path,
        masks=True,
        mask_cut=MASK_CUT,
    )
    resumed = _run(
        tmp_path,
        ckpt,
        raster_path,
        interrupted,
        masks=True,
        mask_cut=MASK_CUT,
        resume=True,
    )
    assert "error" not in resumed, resumed

    def _annotations(bucket: str) -> list[dict]:
        return [
            {k: v for k, v in a.items() if k != "created_at"}
            for a in _document(raster_path, bucket)["annotations"]
        ]

    baseline_annotations = _annotations(uninterrupted)
    assert any(a.get("segmentation") for a in baseline_annotations)
    assert _annotations(interrupted) == baseline_annotations


def test_an_interrupted_pass_leaves_one_identity_and_one_batch_record_and_no_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted pass leaves one identity and one batch record, and no bucket.

    The progress sits under the project, apart from the bucket the pass will publish, and no
    bucket is published for the partial pass.
    """
    ckpt, raster_path = _setup(tmp_path)
    bucket = "preds/2026-01-01"

    _interrupted(tmp_path, monkeypatch, bucket, ckpt, raster_path)

    assert _progress(tmp_path, raster_path, bucket) == [
        "batch-000000",
        "identity",
    ]
    assert not _published(raster_path, bucket)


def test_resume_completes_an_interrupted_pass_with_the_same_detections_and_clears_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt, raster_path = _setup(tmp_path)
    interrupted = "interrupted/2026-01-01"
    _interrupted(tmp_path, monkeypatch, interrupted, ckpt, raster_path)

    resumed = _run(tmp_path, ckpt, raster_path, interrupted, resume=True)

    assert "error" not in resumed, resumed
    assert _progress(tmp_path, raster_path, interrupted) == []
    uninterrupted = "uninterrupted/2026-01-01"
    baseline = _run(tmp_path, ckpt, raster_path, uninterrupted)
    assert "error" not in baseline, baseline
    assert resumed["execution"] == baseline["execution"]
    # As produced, not sorted: a resumed pass reconstructs the identical detection set in the
    # identical order, not merely the same boxes in some order.
    assert _as_produced(raster_path, interrupted) == _as_produced(
        raster_path, uninterrupted
    )


def test_a_fresh_pass_over_recorded_progress_refuses_and_leaves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt, raster_path = _setup(tmp_path)
    bucket = "preds/2026-01-01"
    _interrupted(tmp_path, monkeypatch, bucket, ckpt, raster_path)

    result = _run(tmp_path, ckpt, raster_path, bucket)

    assert "already recorded" in result["error"]
    assert _progress(tmp_path, raster_path, bucket) == [
        "batch-000000",
        "identity",
    ]
    assert not _published(raster_path, bucket)


def test_a_published_bucket_whose_progress_clear_failed_refuses_every_later_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A published bucket whose progress clear failed refuses a fresh pass and a resume.

    With the progress clear raising after publication, a fresh pass refuses as already recorded,
    a resume refuses with an error saying ``already``, and the bucket's document stays as
    published.
    """
    import tcip_mcp.tools.inference_tools as itools

    ckpt, raster_path = _setup(tmp_path)
    bucket = "preds/2026-01-01"

    def _raise_after_publish(*_args: object) -> None:
        raise RuntimeError(
            "simulated crash between publication and the progress clear"
        )

    monkeypatch.setattr(
        itools, "_clear_raster_pass_progress", _raise_after_publish
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(tmp_path, ckpt, raster_path, bucket)
    monkeypatch.undo()

    published = _document(raster_path, bucket)

    fresh = _run(tmp_path, ckpt, raster_path, bucket)
    resumed = _run(tmp_path, ckpt, raster_path, bucket, resume=True)

    assert "already recorded" in fresh["error"]
    assert "already" in resumed["error"]
    assert _document(raster_path, bucket) == published


def test_resume_refuses_a_call_that_differs_from_the_recorded_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt, raster_path = _setup(tmp_path)
    bucket = "preds/2026-01-01"
    _interrupted(tmp_path, monkeypatch, bucket, ckpt, raster_path)

    result = _run(tmp_path, ckpt, raster_path, bucket, resume=True, conf=0.9)

    assert "conf" in result["error"]
    # A refused resume leaves the recorded progress untouched.
    assert _progress(tmp_path, raster_path, bucket) == [
        "batch-000000",
        "identity",
    ]


def test_resume_refuses_when_the_recorded_identity_carries_an_extra_top_level_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume refuses when the recorded identity carries an extra top-level key.

    The refusal names the extra key, ``future_field``.
    """
    from tcip_mcp.tools.inference_tools import _progress_key
    from tcip_store import store

    ckpt, raster_path = _setup(tmp_path)
    bucket = "preds/2026-01-01"
    _interrupted(tmp_path, monkeypatch, bucket, ckpt, raster_path)
    identity_key = _progress_key(tmp_path, tmp_path / "ds", bucket, "identity")
    body = dict(store.read(identity_key))
    body["future_field"] = "a value this reader does not expect"
    with store.transaction(identity_key) as txn:
        txn.write(identity_key, body)

    result = _run(tmp_path, ckpt, raster_path, bucket, resume=True)

    assert "future_field" in result["error"]


def test_content_identity_failure_after_open_refuses_naming_the_raster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A content-identity failure after open refuses, naming the raster, and writes nothing.

    A raster that opens cleanly but fails partway through the sampling read refuses as a returned
    ``{"error": ...}`` naming the raster, and no bucket is published.
    """
    import tcip_mcp.pipelines.raster_source as raster_source_module

    ckpt, raster_path = _setup(tmp_path)
    real_open = raster_source_module.SourceHeader.open

    def _flaky_open(header: SourceHeader, num_channels: int) -> RasterSource:
        reader = real_open(header, num_channels)

        def _raise(*args: object, **kwargs: object) -> None:
            raise OSError("simulated disk read failure")

        reader.read_region = _raise
        return reader

    monkeypatch.setattr(raster_source_module.SourceHeader, "open", _flaky_open)

    result = _run(tmp_path, ckpt, raster_path, "preds/2026-01-01")

    assert str(raster_path) in result["error"]
    assert not _published(raster_path, "preds/2026-01-01")


def test_an_interrupted_pass_under_an_assessment_resumes_under_that_assessment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted pass under an assessment resumes under that assessment.

    A pass published under a reserved-regions assessment resumes under the same assessment and
    execution record, reproducing the detections an uninterrupted pass makes, and a resume naming
    another assessment refuses.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests import _trait_fixtures as fx
    from tests.test_block_calibration import (
        TILE as BLOCK_TILE,
        _assess,
        _attested,
    )

    fx.seed_confirmed_count(tmp_path, measured_subject="bud")
    exp = _attested(tmp_path)
    assessment = _assess(exp)
    raster_path = exp["raster_path"]
    call = {
        "raster_path": str(raster_path),
        "tile_batch_size": 50,
        "device": "cpu",
        "assessment_id": assessment["assessment_id"],
        "stated": Stated(),
    }
    interrupted = "interrupted/2026-01-01"
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_inference(
            tmp_path,
            exp["checkpoint_path"],
            bucket=interrupted,
            **call,
            actor=None,
        )
    monkeypatch.undo()

    other = run_inference(
        tmp_path,
        exp["checkpoint_path"],
        bucket=interrupted,
        resume=True,
        **{**call, "assessment_id": "another"},
        actor=None,
    )
    resumed = run_inference(
        tmp_path,
        exp["checkpoint_path"],
        bucket=interrupted,
        resume=True,
        **call,
        actor=None,
    )
    uninterrupted = "uninterrupted/2026-01-01"
    baseline = run_inference(
        tmp_path,
        exp["checkpoint_path"],
        bucket=uninterrupted,
        **call,
        actor=None,
    )

    assert "error" in other
    assert "error" not in resumed, resumed
    assert "error" not in baseline, baseline
    bucket = read_bucket(resumed["dataset_root"], interrupted)
    assert bucket.assessment_id == assessment["assessment_id"]
    assert (
        EXECUTION_RECORD.dump_python(bucket.execution, mode="json")
        == assessment["execution"]
    )
    assert bucket.execution.tile_size == BLOCK_TILE
    assert _as_produced(raster_path, interrupted) == _as_produced(
        raster_path, uninterrupted
    )
