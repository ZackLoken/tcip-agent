"""What a prediction bucket's stamp records, and where each number's provenance comes from.

Behaviour, not surface: the agent's export door and the GUI's own inference worker record the same
facts about what produced a bucket's counts, a bucket's stamps are never read as image records, and
a detection cap carries the provenance of whoever actually produced it.
"""

from __future__ import annotations

import pytest

# Every stamp a bucket carries beside its per-image records. Spelled out here rather than imported,
# so this file states independently what the enumeration below must skip.
STAMP_FILENAMES = (
    "operating_point.json",
    "classifier_operating_point.json",
    "ordinal_operating_point.json",
    "regression_operating_point.json",
    "resolve_scale.json",
)


def test_bucket_stems_skips_every_stamp_not_only_the_count_one(tmp_path):
    """A bucket carrying a classifier or ordinal calibration stamp beside its predictions has two
    image stems, not six: reading a stamp as an image record invents a stem no image has, and every
    consumer of that stem set (verdict counting, bucket redirection) then counts a file that is
    provenance, not a prediction."""
    from tcip_mcp.prediction_buckets import bucket_stems

    bucket = tmp_path / "bucket"
    bucket.mkdir()
    for name in STAMP_FILENAMES:
        (bucket / name).write_text("{}", encoding="utf-8")
    for stem in ("img_0001", "img_0002"):
        (bucket / f"{stem}.json").write_text("{}", encoding="utf-8")

    assert bucket_stems(bucket) == {"img_0001", "img_0002"}


def test_gui_inference_stamp_records_what_the_agents_export_door_records(tmp_path, monkeypatch):
    """A GUI-produced bucket must be as auditable as an agent-produced one. Both stamp the run's
    identity, when it ran, what the tile scale rested on, and what stopped the bundle from being
    shippable, so a reviewer reconstructing a number never has to know which door wrote it."""
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    out_dir = tmp_path / "out"
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)

    class FakePredictor:
        config = {"data": {"subject": "bud"}}

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1],
                     "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        job_id="stamp1", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), tile=False, conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nms", platform_root=str(tmp_path),
    )
    _worker(job)

    assert job.status == "completed"
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(out_dir)
    for key in ("operating_point", "id_map", "validated", "tile_size_validated",
                "shippable_issues", "checkpoint", "checkpoint_sha256", "experiment_id",
                "images_dir", "raster_path", "produced_at"):
        assert key in sidecar, f"the GUI-produced stamp is missing {key!r}"
    assert sidecar["images_dir"] == str(images_dir)
    assert sidecar["raster_path"] is None
    assert sidecar["produced_at"]


# --- a detection cap carries the provenance of whoever produced it ---

def _records(counts: list[int]) -> list[dict]:
    from tcip_mcp.pipelines.training.evaluation import build_coco_image_record

    return [
        build_coco_image_record(
            100, 100,
            [{"bbox": [10.0, 10.0, 12.0, 12.0], "category_id": 0} for _ in range(n)],
            [{"bbox": [10.0, 10.0, 12.0, 12.0], "category_id": 0, "score": 0.9}
             for _ in range(n)],
            image_id=f"img{i}",
        )
        for i, n in enumerate(counts)
    ]


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_caller_supplied_detection_cap_is_not_labeled_with_a_derivation_it_never_came_from():
    """A cap the caller states is an explicit override. Stamping the resolver's own density formula
    on it attributes a derivation to a number the resolver never derived, and the provenance is what
    a reviewer reconstructs the count from."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    bundle = resolve_operating_point(
        "bud_opening", dataset_hash="d", calibration_records=_records([2, 3]),
        tiled=False, max_dets=250)
    cap = bundle.get("max_dets")

    assert cap.value == 250
    assert cap.source == "explicit"
    assert cap.derived_from == "caller override"


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_a_cap_the_resolver_derives_itself_still_says_how(tmp_path):
    """The rail that stops mislabeling a caller's number must still label the resolver's own, or it
    has traded one wrong provenance for a missing one."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    bundle = resolve_operating_point(
        "bud_opening", dataset_hash="d", calibration_records=_records([2, 3]), tiled=False)
    cap = bundle.get("max_dets")

    assert cap.source == "derived"
    assert "p99 GT objects/image" in cap.derived_from


def test_full_frame_evaluation_leaves_nms_and_cap_unstated_for_the_shared_resolution():
    """The select-point must equal the ship-point: the delivery-grade eval binds no cross-tile NMS
    or detection cap of its own, leaving both unstated (``None``) so ``applied_operating_point``,
    the one resolution ``run_inference`` also calls, supplies the platform defaults. Coverage of
    the signature and of the shared resolution's own defaults; the agreement itself holds by
    construction, one function called from both doors."""
    import inspect

    from tcip_mcp.pipelines.resolution import (
        DEFAULT_CONF, DEFAULT_MAX_DETS, DEFAULT_NMS_IOU, applied_operating_point,
    )
    from tcip_mcp.pipelines.training.eval_runners import run_full_frame_evaluation

    params = inspect.signature(run_full_frame_evaluation).parameters
    assert params["global_nms_iou"].default is None
    assert params["max_dets"].default is None
    assert applied_operating_point(None, None, None) == (DEFAULT_CONF, DEFAULT_NMS_IOU, DEFAULT_MAX_DETS)
