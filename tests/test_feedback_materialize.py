"""Review-verdict materialization (pure, torch-free)."""

from __future__ import annotations

import json

import pytest

from tcip_mcp.pipelines.feedback.materialize import (
    materialize_dataset,
    partition_review_verdicts,
    reviewed_image_names,
    select_unreviewed,
)


def _review_state():
    # Keys match tcip_annotation.review_engine.record_detection_action entries.
    return {"image": {
        "imgA.png": {"img_status": "completed", "detections": [
            {"action": "accepted", "match_type": "TP", "class_name": "bud",
             "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": [0.5, 0.5, 0.2, 0.2]},
            {"action": "rejected", "match_type": "FP", "class_name": "bud",
             "gt_bbox_norm": None, "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]},
        ]},
        "imgB.png": {"img_status": "completed", "detections": [
            {"action": "rejected", "match_type": "FP", "class_name": "bud",
             "gt_bbox_norm": None, "pred_bbox_norm": [0.3, 0.3, 0.1, 0.1]},
        ]},
        "imgC.png": {"img_status": "started", "detections": [
            {"action": "accepted", "match_type": "FN", "class_name": "leaf",
             "gt_bbox_norm": [0.4, 0.4, 0.3, 0.3], "pred_bbox_norm": None},
        ]},
    }}


def test_partition_positives_rejections_and_hard_negatives():
    p = partition_review_verdicts(_review_state())
    assert p["imgA.png"]["status"] == "positive"
    assert len(p["imgA.png"]["positives"]) == 1
    assert p["imgA.png"]["rejected_count"] == 1
    assert p["imgB.png"]["status"] == "hard_negative"


def test_partition_fp_accept_uses_pred_box():
    state = {"image": {"x.png": {"img_status": "completed", "detections": [
        {"action": "accepted", "match_type": "FP", "class_name": "bud",
         "gt_bbox_norm": None, "pred_bbox_norm": [0.6, 0.6, 0.2, 0.2]},
    ]}}}
    pos = partition_review_verdicts(state)["x.png"]["positives"]
    assert len(pos) == 1
    assert pos[0][0] == "bud" and pos[0][1] == pytest.approx(0.6)


def test_materialize_writes_labels_manifest_and_empty_negatives(tmp_path):
    from PIL import Image
    src = tmp_path / "src"
    src.mkdir()
    for name in ("imgA.png", "imgB.png"):
        Image.new("RGB", (64, 64), (120, 120, 120)).save(src / name)
    out = tmp_path / "out"
    state = {"image": {
        "imgA.png": {"img_status": "completed", "detections": [
            {"action": "accepted", "class_name": "bud", "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
        "imgB.png": {"img_status": "completed", "detections": [
            {"action": "rejected", "class_name": "bud", "gt_bbox_norm": None, "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]}]},
    }}
    r = materialize_dataset(state, str(src), str(out))
    assert (r["positive"], r["hard_negative"], r["total_boxes"]) == (1, 1, 1)

    from tcip_annotation import json_io
    anns_a = json_io.read_annotations(str(out / "annotations" / "imgA.json"))
    assert len(anns_a) == 1 and anns_a[0].subject == "bud"

    label_b = out / "annotations" / "imgB.json"
    assert label_b.is_file() and json.loads(label_b.read_text())["annotations"] == []  # empty hard-negative label

    assert (out / "images" / "imgA.png").is_file()
    import tcip_store as ts
    from tcip_mcp.pipelines.feedback.materialize import curated_manifest_key
    man = ts.read(curated_manifest_key(out))
    assert man["counts"]["positive"] == 1 and man["counts"]["hard_negative"] == 1


def test_materialize_skips_missing_source_images(tmp_path):
    src = tmp_path / "src"
    src.mkdir()  # empty
    state = {"image": {"ghost.png": {"img_status": "completed", "detections": [
        {"action": "accepted", "class_name": "bud", "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]}}}
    r = materialize_dataset(state, str(src), str(tmp_path / "out"))
    assert r["missing_images"] == 1 and r["positive"] == 0
    assert not (tmp_path / "out" / "images" / "ghost.png").exists()


def test_only_completed_and_select_unreviewed():
    state = _review_state()
    assert "imgC.png" in partition_review_verdicts(state)
    assert "imgC.png" not in partition_review_verdicts(state, only_completed=True)
    assert reviewed_image_names(state) == {"imgA.png", "imgB.png"}
    assert select_unreviewed(["/a/imgA.png", "/a/imgZ.png"], {"imgA.png"}) == ["/a/imgZ.png"]

def test_hard_negatives_survive_into_training(tmp_path):
    """A rejected-only image is a human-confirmed negative and must train as one.

    The status store is subject-scoped, so materialize has to write the bucket its emitted label
    dir resolves to. An unbucketed write is quarantined as unattributable, which would silently
    discard every rejection verdict the review loop exists to harvest.
    """
    from PIL import Image

    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    src = tmp_path / "src"
    src.mkdir()
    for name in ("imgA.png", "imgB.png"):
        Image.new("RGB", (64, 64), (120, 120, 120)).save(src / name)
    out = tmp_path / "out"
    state = {"image": {
        "imgA.png": {"img_status": "completed", "detections": [
            {"action": "accepted", "class_name": "bud", "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2],
             "pred_bbox_norm": None}]},
        "imgB.png": {"img_status": "completed", "detections": [
            {"action": "rejected", "class_name": "bud", "gt_bbox_norm": None,
             "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]}]},
    }}
    materialize_dataset(state, str(src), str(out))

    labels_out = out / "annotations"
    assert confirmed_negative_names(labels_out, subject="bud", date=None) == {"imgB.png"}
    assert json.loads((labels_out / "imgB.json").read_text())["annotations"] == []
