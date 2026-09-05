"""Geometry, box accounting and inclusion scope of a materialized review dataset.

The verdict log stores normalized center-form boxes; the canonical per-image JSON is pixel-space
xyxy. On a non-square frame the two axes scale by different extents, so a box denormalized against
the wrong axis lands somewhere plausible and trains a displaced target. The same fixtures pin what
counts as a materialized image at all: the caller's ``only_completed`` and ``include_hard_negatives``
requests decide membership, and the box tally counts boxes rather than images.
"""

from __future__ import annotations

import json

from PIL import Image

from tcip_annotation import json_io
from tcip_mcp.pipelines.feedback.materialize import materialize_dataset, partition_review_verdicts


def _image(images_dir, name: str, size) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 140, 60)).save(images_dir / name)


def _accepted(class_name: str, gt, pred=None) -> dict:
    return {"action": "accepted", "class_name": class_name, "gt_bbox_norm": gt,
            "pred_bbox_norm": pred}


def _rejected(class_name: str, pred) -> dict:
    return {"action": "rejected", "class_name": class_name, "gt_bbox_norm": None,
            "pred_bbox_norm": pred}


def _completed(detections: list[dict]) -> dict:
    return {"img_status": "completed", "detections": detections}


def test_positive_boxes_denormalize_against_each_axis_extent(tmp_path):
    """Width scales x, height scales y, on frames whose two extents differ."""
    src = tmp_path / "src"
    _image(src, "wide.png", (96, 48))
    _image(src, "tall.png", (40, 100))
    out = tmp_path / "out"
    state = {"image": {
        "wide.png": _completed([_accepted("bud", [0.5, 0.25, 0.25, 0.5])]),
        "tall.png": _completed([_accepted("bud", [0.25, 0.6, 0.5, 0.2])]),
    }}

    materialize_dataset(state, str(src), str(out))

    wide = json_io.read_annotations(str(out / "annotations" / "wide.json"))
    assert len(wide) == 1
    g = wide[0].geometry
    assert (g.x1, g.y1, g.x2, g.y2) == (36.0, 0.0, 60.0, 24.0)

    tall = json_io.read_annotations(str(out / "annotations" / "tall.json"))
    assert len(tall) == 1
    g = tall[0].geometry
    assert (g.x1, g.y1, g.x2, g.y2) == (0.0, 50.0, 20.0, 70.0)

    stamped = json.loads((out / "annotations" / "wide.json").read_text())
    assert (stamped["width"], stamped["height"]) == (96, 48)


def test_human_edited_box_takes_precedence_over_prediction(tmp_path):
    """An edited verdict carries both boxes; the human's corrected box is the label."""
    entry = {"action": "edited", "class_name": "bud",
             "gt_bbox_norm": [0.25, 0.5, 0.4, 0.2], "pred_bbox_norm": [0.75, 0.25, 0.1, 0.1]}
    positives = partition_review_verdicts({"image": {"e.png": _completed([entry])}})["e.png"]["positives"]
    assert positives == [("bud", 0.25, 0.5, 0.4, 0.2)]

    src = tmp_path / "src"
    _image(src, "e.png", (100, 50))
    out = tmp_path / "out"
    materialize_dataset({"image": {"e.png": _completed([entry])}}, str(src), str(out))

    anns = json_io.read_annotations(str(out / "annotations" / "e.json"))
    assert len(anns) == 1
    g = anns[0].geometry
    assert (g.x1, g.y1, g.x2, g.y2) == (5.0, 20.0, 45.0, 30.0)


def test_a_degenerate_positive_box_is_reported_by_name_not_raised(tmp_path):
    """A verdict box that denormalizes to zero extent must not abort the harvest: it is reported
    by name rather than raised, and materializing the rest of the images still proceeds."""
    src = tmp_path / "src"
    _image(src, "bad.png", (100, 50))
    _image(src, "good.png", (100, 50))
    out = tmp_path / "out"
    state = {"image": {
        "bad.png": _completed([_accepted("bud", [0.5, 0.5, 0.0, 0.3])]),  # zero-width box
        "good.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.3])]),
    }}

    r = materialize_dataset(state, str(src), str(out))

    assert r["positive"] == 1
    assert len(r["boundary_refused"]) == 1
    assert r["boundary_refused"][0]["image"] == "bad.png"
    assert not (out / "annotations" / "bad.json").exists()
    good = json_io.read_annotations(str(out / "annotations" / "good.json"))
    assert len(good) == 1


def test_box_counts_track_every_positive_not_every_image(tmp_path):
    """``total_boxes`` sums boxes across images, and each manifest row reports its own count."""
    src = tmp_path / "src"
    _image(src, "three.png", (120, 60))
    _image(src, "one.png", (60, 120))
    _image(src, "none.png", (80, 40))
    out = tmp_path / "out"
    state = {"image": {
        "three.png": _completed([
            _accepted("bud", [0.2, 0.2, 0.1, 0.1]),
            _accepted("bud", [0.5, 0.4, 0.2, 0.3]),
            _accepted("bud", [0.8, 0.7, 0.15, 0.05]),
        ]),
        "one.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.4])]),
        "none.png": _completed([_rejected("bud", [0.3, 0.3, 0.1, 0.1])]),
    }}

    r = materialize_dataset(state, str(src), str(out))
    assert (r["positive"], r["hard_negative"], r["total_boxes"]) == (2, 1, 4)

    import tcip_store
    from tcip_mcp.pipelines.feedback.materialize import curated_manifest_key

    rows = {e["image"]: e for e in tcip_store.read(curated_manifest_key(out))["images"]}
    assert set(rows) == {"three.png", "one.png", "none.png"}
    assert rows["three.png"]["n_boxes"] == 3
    assert rows["one.png"]["n_boxes"] == 1
    assert rows["none.png"]["n_boxes"] == 0
    assert len(json_io.read_annotations(str(out / "annotations" / "three.json"))) == 3


def test_unreviewed_images_stay_out_of_the_materialized_set(tmp_path):
    """``only_completed`` is honored by the writer, not just by the partition helper."""
    src = tmp_path / "src"
    _image(src, "done.png", (70, 35))
    _image(src, "midway.png", (35, 70))
    state = {"image": {
        "done.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.2])]),
        "midway.png": {"img_status": "started", "detections": [
            _accepted("bud", [0.4, 0.4, 0.3, 0.3])]},
    }}

    strict = tmp_path / "strict"
    r = materialize_dataset(state, str(src), str(strict), only_completed=True)
    assert r["positive"] == 1
    assert (strict / "annotations" / "done.json").is_file()
    assert not (strict / "annotations" / "midway.json").exists()
    assert not (strict / "images" / "midway.png").exists()

    loose = tmp_path / "loose"
    assert materialize_dataset(state, str(src), str(loose))["positive"] == 2
    assert (loose / "annotations" / "midway.json").is_file()


def test_hard_negative_inclusion_follows_the_caller_request(tmp_path):
    """Rejected-only images are harvested when asked for and left out when not."""
    src = tmp_path / "src"
    _image(src, "pos.png", (90, 30))
    _image(src, "neg_one.png", (30, 90))
    _image(src, "neg_two.png", (75, 25))
    state = {"image": {
        "pos.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.2])]),
        "neg_one.png": _completed([_rejected("bud", [0.3, 0.7, 0.1, 0.2])]),
        "neg_two.png": _completed([_rejected("bud", [0.6, 0.2, 0.2, 0.1])]),
    }}

    kept = tmp_path / "kept"
    r = materialize_dataset(state, str(src), str(kept), include_hard_negatives=True)
    assert (r["positive"], r["hard_negative"], r["skipped"]) == (1, 2, 0)
    assert json.loads((kept / "annotations" / "neg_one.json").read_text())["annotations"] == []

    dropped = tmp_path / "dropped"
    r = materialize_dataset(state, str(src), str(dropped), include_hard_negatives=False)
    assert (r["positive"], r["hard_negative"], r["skipped"]) == (1, 0, 2)
    assert not (dropped / "annotations" / "neg_one.json").exists()
    assert not (dropped / "annotations" / "neg_two.json").exists()
    assert (dropped / "annotations" / "pos.json").is_file()
