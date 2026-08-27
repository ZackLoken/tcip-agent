"""Inference-side binding to a named split manifest: the calibration universe becomes the
manifest's held-out side for one capture date instead of every labelled stem with an image.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 32
SUBJECT = "catkin"
DATES = ("2-11-26", "2-12-01")


def _save_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMG, IMG), color=(128, 128, 128)).save(path)


def _two_date_dataset(root: Path, stems=("a", "b", "c")) -> Path:
    for date in DATES:
        images_dir, labels_dir = root / "images" / date, root / "annotations" / date
        for stem in stems:
            _save_png(images_dir / f"{stem}.jpg")
            json_io.write_annotations(
                str(labels_dir / f"{stem}.json"),
                [Annotation(subject=SUBJECT, geometry=BBox(2, 2, 10, 10))], IMG, IMG,
            )
    return root


def _draw(root: Path, out: Path, *, seed: int = 1) -> dict:
    import tcip_store as ts

    from tcip_mcp.tools.data_tools import make_splits, split_manifest_key

    result = make_splits(str(root), output_path=str(out), subject=SUBJECT, seed=seed,
                         train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
    assert "error" not in result, result
    return ts.read(split_manifest_key(out))


# -- calibration_universe_from_manifest ------------------------------------------


def test_calibration_universe_from_manifest_holds_only_val_members_present(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest

    root = _two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    stems, group_by, group_key_map, excluded = calibration_universe_from_manifest(
        manifest, DATES[0], present=["a", "b", "c"])

    val_this_date = {i.split("/", 1)[1] for i in manifest["splits"]["val"]
                     if i.startswith(f"{DATES[0]}/")}
    train_this_date = {i.split("/", 1)[1] for i in manifest["splits"]["train"]
                       if i.startswith(f"{DATES[0]}/")}
    assert set(stems) == val_this_date
    assert set(excluded["excluded_training_stems"]) == train_this_date
    assert group_by == manifest["group_by"]
    assert group_key_map is None


def test_calibration_universe_from_manifest_excludes_a_stem_not_present(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest

    all_stems = ("a", "b", "c", "d", "e", "f")
    root = _two_date_dataset(tmp_path / "ds", stems=all_stems)
    manifest = _draw(root, tmp_path / "m", seed=0)
    val_this_date = [i.split("/", 1)[1] for i in manifest["splits"]["val"]
                     if i.startswith(f"{DATES[0]}/")]
    assert len(val_this_date) >= 3, "fixture must leave room to drop one and still have >=2"
    present_minus_one = [s for s in all_stems if s != val_this_date[0]]

    stems, *_rest = calibration_universe_from_manifest(manifest, DATES[0], present=present_minus_one)

    assert val_this_date[0] not in stems


def test_calibration_universe_from_manifest_refuses_fewer_than_two_groups(tmp_path: Path):
    """The refusal names a remedy: a different draw, or the whole-directory calibration."""
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest

    root = _two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")
    val_this_date = [i.split("/", 1)[1] for i in manifest["splits"]["val"]
                     if i.startswith(f"{DATES[0]}/")]

    with pytest.raises(ValueError, match="split_manifest_dir"):
        calibration_universe_from_manifest(manifest, DATES[0], present=val_this_date[:1])


# -- resolve_locked_cal_holdout_split's split_manifest_dir -----------------------


def test_resolve_locked_cal_holdout_split_records_the_manifest_dir(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split

    locked = resolve_locked_cal_holdout_split(
        ["a", "b", "c", "d"], identity_hash="ident1", scope_root=tmp_path,
        split_manifest_dir="some/manifest/dir",
    )

    assert locked["split_manifest_dir"] == "some/manifest/dir"
    assert locked["redraw_history"][0]["policy"]["split_manifest_dir"] == "some/manifest/dir"


def test_a_whole_directory_lock_and_a_manifest_lock_coexist(tmp_path: Path):
    """Two distinct identities (a whole-directory hash and a universe hash) draw and answer
    from two distinct locks over the same scope root."""
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split

    whole = resolve_locked_cal_holdout_split(
        ["a", "b", "c", "d"], identity_hash="whole_ident", scope_root=tmp_path)
    manifest = resolve_locked_cal_holdout_split(
        ["a", "b"], identity_hash="manifest_ident", scope_root=tmp_path,
        split_manifest_dir="some/manifest/dir")

    assert whole["split_manifest_dir"] is None
    assert manifest["split_manifest_dir"] == "some/manifest/dir"
    assert set(whole["calibration"]) | set(whole["holdout"]) == {"a", "b", "c", "d"}
    assert set(manifest["calibration"]) | set(manifest["holdout"]) == {"a", "b"}

    # Re-resolving each by its own identity still answers from its own lock, unchanged.
    again_whole = resolve_locked_cal_holdout_split(
        ["a", "b", "c", "d"], identity_hash="whole_ident", scope_root=tmp_path)
    again_manifest = resolve_locked_cal_holdout_split(
        ["a", "b"], identity_hash="manifest_ident", scope_root=tmp_path,
        split_manifest_dir="some/manifest/dir")
    assert again_whole == whole
    assert again_manifest == manifest


# -- attach_split_policy_provenance ------------------------------------------------


def test_attach_split_policy_provenance_copies_the_manifest_dir():
    from tcip_mcp.pipelines.operating_point import attach_split_policy_provenance
    from tcip_mcp.pipelines.resolution import ResolvedBundle, derived

    conf = derived("conf", 0.5, derived_from="test", requires_validation=True,
                   validation_kind="annotations", validated_against=None, sweep={})
    bundle = ResolvedBundle(trait="catkin", dataset_hash=None, params={"conf": conf})

    attach_split_policy_provenance(bundle, {"group_by": "stem", "seed": 0, "holdout_ratio": 0.5,
                                            "identity_hash": "abc", "split_manifest_dir": "m/dir"})

    assert bundle.get("conf").sweep["split_policy"]["split_manifest_dir"] == "m/dir"


# -- _reference_identity's label_stems group ---------------------------------------


def test_reference_identity_hashes_a_label_stems_group(tmp_path: Path):
    from tcip_mcp.pipelines.resolution import _reference_identity

    root = _two_date_dataset(tmp_path / "ds")
    labels_dir = root / "annotations" / DATES[0]

    identity = _reference_identity(
        {"label_stems": {"calibration": {"path": str(labels_dir), "stems": ["a", "b"]}}},
        dataset_root=root,
    )

    assert identity["label_stems"]["calibration"]["count"] == 2
    assert identity["label_stems"]["calibration"]["dataset_hash"]


# -- _calibrate_operating_point's manifest branch ----------------------------------


class _CalStub:
    """A predictor with the mutable operating-point surface the calibration sets, predicting
    nothing: enough to drive the manifest checks without a real forward pass."""

    def __init__(self, subject=SUBJECT, attribute=None):
        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None
        self.config = {"data": {"subject": subject, "attribute": attribute}}

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": IMG, "height": IMG,
                 "boxes": [], "scores": [], "labels": [], "count": 0} for p in paths]


_CAL_KWARGS = dict(tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8, global_nms_iou=0.3,
                   postprocess="nms", cross_tile_nms=None, max_dets=None, seed=0, holdout_ratio=0.5)


def test_calibrate_operating_point_binds_to_the_manifests_held_out_side(tmp_path: Path):
    import tcip_mcp.tools.inference_tools as itools

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    bundle, dh, _n_excl, evidence = itools._calibrate_operating_point(
        _CalStub(), "catkin", str(root / "annotations" / DATES[0]),
        str(root / "images" / DATES[0]), split_manifest_dir=str(out), **_CAL_KWARGS)

    val_this_date = {i.split("/", 1)[1] for i in manifest["splits"]["val"]
                     if i.startswith(f"{DATES[0]}/")}
    assert set(evidence["calibration_stems"]) == val_this_date
    assert "label_stems" in evidence["reference_inputs"]
    assert evidence["reference_inputs"]["stated_values"]["split_manifest_dir"] == str(out)
    from tcip_mcp.pipelines.resolution import dataset_hash
    assert dh == dataset_hash(str(root / "annotations" / DATES[0]), stems=sorted(val_this_date))


def test_calibrate_operating_point_manifest_refuses_a_subject_mismatch(tmp_path: Path):
    import tcip_mcp.tools.inference_tools as itools

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    with pytest.raises(ValueError, match="subject"):
        itools._calibrate_operating_point(
            _CalStub(subject="a_different_subject"), "catkin",
            str(root / "annotations" / DATES[0]), str(root / "images" / DATES[0]),
            split_manifest_dir=str(out), **_CAL_KWARGS)


def test_calibrate_operating_point_manifest_conflicts_with_group_by(tmp_path: Path):
    import tcip_mcp.tools.inference_tools as itools

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    kwargs = dict(_CAL_KWARGS)

    with pytest.raises(ValueError, match="group_by"):
        itools._calibrate_operating_point(
            _CalStub(), "catkin", str(root / "annotations" / DATES[0]),
            str(root / "images" / DATES[0]), split_manifest_dir=str(out), group_by="stem",
            **kwargs)


def test_calibrate_operating_point_manifest_refuses_an_images_root_mismatch(tmp_path: Path):
    import tcip_mcp.tools.inference_tools as itools

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    other_images = tmp_path / "elsewhere"
    other_images.mkdir()

    with pytest.raises(ValueError, match="images_root"):
        itools._calibrate_operating_point(
            _CalStub(), "catkin", str(root / "annotations" / DATES[0]), str(other_images),
            split_manifest_dir=str(out), **_CAL_KWARGS)


def test_calibrate_operating_point_manifest_requires_images_dir(tmp_path: Path):
    """A labels-only universe can include a stem whose image is gone, a lock the redraw would
    address that no manifest-restricted calibration ever draws; refuse by name rather than raise
    a bare ``KeyError`` out of the stem-to-image narrowing."""
    import tcip_mcp.tools.inference_tools as itools

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    kwargs = dict(_CAL_KWARGS)

    with pytest.raises(ValueError, match="images_dir"):
        itools._calibrate_operating_point(
            _CalStub(), "catkin", str(root / "annotations" / DATES[0]), None,
            split_manifest_dir=str(out), **kwargs)


def test_calibrate_operating_point_manifest_refuses_a_moved_images_root_by_name(tmp_path: Path):
    """A manifest's recorded images_root that no longer exists on disk answers the named
    refusal, never a bare crash from comparing against a gone path."""
    import tcip_mcp.tools.inference_tools as itools

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    moved = tmp_path / "elsewhere"
    (root / "images" / DATES[0]).rename(moved)

    with pytest.raises(ValueError, match="images_root"):
        itools._calibrate_operating_point(
            _CalStub(), "catkin", str(root / "annotations" / DATES[0]), str(moved),
            split_manifest_dir=str(out), **_CAL_KWARGS)


# -- run_inference's own split_manifest_dir refusal --------------------------------


def test_run_inference_refuses_split_manifest_dir_without_calibration_labels_dir(tmp_path: Path):
    """A manifest with no ``calibration_labels_dir`` scopes a calibration that will never run
    (``trait``/``calibration_labels_dir`` is what turns the manifest into a bounded universe), so
    the call is refused by name rather than silently training without validation."""
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    result = run_inference(
        str(ckpt), image_paths=[str(tmp_path / "a.png")], split_manifest_dir=str(tmp_path / "m"))

    assert "error" in result and "split_manifest_dir" in result["error"]


# -- force_redraw_cal_holdout_split with a manifest --------------------------------


def test_force_redraw_binds_to_the_manifest_and_records_its_dir(tmp_path: Path):
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    result = force_redraw_cal_holdout_split(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(root / "images" / DATES[0]), split_manifest_dir=str(out),
        subject=SUBJECT, reason="test redraw",
    )

    assert "error" not in result
    val_this_date = {i.split("/", 1)[1] for i in manifest["splits"]["val"]
                     if i.startswith(f"{DATES[0]}/")}
    train_this_date = {i.split("/", 1)[1] for i in manifest["splits"]["train"]
                       if i.startswith(f"{DATES[0]}/")}
    new_members = set(result["new_membership"]["calibration"]) | set(
        result["new_membership"]["holdout"])
    assert new_members == val_this_date
    assert new_members.isdisjoint(train_this_date)


def test_force_redraw_manifest_requires_subject(tmp_path: Path):
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    result = force_redraw_cal_holdout_split(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(root / "images" / DATES[0]), split_manifest_dir=str(out),
        reason="test redraw",
    )

    assert "error" in result and "subject" in result["error"]


def test_force_redraw_manifest_requires_images_dir(tmp_path: Path):
    """A labels-only universe can include a stem whose image is gone, a lock the redraw would
    address that no manifest-restricted calibration ever draws; refuse rather than address it."""
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    result = force_redraw_cal_holdout_split(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        split_manifest_dir=str(out), subject=SUBJECT, reason="test redraw",
    )

    assert "error" in result and "images_dir" in result["error"]


def test_force_redraw_refuses_a_moved_images_root_by_name(tmp_path: Path):
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    moved = tmp_path / "elsewhere"
    (root / "images" / DATES[0]).rename(moved)

    result = force_redraw_cal_holdout_split(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(moved), split_manifest_dir=str(out),
        subject=SUBJECT, reason="test redraw",
    )

    assert "error" in result and "images_root" in result["error"]


def test_force_redraw_manifest_addresses_the_same_lock_when_an_image_is_missing(tmp_path: Path):
    """With images_dir given, a manifest redraw's universe excludes a held-out member whose
    image is gone, the same universe a manifest-restricted calibration would draw, so the redraw
    addresses that same lock rather than a second, unreachable one."""
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest, label_image_stems
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    root = _two_date_dataset(tmp_path / "ds", stems=("a", "b", "c", "d", "e", "f", "g", "h"))
    out = tmp_path / "m"
    manifest = _draw(root, out)
    val_this_date = [i.split("/", 1)[1] for i in manifest["splits"]["val"]
                    if i.startswith(f"{DATES[0]}/")]
    missing = val_this_date[0]
    (root / "images" / DATES[0] / f"{missing}.jpg").unlink()

    present, _ = label_image_stems(
        str(root / "annotations" / DATES[0]), str(root / "images" / DATES[0]))
    expected_universe, _gb, _gkm, _excl = calibration_universe_from_manifest(
        manifest, DATES[0], present)
    expected_hash = dataset_hash(str(root / "annotations" / DATES[0]), stems=expected_universe)

    result = force_redraw_cal_holdout_split(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(root / "images" / DATES[0]), split_manifest_dir=str(out),
        subject=SUBJECT, reason="test redraw",
    )

    assert "error" not in result
    assert result["identity_hash"] == expected_hash
