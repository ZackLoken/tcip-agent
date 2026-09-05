"""Inference-side binding to a named split manifest: the calibration universe is the manifest's
third, calibration side, for one capture date, instead of every labelled stem with an image.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 32
SUBJECT = "bud"
DATES = ("2-11-26", "2-12-01")
_STEMS = ("a", "b", "c", "d", "e", "f", "g", "h")


def _save_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMG, IMG), color=(128, 128, 128)).save(path)


def _two_date_dataset(root: Path, stems=_STEMS) -> Path:
    """Two capture dates, eight stems each, all foreground for ``SUBJECT``: enough groups that a
    three-way draw still leaves the calibration side at least four present images across at
    least two groups for either date, so the lock's own halving leaves two per half."""
    for date in DATES:
        images_dir, labels_dir = root / "images" / date, root / "annotations" / date
        for stem in stems:
            _save_png(images_dir / f"{stem}.jpg")
            json_io.write_annotations(
                str(labels_dir / f"{stem}.json"),
                [Annotation(subject=SUBJECT, geometry=BBox(2, 2, 10, 10))], IMG, IMG,
            )
    return root


def _draw(root: Path, out: Path, *, seed: int = 2) -> dict:
    import tcip_store as ts

    from tcip_mcp.tools.data_tools import draw_splits, split_manifest_key

    result = draw_splits(str(root), output_path=str(out), subject=SUBJECT, seed=seed,
                         train_ratio=0.4, val_ratio=0.3, calibration_ratio=0.3)
    assert "error" not in result, result
    return ts.read(split_manifest_key(out))


def _calibration_this_date(manifest: dict, date: str = DATES[0]) -> list[str]:
    return sorted(i.split("/", 1)[1] for i in manifest["splits"]["calibration"]
                 if i.startswith(f"{date}/"))


def _train_this_date(manifest: dict, date: str = DATES[0]) -> set[str]:
    return {i.split("/", 1)[1] for i in manifest["splits"]["train"] if i.startswith(f"{date}/")}


def _val_this_date(manifest: dict, date: str = DATES[0]) -> set[str]:
    return {i.split("/", 1)[1] for i in manifest["splits"]["val"] if i.startswith(f"{date}/")}


# -- calibration_universe_from_manifest ------------------------------------------


def test_calibration_universe_from_manifest_holds_only_calibration_members_present(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest

    root = _two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    stems, group_by, group_key_map, excluded = calibration_universe_from_manifest(
        manifest, DATES[0], present=list(_STEMS))

    assert set(stems) == set(_calibration_this_date(manifest))
    assert set(excluded["excluded_training_stems"]) == _train_this_date(manifest)
    assert set(excluded["excluded_validation_stems"]) == _val_this_date(manifest)
    assert excluded["excluded_unassigned_stems"] == []
    assert group_by == manifest["group_by"]
    assert group_key_map is None


def test_calibration_universe_from_manifest_excludes_a_stem_not_present(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest

    root = _two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")
    calibration_this_date = _calibration_this_date(manifest)
    assert len(calibration_this_date) >= 3, "fixture must leave room to drop one and still have >=2"
    present_minus_one = [s for s in _STEMS if s != calibration_this_date[0]]

    stems, *_rest = calibration_universe_from_manifest(manifest, DATES[0], present=present_minus_one)

    assert calibration_this_date[0] not in stems


def test_calibration_universe_from_manifest_reports_an_unassigned_present_stem(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest

    root = _two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")

    _stems, _gb, _gkm, excluded = calibration_universe_from_manifest(
        manifest, DATES[0], present=list(_STEMS) + ["never_drawn"])

    assert excluded["excluded_unassigned_stems"] == ["never_drawn"]


def test_calibration_universe_from_manifest_refuses_fewer_than_two_groups(tmp_path: Path):
    """The refusal names a remedy: a redraw on this date with a larger calibration ratio or more
    foreground groups."""
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest

    root = _two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")
    calibration_this_date = _calibration_this_date(manifest)

    with pytest.raises(ValueError, match="calibration_ratio"):
        calibration_universe_from_manifest(manifest, DATES[0], present=calibration_this_date[:1])


def test_calibration_universe_from_manifest_floor_is_foreground_aware(tmp_path: Path):
    """A universe with enough raw groups but only one carrying real foreground still refuses:
    the floor counts foreground groups, not bare group presence, when the caller states which
    stems are foreground."""
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest

    root = _two_date_dataset(tmp_path / "ds")
    manifest = _draw(root, tmp_path / "m")
    calibration_this_date = _calibration_this_date(manifest)
    assert len(calibration_this_date) >= 3

    with pytest.raises(ValueError, match="foreground group"):
        calibration_universe_from_manifest(
            manifest, DATES[0], present=list(_STEMS),
            foreground_stems={calibration_this_date[0]})


# -- resolve_manifest_calibration_universe's scope check --------------------------


def test_resolve_manifest_calibration_universe_refuses_a_members_block_with_no_images_root(
    tmp_path: Path,
):
    """A manifest whose members block under this date names no images root refuses at the
    calibration door."""
    import tcip_store as ts
    from tcip_mcp.pipelines.data.splits import (
        manifest_date_key, resolve_manifest_calibration_universe,
    )
    from tcip_mcp.tools.data_tools import split_manifest_key

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    date_key = manifest_date_key(DATES[0])
    manifest["members"][date_key] = {
        k: v for k, v in manifest["members"][date_key].items() if k != "images_root"
    }
    ts.replace(split_manifest_key(out), manifest)

    with pytest.raises(ValueError, match="images root"):
        resolve_manifest_calibration_universe(
            manifest, str(out), str(root / "annotations" / DATES[0]),
            str(root / "images" / DATES[0]), SUBJECT, None, list(_STEMS))


def test_resolve_manifest_calibration_universe_refuses_a_door_with_no_images_dir(
    tmp_path: Path,
):
    """A door naming no images_dir at all refuses at the calibration door."""
    from tcip_mcp.pipelines.data.splits import resolve_manifest_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    with pytest.raises(ValueError, match="states no images_dir"):
        resolve_manifest_calibration_universe(
            manifest, str(out), str(root / "annotations" / DATES[0]), None, SUBJECT, None,
            list(_STEMS))


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
                   validation_kind="annotations", validated_against=None, gate_evidence={})
    bundle = ResolvedBundle(trait="bud_opening", dataset_hash=None, params={"conf": conf})

    attach_split_policy_provenance(bundle, {"group_by": "stem", "seed": 0, "holdout_ratio": 0.5,
                                            "identity_hash": "abc", "split_manifest_dir": "m/dir"})

    assert bundle.get("conf").gate_evidence["split_policy"]["split_manifest_dir"] == "m/dir"


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


# -- calibrate_operating_point's manifest branch ----------------------------------


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


def test_calibrate_operating_point_binds_to_the_manifests_calibration_side(tmp_path: Path):
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    bundle, dh, _n_excl, evidence = calibration.calibrate_operating_point(
        _CalStub(), "bud_opening", str(root / "annotations" / DATES[0]),
        str(root / "images" / DATES[0]), split_manifest_dir=str(out), **_CAL_KWARGS)

    calibration_this_date = set(_calibration_this_date(manifest))
    assert set(evidence["calibration_stems"]) == calibration_this_date
    assert "label_stems" in evidence["reference_inputs"]
    assert evidence["reference_inputs"]["stated_values"]["split_manifest_dir"] == str(out)
    # The persisted evidence's own inputs, what a delivery door reopens the gate with, carry the
    # manifest and its date: nothing pins the producer writing these two without this assertion.
    assert evidence["inputs"]["split_manifest_dir"] == str(out)
    assert evidence["inputs"]["calibration_date"] == DATES[0]
    from tcip_mcp.pipelines.resolution import dataset_hash
    assert dh == dataset_hash(
        str(root / "annotations" / DATES[0]), stems=sorted(calibration_this_date))


def test_calibrate_operating_point_manifest_refuses_a_subject_mismatch(tmp_path: Path):
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    with pytest.raises(ValueError, match="subject"):
        calibration.calibrate_operating_point(
            _CalStub(subject="a_different_subject"), "bud_opening",
            str(root / "annotations" / DATES[0]), str(root / "images" / DATES[0]),
            split_manifest_dir=str(out), **_CAL_KWARGS)


def test_resolve_manifest_calibration_universe_admits_a_doors_empty_string_attribute(
    tmp_path: Path,
):
    """An explicit empty-string attribute the door states normalizes to ``None`` the same way
    the training child's own admission normalizes it, so a door with no attribute draws a
    universe from a manifest drawn with none, rather than being refused for a divergence that
    reads the same fact two ways."""
    from tcip_mcp.pipelines.data.splits import resolve_manifest_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    stems, _group_by, _group_key_map, _excluded, cal_date, subject, attribute = \
        resolve_manifest_calibration_universe(
            manifest, str(out), str(root / "annotations" / DATES[0]),
            str(root / "images" / DATES[0]), SUBJECT, "", list(_STEMS))

    assert stems
    assert cal_date == DATES[0]
    assert (subject, attribute) == (SUBJECT, None)


def test_resolve_manifest_calibration_universe_empty_string_attribute_counts_real_foreground(
    tmp_path: Path,
):
    """The measurement this door's normalization protects: a checkpoint's stamped
    ``attribute=""`` must still count a subject-matching stem's instances as foreground, through
    :func:`~tcip_mcp.pipelines.data.splits.count_label_lines` called with the same raw ``""`` a
    caller that never normalizes its own copy would pass. Before ``count_label_lines`` normalized
    it itself, ``a.attributes.get("")`` matched no record (no annotation carries the key ``""``)
    and every stem counted zero, even though the manifest itself is unscoped by attribute. Indexed
    into the door's return rather than unpacked, so this stays a measurement proof rather than a
    return-arity one."""
    from tcip_mcp.pipelines.data.splits import (
        count_label_lines, resolve_manifest_calibration_universe,
    )

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    labels_dir = root / "annotations" / DATES[0]

    result = resolve_manifest_calibration_universe(
        manifest, str(out), str(labels_dir), str(root / "images" / DATES[0]), SUBJECT, "",
        list(_STEMS))
    stems, cal_date = result[0], result[4]

    assert stems
    assert cal_date == DATES[0]
    counts = {s: count_label_lines(labels_dir, s, subject=SUBJECT, attribute="") for s in stems}
    assert any(c > 0 for c in counts.values())


def test_resolve_manifest_calibration_universe_scope_check_reaches_the_calibration_door(
    tmp_path: Path, monkeypatch,
):
    """Marker proof that resolve_manifest_calibration_universe reaches manifest_scope_issues,
    the one accumulator every manifest-scope consumer shares: a site that stopped calling it
    would pass this test's own scenario silently instead of raising the marker below."""
    import tcip_mcp.pipelines.data.splits as splits_mod

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    monkeypatch.setattr(
        splits_mod, "manifest_scope_issues",
        lambda *a, **k: (["MARKER-CALIBRATION-DOOR-SCOPE-ISSUE"], None),
    )

    with pytest.raises(ValueError, match="MARKER-CALIBRATION-DOOR-SCOPE-ISSUE"):
        splits_mod.resolve_manifest_calibration_universe(
            manifest, str(out), str(root / "annotations" / DATES[0]),
            str(root / "images" / DATES[0]), SUBJECT, None, list(_STEMS),
        )


def test_calibrate_operating_point_manifest_conflicts_with_group_by(tmp_path: Path):
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    kwargs = dict(_CAL_KWARGS)

    with pytest.raises(ValueError, match="group_by"):
        calibration.calibrate_operating_point(
            _CalStub(), "bud_opening", str(root / "annotations" / DATES[0]),
            str(root / "images" / DATES[0]), split_manifest_dir=str(out), group_by="stem",
            **kwargs)


def test_calibrate_operating_point_manifest_refuses_an_images_root_mismatch(tmp_path: Path):
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    other_images = tmp_path / "elsewhere"
    other_images.mkdir()

    with pytest.raises(ValueError, match="images_root"):
        calibration.calibrate_operating_point(
            _CalStub(), "bud_opening", str(root / "annotations" / DATES[0]), str(other_images),
            split_manifest_dir=str(out), **_CAL_KWARGS)


def test_calibrate_operating_point_manifest_requires_images_dir(tmp_path: Path):
    """A labels-only universe can include a stem whose image is gone, a lock the redraw would
    address that no manifest-restricted calibration ever draws; refuse by name rather than raise
    a bare ``KeyError`` out of the stem-to-image narrowing."""
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    kwargs = dict(_CAL_KWARGS)

    with pytest.raises(ValueError, match="images_dir"):
        calibration.calibrate_operating_point(
            _CalStub(), "bud_opening", str(root / "annotations" / DATES[0]), None,
            split_manifest_dir=str(out), **kwargs)


def test_calibrate_operating_point_manifest_refuses_a_moved_images_root_by_name(tmp_path: Path):
    """A manifest's recorded images_root that no longer exists on disk answers the named
    refusal, never a bare crash from comparing against a gone path."""
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    moved = tmp_path / "elsewhere"
    (root / "images" / DATES[0]).rename(moved)

    with pytest.raises(ValueError, match="images_root"):
        calibration.calibrate_operating_point(
            _CalStub(), "bud_opening", str(root / "annotations" / DATES[0]), str(moved),
            split_manifest_dir=str(out), **_CAL_KWARGS)


def test_calibrate_operating_point_refuses_a_checkpoint_bound_to_a_different_manifest(
    tmp_path: Path,
):
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    other_out = tmp_path / "m2"
    _draw(root, other_out, seed=3)

    bound = _CalStub()
    bound.config["data"]["split"] = {"manifest_binding": {"manifest_dir": str(other_out)}}

    with pytest.raises(ValueError, match="bound to split manifest"):
        calibration.calibrate_operating_point(
            bound, "bud_opening", str(root / "annotations" / DATES[0]),
            str(root / "images" / DATES[0]), split_manifest_dir=str(out), **_CAL_KWARGS)


def test_calibrate_operating_point_admits_a_bound_checkpoint_under_its_own_manifest_respelled(
    tmp_path: Path,
):
    """The bound-checkpoint manifest comparison resolves both paths through filesystem identity,
    not a bare string comparison: a trailing separator, forward slashes or a relative spelling of
    the same manifest directory is still the checkpoint's own manifest."""
    import os

    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    bound = _CalStub()
    bound.config["data"]["split"] = {"manifest_binding": {"manifest_dir": str(out)}}

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        respellings = [str(out) + os.sep, str(out).replace(os.sep, "/"), os.path.relpath(out)]
        for spelling in respellings:
            _bundle, _dh, _n_excl, evidence = calibration.calibrate_operating_point(
                bound, "bud_opening", str(root / "annotations" / DATES[0]),
                str(root / "images" / DATES[0]), split_manifest_dir=spelling, **_CAL_KWARGS)
            assert evidence["reference_inputs"]["stated_values"]["split_manifest_dir"] == spelling
    finally:
        os.chdir(cwd)


# -- run_inference's own split_manifest_dir refusal --------------------------------


def test_run_inference_refuses_split_manifest_dir_without_calibration_labels_dir(tmp_path: Path):
    """A manifest with no ``calibration_labels_dir`` scopes a calibration that will never run
    (``trait``/``calibration_labels_dir`` is what turns the manifest into a bounded universe), so
    the call is refused by name rather than silently training without validation."""
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    result = run_inference(
        str(ckpt), images_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
        split_manifest_dir=str(tmp_path / "m"))

    assert "error" in result and "split_manifest_dir" in result["error"]


# -- redraw_calibration_holdout with a manifest --------------------------------


def test_force_redraw_binds_to_the_manifest_and_records_its_dir(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    result = redraw_calibration_holdout(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(root / "images" / DATES[0]), split_manifest_dir=str(out),
        subject=SUBJECT, reason="test redraw",
    )

    assert "error" not in result
    calibration_this_date = set(_calibration_this_date(manifest))
    train_this_date = _train_this_date(manifest)
    new_members = set(result["new_membership"]["calibration"]) | set(
        result["new_membership"]["holdout"])
    assert new_members == calibration_this_date
    assert new_members.isdisjoint(train_this_date)


def test_force_redraw_manifest_requires_subject(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    result = redraw_calibration_holdout(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(root / "images" / DATES[0]), split_manifest_dir=str(out),
        reason="test redraw",
    )

    assert "error" in result and "subject" in result["error"]


def test_force_redraw_manifest_requires_images_dir(tmp_path: Path):
    """A labels-only universe can include a stem whose image is gone, a lock the redraw would
    address that no manifest-restricted calibration ever draws; refuse rather than address it."""
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    result = redraw_calibration_holdout(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        split_manifest_dir=str(out), subject=SUBJECT, reason="test redraw",
    )

    assert "error" in result and "images_dir" in result["error"]


def test_force_redraw_refuses_a_moved_images_root_by_name(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    moved = tmp_path / "elsewhere"
    (root / "images" / DATES[0]).rename(moved)

    result = redraw_calibration_holdout(
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
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    calibration_this_date = _calibration_this_date(manifest)
    missing = calibration_this_date[0]
    (root / "images" / DATES[0] / f"{missing}.jpg").unlink()

    present, _ = label_image_stems(
        str(root / "annotations" / DATES[0]), str(root / "images" / DATES[0]))
    expected_universe, _gb, _gkm, _excl = calibration_universe_from_manifest(
        manifest, DATES[0], present)
    expected_hash = dataset_hash(str(root / "annotations" / DATES[0]), stems=expected_universe)

    result = redraw_calibration_holdout(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(root / "images" / DATES[0]), split_manifest_dir=str(out),
        subject=SUBJECT, reason="test redraw",
    )

    assert "error" not in result
    assert result["identity_hash"] == expected_hash


# -- a manifest-restricted calibration's evidence, through the real count door -----


def test_manifest_calibrations_evidence_earns_a_validated_record_through_export(
        tmp_path: Path, monkeypatch):
    """A manifest-restricted calibration's evidence, driven through the real count door
    (``run_inference``), earns a record whose reference identity carries the manifest's
    universe, and the delivery reader's own verification of the stamp's binding passes against
    the bucket as it was actually written."""
    import tcip_mcp.pipelines.calibration as calibration
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    import tcip_mcp.tools.inference_tools as itools
    from tests._dense_op_fixtures import dense_records

    from tcip_mcp.experiments import find_validation
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import (
        dataset_hash, read_operating_point_sidecar, verify_stamp_binding,
    )

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    universe = ["a", "b"]
    dh = dataset_hash(root / "annotations" / DATES[0], stems=universe)

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    inputs = {
        "dataset_hash": dh, "tiled": False, "staged_conf_floor": 0.01,
        "calibration_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "holdout_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="h", shift=5.0,
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
    }
    bundle = resolve_operating_point("bud_opening", experiment_id=None, **inputs)
    evidence = {
        "resolver": "resolve_operating_point", "inputs": inputs,
        "reference_inputs": {
            "label_stems": {"calibration": {
                "path": str(root / "annotations" / DATES[0]), "stems": universe}},
            "stated_values": {"split_manifest_dir": str(out)},
        },
        "calibration_stems": universe,
    }
    monkeypatch.setattr(calibration, "calibrate_operating_point",
                        lambda *a, **k: (bundle, dh, 0, evidence))

    class _BucketStub:
        def __init__(self) -> None:
            from types import SimpleNamespace

            self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
            self.device = "cpu"
            self.score_threshold = 0.5
            self.train_tile_size = None
            self.train_overlap = None

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": IMG, "height": IMG,
                     "boxes": [[2, 2, 10, 10]], "scores": [0.95], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda checkpoint, **kw: _BucketStub())
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    result = itools.run_inference(
        str(ckpt), images_dir=str(root / "images" / DATES[0]),
        output_dir=str(root / "predictions" / "baseline" / DATES[0]),
        device="cpu", tile=False, trait="bud_opening",
        calibration_labels_dir=str(root / "annotations" / DATES[0]),
        split_manifest_dir=str(out))
    assert "error" not in result, result
    bucket = result["output_dir"]

    stamp = read_operating_point_sidecar(bucket)
    binding = verify_stamp_binding(stamp, bucket, document="operating_point", trait="bud_opening")
    assert binding.ok is True
    assert binding.claimed is True

    pointer = stamp["validated_by"]
    row = find_validation(pointer["experiment_id"], pointer["record_digest"])
    identity = row["reference_identity"]
    assert identity["label_stems"]["calibration"]["count"] == len(universe)
    assert identity["stated_values"]["split_manifest_dir"] == str(out)
    assert row["selection_disjointness"]["applicable"] is False


def test_count_door_round_trip_earns_a_checked_selection_disjointness(tmp_path, monkeypatch):
    """``draw_splits`` draws three sides; a real bound launch binds to the manifest's train/val;
    ``run_inference`` calibrates under the manifest with that run as producer; the sealed
    row carries ``label_stems.calibration`` and a checked, leak-free ``selection_disjointness``;
    and ``verify_stamp_binding`` verifies the delivered bucket."""
    import tcip_mcp.pipelines.calibration as calibration
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    import tcip_mcp.tools.inference_tools as itools
    from tests._dense_op_fixtures import dense_records

    from tcip_mcp.experiments import create_experiment, find_validation
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import (
        dataset_hash, read_operating_point_sidecar, verify_stamp_binding,
    )
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)

    data_cfg = {
        "images_dir": str(root / "images" / DATES[0]),
        "labels_dir": str(root / "annotations" / DATES[0]),
        "subject": SUBJECT, "attribute": None,
        "split": {"manifest_dir": str(out)},
    }
    experiment_id = "exp_round_trip_bound"
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    create_experiment(experiment_id, {})
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg)

    universe = _calibration_this_date(manifest)
    dh = dataset_hash(root / "annotations" / DATES[0], stems=universe)

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    inputs = {
        "dataset_hash": dh, "tiled": False, "staged_conf_floor": 0.01,
        "calibration_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "holdout_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="h", shift=5.0,
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "split_manifest_dir": str(out), "calibration_date": DATES[0],
    }
    bundle = resolve_operating_point("bud_opening", experiment_id=experiment_id, **inputs)
    assert bundle.get("conf").gate_evidence["selection_disjointness"]["checked"] is True
    assert not bundle.get("conf").gate_evidence["selection_disjointness"]["leaked_groups"]
    assert not bundle.get("conf").gate_evidence["selection_disjointness"]["leaked_stems"]
    evidence = {
        "resolver": "resolve_operating_point", "inputs": inputs,
        "reference_inputs": {
            "label_stems": {"calibration": {
                "path": str(root / "annotations" / DATES[0]), "stems": universe}},
            "stated_values": {"split_manifest_dir": str(out)},
        },
        "calibration_stems": universe,
    }
    monkeypatch.setattr(calibration, "calibrate_operating_point",
                        lambda *a, **k: (bundle, dh, 0, evidence))

    class _BucketStub:
        def __init__(self) -> None:
            self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
            self.device = "cpu"
            self.score_threshold = 0.5
            self.train_tile_size = None
            self.train_overlap = None

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": IMG, "height": IMG,
                     "boxes": [[2, 2, 10, 10]], "scores": [0.95], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(predictor_mod, "build_predictor", lambda checkpoint, **kw: _BucketStub())
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    result = itools.run_inference(
        str(ckpt), images_dir=str(root / "images" / DATES[0]),
        output_dir=str(root / "predictions" / "bound" / DATES[0]),
        device="cpu", tile=False, trait="bud_opening",
        calibration_labels_dir=str(root / "annotations" / DATES[0]),
        split_manifest_dir=str(out), experiment_id=experiment_id)
    assert "error" not in result, result
    bucket = result["output_dir"]

    stamp = read_operating_point_sidecar(bucket)
    binding = verify_stamp_binding(stamp, bucket, document="operating_point", trait="bud_opening")
    assert binding.ok is True
    assert binding.claimed is True

    pointer = stamp["validated_by"]
    row = find_validation(pointer["experiment_id"], pointer["record_digest"])
    assert row["reference_identity"]["label_stems"]["calibration"]["count"] == len(universe)
    assert row["selection_disjointness"]["applicable"] is True
    assert row["selection_disjointness"]["checked"] is True
