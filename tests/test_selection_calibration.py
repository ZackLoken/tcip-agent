"""Inference-side binding to a named selection: the calibration universe is the selection's
third, calibration side under the labels directory the door names, instead of every labelled
stem with an image.
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


def _draw(root: Path, out: Path, *, seed: int = 2):
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(str(root), output_path=str(out), subject=SUBJECT, seed=seed,
                         train_ratio=0.4, val_ratio=0.3, calibration_ratio=0.3)
    assert "error" not in result, result
    return read_selection(out)


def _side_this_date(selection, side: str, date: str = DATES[0]) -> set[str]:
    return {Path(s.ground_truth).stem for s in selection.on(side)
            if Path(s.ground_truth).parent.name == date}


def _calibration_this_date(selection, date: str = DATES[0]) -> list[str]:
    return sorted(_side_this_date(selection, "calibration", date))


def _labels_dir(root: Path, date: str = DATES[0]) -> Path:
    return root / "annotations" / date


# -- selection_calibration_universe --------------------------------------------


def test_selection_calibration_universe_holds_only_calibration_samples_present(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import selection_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    drawn = _draw(root, tmp_path / "m")

    stems, group_by, group_key_map, excluded, _samples = selection_calibration_universe(
        drawn, _labels_dir(root), present=list(_STEMS))

    assert set(stems) == set(_calibration_this_date(drawn))
    assert set(excluded["excluded_training_stems"]) == _side_this_date(drawn, "train")
    assert set(excluded["excluded_validation_stems"]) == _side_this_date(drawn, "val")
    assert excluded["excluded_unassigned_stems"] == []
    assert group_by == "explicit_map"
    assert set(group_key_map) == set(stems)


def test_selection_calibration_universe_reads_only_the_named_labels_directory(tmp_path: Path):
    """The other date's calibration samples belong to another directory: a door narrows to the
    one it named rather than to a capture date it compared against a record."""
    from tcip_mcp.pipelines.data.splits import selection_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    drawn = _draw(root, tmp_path / "m")

    stems, *_rest = selection_calibration_universe(
        drawn, _labels_dir(root, DATES[1]), present=list(_STEMS),
        min_foreground_groups={"calibration": 1})

    assert set(stems) == set(_calibration_this_date(drawn, DATES[1]))
    assert set(stems) != set(_calibration_this_date(drawn, DATES[0]))


def test_selection_calibration_universe_refuses_a_recorded_member_the_listing_lost(
    tmp_path: Path,
):
    """The selection says which samples were held out. A member the door's own listing no longer
    holds is data that moved under the draw, named here rather than dropped into a universe one
    stem smaller than the one recorded."""
    from tcip_mcp.pipelines.data.splits import selection_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    drawn = _draw(root, tmp_path / "m")
    calibration_this_date = _calibration_this_date(drawn)
    assert len(calibration_this_date) >= 3, "fixture must leave room to drop one and still have >=2"
    present_minus_one = [s for s in _STEMS if s != calibration_this_date[0]]

    with pytest.raises(ValueError, match=calibration_this_date[0]) as raised:
        selection_calibration_universe(drawn, _labels_dir(root), present=present_minus_one)

    assert "draw the selection again" in str(raised.value)


def test_selection_calibration_universe_reports_an_unassigned_present_stem(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import selection_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    drawn = _draw(root, tmp_path / "m")

    _stems, _gb, _gkm, excluded, _samples = selection_calibration_universe(
        drawn, _labels_dir(root), present=list(_STEMS) + ["never_drawn"])

    assert excluded["excluded_unassigned_stems"] == ["never_drawn"]


def test_selection_calibration_universe_refuses_fewer_than_two_groups(tmp_path: Path):
    """The refusal names a remedy: a fresh draw with a larger calibration ratio or more
    foreground groups under this labels directory."""
    from tcip_mcp.pipelines.data.splits import selection_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    drawn = _draw(root, tmp_path / "m")
    held_out = {s.group for s in drawn.on("calibration")
                if Path(s.ground_truth).parent.name == DATES[0]}

    with pytest.raises(ValueError, match="calibration_ratio"):
        selection_calibration_universe(
            drawn, _labels_dir(root), present=list(_STEMS),
            min_foreground_groups={"calibration": len(held_out) + 1})


def test_selection_calibration_universe_floor_is_foreground_aware(tmp_path: Path):
    """A universe with enough raw groups but only one carrying real foreground still refuses:
    the floor counts foreground groups, not bare group presence, when the caller states which
    stems are foreground."""
    from tcip_mcp.pipelines.data.splits import selection_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    drawn = _draw(root, tmp_path / "m")
    calibration_this_date = _calibration_this_date(drawn)
    assert len(calibration_this_date) >= 3

    with pytest.raises(ValueError, match="foreground group"):
        selection_calibration_universe(
            drawn, _labels_dir(root), present=list(_STEMS),
            foreground_stems={calibration_this_date[0]})


def test_selection_calibration_universe_refuses_a_calibration_sample_naming_a_rect(
    tmp_path: Path,
):
    """No loader windows to a recorded rect. A door that predicted over the whole source would
    measure a region against the pixels around it, so every consumer of a selection's samples
    passes through the loaders' own refusal rather than half of them honoring the field."""
    from dataclasses import replace

    from tcip_mcp.pipelines.data.selection import write_selection
    from tcip_mcp.pipelines.data.splits import selection_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    regioned = tuple(replace(s, rect=(0, 0, IMG // 2, IMG // 2)) if s.side == "calibration" else s
                     for s in drawn.samples)
    write_selection(out, replace(drawn, samples=regioned))

    from tcip_mcp.pipelines.data.selection import read_selection

    with pytest.raises(ValueError, match="pixel rect"):
        selection_calibration_universe(
            read_selection(out), _labels_dir(root), present=list(_STEMS))


def test_a_calibration_door_refuses_a_selections_recorded_rect(tmp_path: Path):
    """The same refusal reached through the real door: a rect-bearing calibration side stops the
    calibration instead of being measured over whole sources."""
    from dataclasses import replace

    import tcip_mcp.pipelines.calibration as calibration
    from tcip_mcp.pipelines.data.selection import write_selection

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    regioned = tuple(replace(s, rect=(0, 0, IMG // 2, IMG // 2)) if s.side == "calibration" else s
                     for s in drawn.samples)
    write_selection(out, replace(drawn, samples=regioned))

    with pytest.raises(ValueError, match="pixel rect"):
        calibration.calibrate_operating_point(
            _CalStub(), "bud_opening", str(root / "annotations" / DATES[0]),
            str(root / "images" / DATES[0]), selection_dir=str(out), **_CAL_KWARGS)


def test_resolve_selection_calibration_universe_reads_the_scope_off_the_selection(tmp_path: Path):
    """The door states no subject or attribute of its own: the selection records the scope it
    admitted under, and the foreground count is taken under that scope."""
    from tcip_mcp.pipelines.data.splits import resolve_selection_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    drawn = _draw(root, tmp_path / "m")

    stems, group_by, group_key_map, excluded, subject, attribute, samples = \
        resolve_selection_calibration_universe(drawn, _labels_dir(root), list(_STEMS))

    assert (subject, attribute) == (SUBJECT, None)
    assert set(stems) == set(_calibration_this_date(drawn))
    assert group_by == "explicit_map" and set(group_key_map) == set(stems)
    assert set(excluded) == {
        "excluded_training_stems", "excluded_validation_stems", "excluded_unassigned_stems"}


def test_resolve_selection_calibration_universe_refuses_a_directory_the_draw_never_held(
    tmp_path: Path,
):
    """A labels directory the selection holds no calibration sample under gives an empty
    universe, refused by the floor naming the directory rather than answering nothing."""
    from tcip_mcp.pipelines.data.splits import resolve_selection_calibration_universe

    root = _two_date_dataset(tmp_path / "ds")
    drawn = _draw(root, tmp_path / "m")
    elsewhere = root / "annotations" / "9-9-99"
    elsewhere.mkdir(parents=True)

    with pytest.raises(ValueError, match=str(elsewhere.name)):
        resolve_selection_calibration_universe(drawn, elsewhere, list(_STEMS))


def test_a_selection_restricted_calibration_reads_its_own_recorded_sources(tmp_path: Path):
    """A second directory holds identically named images with different pixels. The calibration
    must measure on the images the draw held out, which is what each sample records, rather than
    on whatever the caller's own images directory happens to list under the same names."""
    from tcip_mcp.pipelines.data.splits import (
        label_image_stems, resolve_selection_calibration_universe,
    )
    from tcip_mcp.pipelines.image_utils import resolve_source_path

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    decoy = tmp_path / "decoy" / "images"
    decoy.mkdir(parents=True)
    for stem in _STEMS:
        from PIL import Image

        Image.new("RGB", (IMG, IMG), color=(5, 5, 5)).save(decoy / f"{stem}.jpg")

    labels_dir = _labels_dir(root)
    present, _ = label_image_stems(str(labels_dir), str(decoy))
    stems, _gb, _gkm, _excl, _subject, _attribute, samples = \
        resolve_selection_calibration_universe(drawn, str(labels_dir), present)

    assert stems
    for stem in stems:
        # Each universe stem answers with its own recorded source under the real dataset, never
        # the same-named decoy the caller's listing would have handed a directory-driven read.
        source = Path(str(resolve_source_path(samples[stem].source)))
        assert source.parent == root / "images" / DATES[0]
        assert decoy not in source.parents
        assert samples[stem].ground_truth == str(labels_dir / f"{stem}.json")


# -- resolve_locked_cal_holdout_split's selection_dir ---------------------------


def test_resolve_locked_cal_holdout_split_records_the_selection_dir(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split

    locked = resolve_locked_cal_holdout_split(
        ["a", "b", "c", "d"], identity_hash="ident1", scope_root=tmp_path,
        selection_dir="some/selection/dir",
    )

    assert locked["selection_dir"] == "some/selection/dir"
    assert locked["redraw_history"][0]["policy"]["selection_dir"] == "some/selection/dir"


def test_a_whole_directory_lock_and_a_selection_lock_coexist(tmp_path: Path):
    """Two distinct identities (a whole-directory hash and a universe hash) draw and answer
    from two distinct locks over the same scope root."""
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split

    whole = resolve_locked_cal_holdout_split(
        ["a", "b", "c", "d"], identity_hash="whole_ident", scope_root=tmp_path)
    scoped = resolve_locked_cal_holdout_split(
        ["a", "b"], identity_hash="selection_ident", scope_root=tmp_path,
        selection_dir="some/selection/dir")

    assert whole["selection_dir"] is None
    assert scoped["selection_dir"] == "some/selection/dir"
    assert set(whole["calibration"]) | set(whole["holdout"]) == {"a", "b", "c", "d"}
    assert set(scoped["calibration"]) | set(scoped["holdout"]) == {"a", "b"}

    # Re-resolving each by its own identity still answers from its own lock, unchanged.
    again_whole = resolve_locked_cal_holdout_split(
        ["a", "b", "c", "d"], identity_hash="whole_ident", scope_root=tmp_path)
    again_scoped = resolve_locked_cal_holdout_split(
        ["a", "b"], identity_hash="selection_ident", scope_root=tmp_path,
        selection_dir="some/selection/dir")
    assert again_whole == whole
    assert again_scoped == scoped


# -- attach_split_policy_provenance --------------------------------------------


def test_attach_split_policy_provenance_copies_the_selection_dir():
    from tcip_mcp.pipelines.operating_point import attach_split_policy_provenance
    from tcip_mcp.pipelines.resolution import ResolvedBundle, derived

    conf = derived("conf", 0.5, derived_from="test", requires_validation=True,
                   validation_kind="annotations", validated_against=None, gate_evidence={})
    bundle = ResolvedBundle(trait="bud_opening", dataset_hash=None, params={"conf": conf})

    attach_split_policy_provenance(bundle, {"group_by": "stem", "seed": 0, "holdout_ratio": 0.5,
                                            "identity_hash": "abc", "selection_dir": "m/dir"})

    assert bundle.get("conf").gate_evidence["split_policy"]["selection_dir"] == "m/dir"


# -- _reference_identity's label_stems group -----------------------------------


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


# -- calibrate_operating_point's selection branch ------------------------------


class _CalStub:
    """A predictor with the mutable operating-point surface the calibration sets, predicting
    nothing: enough to drive the selection checks without a real forward pass."""

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


def test_calibrate_operating_point_binds_to_the_selections_calibration_side(tmp_path: Path):
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    bundle, dh, _n_excl, evidence = calibration.calibrate_operating_point(
        _CalStub(), "bud_opening", str(root / "annotations" / DATES[0]),
        str(root / "images" / DATES[0]), selection_dir=str(out), **_CAL_KWARGS)

    calibration_this_date = set(_calibration_this_date(drawn))
    assert set(evidence["calibration_stems"]) == calibration_this_date
    assert "label_stems" in evidence["reference_inputs"]
    assert evidence["reference_inputs"]["stated_values"]["selection_dir"] == str(out)
    # The persisted evidence's own inputs, what a delivery door reopens the gate with, carry the
    # selection and the directory it read: nothing pins the producer writing these two otherwise.
    assert evidence["inputs"]["selection_dir"] == str(out)
    assert evidence["inputs"]["calibration_labels_dir"] == str(root / "annotations" / DATES[0])
    assert evidence["inputs"]["selection_sha256"]
    from tcip_mcp.pipelines.resolution import dataset_hash
    assert dh == dataset_hash(
        str(root / "annotations" / DATES[0]), stems=sorted(calibration_this_date))


def test_calibrate_operating_point_refuses_a_checkpoint_trained_for_another_class_space(
    tmp_path: Path,
):
    """A model only speaks its own training vocabulary, so a checkpoint trained for one subject
    cannot be measured against a selection drawn for another: the refusal names both rather than
    letting the class-id read fail deep inside the reference build."""
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    with pytest.raises(ValueError, match="training vocabulary"):
        calibration.calibrate_operating_point(
            _CalStub(subject="a_different_subject"), "bud_opening",
            str(root / "annotations" / DATES[0]), str(root / "images" / DATES[0]),
            selection_dir=str(out), **_CAL_KWARGS)


def test_calibrate_operating_point_selection_conflicts_with_group_by(tmp_path: Path):
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    with pytest.raises(ValueError, match="group_by"):
        calibration.calibrate_operating_point(
            _CalStub(), "bud_opening", str(root / "annotations" / DATES[0]),
            str(root / "images" / DATES[0]), selection_dir=str(out), group_by="stem",
            **_CAL_KWARGS)


def test_calibrate_operating_point_selection_requires_images_dir(tmp_path: Path):
    """A labels-only universe can include a stem whose image is gone, a lock the redraw would
    address that no selection-restricted calibration ever draws; refuse by name rather than raise
    a bare ``KeyError`` out of the stem-to-image narrowing."""
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    with pytest.raises(ValueError, match="images_dir"):
        calibration.calibrate_operating_point(
            _CalStub(), "bud_opening", str(root / "annotations" / DATES[0]), None,
            selection_dir=str(out), **_CAL_KWARGS)


def test_calibrate_operating_point_refuses_a_checkpoint_bound_to_a_different_selection(
    tmp_path: Path,
):
    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    other_out = tmp_path / "m2"
    _draw(root, other_out, seed=3)

    bound = _CalStub()
    bound.config["data"]["split"] = {"selection_binding": {"selection_dir": str(other_out)}}

    with pytest.raises(ValueError, match="bound to the selection"):
        calibration.calibrate_operating_point(
            bound, "bud_opening", str(root / "annotations" / DATES[0]),
            str(root / "images" / DATES[0]), selection_dir=str(out), **_CAL_KWARGS)


def test_calibrate_operating_point_admits_a_bound_checkpoint_under_its_own_selection_respelled(
    tmp_path: Path,
):
    """The bound-checkpoint comparison resolves both paths through filesystem identity, not a
    bare string comparison: a trailing separator, forward slashes or a relative spelling of the
    same selection directory is still the checkpoint's own selection."""
    import os

    import tcip_mcp.pipelines.calibration as calibration

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    bound = _CalStub()
    bound.config["data"]["split"] = {"selection_binding": {"selection_dir": str(out)}}

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        respellings = [str(out) + os.sep, str(out).replace(os.sep, "/"), os.path.relpath(out)]
        for spelling in respellings:
            _bundle, _dh, _n_excl, evidence = calibration.calibrate_operating_point(
                bound, "bud_opening", str(root / "annotations" / DATES[0]),
                str(root / "images" / DATES[0]), selection_dir=spelling, **_CAL_KWARGS)
            assert evidence["reference_inputs"]["stated_values"]["selection_dir"] == spelling
    finally:
        os.chdir(cwd)


# -- evaluate_model under a selection ------------------------------------------


def _evaluate_under(root: Path, out: Path, tmp_path: Path, **kwargs):
    from tcip_mcp.tools.training_tools import evaluate_model

    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    return evaluate_model(
        ckpt, str(root / "images" / DATES[0]), str(root / "annotations" / DATES[0]),
        task="detection", selection_dir=str(out), **kwargs)


def _corner_dataset(root: Path) -> Path:
    """The two-date dataset with every box in the bottom-right quadrant, so a keep region over one
    quadrant either holds every source's ground truth or none of it."""
    for date in DATES:
        images_dir, labels_dir = root / "images" / date, root / "annotations" / date
        for stem in _STEMS:
            _save_png(images_dir / f"{stem}.jpg")
            json_io.write_annotations(
                str(labels_dir / f"{stem}.json"),
                [Annotation(subject=SUBJECT, geometry=BBox(20, 20, 28, 28))], IMG, IMG,
            )
    return root


_QUADRANT_TILING = {"tile_size": IMG // 2, "overlap": 0, "skip_empty": True}


def test_selected_evaluation_refuses_a_tiling_that_drops_a_held_out_source(
    tmp_path: Path, monkeypatch,
):
    """Admission answers for the samples; this answers for the loader built from them. A tiling
    whose keep regions hold no ground truth of a source leaves that source with no tile at all, so
    the measurement would be taken over part of the universe the draw held out and reported as the
    whole of it."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _corner_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    result = _evaluate_under(
        root, out, tmp_path,
        tiling={**_QUADRANT_TILING, "keep_regions": [(0, 0, IMG // 2, IMG // 2)]})

    assert "error" in result
    assert "indexes nothing" in result["error"]
    for stem in _calibration_this_date(drawn)[:1]:
        assert stem in result["error"]


def test_selected_evaluation_admits_a_tiling_that_keeps_every_held_out_source(
    tmp_path: Path, monkeypatch,
):
    """The admitting half of the same rail: a keep region holding every source's ground truth
    retains all of them, and the measurement runs over the universe the draw held out."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _corner_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    result = _evaluate_under(
        root, out, tmp_path,
        tiling={**_QUADRANT_TILING, "keep_regions": [(IMG // 2, IMG // 2, IMG, IMG)]})

    assert "error" not in result, result
    assert result["evaluated_stem_count"] == len(_calibration_this_date(drawn))


def test_selected_evaluation_measures_the_universe_the_draw_held_out(tmp_path: Path,
                                                                     monkeypatch):
    """The admitting half: an untouched calibration side is measured whole, and the record says
    how many of its samples the loader indexed."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    result = _evaluate_under(root, out, tmp_path)

    assert "error" not in result, result
    assert result["evaluated_stem_count"] == len(_calibration_this_date(drawn))
    assert result["selection_dir"] == str(out)


def test_selected_evaluation_refuses_a_calibration_label_emptied_since_the_draw(
    tmp_path: Path, monkeypatch,
):
    """A held-out label emptied with nobody confirming that image negative would be scored as an
    image with no objects, turning a real object into a false positive against the operating
    point. The evaluation refuses by name instead, the way a bound run's own loaders do."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    emptied = next(s for s in drawn.on("calibration")
                   if Path(s.ground_truth).parent.name == DATES[0])
    json_io.write_annotations(emptied.ground_truth, [], IMG, IMG, keep_empty=True)

    result = _evaluate_under(root, out, tmp_path)

    assert "error" in result and "no longer admissible" in result["error"]


# -- run_inference's own selection_dir refusal ---------------------------------


def test_run_inference_refuses_selection_dir_without_calibration_labels_dir(tmp_path: Path):
    """A selection with no ``calibration_labels_dir`` scopes a calibration that will never run
    (``trait``/``calibration_labels_dir`` is what turns it into a bounded universe), so the call
    is refused by name rather than silently dropping the scope."""
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    result = run_inference(
        str(ckpt), images_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
        selection_dir=str(tmp_path / "m"))

    assert "error" in result and "selection_dir" in result["error"]


# -- redraw_calibration_holdout with a selection -------------------------------


def test_force_redraw_binds_to_the_selection_and_records_its_dir(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    result = redraw_calibration_holdout(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(root / "images" / DATES[0]), selection_dir=str(out),
        subject=SUBJECT, reason="test redraw",
    )

    assert "error" not in result
    calibration_this_date = set(_calibration_this_date(drawn))
    new_members = set(result["new_membership"]["calibration"]) | set(
        result["new_membership"]["holdout"])
    assert new_members == calibration_this_date
    assert new_members.isdisjoint(_side_this_date(drawn, "train"))


def test_force_redraw_selection_requires_subject(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    result = redraw_calibration_holdout(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(root / "images" / DATES[0]), selection_dir=str(out),
        reason="test redraw",
    )

    assert "error" in result and "subject" in result["error"]


def test_force_redraw_selection_requires_images_dir(tmp_path: Path):
    """A labels-only universe can include a stem whose image is gone, a lock the redraw would
    address that no selection-restricted calibration ever draws; refuse rather than address it."""
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    result = redraw_calibration_holdout(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        selection_dir=str(out), subject=SUBJECT, reason="test redraw",
    )

    assert "error" in result and "images_dir" in result["error"]


def test_force_redraw_selection_refuses_the_same_missing_image_the_universe_names(tmp_path: Path):
    """A held-out sample whose image is gone is not a smaller universe: the redraw and the
    universe refuse on the same member by name, so neither can lock a split the other would never
    draw."""
    from tcip_mcp.pipelines.data.splits import label_image_stems, selection_calibration_universe
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)
    missing = _calibration_this_date(drawn)[0]
    (root / "images" / DATES[0] / f"{missing}.jpg").unlink()

    present, _ = label_image_stems(
        str(root / "annotations" / DATES[0]), str(root / "images" / DATES[0]))
    with pytest.raises(ValueError, match=missing):
        selection_calibration_universe(drawn, _labels_dir(root), present)

    result = redraw_calibration_holdout(
        dataset_root=str(root), labels_dir=str(root / "annotations" / DATES[0]),
        images_dir=str(root / "images" / DATES[0]), selection_dir=str(out),
        subject=SUBJECT, reason="test redraw",
    )

    assert "error" in result and missing in result["error"]


# -- a selection-restricted calibration's evidence, through the real count door ---


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


def _dense_inputs(dh: str, **extra) -> dict:
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    return {
        "dataset_hash": dh, "tiled": False, "staged_conf_floor": 0.01,
        "calibration_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "holdout_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="h", shift=5.0,
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        **extra,
    }


def test_selection_calibrations_evidence_earns_a_validated_record_through_export(
        tmp_path: Path, monkeypatch):
    """A selection-restricted calibration's evidence, driven through the real count door
    (``run_inference``), earns a record whose reference identity carries the selection's
    universe, and the delivery reader's own verification of the stamp's binding passes against
    the bucket as it was actually written."""
    import tcip_mcp.pipelines.calibration as calibration
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    import tcip_mcp.tools.inference_tools as itools

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

    inputs = _dense_inputs(dh)
    bundle = resolve_operating_point("bud_opening", experiment_id=None, **inputs)
    evidence = {
        "resolver": "resolve_operating_point", "inputs": inputs,
        "reference_inputs": {
            "label_stems": {"calibration": {
                "path": str(root / "annotations" / DATES[0]), "stems": universe}},
            "stated_values": {"selection_dir": str(out)},
        },
        "calibration_stems": universe,
    }
    monkeypatch.setattr(calibration, "calibrate_operating_point",
                        lambda *a, **k: (bundle, dh, 0, evidence))
    monkeypatch.setattr(predictor_mod, "build_predictor", lambda checkpoint, **kw: _BucketStub())
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    result = itools.run_inference(
        str(ckpt), images_dir=str(root / "images" / DATES[0]),
        output_dir=str(root / "predictions" / "baseline" / DATES[0]),
        device="cpu", tile=False, trait="bud_opening",
        calibration_labels_dir=str(root / "annotations" / DATES[0]),
        selection_dir=str(out))
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
    assert identity["stated_values"]["selection_dir"] == str(out)
    assert row["selection_disjointness"]["applicable"] is False


def test_a_calibration_date_holding_no_training_members_is_checked_not_unresolvable(tmp_path):
    """A draw can put every training sample on one date and a whole calibration side on another.
    The producer's foreground floor is global, so that is legitimate work: the checks read the
    directory the calibration named, find its training side explicitly empty, and answer checked
    with no leak, never "this run recorded no training provenance"."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.selection import Selection, write_selection
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition
    from tcip_mcp.pipelines.operating_point import (
        _selection_disjointness, _train_disjointness,
    )

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    # The same samples, re-sided so date A trains and validates and date B is the whole
    # calibration side: every field but the side is exactly what the draw wrote.
    def _sided(sample):
        from dataclasses import replace

        on_a = Path(sample.ground_truth).parent.name == DATES[0]
        if not on_a:
            return replace(sample, side="calibration")
        return replace(sample, side="train" if sample.group.endswith(("a", "b", "c", "d"))
                       else "val")

    write_selection(out, Selection(
        samples=tuple(_sided(s) for s in drawn.samples), subject=drawn.subject,
        attribute=drawn.attribute, id_map=drawn.id_map, seed=drawn.seed,
        group_by=drawn.group_by))

    data_cfg = {"split": {"selection_dir": str(out)}}
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    create_experiment("exp-calibration-only-date", {})
    persist_run_partition("exp-calibration-only-date", train_ds, val_ds, data_cfg,
                          partition=partition)

    cal_labels = str(_labels_dir(root, DATES[1]))
    cal_ids = {Path(s.ground_truth).stem for s in drawn.samples
               if Path(s.ground_truth).parent.name == DATES[1]}

    trained = _train_disjointness(
        "exp-calibration-only-date", cal_ids, set(), calibration_labels_dir=cal_labels)
    assert trained["unresolvable"] is False and trained["checked"] is True
    assert not trained["leaked_stems"]

    selected = _selection_disjointness(
        "exp-calibration-only-date", cal_ids, set(),
        selection_dir=str(out), calibration_labels_dir=cal_labels)
    assert selected["applicable"] is True
    assert selected["unresolvable"] is False
    assert not selected["leaked_stems"]


def test_a_crop_annotated_after_the_draw_is_grouped_by_the_policy_the_selection_recorded(
    tmp_path: Path,
):
    """A bound run records a per-stem map over its own members and the policy that drew them. A
    sibling crop of a training parent, annotated after the draw, is in no map; the recorded policy
    is what says which parent it belongs to, so a calibration universe holding it is caught as a
    leak of that parent's group rather than read as an unrelated stem."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    data_cfg = {"split": {"selection_dir": str(out)}}
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    create_experiment("exp-late-crop", {})
    persist_run_partition("exp-late-crop", train_ds, val_ds, data_cfg, partition=partition)

    labels_dir = _labels_dir(root)
    trained_here = sorted(_side_this_date(drawn, "train"))
    assert trained_here, "the fixture must train on this date for the leak to be a leak"
    parent = trained_here[0]
    late_crop = f"{parent}_9_0"
    _save_png(root / "images" / DATES[0] / f"{late_crop}.jpg")
    json_io.write_annotations(
        str(labels_dir / f"{late_crop}.json"),
        [Annotation(subject=SUBJECT, geometry=BBox(2, 2, 10, 10))], IMG, IMG)

    resolved = _train_disjointness(
        "exp-late-crop", {late_crop}, set(), calibration_labels_dir=str(labels_dir))

    recorded_key = next(s.group for s in drawn.on("train")
                        if Path(s.ground_truth).stem == parent
                        and Path(s.ground_truth).parent.name == DATES[0])
    assert resolved["leaked_groups"] == [recorded_key]


def test_a_sample_backed_loaders_records_name_the_members_the_partition_recorded(tmp_path: Path):
    """One member vocabulary across the leakage join. The loaders index by source identity so two
    dates' same-named images stay apart, and the run's recorded partition names each member by its
    bare ground-truth stem; the records a model pass writes are spelled the record's way, or every
    join a delivered number rests on matches nothing."""
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.operating_point import records_over_loader
    from tcip_mcp.pipelines.training.collation import task_collate

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    train_ds, _val_ds, partition = auto_train_val(
        "detection", {"split": {"selection_dir": str(out)}}, None)

    class _NoDetections:
        """A detector that finds nothing: the record ids are the point here, not the boxes."""

        def eval(self):
            return self

        def __call__(self, images):
            return [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                     "labels": torch.zeros(0, dtype=torch.int64)} for _ in images]

    loader = DataLoader(train_ds, batch_size=2, collate_fn=task_collate("detection"))
    records = records_over_loader(
        _NoDetections(), loader, torch.device("cpu"), "detection")

    assert {r["image_id"] for r in records} == set(partition["train"])


class _BespokeStemDataset:
    """An agent's own detection dataset: a plain Torch dataset naming its samples through the
    ``stems`` list the seam has always taken them from."""

    def __init__(self, stems) -> None:
        self.stems = list(stems)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        return torch.zeros(3, IMG, IMG), {"boxes": torch.zeros(0, 4),
                                          "labels": torch.zeros(0, dtype=torch.int64),
                                          "image_id": idx}


def build_bespoke_stem_dataset(**_kwargs) -> _BespokeStemDataset:
    """The ``dataset_source`` builder :func:`test_a_bespoke_datasets_own_stems_name_its_records`
    registers through the seam's dotted escape."""
    return _BespokeStemDataset(["a", "b"])


def test_a_bespoke_datasets_own_stems_name_its_records(tmp_path: Path):
    """The ``dataset_source`` seam takes any Torch dataset, and ``stems`` is the interface it has
    always named its samples through. A measurement that stopped reading it would name every
    record by its integer index, and every join a delivered number rests on would match nothing."""
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.operating_point import records_over_loader
    from tcip_mcp.pipelines.training.collation import task_collate

    dataset = build_dataset(
        "detection",
        dataset_source={"builder": f"{__name__}:build_bespoke_stem_dataset", "task": "detection"})

    class _NoDetections:
        def eval(self):
            return self

        def __call__(self, images):
            return [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                     "labels": torch.zeros(0, dtype=torch.int64)} for _ in images]

    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("detection"))
    records = records_over_loader(_NoDetections(), loader, torch.device("cpu"), "detection")

    assert [r["image_id"] for r in records] == ["a", "b"]


def test_count_door_round_trip_earns_a_checked_selection_disjointness(tmp_path, monkeypatch):
    """``draw_splits`` draws three sides; ``auto_train_val`` binds an experiment to the
    selection's train/val and persists that partition (no trainer runs here, and the calibration
    itself is stubbed); ``run_inference`` calibrates under it with that run as producer; the
    sealed row carries ``label_stems.calibration`` and a checked, leak-free
    ``selection_disjointness``; and ``verify_stamp_binding`` verifies the delivered bucket."""
    import tcip_mcp.pipelines.calibration as calibration
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    import tcip_mcp.tools.inference_tools as itools

    from tcip_mcp.experiments import create_experiment, find_validation
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import (
        dataset_hash, read_operating_point_sidecar, verify_stamp_binding,
    )
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition

    root = _two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    drawn = _draw(root, out)

    data_cfg = {"split": {"selection_dir": str(out)}}
    experiment_id = "exp_round_trip_bound"
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    create_experiment(experiment_id, {})
    persist_run_partition(experiment_id, train_ds, val_ds, data_cfg, partition=partition)

    universe = _calibration_this_date(drawn)
    dh = dataset_hash(root / "annotations" / DATES[0], stems=universe)

    inputs = _dense_inputs(
        dh, selection_dir=str(out),
        calibration_labels_dir=str(root / "annotations" / DATES[0]))
    bundle = resolve_operating_point("bud_opening", experiment_id=experiment_id, **inputs)
    disjointness = bundle.get("conf").gate_evidence["selection_disjointness"]
    assert disjointness["checked"] is True
    assert not disjointness["leaked_groups"] and not disjointness["leaked_stems"]
    evidence = {
        "resolver": "resolve_operating_point", "inputs": inputs,
        "reference_inputs": {
            "label_stems": {"calibration": {
                "path": str(root / "annotations" / DATES[0]), "stems": universe}},
            "stated_values": {"selection_dir": str(out)},
        },
        "calibration_stems": universe,
    }
    monkeypatch.setattr(calibration, "calibrate_operating_point",
                        lambda *a, **k: (bundle, dh, 0, evidence))
    monkeypatch.setattr(predictor_mod, "build_predictor", lambda checkpoint, **kw: _BucketStub())
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    result = itools.run_inference(
        str(ckpt), images_dir=str(root / "images" / DATES[0]),
        output_dir=str(root / "predictions" / "bound" / DATES[0]),
        device="cpu", tile=False, trait="bud_opening",
        calibration_labels_dir=str(root / "annotations" / DATES[0]),
        selection_dir=str(out), experiment_id=experiment_id)
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
