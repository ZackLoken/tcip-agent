"""Rails for the selection-disjointness label-movement check: a calibration label that moved
between a split's draw and its calibration is named on the sealed row, through the
label_digests block a bound run's own split.json carries, never a refusal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_store import RECORD_JSON  # noqa: E402

IMG = 32
SUBJECT = "leaf"
TRAIT = "leaf_area"
DATES = ("2026-02-11", "2026-02-25")
_STEMS = ("a", "b", "c", "d", "e", "f", "g", "h")


def _seed_trait_spec(project_root: Path) -> None:
    """A trait spec whose only job is to give ``resolve_operating_point`` count-bias/localization
    fields to read; ``holdout_match_quality_floor`` is set loose enough for the dense, synthetic
    references these rails build to clear it, the way a real trait's own confirmed value would."""
    import tcip_store as ts

    from tcip_mcp import traits

    specs_dir = project_root / ".tcip" / "state" / "trait_specs"
    spec = {
        "name": TRAIT, "count_objective": "count_unbiased", "localization": "center_match",
        "localization_tolerance": "half_class_avg_size", "localization_tolerance_frac": 0.5,
        "holdout_match_quality_floor": 0.5, "positive_class_name": "", "milestone_fractions": [],
        "milestone_on": "", "majority_milestone": "", "majority_provisional": False,
        "phenology_prefix": "leaf_out", "majority_label": "", "sliver_policy": "class_avg_size",
        "sliver_frac": 0.5, "count_bias_tolerance_frac": 0.01,
        "delivers": ["leaf_out_05per_date", "leaf_out_50per_date"],
        "notes": "Test-only, not a domain-expert-confirmed measurement.",
    }
    ts.replace(traits.trait_spec_key(specs_dir, TRAIT), spec, expect=ts.Version.ABSENT)


def _save_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMG, IMG), color=(128, 128, 128)).save(path)


def _dataset(root: Path, stems=_STEMS) -> Path:
    """Two capture dates, eight stems each, enough groups that a three-way draw leaves both
    train and val non-empty for either date."""
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

    from tcip_mcp.tools.data_tools import make_splits, split_manifest_key

    result = make_splits(str(root), output_path=str(out), subject=SUBJECT, seed=seed,
                         train_ratio=0.4, val_ratio=0.3, calibration_ratio=0.3)
    assert "error" not in result, result
    return ts.read(split_manifest_key(out))


def _calibration_stems(manifest: dict, date: str = DATES[0]) -> list[str]:
    return sorted(i.split("/", 1)[1] for i in manifest["splits"]["calibration"]
                 if i.startswith(f"{date}/"))


def _label_path(root: Path, date: str, stem: str) -> Path:
    return root / "annotations" / date / f"{stem}.json"


def _rewrite_label(root: Path, date: str, stem: str, *, offset: float) -> None:
    from tcip_mcp.tools.annotation_tools import save_annotations

    image_path = str(root / "images" / date / f"{stem}.jpg")
    res = save_annotations(
        image_path,
        annotations=[{"subject": SUBJECT, "bbox": [2 + offset, 2, 10 + offset, 10]}],
    )
    assert "error" not in res, res


def _bind_run(root: Path, out: Path, experiment_id: str, *, date: str = DATES[0]) -> dict | None:
    """A run bound to the manifest at ``out``, for ``date``: the exact sequence
    ``subprocess_worker.run`` follows (``auto_train_val`` then ``persist_split_manifest``),
    called directly so no real training subprocess is needed. Returns the run's own
    ``label_digests`` block (``auto_train_val``'s third return value), never read back through
    ``data_cfg``.
    """
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    data_cfg = {
        "images_dir": str(root / "images" / date),
        "labels_dir": str(root / "annotations" / date),
        "subject": SUBJECT, "attribute": None,
        "split": {"manifest_dir": str(out)},
    }
    train_ds, val_ds, label_digests = auto_train_val("detection", data_cfg, None)
    create_experiment(experiment_id, {})
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg,
                            label_digests=label_digests)
    return label_digests


def _seal(
    root: Path, out: Path, experiment_id: str, project_root: Path, *, date: str = DATES[0],
    calibration_labels_dir: str | None = None, split_manifest_sha256: str | None = None,
    real_stem_ids: bool = True,
) -> tuple[dict, bool]:
    """Calibrate a stub detector under the bound manifest; returns ``(selection_disjointness,
    is_shippable)``, a dense, hand-verifiable reference so most scenarios clear the gate (a
    redraw can shrink the calibration/holdout side enough to float an unrelated sufficiency
    floor, which is why shippability is returned rather than asserted here). ``TRAIT`` is a
    registered trait purely so ``resolve_operating_point`` has a spec to read its count-bias/
    localization fields from; it names no relation to ``SUBJECT``, the dataset's own annotated
    class. ``real_stem_ids=False`` keeps the synthetic ids ``dense_records`` mints (a caller
    proving a run entirely unrelated to this manifest, whose own val stems drawn from the same
    small pool could otherwise collide with the calibration side by pure chance).
    """
    from tests._dense_op_fixtures import dense_records

    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import dataset_hash

    _seed_trait_spec(project_root)
    universe = _calibration_stems(_manifest(out), date)
    dh = dataset_hash(root / "annotations" / date, stems=universe)
    objects_per_image = 80
    n_images = len(universe)
    miss, fp = [0] * n_images, [1] * n_images
    # image_id set to the calibration side's own real stems: calibration_labels_moved intersects
    # the moved set against exactly these ids, so a synthetic id here would never intersect.
    cal_records = dense_records(
        n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    if real_stem_ids:
        for record, stem in zip(cal_records, universe):
            record["image_id"] = stem
    hold_records = dense_records(
        n_images=n_images, objects_per_image=objects_per_image, id_prefix="h", shift=5.0,
        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    bundle = resolve_operating_point(
        TRAIT, experiment_id=experiment_id, dataset_hash=dh, tiled=False,
        staged_conf_floor=0.01,
        calibration_records=cal_records, holdout_records=hold_records,
        split_manifest_dir=str(out), calibration_date=date,
        calibration_labels_dir=calibration_labels_dir,
        split_manifest_sha256=split_manifest_sha256,
    )
    conf = bundle.get("conf")
    assert conf is not None and conf.sweep is not None, bundle
    return conf.sweep["selection_disjointness"], conf.is_shippable


def _manifest(out: Path) -> dict:
    import tcip_store as ts

    from tcip_mcp.tools.data_tools import split_manifest_key

    return ts.read(split_manifest_key(out))


def _manifest_sha256(out: Path) -> str:
    return hashlib.sha256(RECORD_JSON.encode(_manifest(out))).hexdigest()


# -- rail: a label rewritten between the draw and the run's bind ------------------------------


def test_a_label_rewritten_between_draw_and_bind_names_the_stem_and_still_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    stem = _calibration_stems(manifest)[0]

    _rewrite_label(root, DATES[0], stem, offset=1.0)
    _bind_run(root, out, "exp_moved_before_bind")
    sd, shippable = _seal(root, out, "exp_moved_before_bind", tmp_path)

    assert sd["labels_moved_draw_to_run"] == [stem]
    assert sd["calibration_labels_moved"] == [stem]
    assert sd["checked"] is True
    assert shippable is True


# -- rail: the same move, restored before calibration, names both windows ---------------------


def test_a_label_restored_before_calibration_is_named_in_both_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    stem = _calibration_stems(manifest)[0]
    label_path = _label_path(root, DATES[0], stem)
    original = label_path.read_bytes()

    _rewrite_label(root, DATES[0], stem, offset=1.0)
    _bind_run(root, out, "exp_restored")
    label_path.write_bytes(original)
    assert hashlib.sha256(label_path.read_bytes()).hexdigest()[:16] == hashlib.sha256(
        original).hexdigest()[:16]

    sd, shippable = _seal(root, out, "exp_restored", tmp_path,
                         calibration_labels_dir=str(root / "annotations" / DATES[0]))

    assert sd["labels_moved_draw_to_run"] == [stem]
    assert sd["labels_moved_run_to_now"] == [stem]
    assert sd["calibration_labels_moved"] == [stem]
    assert shippable is True


# -- rail: a label rewritten after the run, before calibration, names the second window --------


def test_a_label_rewritten_after_the_run_names_the_run_to_now_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    stem = _calibration_stems(manifest)[0]

    _bind_run(root, out, "exp_moved_after_run")
    _rewrite_label(root, DATES[0], stem, offset=1.0)
    sd, shippable = _seal(root, out, "exp_moved_after_run", tmp_path,
                         calibration_labels_dir=str(root / "annotations" / DATES[0]))

    assert sd["labels_moved_draw_to_run"] == []
    assert sd["labels_moved_run_to_now"] == [stem]
    assert sd["calibration_labels_moved"] == [stem]
    assert shippable is True


# -- rail: the second window scopes to the calibration's own universe --------------------------


def test_the_second_window_never_names_a_train_or_val_stem_absent_from_a_subset_directory(
    tmp_path: Path,
) -> None:
    """``calibration_labels_dir`` may be one of several already-split per-image directories (a
    classifier calibration's own GT dir), holding only the calibration side's own files. The
    second window must not read a train- or val-side stem's mere absence from that directory as
    a move: it is out of scope for that directory, not moved."""
    from tcip_mcp.pipelines.operating_point import _resolve_label_movement
    from tcip_mcp.pipelines.resolution import label_digests as compute_label_digests

    cal_dir = tmp_path / "cal_only"
    cal_dir.mkdir()
    (cal_dir / "c1.json").write_bytes(b'{"a": 1}')

    at_run = {
        "t1": "0" * 16, "v1": "1" * 16,
        "c1": compute_label_digests(cal_dir, ["c1"])["c1"],
    }
    label_digests_block = {"at_split": dict(at_run), "at_run": dict(at_run), "manifest_sha256": "m"}

    moved = _resolve_label_movement(label_digests_block, {"c1"}, str(cal_dir), None)

    assert moved["labels_moved_run_to_now"] == []
    assert moved["calibration_labels_moved"] == []


def test_the_second_window_still_names_a_moved_calibration_side_stem(tmp_path: Path) -> None:
    """The scoping in the test above does not blind the window to a genuine move on the
    calibration's own side."""
    from tcip_mcp.pipelines.operating_point import _resolve_label_movement

    cal_dir = tmp_path / "cal_only"
    cal_dir.mkdir()
    (cal_dir / "c1.json").write_bytes(b'{"a": 1}')

    at_run = {"t1": "0" * 16, "c1": "stale-digest-not-matching-the-file-on-disk"}
    label_digests_block = {"at_split": dict(at_run), "at_run": dict(at_run), "manifest_sha256": "m"}

    moved = _resolve_label_movement(label_digests_block, {"c1"}, str(cal_dir), None)

    assert moved["labels_moved_run_to_now"] == ["c1"]
    assert moved["calibration_labels_moved"] == ["c1"]


# -- rail: nothing touched delivers with every list empty, on both surfaces --------------------


def test_nothing_touched_delivers_with_every_list_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    _bind_run(root, out, "exp_untouched")
    manifest_sha = _manifest_sha256(out)

    sd, shippable = _seal(root, out, "exp_untouched", tmp_path,
                         calibration_labels_dir=str(root / "annotations" / DATES[0]),
                         split_manifest_sha256=manifest_sha)

    assert sd["labels_moved_draw_to_run"] == []
    assert sd["labels_moved_run_to_now"] == []
    assert sd["calibration_labels_moved"] == []
    assert sd["manifest_redrawn"] is False
    assert shippable is True


_REVIEW_IDENTITY = {"checkpoint_sha256": "sha-review", "experiment_id": None}


def _review_entry(gt, pred, conf):
    return {"match_type": "TP", "action": "accepted", "class_id": 0,
            "gt_bbox_norm": gt, "pred_bbox_norm": pred, "conf": conf,
            "producer_identity": _REVIEW_IDENTITY, "conf_threshold": None,
            "missed_object_attested": False}


def _review_state_over_stems(stems: list[str]) -> dict:
    """One dense, well-separated confirmed match per stem, image ids the same real stems the
    manifest and the bound run name, so a locked calibration-side review image lines up with a
    label this test can move on disk."""
    images = {}
    for i, stem in enumerate(stems):
        jitter = i * 0.01
        box = [0.2 + jitter, 0.2, 0.05, 0.05]
        images[f"{stem}.jpg"] = {
            "img_status": "completed", "gt_preexisting": True, "detections": [_review_entry(box, box, 0.9)],
        }
    return {"image": images}


def test_the_review_path_genuinely_runs_and_seals_null_second_window_when_nothing_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolve_operating_point_from_review`` names no manifest and holds no labels directory
    of its own, so ``labels_moved_run_to_now``/``manifest_redrawn`` stay null there, and the row
    still delivers. Driven through the real review path, not through ``resolve_operating_point``
    called with ``split_manifest_dir`` the way a caller-named-manifest calibration does."""
    from tcip_mcp.pipelines.feedback import resolve_operating_point_from_review

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    _bind_run(root, out, "exp_review_untouched", date=DATES[0])
    _seed_trait_spec(tmp_path)

    state = _review_state_over_stems(list(_STEMS))
    bundle = resolve_operating_point_from_review(
        state, TRAIT, scope_root=root, bucket_identities=[_REVIEW_IDENTITY],
        staged_conf_floor=0.01, tiled=False, experiment_id="exp_review_untouched",
        calibration_date=DATES[0])
    sd = bundle.get("conf").sweep["selection_disjointness"]

    assert sd["applicable"] is True
    assert sd["labels_moved_draw_to_run"] == []
    assert sd["calibration_labels_moved"] == []
    assert sd["labels_moved_run_to_now"] is None
    assert sd["manifest_redrawn"] is None


def test_the_review_path_names_a_calibration_side_label_moved_before_the_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label rewritten between the draw and the bind is named on the review-sealed row's own
    ``labels_moved_draw_to_run``/``calibration_labels_moved`` (the only window the review path
    can populate, since it holds no labels directory of its own for the second), and
    ``describe_review_validation``'s sentence names it on whichever branch the gate lands on:
    the only exercise, in the repository, of ``_selection_movement_sentence``."""
    from tcip_mcp.pipelines.feedback import describe_review_validation, resolve_operating_point_from_review

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    for stem in _STEMS:
        _rewrite_label(root, DATES[0], stem, offset=1.0)
    _bind_run(root, out, "exp_review_moved", date=DATES[0])
    _seed_trait_spec(tmp_path)

    state = _review_state_over_stems(list(_STEMS))
    bundle = resolve_operating_point_from_review(
        state, TRAIT, scope_root=root, bucket_identities=[_REVIEW_IDENTITY],
        staged_conf_floor=0.01, tiled=False, experiment_id="exp_review_moved",
        calibration_date=DATES[0])
    sd = bundle.get("conf").sweep["selection_disjointness"]

    assert sd["applicable"] is True
    assert set(sd["labels_moved_draw_to_run"]) == set(_STEMS)
    assert sd["calibration_labels_moved"], sd
    assert set(sd["calibration_labels_moved"]).issubset(set(_STEMS))

    desc = describe_review_validation(bundle, reviewed_image_count=len(_STEMS))
    assert "changed since this split was drawn" in desc["reason"]


# -- rail: a calibration member withdrawn between the draw and the run ------------------------


def test_a_withdrawn_calibration_member_is_named_through_the_absent_file_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.dataset_layout import record_image_statuses, status_bucket

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    stem = _calibration_stems(manifest)[0]

    _label_path(root, DATES[0], stem).unlink()
    record_image_statuses(
        root, status_bucket(SUBJECT, DATES[0]), {f"{stem}.jpg": "negative"}, recorded_by="user:t")

    _bind_run(root, out, "exp_withdrawn")
    sd, _shippable = _seal(root, out, "exp_withdrawn", tmp_path)

    assert sd["labels_moved_draw_to_run"] == [stem]
    assert sd["calibration_labels_moved"] == [stem]


# -- rail: a redraw between the run and the calibration is named separately from a moved label -


def test_a_redraw_between_run_and_calibration_is_named_beside_a_moved_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out, seed=2)
    stem = _calibration_stems(manifest)[0]

    _rewrite_label(root, DATES[0], stem, offset=1.0)
    _bind_run(root, out, "exp_redrawn")
    original_manifest_sha = _manifest_sha256(out)

    _draw(root, out, seed=7)  # a redraw into the same directory, after the run bound
    redrawn_manifest_sha = _manifest_sha256(out)
    assert redrawn_manifest_sha != original_manifest_sha

    sd, _shippable = _seal(root, out, "exp_redrawn", tmp_path,
                          calibration_labels_dir=str(root / "annotations" / DATES[0]),
                          split_manifest_sha256=redrawn_manifest_sha)

    assert sd["manifest_redrawn"] is True
    assert stem in sd["labels_moved_draw_to_run"]


def test_an_unbound_run_calibrated_under_a_caller_named_manifest_seals_null_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run with no ``manifest_binding`` at all (never went through the manifest branch of
    ``auto_train_val``), calibrated under a caller-named manifest anyway: the four
    label-movement keys are ``null`` with the reason, and the row still delivers."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    flat_cfg = {
        "images_dir": str(root / "images" / DATES[0]),
        "labels_dir": str(root / "annotations" / DATES[0]),
        "subject": SUBJECT, "attribute": None,
    }
    train_ds, val_ds, label_digests = auto_train_val("detection", flat_cfg, None)
    assert label_digests is None
    create_experiment("exp_unbound", {})
    persist_split_manifest("exp_unbound", train_ds, val_ds, flat_cfg, label_digests=label_digests)

    sd, shippable = _seal(root, out, "exp_unbound", tmp_path, real_stem_ids=False)

    assert sd["labels_moved_draw_to_run"] is None
    assert sd["labels_moved_run_to_now"] is None
    assert sd["calibration_labels_moved"] is None
    assert sd["manifest_redrawn"] is None
    assert sd["reason"]
    assert sd["checked"] is True
    assert shippable is True


# -- rail: read_split_manifest_dir requires label_digests on every members block ---------------


def test_read_split_manifest_dir_refuses_a_members_block_without_label_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manifest ``make_splits`` itself wrote, stripped of one date's ``label_digests`` the way
    an old, pre-family manifest would carry it: every other required key present, so the refusal
    is provably about this key, not a stand-in shaped so loosely it would refuse for any reason."""
    import tcip_store as ts

    from tcip_mcp.tools.data_tools import read_split_manifest_dir, split_manifest_key

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    # Built without the key rather than deleted from a block that has it, so this arrangement
    # cannot itself raise regardless of whether the block ever carried label_digests.
    manifest["members"][DATES[0]] = {
        k: v for k, v in manifest["members"][DATES[0]].items() if k != "label_digests"
    }
    ts.replace(split_manifest_key(out), manifest)

    with pytest.raises(ValueError, match="label_digests"):
        read_split_manifest_dir(out)


def test_read_split_manifest_dir_refuses_a_members_block_with_an_empty_label_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty (or null) ``label_digests`` is not merely absent, and admitting it would let
    ``_resolve_label_movement`` read an empty ``at_split`` as "checked, nothing moved" rather
    than "not checked": ``make_splits`` never writes a members block for a date with no admitted
    stems, so a legitimate block's ``label_digests`` is never empty either."""
    import tcip_store as ts

    from tcip_mcp.tools.data_tools import read_split_manifest_dir, split_manifest_key

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    manifest = _draw(root, out)
    manifest["members"][DATES[0]]["label_digests"] = {}
    ts.replace(split_manifest_key(out), manifest)

    with pytest.raises(ValueError, match="label_digests"):
        read_split_manifest_dir(out)


def test_read_split_manifest_dir_admits_a_manifest_make_splits_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools.data_tools import read_split_manifest_dir

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    manifest = read_split_manifest_dir(out)
    for date, block in manifest["members"].items():
        assert "label_digests" in block, date


# -- rail: label_digests' absent-file convention, and dataset_hash's formula is unchanged ------


def test_label_digests_gives_the_absent_file_digest_and_dataset_hash_is_unchanged(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.resolution import dataset_hash, label_digests

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "present.json").write_bytes(b'{"a": 1}')

    digests = label_digests(labels_dir, ["present", "absent"])
    assert digests["absent"] == hashlib.sha256(b"").hexdigest()[:16]
    assert digests["present"] == hashlib.sha256(b'{"a": 1}').hexdigest()[:16]

    expected = hashlib.sha256()
    for stem in ("absent", "present"):
        expected.update(stem.encode("utf-8"))
        expected.update(b"\0")
        expected.update((labels_dir / f"{stem}.json").read_bytes()
                        if (labels_dir / f"{stem}.json").is_file() else b"")
        expected.update(b"\0")
    assert dataset_hash(labels_dir, stems=["absent", "present"]) == expected.hexdigest()[:16]

    opened: list[Path] = []
    real_read_bytes = Path.read_bytes

    def spy(self: Path) -> bytes:
        opened.append(self)
        return real_read_bytes(self)

    import unittest.mock as mock

    with mock.patch.object(Path, "read_bytes", spy):
        label_digests(labels_dir, ["present", "absent"])
    assert opened.count(labels_dir / "present.json") == 1


def test_manifest_digest_is_the_one_function_the_bind_write_and_the_calibration_read_both_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolution.manifest_digest`` is the sha256 hex digest over ``RECORD_JSON.encode``, and
    the run's own ``split.json`` (written by ``persist_split_manifest``, the bind side) already
    carries that value for the manifest it bound to, the same value a caller's own re-encoding
    produces: the two spellings this test's own independent oracle (``_manifest_sha256``) and the
    production side must agree on."""
    from tcip_mcp.experiments import read_split_manifest
    from tcip_mcp.pipelines.resolution import manifest_digest

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    label_digests_block = _bind_run(root, out, "exp_manifest_digest")

    manifest = _manifest(out)
    assert manifest_digest(manifest) == _manifest_sha256(out)

    split = read_split_manifest("exp_manifest_digest")
    assert split["label_digests"]["manifest_sha256"] == manifest_digest(manifest)
    assert label_digests_block["manifest_sha256"] == manifest_digest(manifest)


def test_dataset_hash_and_label_digests_reads_each_label_once_and_agrees_with_the_apart_calls(
    tmp_path: Path,
) -> None:
    """The pair ``make_splits`` actually calls, ``dataset_hash_and_label_digests``, reads every
    label's bytes once (not twice, once per digest, the way calling ``dataset_hash`` and
    ``label_digests`` apart would) and returns the same values those two calls would have: the
    earlier spy above proves only ``label_digests`` alone reads once, and says nothing about the
    pair the draw runs."""
    from tcip_mcp.pipelines.resolution import (
        dataset_hash, dataset_hash_and_label_digests, label_digests,
    )

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "present.json").write_bytes(b'{"a": 1}')
    stems = ["present", "absent"]

    combined_hash, combined_digests = dataset_hash_and_label_digests(labels_dir, stems)
    assert combined_hash == dataset_hash(labels_dir, stems=stems)
    assert combined_digests == label_digests(labels_dir, stems)

    opened: list[Path] = []
    real_read_bytes = Path.read_bytes

    def spy(self: Path) -> bytes:
        opened.append(self)
        return real_read_bytes(self)

    import unittest.mock as mock

    with mock.patch.object(Path, "read_bytes", spy):
        dataset_hash_and_label_digests(labels_dir, stems)
    assert opened.count(labels_dir / "present.json") == 1


def test_make_splits_calls_the_combined_helper_not_dataset_hash_and_label_digests_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The draw calls ``dataset_hash_and_label_digests`` once per date, the single-pass helper,
    rather than a separate ``dataset_hash`` and ``label_digests`` call each opening every file."""
    import unittest.mock as mock

    from tcip_mcp.pipelines import resolution

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"

    real_combined = resolution.dataset_hash_and_label_digests
    calls: list[tuple] = []

    def spy(labels_dir, stems):
        calls.append((labels_dir, tuple(stems)))
        return real_combined(labels_dir, stems)

    with mock.patch("tcip_mcp.pipelines.resolution.dataset_hash_and_label_digests", spy):
        _draw(root, out)

    assert len(calls) == len(DATES), calls


# -- rail: the durable config carries no per-stem digests after a bound run --------------------


def test_auto_train_vals_third_return_value_never_lands_in_the_split_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``label_digests`` rides only as ``auto_train_val``'s own third return value: ``data_cfg
    ["split"]`` (the block copied whole into the durable experiment config and embedded in every
    checkpoint) never gains the key, so neither does anything downstream that merges it. This is
    coverage of the source the durable config, every checkpoint's embedded config and every
    trial's resolved config each copy verbatim (design test 23): the durable config's own read
    back below reads the one place ``subprocess_worker.run`` patches it into, without a real
    training subprocess; a checkpoint's embedded config and a trial's resolved config are not
    independently read back here."""
    import tcip_store as ts

    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_split
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    data_cfg = {
        "images_dir": str(root / "images" / DATES[0]),
        "labels_dir": str(root / "annotations" / DATES[0]),
        "subject": SUBJECT, "attribute": None,
        "split": {"manifest_dir": str(out)},
    }
    _train_ds, _val_ds, label_digests = auto_train_val("detection", data_cfg, None)

    assert label_digests is not None
    assert "label_digests" not in data_cfg["split"]
    assert "label_digests" not in data_cfg["split"]["manifest_binding"]

    create_experiment("exp_split_config_readback", {})
    _patch_experiment_config_split("exp_split_config_readback", data_cfg["split"])
    durable = ts.read(config_key("exp_split_config_readback"))
    assert "label_digests" not in durable["data"]["split"]
    assert "label_digests" not in durable["data"]["split"]["manifest_binding"]
