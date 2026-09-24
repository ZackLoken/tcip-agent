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
        "holdout_match_quality_floor": 0.5, "positive_value": "", "milestone_fractions": [],
        "milestone_on": "", "majority_milestone": "", "majority_provisional": False,
        "phenology_prefix": "leaf_out", "majority_label": "", "sliver_policy": "class_avg_size",
        "sliver_frac": 0.5, "count_bias_tolerance_frac": 0.01,
        "delivers": ["leaf_out_05per_date", "leaf_out_50per_date"],
        "notes": "Test-only, not a domain-expert-confirmed measurement.",
        "schema_version": traits.TRAIT_SPEC_SCHEMA_VERSION,
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


def _draw(root: Path, out: Path, *, seed: int = 2):
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(str(root), output_path=str(out), subject=SUBJECT, seed=seed,
                         train_ratio=0.4, val_ratio=0.3, calibration_ratio=0.3)
    assert "error" not in result, result
    return read_selection(out)


def _calibration_stems(selection, date: str = DATES[0]) -> list[str]:
    return sorted(Path(s.ground_truth).stem for s in selection.on("calibration")
                 if Path(s.ground_truth).parent.name == date)


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
    ``subprocess_worker.run`` follows (``auto_train_val`` then ``persist_run_partition``),
    called directly so no real training subprocess is needed. Returns the run's own
    recorded partition (``auto_train_val``'s third return value), never read back through
    ``data_cfg``.
    """
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition

    data_cfg = {"split": {"selection_dir": str(out)}}
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    create_experiment(experiment_id, {})
    persist_run_partition(experiment_id, data_cfg, partition=partition)
    return partition


def _seal(
    root: Path, out: Path, experiment_id: str, project_root: Path, *, date: str = DATES[0],
    calibration_labels_dir: str | None = None, selection_sha256: str | None = None,
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
        selection_dir=str(out),
        calibration_labels_dir=calibration_labels_dir or str(root / "annotations" / date),
        selection_sha256=selection_sha256,
    )
    conf = bundle.get("conf")
    assert conf is not None and conf.gate_evidence is not None, bundle
    return conf.gate_evidence["selection_disjointness"], conf.is_shippable


def _manifest(out: Path):
    from tcip_mcp.pipelines.data.selection import read_selection

    return read_selection(out)


def _manifest_sha256(out: Path) -> str:
    from tcip_mcp.pipelines.data.selection import selection_document

    return hashlib.sha256(RECORD_JSON.encode(selection_document(_manifest(out)))).hexdigest()


# -- rail: a label rewritten between the draw and the run's bind ------------------------------


def test_a_label_rewritten_between_draw_and_bind_names_the_stem_and_still_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
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
    from tcip_mcp.pipelines.resolution import ground_truth_digest

    cal_dir = tmp_path / "cal_only"
    cal_dir.mkdir()
    (cal_dir / "c1.json").write_bytes(b'{"a": 1}')

    at_run = {
        "t1": "0" * 16, "v1": "1" * 16,
        "c1": ground_truth_digest(cal_dir / "c1.json"),
    }
    paths = {"t1": str(tmp_path / "elsewhere" / "t1.json"),
             "v1": str(tmp_path / "elsewhere" / "v1.json"),
             "c1": str(cal_dir / "c1.json")}
    label_digests_block = {"at_split": dict(at_run), "at_run": dict(at_run),
                           "ground_truth": paths}

    moved = _resolve_label_movement(label_digests_block, {"c1"}, str(cal_dir), None, "m")

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
    paths = {"t1": str(tmp_path / "elsewhere" / "t1.json"), "c1": str(cal_dir / "c1.json")}
    label_digests_block = {"at_split": dict(at_run), "at_run": dict(at_run),
                           "ground_truth": paths}

    moved = _resolve_label_movement(label_digests_block, {"c1"}, str(cal_dir), None, "m")

    assert moved["labels_moved_run_to_now"] == ["c1"]
    assert moved["calibration_labels_moved"] == ["c1"]


# -- rail: nothing touched delivers with every list empty, on both surfaces --------------------


def test_nothing_touched_delivers_with_every_list_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    _bind_run(root, out, "exp_untouched")
    manifest_sha = _manifest_sha256(out)

    sd, shippable = _seal(root, out, "exp_untouched", tmp_path,
                         calibration_labels_dir=str(root / "annotations" / DATES[0]),
                         selection_sha256=manifest_sha)

    assert sd["labels_moved_draw_to_run"] == []
    assert sd["labels_moved_run_to_now"] == []
    assert sd["calibration_labels_moved"] == []
    assert sd["selection_redrawn"] is False
    assert shippable is True


_REVIEW_IDENTITY = {"checkpoint_sha256": "sha-review", "experiment_id": None}


def _review_entry(gt, pred, conf):
    return {"match_type": "TP", "action": "accepted", "class_id": 0,
            "iscrowd": False, "reviewed_by": "", "class_name": "", "gt_bbox_norm": gt, "pred_bbox_norm": pred, "conf": conf,
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
    """``resolve_operating_point_from_review`` names no selection of its own, so
    ``selection_redrawn`` stays null there, and the row still delivers. Driven through the real
    review path, not through ``resolve_operating_point`` called with ``selection_dir`` the way a
    caller-named-selection calibration does."""
    from tcip_mcp.pipelines.feedback import resolve_operating_point_from_review

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    _bind_run(root, out, "exp_review_untouched", date=DATES[0])
    _seed_trait_spec(tmp_path)

    state = _review_state_over_stems(list(_STEMS))
    bundle = resolve_operating_point_from_review(
        state, TRAIT, scope_root=root, bucket_identities=[_REVIEW_IDENTITY],
        staged_conf_floor=0.01, tiled=False, experiment_id="exp_review_untouched",
        calibration_labels_dir=str(root / "annotations" / DATES[0]))
    sd = bundle.get("conf").gate_evidence["selection_disjointness"]

    assert sd["applicable"] is True
    assert sd["labels_moved_draw_to_run"] == []
    assert sd["calibration_labels_moved"] == []
    assert sd["labels_moved_run_to_now"] == []
    assert sd["selection_redrawn"] is None


def test_the_review_path_names_a_calibration_side_label_moved_before_the_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label rewritten between the draw and the bind is named on the review-sealed row's own
    ``labels_moved_draw_to_run``/``calibration_labels_moved``, and
    ``describe_review_validation``'s sentence names it on whichever branch the gate lands on:
    the only exercise, in the repository, of ``_selection_movement_sentence``."""
    from tcip_mcp.pipelines.feedback import describe_review_validation, resolve_operating_point_from_review

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
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
        calibration_labels_dir=str(root / "annotations" / DATES[0]))
    sd = bundle.get("conf").gate_evidence["selection_disjointness"]

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

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
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
                          selection_sha256=redrawn_manifest_sha)

    assert sd["selection_redrawn"] is True
    assert stem in sd["labels_moved_draw_to_run"]


def test_a_record_carrying_no_selection_digest_leaves_the_redraw_window_unsealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound record that carries no digest of its own is no evidence either way: the redraw
    window is left unsealed, the way the other windows are when the record holds nothing for
    them. Comparing the calibration's own digest against nothing would report an unchanged
    selection as redrawn."""
    from tcip_store import store

    from tcip_mcp.experiments import read_run_partition, split_key

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    _bind_run(root, out, "exp_no_recorded_digest")

    record = read_run_partition("exp_no_recorded_digest")
    assert record["selection_binding"].pop("selection_sha256")
    store.replace(split_key("exp_no_recorded_digest"), record)

    sd, _shippable = _seal(root, out, "exp_no_recorded_digest", tmp_path,
                           calibration_labels_dir=str(root / "annotations" / DATES[0]),
                           selection_sha256=_manifest_sha256(out))

    assert sd["selection_redrawn"] is None
    assert sd["labels_moved_draw_to_run"] == []


def test_an_unbound_run_calibrated_under_a_caller_named_manifest_seals_null_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run with no ``selection_binding`` at all (never went through the manifest branch of
    ``auto_train_val``), calibrated under a caller-named manifest anyway: the four
    label-movement keys are ``null`` with the reason, and the row still delivers."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    flat_cfg = {
        "images_dir": str(root / "images" / DATES[0]),
        "labels_dir": str(root / "annotations" / DATES[0]),
        "subject": SUBJECT, "attribute": None,
    }
    train_ds, val_ds, partition = auto_train_val("detection", flat_cfg, None)
    # A drawn run records a partition of its own; what it must not carry is a selection binding.
    assert partition is not None
    create_experiment("exp_unbound", {})
    persist_run_partition("exp_unbound", flat_cfg, partition=partition)

    sd, shippable = _seal(root, out, "exp_unbound", tmp_path, real_stem_ids=False)

    assert sd["labels_moved_draw_to_run"] is None
    assert sd["labels_moved_run_to_now"] is None
    assert sd["calibration_labels_moved"] is None
    assert sd["selection_redrawn"] is None
    assert sd["reason"]
    assert sd["checked"] is True
    assert shippable is True


# -- rail: every drawn sample carries the digest a later movement check reads -----------------


def test_every_drawn_sample_carries_its_own_label_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The digest a later movement check reads rides on each sample, so a selection cannot be
    written whose samples the check has nothing to compare against."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"

    drawn = _draw(root, out)

    assert drawn.samples
    assert all(sample.ground_truth_digest for sample in drawn.samples)


def test_a_withdrawn_ground_truth_digests_as_empty_bytes_in_both_readers(
    tmp_path: Path,
) -> None:
    """One convention for a file that is gone, in the per-member digest and in the combined hash
    over a directory: the digest of empty bytes, never an absent key, so a withdrawn label reads
    as moved rather than as never recorded."""
    from tcip_mcp.pipelines.resolution import dataset_hash, ground_truth_digest

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "present.json").write_bytes(b'{"a": 1}')

    assert ground_truth_digest(labels_dir / "absent.json") == hashlib.sha256(b"").hexdigest()[:16]
    assert ground_truth_digest(labels_dir / "present.json") == \
        hashlib.sha256(b'{"a": 1}').hexdigest()[:16]

    expected = hashlib.sha256()
    for stem in ("absent", "present"):
        expected.update(stem.encode("utf-8"))
        expected.update(b"\0")
        expected.update((labels_dir / f"{stem}.json").read_bytes()
                        if (labels_dir / f"{stem}.json").is_file() else b"")
        expected.update(b"\0")
    assert dataset_hash(labels_dir, stems=["absent", "present"]) == expected.hexdigest()[:16]


def test_selection_digest_is_the_one_function_the_bind_write_and_the_calibration_read_both_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolution.selection_digest`` is the sha256 hex digest over ``RECORD_JSON.encode`` of the
    selection's own document, and the run's own ``split.json`` (written by
    ``persist_run_partition``, the bind side) already carries that value in its binding block for
    the selection it bound to, the same value a caller's own re-encoding produces: the two
    spellings this test's own independent oracle (``_manifest_sha256``) and the production side
    must agree on. It is one fact for the run, so it is recorded once, never per scope."""
    from tcip_mcp.experiments import read_run_partition
    from tcip_mcp.pipelines.resolution import selection_digest

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)
    partition = _bind_run(root, out, "exp_selection_digest")

    selection = _manifest(out)
    assert selection_digest(selection) == _manifest_sha256(out)

    split = read_run_partition("exp_selection_digest")
    assert split["selection_binding"]["selection_sha256"] == selection_digest(selection)
    for block in (*split["members"].values(), *partition.values()):
        assert "selection_sha256" not in block["label_digests"]


def test_draw_splits_digests_each_document_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The draw digests each member's own ground truth once, however many members that file
    answers for, so a two-date draw opens each document exactly once."""
    import unittest.mock as mock

    from tcip_mcp.pipelines import resolution

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"


    real_digest = resolution.ground_truth_digest
    opened: list[str] = []

    def spy(path):
        opened.append(str(path))
        return real_digest(path)

    with mock.patch("tcip_mcp.pipelines.resolution.ground_truth_digest", spy):
        _draw(root, out)

    assert opened, "the draw digested nothing"
    assert len(opened) == len(set(opened)), opened
    assert all(p.endswith(".json") for p in opened), opened


# -- rail: the durable config carries no per-stem digests after a bound run --------------------


def test_auto_train_vals_third_return_value_never_lands_in_the_split_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-sample digests and group keys ride only as ``auto_train_val``'s own third return
    value: ``data_cfg["split"]`` (the block copied whole into the durable experiment config and
    embedded in every checkpoint) never gains them, so neither does anything downstream that
    merges it. The durable config's own read back below reads the one place
    ``subprocess_worker.run`` patches it into, without a real training subprocess; a checkpoint's
    embedded config and a trial's resolved config are not independently read back here."""
    import tcip_store as ts

    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_split
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = _dataset(tmp_path / "ds")
    out = tmp_path / "m"
    _draw(root, out)

    data_cfg = {"split": {"selection_dir": str(out)}}
    _train_ds, _val_ds, partition = auto_train_val("detection", data_cfg, None)

    assert partition and all(block["label_digests"] for block in partition.values())
    for block in (data_cfg["split"], data_cfg["split"]["selection_binding"]):
        assert "label_digests" not in block
        assert "members" not in block

    create_experiment("exp_split_config_readback", {})
    _patch_experiment_config_split("exp_split_config_readback", data_cfg["split"])
    durable = ts.read(config_key("exp_split_config_readback"))
    for block in (durable["data"]["split"], durable["data"]["split"]["selection_binding"]):
        assert "label_digests" not in block
        assert "members" not in block
