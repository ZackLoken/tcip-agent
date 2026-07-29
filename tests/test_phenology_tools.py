"""Tests for the phenology MCP tools (build_plant_mapping + compute_phenology).

The tools are the agent-facing surface for the per-plant bloom pipeline. These tests pin:
(1) build_plant_mapping wraps build + persist and reports a compact summary + error paths;
(2) compute_phenology writes the canonical column schema from classified predictions + a
persisted plant mapping, resolving the positive class id from the prediction buckets' own
recorded id_map (K4/K5) and gating both the classifier (K3, reconciled from
classifier_operating_point.json) and the count operating point; and (3) its measurement-
integrity guard refuses to deliver a CSV when no bucket ever classified along the trait's
positive-class axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.traits import CENTER_MATCH, get_trait
from tcip_mcp.tools.phenology_tools import (
    _classification_items,
    build_plant_mapping,
    calibrate_classifier_operating_point,
    compute_phenology,
)

# Round 10 (2026-07-29): no built-in traits — seed_catkin_trait_spec (conftest.py) writes a real
# catkin.yml into this test's pinned project root so get_trait("catkin") keeps resolving by default.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")


def _plant_csv(path: Path) -> None:
    path.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "P1,acc-9,-90.058,43.197\n",
        encoding="utf-8",
    )


def test_build_plant_mapping_wraps_build_and_persists(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    (images_root / "2026-02-11").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(images_root / "2026-02-11" / "img1.jpg")
    csv_path = tmp_path / "plants.csv"
    _plant_csv(csv_path)
    out = tmp_path / "state" / "plant_mapping.json"

    res = build_plant_mapping(
        images_root=str(images_root),
        plant_csv_paths=[str(csv_path)],
        output_mapping_path=str(out),
    )

    assert "error" not in res
    assert res["mapping_path"] == str(out)
    assert res["n_dates"] == 1
    assert res["n_images"] == 1
    assert res["n_mapped"] + res["n_unmapped"] == 1
    assert "2026-02-11" in res["per_date"]
    assert out.is_file()
    persisted = json.loads(out.read_text(encoding="utf-8"))
    assert list(persisted.keys()) == ["2026-02-11"]
    assert persisted["2026-02-11"][0]["stem"] == "img1"
    assert "confidence" not in persisted["2026-02-11"][0]


def test_build_plant_mapping_missing_images_root(tmp_path: Path) -> None:
    res = build_plant_mapping(
        images_root=str(tmp_path / "nope"),
        plant_csv_paths=[str(tmp_path / "plants.csv")],
        output_mapping_path=str(tmp_path / "m.json"),
    )
    assert "error" in res
    assert "images_root not found" in res["error"]


def test_build_plant_mapping_missing_csv(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    (images_root / "2026-02-11").mkdir(parents=True)
    res = build_plant_mapping(
        images_root=str(images_root),
        plant_csv_paths=[str(tmp_path / "missing.csv")],
        output_mapping_path=str(tmp_path / "m.json"),
    )
    assert "error" in res
    assert "plant CSV" in res["error"]


def _write_mapping(path: Path, mapping: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping), encoding="utf-8")


def _write_preds(dir_path: Path, stem: str, subjects: list[str]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    anns = [Annotation(subject=s, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9) for s in subjects]
    json_io.write_annotations(dir_path / f"{stem}.json", anns, 8, 8)


def _write_op_sidecar(dir_path: Path, *, validated: bool, conf: float = 0.4,
                      id_map: dict | None = None, experiment_id: str | None = None) -> None:
    """The operating_point.json a calibrated export_predictions writes."""
    ref = "validated_held_out" if validated else "false"
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "operating_point.json").write_text(json.dumps({
        "validated": validated,
        "operating_point": {"conf": {"value": conf, "validated_vs_gt": ref}},
        "id_map": id_map,
        "experiment_id": experiment_id,
    }), encoding="utf-8")


def _write_classifier_sidecar(dir_path: Path, *, validated: bool, trait: str | None = None,
                              experiment_id: str | None = None) -> None:
    """The classifier_operating_point.json calibrate_classifier_operating_point writes (K3)."""
    ref = "validated_held_out" if validated else "false"
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "classifier_operating_point.json").write_text(json.dumps({
        "validated": validated,
        "operating_point": {"classifier": {"value": "elongated", "validated_vs_gt": ref}},
        "trait": trait,
        "experiment_id": experiment_id,
    }), encoding="utf-8")


ID_MAP = {"dormant": 0, "elongated": 1}


def test_compute_phenology_delivers_when_both_validated(tmp_path: Path) -> None:
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["dormant"])
    _write_preds(d2, "P1_b", ["elongated"])
    _write_op_sidecar(d1, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, validated=True, id_map=ID_MAP)
    _write_classifier_sidecar(d1, validated=True, trait="catkin")
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    _write_mapping(mapping_path, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "catkin_phenology.csv"

    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
        operating_point_conf=0.4,
        operating_point_validated="validated_held_out",
    )

    assert "error" not in res, res
    assert res["elongation_classified"] is True
    assert out_csv.exists()


def test_compute_phenology_rejects_classifier_stamp_from_unrelated_run(tmp_path: Path) -> None:
    """Stage-6 review N3: a genuinely-validated classifier_operating_point.json calibrated for a
    DIFFERENT trait/experiment must not validate an unrelated delivery -- classifier_pred_dirs is a
    separate, caller-supplied list, so reconcile_classifier_validity's own on-disk check alone can't
    see this; the stamp's own recorded trait/experiment_id must agree with what's being delivered."""
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["dormant"])
    _write_preds(d2, "P1_b", ["elongated"])
    _write_op_sidecar(d1, validated=True, id_map=ID_MAP, experiment_id="run-B")
    _write_op_sidecar(d2, validated=True, id_map=ID_MAP, experiment_id="run-B")
    # Genuinely validated (validated=True), but calibrated for a DIFFERENT trait and a DIFFERENT
    # producing run than the one being delivered here.
    other_trait_dir = tmp_path / "unrelated_calibration"
    _write_classifier_sidecar(other_trait_dir, validated=True,
                              trait="some_other_trait", experiment_id="run-A-different-model")
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    _write_mapping(mapping_path, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "catkin_phenology.csv"

    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(other_trait_dir)],
        operating_point_conf=0.4,
        operating_point_validated="validated_held_out",
    )

    assert "error" in res
    assert res["positive_state_classifier_validated"] == "false"
    assert not out_csv.exists()


def test_compute_phenology_rejects_classifier_stamp_with_no_trait_recorded(tmp_path: Path) -> None:
    """Stage-6 review NEW-7, round 4: the real writer (calibrate_classifier_operating_point) always
    records a real trait name -- unlike experiment_id, there is no legitimate producer path that
    omits it. A sidecar with trait=None (a hand-edited or foreign file, not one the real writer could
    produce) must not be trusted just because neither the trait-mismatch nor the experiment-mismatch
    branch fires against a null -- both being null used to bypass the binding check entirely."""
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["dormant"])
    _write_preds(d2, "P1_b", ["elongated"])
    _write_op_sidecar(d1, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, validated=True, id_map=ID_MAP)
    # Genuinely "validated", but with NEITHER trait NOR experiment_id recorded -- the shape a
    # hand-edited/foreign sidecar could carry, never one calibrate_classifier_operating_point writes.
    _write_classifier_sidecar(d1, validated=True, trait=None, experiment_id=None)
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    _write_mapping(mapping_path, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "catkin_phenology.csv"

    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
        operating_point_conf=0.4,
        operating_point_validated="validated_held_out",
    )

    assert "error" in res
    assert res["positive_state_classifier_validated"] == "false"
    assert not out_csv.exists()


def test_compute_phenology_refuses_unvalidated_classifier(tmp_path: Path) -> None:
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["dormant"])
    _write_preds(d2, "P1_b", ["elongated"])
    _write_op_sidecar(d1, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, validated=True, id_map=ID_MAP)
    # No classifier_operating_point.json anywhere -> classifier dimension floors to unvalidated.
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    _write_mapping(mapping_path, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert res["positive_state_classifier_validated"] == "false"
    assert not out_csv.exists()


def test_compute_phenology_acknowledge_unvalidated_stamps_false(tmp_path: Path) -> None:
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["dormant"])
    _write_preds(d2, "P1_b", ["elongated"])
    _write_op_sidecar(d1, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, validated=True, id_map=ID_MAP)
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    _write_mapping(mapping_path, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        acknowledge_unvalidated=True,  # provisional delivery, clearly flagged
    )
    assert "error" not in res, res
    assert res["positive_state_classifier_validated"] == "false"
    assert out_csv.exists()


def test_compute_phenology_refuses_asymmetric_validation(tmp_path: Path) -> None:
    # Classifier validated but the count operating point isn't. The gate requires both.
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["dormant"])
    _write_preds(d2, "P1_b", ["elongated"])
    _write_op_sidecar(d1, validated=False, id_map=ID_MAP)
    _write_op_sidecar(d2, validated=False, id_map=ID_MAP)
    _write_classifier_sidecar(d1, validated=True, trait="catkin")
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    _write_mapping(mapping_path, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )
    assert "error" in res
    assert not out_csv.exists()


def test_compute_phenology_acknowledge_stamps_each_dimension_independently(tmp_path: Path) -> None:
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["dormant"])
    _write_preds(d2, "P1_b", ["elongated"])
    _write_op_sidecar(d1, validated=True, id_map=ID_MAP)
    _write_op_sidecar(d2, validated=True, id_map=ID_MAP)
    # Classifier not validated; op point IS validated on disk.
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    _write_mapping(mapping_path, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        operating_point_validated="validated_held_out",
        acknowledge_unvalidated=True,
    )
    assert "error" not in res, res
    assert res["positive_state_classifier_validated"] == "false"  # never upgraded
    assert res["operating_point_validated"] == "validated_held_out"
    assert out_csv.exists()


def test_compute_phenology_refuses_unclassified_predictions(tmp_path: Path) -> None:
    # Predictions from a bare detector (no elongation axis at all) — the round-3/4/5 canonical
    # case: this must refuse, never report full coverage.
    d1 = tmp_path / "2026-02-11"
    _write_preds(d1, "P1_a", ["catkin"])
    _write_op_sidecar(d1, validated=True, id_map={"catkin": 0})
    mapping_path = tmp_path / "plant_mapping.json"
    _write_mapping(mapping_path, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
    })
    out_csv = tmp_path / "out" / "catkin_phenology.csv"

    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1)},
        output_csv_path=str(out_csv),
    )

    assert "error" in res
    assert not out_csv.exists()


def test_compute_phenology_missing_mapping(tmp_path: Path) -> None:
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(tmp_path / "nope.json"),
        predictions_by_date={},
        output_csv_path=str(tmp_path / "out.csv"),
    )
    assert "error" in res
    assert "not found" in res["error"]


def test_compute_phenology_unknown_trait_refuses(tmp_path: Path) -> None:
    res = compute_phenology(
        trait="not-a-real-trait",
        mapping_path=str(tmp_path / "nope.json"),
        predictions_by_date={},
        output_csv_path=str(tmp_path / "out.csv"),
    )
    assert "error" in res


# ── calibrate_classifier_operating_point (K3) — the classifier-validity producer ────────────

def _write_calibration_image(
    gt_dir: Path, pred_dir: Path, stem: str, calls: list[tuple[bool | None, bool]], *,
    image_offset: float,
) -> None:
    """One (GT, pred) file pair with several classified instances, GT and pred boxes at the
    SAME position per instance (spaced far apart from each other) so they match regardless of
    the exact derived center-match tolerance. ``calls`` is
    ``[(is_true_positive, is_pred_positive), ...]`` — ``is_true_positive=None`` writes a GT
    instance with NO ``elongation`` attribute at all (never assessed), instead of a real value.
    ``image_offset`` shifts every box in this image by a unique amount so distinct images never
    collide on content hash — two images with the same classification PATTERN must still carry
    different geometry, the same way two different real photos would.
    """
    gt_anns, pred_anns = [], []
    for i, (is_tp, is_pred_pos) in enumerate(calls):
        x = image_offset + i * 20.0
        box = BBox(x, 0.0, x + 8.0, 8.0)
        attrs = {} if is_tp is None else {"elongation": "elongated" if is_tp else "dormant"}
        gt_anns.append(Annotation(subject="catkin", geometry=box, attributes=attrs))
        pred_anns.append(Annotation(subject="elongated" if is_pred_pos else "dormant",
                                    geometry=box, score=0.9))
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    w = int(image_offset + len(calls) * 20.0) + 8
    json_io.write_annotations(gt_dir / f"{stem}.json", gt_anns, w, 8)
    json_io.write_annotations(pred_dir / f"{stem}.json", pred_anns, w, 8)


def _write_split(gt_dir: Path, pred_dir: Path, *, prefix: str, n_images: int,
                 per_image_calls, offset: int = 0, offset_stride: float = 1000.0) -> None:
    """``per_image_calls(image_index) -> list[(is_tp, is_pred_pos)]`` builds one split of
    ``n_images`` files, each named ``f"{prefix}_{offset+i}"`` — a distinct stem per split so
    calibration and holdout are disjoint by construction unless a test deliberately reuses one.
    Each image also gets a unique geometry offset (``offset_stride`` apart), so two splits built
    with the SAME ``offset``/``prefix`` sequence (as a genuine-duplication test wants) land on
    identical geometry, while two splits meant to be independent don't collide by accident.
    """
    for i in range(n_images):
        _write_calibration_image(gt_dir, pred_dir, f"{prefix}_{offset + i}", per_image_calls(i),
                                 image_offset=(offset + i) * offset_stride)


def _one_positive_one_negative(_i: int) -> list[tuple[bool, bool]]:
    """Every image: one correctly-classified positive + one correctly-classified negative."""
    return [(True, True), (False, False)]


def _alternating_calls(i: int) -> list[tuple[bool, bool]]:
    """Every image: one correctly-classified instance, alternating positive/negative by index."""
    return [(i % 2 == 0, i % 2 == 0), (i % 2 == 1, i % 2 == 1)]


def test_calibrate_classifier_operating_point_passes_for_well_formed_reference(tmp_path: Path) -> None:
    cal_gt, cal_pred = tmp_path / "cal_gt", tmp_path / "cal_pred"
    hold_gt, hold_pred = tmp_path / "hold_gt", tmp_path / "hold_pred"
    # 20 images/split, 2 correctly-classified instances each (1 positive, 1 negative) -> perfect,
    # balanced, disjoint (different stems) references on both sides.
    calls = _one_positive_one_negative
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=calls)
    _write_split(hold_gt, hold_pred, prefix="hold", n_images=20, per_image_calls=calls, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="catkin", subject="catkin", attribute="elongation",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), experiment_id=None,
    )

    assert res["passed"] is True, res
    assert res["failures"] == []
    sidecar = json.loads((tmp_path / "out" / "classifier_operating_point.json").read_text())
    assert sidecar["validated"] is True


def test_calibrate_classifier_operating_point_refuses_genuinely_duplicated_holdout(tmp_path: Path) -> None:
    """A holdout whose GT content is cloned from calibration (same classification calls, same
    geometry) must refuse content_duplicated even under DIFFERENT image ids — the whole point of
    a content hash, not an image-id check (stage-6 review, K3's critical finding)."""
    cal_gt, cal_pred = tmp_path / "cal_gt", tmp_path / "cal_pred"
    hold_gt, hold_pred = tmp_path / "hold_gt", tmp_path / "hold_pred"
    calls = _alternating_calls
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=calls)
    # Different stems ("dup" vs "cal"), but identical per-image geometry+calls -> disjoint by
    # image_id, still content-duplicated.
    _write_split(hold_gt, hold_pred, prefix="dup", n_images=20, per_image_calls=calls)

    res = calibrate_classifier_operating_point(
        trait_name="catkin", subject="catkin", attribute="elongation",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), experiment_id=None,
    )

    assert res["passed"] is False
    assert "content_duplicated" in res["failures"]


def test_calibrate_classifier_operating_point_partial_flip_fails_compensating_error_floor(
    tmp_path: Path,
) -> None:
    """A classifier that flips a substantial, symmetric fraction of calls (net count-bias ~0)
    must still fail — the compensating-error floor's whole reason to exist (stage-6 review: the
    prior kappa>0 floor let a 40%-wrong classifier through)."""
    cal_gt, cal_pred = tmp_path / "cal_gt", tmp_path / "cal_pred"
    hold_gt, hold_pred = tmp_path / "hold_gt", tmp_path / "hold_pred"
    good = _one_positive_one_negative
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=30, per_image_calls=good)

    def flipped(i):
        # 40% of images get a fully-flipped pair (both calls wrong) -- symmetric, net bias 0.
        if i % 5 < 2:
            return [(True, False), (False, True)]
        return [(True, True), (False, False)]

    _write_split(hold_gt, hold_pred, prefix="hold", n_images=50, per_image_calls=flipped, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="catkin", subject="catkin", attribute="elongation",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), experiment_id=None,
    )

    assert res["passed"] is False
    assert "compensating_error_floor_failed" in res["failures"], res["failures"]
    sidecar = json.loads((tmp_path / "out" / "classifier_operating_point.json").read_text())
    sweep_data = sidecar["sweep_data"]
    assert sweep_data["kappa"] is not None
    assert sweep_data["kappa"] <= sweep_data["kappa_floor"]
    assert sweep_data["kappa_floor_source"] == "platform_provisional_default"  # catkin sets none


def test_resolve_classifier_operating_point_refuses_single_image_holdout() -> None:
    """Stage-6 review Finding C/N4: a single-image holdout has no images to vary the count-bias
    across, so its std is trivially 0 -- the SE penalty the equivalence test relies on vanishes.
    Must refuse (insufficient_holdout_images), the same minimum the detection path requires,
    rather than pass at exactly the tolerance with zero uncertainty discount."""
    from tcip_mcp.pipelines.operating_point import resolve_classifier_operating_point

    cal = [
        {"image_id": "c0", "is_true_positive": True, "is_pred_positive": True, "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"image_id": "c1", "is_true_positive": False, "is_pred_positive": False, "bbox": [20.0, 0.0, 30.0, 10.0]},
    ]
    # 20 instances, ALL on one image -- a real net bias of +1 (matching trait.count_bias_tolerance's
    # default of 1.0 exactly), which the equivalence test's SE=0 (n_images=1) would let through.
    hold = [
        {"image_id": "h0", "is_true_positive": i < 10, "is_pred_positive": i < 11,
         "bbox": [float(100 + i), 0.0, float(110 + i), 10.0]}
        for i in range(20)
    ]
    res = resolve_classifier_operating_point(
        "catkin", calibration_items=cal, holdout_items=hold, experiment_id=None)

    assert res["passed"] is False
    assert "insufficient_holdout_images" in res["failures"], res["failures"]


def test_resolve_classifier_operating_point_honors_trait_authored_agreement_floor(
    tmp_path: Path, monkeypatch,
) -> None:
    """Stage-6 review Finding B: TraitSpec.classifier_agreement_floor, when a trait authors one,
    must be the floor actually applied -- not the platform's provisional default."""
    from dataclasses import replace

    from tcip_mcp.pipelines import operating_point as op_mod
    from tests._trait_fixtures import CATKIN

    strict_catkin = replace(CATKIN, classifier_agreement_floor=0.9)
    monkeypatch.setattr(op_mod, "get_trait", lambda name: strict_catkin)

    # A holdout with kappa=0.8 -- clears the platform's provisional default (0.41) but not the
    # trait's own stricter authored floor (0.9).
    def make_items(n, flips):
        items = []
        for i in range(n):
            is_tp = i < n // 2
            pred = (not is_tp) if i in flips else is_tp
            items.append({"image_id": f"i{i}", "is_true_positive": is_tp, "is_pred_positive": pred,
                         "bbox": [float(i), 0.0, float(i + 10), 10.0]})
        return items

    cal = make_items(20, flips=set())
    hold = make_items(100, flips=set(range(10)))  # 10% symmetric flip -> kappa=0.8
    res = op_mod.resolve_classifier_operating_point(
        "catkin", calibration_items=cal, holdout_items=hold, experiment_id=None)

    assert 0.75 < res["sweep_data"]["kappa"] < 0.85, res["sweep_data"]
    assert res["sweep_data"]["kappa_floor"] == 0.9
    assert res["sweep_data"]["kappa_floor_source"] == "trait"
    assert res["passed"] is False
    assert "compensating_error_floor_failed" in res["failures"]


def test_calibrate_classifier_operating_point_foreign_checkpoint_stamp_still_reachable(
    tmp_path: Path,
) -> None:
    """experiment_id=None (a foreign/unregistered checkpoint) skips train-disjointness rather than
    failing closed -- the classifier-validity stamp is still reachable for an otherwise clean
    reference (K3's foreign-checkpoint case)."""
    cal_gt, cal_pred = tmp_path / "cal_gt", tmp_path / "cal_pred"
    hold_gt, hold_pred = tmp_path / "hold_gt", tmp_path / "hold_pred"
    calls = _one_positive_one_negative
    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=calls)
    _write_split(hold_gt, hold_pred, prefix="hold", n_images=20, per_image_calls=calls, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="catkin", subject="catkin", attribute="elongation",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), experiment_id=None,
    )

    assert res["passed"] is True, res
    assert "train_disjointness_unresolvable" not in res["failures"]
    assert "train_disjointness_leaked" not in res["failures"]


def test_calibrate_classifier_operating_point_unassessed_gt_never_fabricates_a_negative(
    tmp_path: Path,
) -> None:
    """Stage-6 review N1: a GT instance never assessed for `attribute` (no elongation key at all)
    must be excluded from the reference, never coerced into "not positive". A perfect classifier
    scored against a reference where every image ALSO carries unassessed instances must still pass
    cleanly -- the unassessed instances contribute no fabricated disagreement."""
    cal_gt, cal_pred = tmp_path / "cal_gt", tmp_path / "cal_pred"
    hold_gt, hold_pred = tmp_path / "hold_gt", tmp_path / "hold_pred"

    def perfect_plus_unassessed(_i: int) -> list[tuple[bool | None, bool]]:
        # 2 correctly-classified instances (1 pos, 1 neg) + 2 never-assessed instances. The
        # classifier's own call on the unassessed instances is irrelevant -- they must not enter
        # the reference at all, so their is_pred_positive value here is deliberately inconsistent
        # (would fabricate a disagreement if the bug were still present).
        return [(True, True), (False, False), (None, True), (None, False)]

    _write_split(cal_gt, cal_pred, prefix="cal", n_images=20, per_image_calls=perfect_plus_unassessed)
    _write_split(hold_gt, hold_pred, prefix="hold", n_images=20,
                per_image_calls=perfect_plus_unassessed, offset=1000)

    res = calibrate_classifier_operating_point(
        trait_name="catkin", subject="catkin", attribute="elongation",
        calibration_gt_dir=str(cal_gt), calibration_pred_dir=str(cal_pred),
        holdout_gt_dir=str(hold_gt), holdout_pred_dir=str(hold_pred),
        output_dir=str(tmp_path / "out"), experiment_id=None,
    )

    assert res["passed"] is True, res
    sidecar = json.loads((tmp_path / "out" / "classifier_operating_point.json").read_text())
    sweep_data = sidecar["sweep_data"]
    assert sweep_data["kappa"] == 1.0, sweep_data  # a perfect classifier, not degraded by phantom errors
    assert sweep_data["count_bias"] == 0.0, sweep_data
    assert res["n_holdout_items"] == 20 * 2  # only the 2 assessed instances/image counted, not 4


def test_classification_items_derives_center_match_tolerance_across_the_whole_split(
    tmp_path: Path,
) -> None:
    """Stage-6 review Finding A/N5: the center-match tolerance must be derived from the WHOLE
    split's GT, not one image at a time. A small-object image's real pair, offset by more than
    that image's own (too-tight) per-image tolerance but less than the split-wide tolerance
    (pulled up by a large-object image elsewhere in the same split), must still match."""
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    spec = get_trait("catkin")
    assert spec.localization == CENTER_MATCH and spec.localization_tolerance_frac == 0.5

    # "big": a 200x200 GT box -> char_size=200 -> per-image tolerance would be 100px; offset 40px
    # (comfortably under both the per-image AND the split-wide tolerance derived below).
    big_box = BBox(0.0, 0.0, 200.0, 200.0)
    json_io.write_annotations(gt_dir / "big.json", [
        Annotation(subject="catkin", geometry=big_box, attributes={"elongation": "elongated"}),
    ], 400, 400)
    json_io.write_annotations(pred_dir / "big.json", [
        Annotation(subject="elongated", geometry=BBox(40.0, 0.0, 240.0, 200.0), score=0.9),
    ], 400, 400)

    # "small": a 20x20 GT box -> char_size=20 -> per-image tolerance would be only 10px; the
    # prediction is offset 15px, which the OLD per-image derivation would drop as unmatched.
    small_box = BBox(1000.0, 0.0, 1020.0, 20.0)
    json_io.write_annotations(gt_dir / "small.json", [
        Annotation(subject="catkin", geometry=small_box, attributes={"elongation": "dormant"}),
    ], 1100, 100)
    json_io.write_annotations(pred_dir / "small.json", [
        Annotation(subject="dormant", geometry=BBox(1015.0, 0.0, 1035.0, 20.0), score=0.9),
    ], 1100, 100)

    # Split-wide avg char_size = (200 + 20) / 2 = 110 -> tolerance = 0.5 * 110 = 55px, comfortably
    # above both the small image's 15px offset and the big image's 40px offset.
    items = _classification_items(str(gt_dir), str(pred_dir), subject="catkin",
                                  positive_value="elongated", attribute="elongation", spec=spec)

    by_image = {it["image_id"]: it for it in items}
    assert "small" in by_image, (
        "the small-object image's pair was dropped -- tolerance was derived per-image, not "
        "across the whole split"
    )
    assert by_image["small"]["is_true_positive"] is False  # "dormant"
    assert by_image["small"]["is_pred_positive"] is False
    assert "big" in by_image
    assert by_image["big"]["is_true_positive"] is True  # "elongated"


def test_classification_items_scopes_gt_to_the_run_subject(tmp_path: Path) -> None:
    """Stage-6 review NEW-5, round 4: a labels dir isn't guaranteed to hold only one kind of
    annotation -- a dataset that also isolates an enabling subject (e.g. "bush", root CLAUDE.md's
    "a subject is not a trait") must not let that unrelated box enter the match pool. Here a "bush"
    annotation sits EXACTLY on the prediction's center (distance 0) while the real "catkin" GT is a
    few px off -- if subject weren't scoped, greedy center-match (closest first) would steal the
    match for "bush" and either drop the real catkin pair or attribute it to the wrong box/attribute
    entirely."""
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    spec = get_trait("catkin")

    # catkin center (125, 125), offset ~7px from the prediction -- comfortably inside tolerance.
    # bush center (120, 120), an EXACT match to the prediction -- if it entered the pool, greedy
    # center-match (ascending distance) would steal the match for bush (distance 0 beats ~7px)
    # ahead of the real catkin pair, regardless of list order.
    catkin_box = BBox(105.0, 105.0, 145.0, 145.0)
    bush_box = BBox(114.0, 114.0, 126.0, 126.0)
    json_io.write_annotations(gt_dir / "a.json", [
        Annotation(subject="catkin", geometry=catkin_box, attributes={"elongation": "elongated"}),
        Annotation(subject="bush", geometry=bush_box),
    ], 400, 400)
    json_io.write_annotations(pred_dir / "a.json", [
        Annotation(subject="elongated", geometry=BBox(114.0, 114.0, 126.0, 126.0), score=0.9),
    ], 400, 400)

    items = _classification_items(str(gt_dir), str(pred_dir), subject="catkin",
                                  positive_value="elongated", attribute="elongation", spec=spec)

    assert len(items) == 1
    assert items[0]["bbox"] == [105.0, 105.0, 145.0, 145.0]  # the catkin box, not bush's
    assert items[0]["is_true_positive"] is True  # catkin's own "elongated" attribute
