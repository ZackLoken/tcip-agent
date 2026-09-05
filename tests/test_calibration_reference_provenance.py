"""A calibration reference has to be a measurement, not the model's own output.

Point a calibration or held-out reference at a bucket of predictions and every numeric gate
clears, because the model agrees with itself: the split is disjoint, the counts are unbiased, and
the operating point earns a held-out stamp with no independent measurement anywhere in the chain.
The provenance the prediction records already carry is the only thing that can catch it, so both
reference reads (the detection calibration's labels dir and the classifier calibration's GT dirs)
refuse such a directory whole rather than validating against it, while ordinary ground truth,
ground truth a person authored, and a prediction a reviewer accepted all still calibrate.

Dropping the score off that output does not change what it is: ground truth an agent authored,
with no reviewer's accepted_by, is the model answering for itself under a different key, so the
rail reads authorship as well as the score. An unattributed label, a person's label, and an
agent's label a reviewer confirmed all remain admissible.

Also covers the conf-floor mismatch reaching the delivered issue list: it is provenance, not a
gate, so it travels with a run's ``shippable_issues`` without changing whether that run validated.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 32
PRODUCER = "model:m_best@c9f632ba98b2"  # the shape run_inference stamps on every prediction


def _save_png(path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMG, IMG), color=(128, 128, 128)).save(path)


def _reference(root, stems, annotations):
    """An images dir plus a reference dir holding ``annotations(stem)`` for each stem."""
    images_dir, labels_dir = root / "images", root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for s in stems:
        _save_png(images_dir / f"{s}.png")
        json_io.write_annotations(str(labels_dir / f"{s}.json"), annotations(s), IMG, IMG,
                                  keep_empty=True)
    return images_dir, labels_dir


def _prediction(subject="bud", score=0.87, box=(2, 2, 10, 10), *, state=None):
    attributes = {"state": state} if state is not None else {}
    return Annotation(subject=subject, geometry=BBox(*box), score=score, created_by=PRODUCER,
                      created_at="2026-01-01T00:00:00+00:00", attributes=attributes)


def _hand_annotation(box=(2, 2, 10, 10), **kw):
    return Annotation(subject="bud", geometry=BBox(*box), **kw)


class _CalStub:
    """A predictor with the mutable operating-point surface the calibration sets, predicting
    nothing: enough to drive the reference read without a forward pass."""

    def __init__(self):
        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": IMG, "height": IMG,
                 "boxes": [], "scores": [], "labels": [], "count": 0} for p in paths]


_CAL_KWARGS = dict(tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8, global_nms_iou=0.3,
                   postprocess="nms", cross_tile_nms=None, max_dets=None, group_by="stem",
                   seed=0, holdout_ratio=0.5)


def _calibrate(labels_dir, images_dir):
    import tcip_mcp.pipelines.calibration as calibration

    return calibration.calibrate_operating_point(
        _CalStub(), "bud_opening", str(labels_dir), str(images_dir), **_CAL_KWARGS)


def _classification_items(gt_dir, pred_dir):
    from tcip_mcp.tools.phenology_tools import _classification_items

    return _classification_items(str(gt_dir), str(pred_dir), trait_name="bud_opening", subject="bud",
                                 positive_value="open", attribute="state")


# --- the reference reads refuse a directory of the model's own predictions --------------------

def test_detection_calibration_refuses_a_reference_of_the_models_own_predictions(tmp_path):
    """A calibration labels dir pointed at a prediction bucket earns a held-out stamp at every
    numeric gate, so the refusal has to come from the records' own provenance."""
    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _reference(
        tmp_path / "ds", stems, lambda s: [_prediction(), _prediction(box=(15, 15, 25, 25))])

    with pytest.raises(ValueError) as exc:
        _calibrate(labels_dir, images_dir)

    message = str(exc.value)
    assert "16 of 16 annotations" in message  # the whole reference, counted, not a sampled subset
    assert str(labels_dir) in message
    assert "accept the model's proposals through review" in message


def test_classifier_calibration_refuses_a_gt_dir_of_the_models_own_predictions(tmp_path):
    """The classifier calibration's GT side is a reference too: pointed at predictions it matches
    the classifier's calls against the classifier's own calls."""
    stems = ["a", "b"]
    _images, gt_dir = _reference(tmp_path / "gt", stems, lambda s: [_prediction()])
    _pred_images, pred_dir = _reference(tmp_path / "pred", stems,
                                        lambda s: [_prediction(subject="open")])

    with pytest.raises(ValueError) as exc:
        _classification_items(gt_dir, pred_dir)

    assert str(gt_dir) in str(exc.value)
    assert "measures the model against itself" in str(exc.value)


def test_the_classifier_tool_reports_an_inadmissible_reference_as_a_refusal(tmp_path):
    """The refusal reaches the caller as a plain error rather than an exception escaping the tool."""
    from tcip_mcp.tools.phenology_tools import calibrate_classifier_operating_point

    stems = ["a", "b"]
    _gt_images, gt_dir = _reference(tmp_path / "gt", stems, lambda s: [_prediction()])
    _pred_images, pred_dir = _reference(tmp_path / "pred", stems,
                                        lambda s: [_prediction(subject="open")])

    result = calibrate_classifier_operating_point(
        trait_name="bud_opening", subject="bud", attribute="state",
        calibration_gt_dir=str(gt_dir), calibration_pred_dir=str(pred_dir),
        holdout_gt_dir=str(gt_dir), holdout_pred_dir=str(pred_dir),
        output_dir=str(tmp_path / "out"), dataset_root=str(tmp_path / "gt"))

    assert "carry a prediction score" in result.get("error", "")


def test_a_mixed_reference_refuses_whole_rather_than_calibrating_on_its_clean_subset(tmp_path):
    """Dropping the offending records would validate against a reference nobody chose."""
    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _reference(
        tmp_path / "ds", stems,
        lambda s: [_hand_annotation()] + ([_prediction()] if s == "src0_0_0" else []))

    with pytest.raises(ValueError, match="1 of 9 annotations"):
        _calibrate(labels_dir, images_dir)


# --- the reference reads refuse ground truth an agent authored and nobody ruled on ------------

def test_detection_calibration_refuses_ground_truth_an_agent_authored_that_nobody_ruled_on(tmp_path):
    """The same output with its score dropped: no reviewer took responsibility for any of it, so
    the model is still the only thing standing behind the reference."""
    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _reference(
        tmp_path / "ds", stems,
        lambda s: [_hand_annotation(created_by=PRODUCER, created_at="2026-01-01T00:00:00+00:00")])

    with pytest.raises(ValueError) as exc:
        _calibrate(labels_dir, images_dir)

    message = str(exc.value)
    assert "8 of 8 annotations" in message
    assert PRODUCER in message  # the refusal names the authorship it is refusing
    assert "no human has adjudicated is not a calibration or holdout reference" in message
    assert "review-confirmation loop" in message  # the lighter alternative, named


def test_the_reference_rail_reads_a_bare_tool_name_as_a_machine_author(tmp_path):
    """A tool producer stays bare under the identity convention, so a bare author is a machine
    author rather than a person whose prefix was left off."""
    _images, labels_dir = _reference(tmp_path / "ds", ["a"],
                                     lambda s: [_hand_annotation(created_by="sam")])

    with pytest.raises(ValueError) as exc:
        json_io.require_reference_ground_truth(labels_dir)

    assert "authored by sam" in str(exc.value)


def test_one_agent_authored_record_refuses_the_whole_reference(tmp_path):
    """The authorship rule keeps the score rule's shape: calibrating on the admissible subset
    would validate against a reference nobody chose."""
    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _reference(
        tmp_path / "ds", stems,
        lambda s: [_hand_annotation()] + ([_hand_annotation(created_by="sam")]
                                          if s == "src0_0_0" else []))

    with pytest.raises(ValueError, match="1 of 9 annotations"):
        _calibrate(labels_dir, images_dir)


# --- the rail admits the references that were always legitimate -------------------------------

def test_hand_annotated_ground_truth_still_calibrates(tmp_path):
    """No score and no author: ordinary ground truth, unchanged by the rail."""
    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _reference(tmp_path / "ds", stems, lambda s: [_hand_annotation()])

    bundle, dataset_hash, n_excluded, _evidence = _calibrate(labels_dir, images_dir)

    assert bundle.get("conf") is not None
    assert dataset_hash and n_excluded == 0


def test_ground_truth_a_person_authored_still_calibrates(tmp_path):
    """The GUI stamps a human author on every shape it saves; that is a reference, not output."""
    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _reference(
        tmp_path / "ds", stems,
        lambda s: [_hand_annotation(created_by="user:breeder",
                                    created_at="2026-01-01T00:00:00+00:00")])

    bundle, _dataset_hash, _n_excluded, _evidence = _calibrate(labels_dir, images_dir)

    assert bundle.get("conf") is not None


def test_a_prediction_a_reviewer_accepted_still_calibrates(tmp_path):
    """Review acceptance keeps the producing model as the creator, adds the reviewer's sign-off,
    and drops the score: model-origin ground truth a human stands behind."""
    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _reference(
        tmp_path / "ds", stems,
        lambda s: [_hand_annotation(created_by=PRODUCER, created_at="2026-01-01T00:00:00+00:00",
                                    accepted_by="user:breeder",
                                    accepted_at="2026-01-02T00:00:00+00:00")])

    bundle, _dataset_hash, _n_excluded, _evidence = _calibrate(labels_dir, images_dir)

    assert bundle.get("conf") is not None


def test_the_classifier_reference_admits_ground_truth_beside_a_scored_prediction_bucket(tmp_path):
    """Only the GT side is held to the rule: the prediction side is predictions by definition."""
    stems = ["a", "b"]
    gt_dir = tmp_path / "gt" / "labels"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        json_io.write_annotations(
            str(gt_dir / f"{stem}.json"),
            [Annotation(subject="bud", geometry=BBox(2, 2, 12, 12),
                        attributes={"state": "open"}),
             Annotation(subject="bud", geometry=BBox(16, 16, 26, 26),
                        attributes={"state": "closed"})], IMG, IMG)
    _pred_images, pred_dir = _reference(
        tmp_path / "pred", stems,
        lambda s: [_prediction(box=(2, 2, 12, 12), state="open"),
                   _prediction(box=(16, 16, 26, 26), state="closed")])

    items = _classification_items(gt_dir, pred_dir)

    assert len(items) == 4  # two matched instances per image, both images
    assert {i["is_true_positive"] for i in items} == {True, False}


# --- the conf-floor mismatch travels to the delivery surface without becoming a gate ----------

def _floor_mismatched_bundle():
    """A reference that passes its held-out gate while its own lowest score sits far above the
    floor the calibration asserts it was staged at."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [0] * n_images
    inputs = {
        "dataset_hash": "H",
        "calibration_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
            miss_pattern=miss, fp_pattern=fp, score=0.9),
        "holdout_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="h", shift=5.0,
            miss_pattern=miss, fp_pattern=fp, score=0.9),
        "tiled": False, "staged_conf_floor": 0.001,
    }
    return resolve_operating_point("bud_opening", experiment_id=None, **inputs), inputs


class _OneDetectionStub(_CalStub):
    def __init__(self, in_chans=3):
        super().__init__()
        self.in_chans = in_chans

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": IMG, "height": IMG,
                 "boxes": [[4, 4, 12, 12]], "scores": [0.9], "labels": [1], "count": 1}
                for p in paths]


def _run_with_bundle(tmp_path, monkeypatch, calibration):
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import registered_checkpoint, run_inference_verified

    bundle, inputs = calibration
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(tmp_path)}}}
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))
    monkeypatch.setattr(predictor_mod, "build_predictor",
                        lambda checkpoint, **kw: _OneDetectionStub())
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    image = tmp_path / "capture.png"
    _save_png(image)
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    return run_inference_verified(
        str(ckpt), image_paths=[str(image)], images_dir=str(tmp_path), device="cpu", tile=False,
        trait="bud_opening", calibration_labels_dir=str(tmp_path))


def test_a_conf_floor_mismatch_reaches_the_delivered_issues_without_changing_the_gate(
    tmp_path, monkeypatch,
):
    """The mismatch is provenance about the reference, not a reason to refuse: it has to be
    readable at the delivery surface, and the run it describes still validates."""
    bundle, inputs = _floor_mismatched_bundle()
    assert bundle.get("conf").gate_evidence["conf_floor_mismatch"] is True
    assert "conf_floor_mismatch" not in (bundle.get("conf").gate_evidence["failures"] or [])

    r = _run_with_bundle(tmp_path, monkeypatch, (bundle, inputs))

    assert "error" not in r, r
    assert any("low-conf tail" in issue for issue in r["shippable_issues"]), r["shippable_issues"]
    assert r["validated"] is True  # surfaced, never gating
    assert r["gate_evidence_summary"]["conf_floor_mismatch"] is True


def test_a_reference_without_the_mismatch_carries_no_such_issue(tmp_path, monkeypatch):
    """The companion: nothing is appended when the reference's own scores reach the floor."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    inputs = {
        "dataset_hash": "H",
        "calibration_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "holdout_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="h", shift=5.0,
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "tiled": False, "staged_conf_floor": 0.01,
    }
    bundle = resolve_operating_point("bud_opening", experiment_id=None, **inputs)
    assert bundle.get("conf").gate_evidence["conf_floor_mismatch"] is False

    r = _run_with_bundle(tmp_path, monkeypatch, (bundle, inputs))

    assert r["shippable_issues"] == []
    assert r["validated"] is True
