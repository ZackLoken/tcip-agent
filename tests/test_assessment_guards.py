"""The assessment's rails, each beside the legitimate call it must still admit.

The chain's own producers build every record here: a capture ingested, a selection drawn, a tiny
detector trained on it, the trait confirmed, the assessment run and the bucket published. A
reference is re-sided only through the admission's own sample producer, never hand-written.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, Polygon  # noqa: E402

from tests import _trait_fixtures as fx  # noqa: E402
from tests._chain_fixtures import (  # noqa: E402
    DATE, IMG, SUBJECT, assess, confirm_count_trait, draw_reference_selection, object_at,
    run_the_chain, synthetic_capture, train_on,
)
from tests._dense_op_fixtures import dense_records, good_cal_holdout  # noqa: E402


def _trained(project: Path, experiment_id: str):
    """A capture, its drawn selection and a checkpoint trained on the selection's train side;
    ``(images_dir, selection, selection_dir, checkpoint_path)``."""
    root = project / "ds"
    images_dir = synthetic_capture(root)
    selection_dir = project / "selection"
    selection = draw_reference_selection(project, root, selection_dir)
    return images_dir, selection, selection_dir, train_on(selection_dir, project, experiment_id)


def _add_frame(images_dir: Path, stem: str, *, like: str, same_pixels: bool) -> None:
    """Add frame ``stem`` labelled exactly as frame ``like``: its image a byte copy of ``like``'s
    when ``same_pixels``, else a frame of its own (another shade) holding the same box."""
    from PIL import Image, ImageDraw

    from tests._producer_fixtures import image_label_key, label_image

    if same_pixels:
        shutil.copyfile(images_dir / f"{like}.png", images_dir / f"{stem}.png")
    else:
        x0, y0, size = object_at(int(like[1:]))
        frame = Image.new("RGB", (IMG, IMG), color=(90, 60, 30))
        ImageDraw.Draw(frame).rectangle([x0, y0, x0 + size - 1, y0 + size - 1],
                                        fill=(250, 240, 200))
        frame.save(images_dir / f"{stem}.png")
    label_image(images_dir / f"{stem}.png", json_io.read_label_document(
        image_label_key(images_dir / f"{like}.png")).annotations, IMG, IMG)


def _resided(project: Path, images_dir: Path, drawn, extra: dict[str, str], out: Path) -> Path:
    """A reference selection of its own: the drawn selection's calibration and holdout sides with
    the added frames ``extra`` (member to side) joined to them, sided and grouped by the admission's
    own sample producer under the draw's own policy; its dir."""
    from tcip_mcp.pipelines.data.label_queries import admit
    from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES, Selection, write_selection
    from tcip_mcp.pipelines.data.splits import recorded_group_key_fn

    admitted = admit(images_dir, scope=drawn.scope)
    sides = {s.member: s.side for s in drawn.samples if s.side in REFERENCE_SIDES} | extra
    samples = admitted.samples(sides, recorded_group_key_fn(drawn.group_by, date=admitted.date))
    write_selection(out, Selection(samples=tuple(samples), scope=drawn.scope, seed=drawn.seed,
                                   group_by=drawn.group_by,
                                   dataset_fingerprint=drawn.dataset_fingerprint),
                    project=project)
    return out


def _a_train_member(selection) -> str:
    return next(s.member for s in selection.samples if s.side == "train")


# -- identity: content, never annotation geometry --------------------------------------


def test_a_distinct_image_holding_a_training_frames_boxes_is_a_legitimate_reference(tmp_path):
    """Two different images can carry identical boxes: the second is not the first, so a reference
    holding it is held out. Identity is the image's content, never the geometry drawn on it."""
    images_dir, drawn, _sel, checkpoint = _trained(tmp_path, "exp-twin")
    confirm_count_trait(tmp_path)
    like = _a_train_member(drawn)
    _add_frame(images_dir, "twin", like=like, same_pixels=False)
    reference = _resided(tmp_path, images_dir, drawn, {"twin": "holdout"},
                         tmp_path / "reference")

    record = assess(tmp_path, checkpoint, reference)

    assert "error" not in record, record
    for side in ("training", "selection"):
        assert record["disjointness"][side] == {"groups": [], "source_digests": []}, record
    assert "reference_shares_training" not in record["failures"]


def test_a_training_image_copied_under_a_new_name_into_the_reference_fails_disjointness(tmp_path):
    """A byte copy of a training frame under another name is the same image: the reference holding
    it shares its content with training, whatever it is called or grouped under."""
    images_dir, drawn, _sel, checkpoint = _trained(tmp_path, "exp-dup")
    confirm_count_trait(tmp_path)
    like = _a_train_member(drawn)
    _add_frame(images_dir, "dup", like=like, same_pixels=True)
    reference = _resided(tmp_path, images_dir, drawn, {"dup": "holdout"},
                         tmp_path / "reference")

    record = assess(tmp_path, checkpoint, reference)

    assert "error" not in record, record
    assert record["passed"] is False
    assert "reference_shares_training" in record["failures"], record["failures"]
    assert record["disjointness"]["training"]["groups"] == [], record["disjointness"]
    assert len(record["disjointness"]["training"]["source_digests"]) == 1, record["disjointness"]


def test_a_reference_frame_in_a_training_group_fails_even_when_its_pixels_are_its_own(tmp_path):
    """A tile of a training source is not held out from it: a reference frame grouped with a
    training frame fails on the group alone, its own pixels notwithstanding."""
    images_dir, drawn, _sel, checkpoint = _trained(tmp_path, "exp-group")
    confirm_count_trait(tmp_path)
    like = _a_train_member(drawn)
    sibling = f"{like}_5_5"
    _add_frame(images_dir, sibling, like=like, same_pixels=False)
    reference = _resided(tmp_path, images_dir, drawn, {sibling: "holdout"},
                         tmp_path / "reference")

    record = assess(tmp_path, checkpoint, reference)

    assert "error" not in record, record
    assert "reference_shares_training" in record["failures"], record["failures"]
    assert record["disjointness"]["training"]["source_digests"] == [], record["disjointness"]
    assert len(record["disjointness"]["training"]["groups"]) == 1, record["disjointness"]


def test_a_mosaic_reference_band_is_held_out_only_inside_a_non_training_region():
    """Against a within-image split's recorded regions, a reference rect is held out when it lies
    wholly inside one non-training region; one straddling the training boundary is not."""
    from tcip_mcp.pipelines.operating_point import spatial_disjointness

    spatial = {"train_region": [[0, 0, 500, 1000]], "val_region": [[500, 0, 650, 1000]],
               "calibration_region": [[650, 0, 800, 1000]],
               "holdout_region": [[800, 0, 1000, 1000]]}

    assert spatial_disjointness(spatial, [(550, 100, 600, 300), (680, 100, 780, 300),
                                          (850, 100, 950, 300)]) == []
    assert spatial_disjointness(spatial, [(400, 100, 600, 300)]) == ["[400, 100, 600, 300]"]
    with pytest.raises(ValueError, match="calibration_region"):
        spatial_disjointness({k: v for k, v in spatial.items() if k != "calibration_region"}, [])


# -- the count and classifier criteria ------------------------------------------------


def _count_entry(**fields):
    return fx.with_fields(fx.COUNT_SPEC, **fields)


def test_an_accurate_reference_passes_the_count_criterion():
    from tcip_mcp.pipelines.operating_point import count_criterion

    cal, hold = good_cal_holdout()
    _conf, _evidence, failures = count_criterion(
        cal, hold, _count_entry(count_error_tolerance=2.0), staged_conf_floor=0.01,
        staged_conf_floor_attribute_path="self")

    assert failures == [], failures


def test_a_known_count_bias_fails_the_outcome_tolerance_with_no_dispersion_at_all():
    """Every held-out image over-counts by the same ten objects: the bias is certain, its spread
    zero, and the count is still wrong."""
    from tcip_mcp.pipelines.operating_point import count_criterion

    cal, _ = good_cal_holdout()
    hold = dense_records(id_prefix="h", miss_pattern=[0] * 20, fp_pattern=[10] * 20, score=0.9)
    _conf, evidence, failures = count_criterion(
        cal, hold, _count_entry(count_error_tolerance=20.0), staged_conf_floor=0.01,
        staged_conf_floor_attribute_path="self")

    assert evidence["holdout_at_conf"]["count_bias_std_present"] == 0.0
    assert "count_bias_exceeds_tolerance" in failures, failures


def _items(image: str, calls: list[tuple[bool, bool]]) -> list[dict]:
    return [{"image_id": image, "is_true_positive": t, "is_pred_positive": p} for t, p in calls]


def test_an_agreeing_classifier_passes_the_classifier_criterion():
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    calls = [(True, True), (True, True), (False, False), (False, False)]
    hold = _items("a", calls) + _items("b", calls)
    evidence, failures = classifier_criterion(hold, hold, _count_entry())

    assert failures == [], (failures, evidence)


def test_compensating_classification_errors_cannot_validate_through_pooled_counts():
    """Each image's two positives called negative and two negatives called positive: its positive
    count is exactly right, and every call is wrong."""
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    calls = [(True, False), (True, False), (False, True), (False, True)]
    hold = _items("a", calls) + _items("b", calls)
    evidence, failures = classifier_criterion(hold, hold, _count_entry())

    assert evidence["positive_count_bias"] == 0.0
    assert "classifier_agreement_below_floor" in failures, (failures, evidence)


# -- the trait revision -----------------------------------------------------------------


def test_an_unauthored_tolerance_refuses_the_revision_with_the_breeders_question(tmp_path):
    """No default stands in for a tolerance the criterion compares against: a count revision that
    leaves one unauthored is refused when proposed, asking the breeder, and none is recorded."""
    from tcip_mcp.traits import QUESTIONS, UnauthoredFieldError, trait_names

    with pytest.raises(UnauthoredFieldError) as refused:
        confirm_count_trait(tmp_path, count_bias_tolerance_frac=None)

    assert "count_bias_tolerance_frac" in str(refused.value)
    assert QUESTIONS["count_bias_tolerance_frac"] in str(refused.value)
    assert fx.COUNT_TRAIT not in trait_names(tmp_path)


def test_a_delivery_under_another_revision_than_the_assessments_refuses_naming_both(tmp_path):
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    chain = run_the_chain(tmp_path, experiment_id="exp-revision")
    confirm_count_trait(tmp_path, count_error_tolerance=3.0)
    out_csv = tmp_path / "counts.csv"

    delivered = deliver_per_image_counts(tmp_path, str(chain.root), chain.bucket, str(out_csv),
                                         trait=fx.COUNT_TRAIT)

    assert "error" in delivered, delivered
    assert "revision 1" in delivered["error"] and "revision 2" in delivered["error"], delivered
    assert not out_csv.exists()


# -- the one gate over the producers ------------------------------------------------------


def test_one_producer_clears_the_gate_and_a_series_of_two_refuses(tmp_path):
    """Two checkpoints over the same capture are two measurements: a delivery naming buckets from
    both refuses outright, naming each, while either one alone clears."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.delivery import DeliveryRefusedError, Result, gate
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.tools.inference_tools import run_inference

    nothing = Result((), (), population=())
    chain = run_the_chain(tmp_path, experiment_id="exp-series-a")
    other = train_on(chain.selection_dir, tmp_path, "exp-series-b")
    second = f"other/{DATE}"
    published = run_inference(tmp_path, checkpoint_path=other, images_dir=str(chain.images_dir),
                              bucket=second)
    assert "error" not in published, published
    revision = latest_confirmed(fx.COUNT_TRAIT, tmp_path)
    first = chain.read()

    assert gate(tmp_path, [first], delivery_kind="per_image_count", revision=revision,
                result=nothing).validated
    with pytest.raises(DeliveryRefusedError, match="more than one checkpoint") as refused:
        gate(tmp_path, [first, read_bucket(chain.root, second)],
             delivery_kind="per_image_count", revision=revision, result=nothing)
    assert chain.bucket in str(refused.value) and second in str(refused.value)


def test_an_assessment_whose_retained_reference_is_gone_answers_for_nothing(tmp_path):
    """An assessment whose retained reference copies are removed reads as a changed reference,
    so the gate refuses the bucket it validated."""
    from tcip_mcp.assessment import REFERENCE_DIR, assessment_dir
    from tcip_mcp.delivery import DeliveryRefusedError, Result, gate
    from tcip_mcp.operationalization import latest_confirmed

    chain = run_the_chain(tmp_path, experiment_id="exp-forged")
    shutil.rmtree(assessment_dir(tmp_path, chain.assessment["assessment_id"]) / REFERENCE_DIR)

    with pytest.raises(DeliveryRefusedError, match="changed since"):
        gate(tmp_path, [chain.read()],
             delivery_kind="per_image_count",
             revision=latest_confirmed(fx.COUNT_TRAIT, tmp_path),
             result=Result((), (), population=()))


def test_a_reference_source_edited_after_the_assessment_answers_for_nothing(tmp_path):
    """The pixels an assessment measured are part of its reference: a source image edited after
    it refuses the bucket it validated, naming the image, which cleared before the edit."""
    from PIL import Image

    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.delivery import DeliveryRefusedError, Result, gate
    from tcip_mcp.operationalization import latest_confirmed

    chain = run_the_chain(tmp_path, experiment_id="exp-edited-source")
    nothing = Result((), (), population=())
    revision = latest_confirmed(fx.COUNT_TRAIT, tmp_path)
    bucket = chain.read()
    assert gate(tmp_path, [bucket], delivery_kind="per_image_count", revision=revision,
                result=nothing).validated
    assessment = read_assessment(tmp_path, chain.assessment["assessment_id"])
    source = assessment.reference.samples[0].source
    with Image.open(source) as frame:
        edited = frame.copy()
    edited.putpixel((0, 0), (255, 0, 0))
    edited.save(source)

    with pytest.raises(DeliveryRefusedError, match="changed since") as refused:
        gate(tmp_path, [bucket], delivery_kind="per_image_count", revision=revision,
             result=nothing)
    assert source in str(refused.value)


# -- a missing fact refuses by name -----------------------------------------------------


def test_an_evaluation_record_missing_its_ground_truth_field_refuses_by_name():
    from tcip_mcp.pipelines.training.evaluation import gt_objects

    with pytest.raises(KeyError, match="gt"):
        gt_objects({"width": IMG, "height": IMG, "dt": []})


def test_a_scalar_prediction_carrying_no_output_refuses_the_assessment_by_name(tmp_path):
    from types import SimpleNamespace

    from PIL import Image

    from tcip_mcp.assessment import AssessmentRefusedError, _scalar
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample
    from tcip_mcp.pipelines.execution import Pass, untiled_execution

    class Silent:
        def predict_batch(self, sources, execution, *, tile_batch_size):
            return [{"image": s, "regression_values": []} for s in sources]

    checkpoint = SimpleNamespace(task="regression", payload={}, path="regressor.pt")
    p = Pass(checkpoint=checkpoint, predictor=Silent(), scope=ClassScope(),
             execution=untiled_execution(checkpoint, conf=None, max_dets=None), tile_batch_size=1)
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    (tmp_path / "t.csv").write_text("image,value\na,1.0\n", encoding="utf-8")
    sample = Sample(member="a", source=str(tmp_path / "a.png"),
                    ground_truth=str(tmp_path / "t.csv"),
                    group="a", side="holdout", row_key="a")
    entry = fx.COUNT_SPEC.model_copy(update={"regression_criterion": "r_squared"})

    with pytest.raises(AssessmentRefusedError, match="_values"):
        _scalar(p, [sample], entry, {sample.location: "digest"})
    # The statistic is the revision's own: one the platform registers no scorer for refuses by
    # name before any image is predicted, whatever a caller might have preferred.
    unregistered = entry.model_copy(update={"regression_criterion": "pearson_r"})
    with pytest.raises(AssessmentRefusedError, match="'pearson_r'"):
        _scalar(p, [sample], unregistered, {sample.location: "digest"})

    class Short:
        def predict_batch(self, sources, execution, *, tile_batch_size):
            return []

    p.predictor = Short()
    with pytest.raises(AssessmentRefusedError, match="0 predictions for 1 holdout samples"):
        _scalar(p, [sample], entry, {sample.location: "digest"})


def test_a_polygon_that_fails_to_rasterize_refuses_rather_than_training_an_empty_mask(
        tmp_path, monkeypatch):
    from PIL import ImageDraw

    from tests._producer_fixtures import dataset_over

    from tests._producer_fixtures import label_image, write_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    write_image(images_dir / "a.png", (IMG, IMG))
    label_image(images_dir / "a.png",
                [Annotation(subject=SUBJECT,
                            geometry=Polygon(rings=[[(2, 2), (20, 2), (20, 20)]]))],
                IMG, IMG)
    dataset = dataset_over("instance_seg", images_dir, subject=SUBJECT)

    def refuse(self, *args, **kwargs):
        raise ValueError("the ring would not draw")

    monkeypatch.setattr(ImageDraw.ImageDraw, "polygon", refuse)
    with pytest.raises(ValueError, match="would not draw"):
        dataset[0]


# -- a restored execution record runs exactly as recorded ---------------------------------


def test_a_restored_record_runs_as_recorded_and_a_changed_overlap_or_merge_refuses_by_name(
        tmp_path):
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, Stated, prepare_pass

    _images, _drawn, _sel, checkpoint_path = _trained(tmp_path, "exp-restore")
    checkpoint = load_registered_checkpoint(checkpoint_path, project=tmp_path)
    recorded = prepare_pass(checkpoint, Stated(tile=True, overlap=0.2)).execution
    assert recorded.tiled, recorded

    assert prepare_pass(checkpoint, Stated(overlap=0.2), restored=recorded).execution == recorded
    with pytest.raises(ExecutionRefusedError, match="overlap stated 0.5, recorded 0.2"):
        prepare_pass(checkpoint, Stated(overlap=0.5), restored=recorded)
    with pytest.raises(ExecutionRefusedError, match="postprocess stated 'nmm', recorded 'nms'"):
        prepare_pass(checkpoint, Stated(postprocess="nmm"), restored=recorded)
    # A blank merge is a name the vocabulary refuses, never the default an omitted one takes.
    with pytest.raises(ValueError, match="Unknown cross-tile merge ''.*nms"):
        prepare_pass(checkpoint, Stated(tile=True, overlap=0.2, postprocess=""))
