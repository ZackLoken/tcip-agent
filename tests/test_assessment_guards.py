"""The assessment's disjointness, criterion, revision and gate checks.

The chain's own producers build the chain's records here: a capture ingested, a selection drawn,
a tiny detector trained on it, the trait confirmed, the assessment run and the bucket published. A
reference is re-sided only through the admission's own sample producer.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, Polygon  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tests import _trait_fixtures as fx  # noqa: E402
from tests._chain_fixtures import (  # noqa: E402
    DATE,
    IMG,
    SUBJECT,
    assess,
    confirm_count_trait,
    draw_reference_selection,
    object_at,
    run_the_chain,
    synthetic_capture,
    train_on,
)
from tests._dense_op_fixtures import dense_records, good_cal_holdout  # noqa: E402
from tests._training_values import (  # noqa: E402
    COUNT_TEST,
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    MASK_CUT,
    RATIONALE,
    collection_pass,
)
from tests._verified_checkpoint_fixtures import SAMPLE_DETECTOR_PASS  # noqa: E402


def _trained(project: Path, experiment_id: str):
    """Ingest a capture, draw its selection and train a checkpoint on the selection's train side.

    Returns ``(images_dir, selection, selection_dir, checkpoint_path)``.
    """
    root = project / "ds"
    images_dir = synthetic_capture(root)
    selection_dir = project / "selection"
    selection = draw_reference_selection(project, root, selection_dir)
    return (
        images_dir,
        selection,
        selection_dir,
        train_on(selection_dir, project, experiment_id),
    )


def _add_frame(
    images_dir: Path, stem: str, *, like: str, same_pixels: bool
) -> None:
    """Add frame ``stem`` labeled with frame ``like``'s stored annotations.

    Its image is a byte copy of ``like``'s when ``same_pixels``, else a frame painted on its own
    background with a square at :func:`object_at` of ``like``'s index.
    """
    from tests._producer_fixtures import (
        image_label_key,
        label_image,
        painted_frame,
    )

    if same_pixels:
        shutil.copyfile(images_dir / f"{like}.png", images_dir / f"{stem}.png")
    else:
        x0, y0, size = object_at(int(like[1:]))
        painted_frame(
            IMG,
            IMG,
            (90, 60, 30),
            [((x0, y0, x0 + size, y0 + size), (250, 240, 200))],
        ).save(images_dir / f"{stem}.png")
    label_image(
        images_dir / f"{stem}.png",
        json_io.read_label_document(
            image_label_key(images_dir / f"{like}.png")
        ).annotations,
        IMG,
        IMG,
    )


def _resided(
    project: Path, images_dir: Path, drawn, extra: dict[str, str], out: Path
) -> Path:
    """Write a reference selection of its own at ``out`` and return ``out``.

    It holds the drawn selection's calibration and holdout sides with the added frames ``extra``
    (member to side) joined to them, sided and grouped by the admission's own sample producer
    under the draw's own policy.
    """
    from tcip_mcp.pipelines.data.label_queries import admit
    from tcip_mcp.pipelines.data.selection import (
        REFERENCE_SIDES,
        Selection,
        write_selection,
    )
    from tcip_mcp.pipelines.data.splits import recorded_group_key_fn

    admitted = admit(images_dir, scope=drawn.scope)
    sides = {
        s.member: s.side for s in drawn.samples if s.side in REFERENCE_SIDES
    } | extra
    samples = admitted.samples(
        sides, recorded_group_key_fn(drawn.group_by, date=admitted.date)
    )
    write_selection(
        out,
        Selection(
            samples=tuple(samples),
            scope=drawn.scope,
            seed=drawn.seed,
            group_by=drawn.group_by,
            ratios=drawn.ratios,
            stratify_foreground=drawn.stratify_foreground,
            rationale=drawn.rationale,
            dataset_fingerprint=drawn.dataset_fingerprint,
        ),
        project=project,
    )
    return out


def _a_train_member(selection) -> str:
    return next(s.member for s in selection.samples if s.side == "train")


# -- identity: content, never annotation geometry --------------------------------------


def test_a_distinct_image_holding_a_training_frames_boxes_is_a_legitimate_reference(
    tmp_path,
):
    """A distinct image holding a training frame's boxes is a disjoint reference.

    Two different images can carry identical boxes: the second is not the first, so a reference
    holding it is held out. Identity is the image's content, never the geometry drawn on it.
    """
    images_dir, drawn, _sel, checkpoint = _trained(tmp_path, "exp-twin")
    confirm_count_trait(tmp_path)
    like = _a_train_member(drawn)
    _add_frame(images_dir, "twin", like=like, same_pixels=False)
    reference = _resided(
        tmp_path,
        images_dir,
        drawn,
        {"twin": "holdout"},
        tmp_path / "reference",
    )

    record = assess(tmp_path, checkpoint, reference)

    assert "error" not in record, record
    for side in ("training", "selection"):
        assert record["disjointness"][side] == {
            "groups": [],
            "source_digests": [],
        }, record
    assert "reference_shares_training" not in record["failures"]


def test_a_training_image_copied_under_a_new_name_into_the_reference_fails_disjointness(
    tmp_path,
):
    """A training image copied under a new name into the reference fails disjointness.

    A reference holding a byte copy of a training frame under another name fails with
    ``reference_shares_training``.
    """
    images_dir, drawn, _sel, checkpoint = _trained(tmp_path, "exp-dup")
    confirm_count_trait(tmp_path)
    like = _a_train_member(drawn)
    _add_frame(images_dir, "dup", like=like, same_pixels=True)
    reference = _resided(
        tmp_path, images_dir, drawn, {"dup": "holdout"}, tmp_path / "reference"
    )

    record = assess(tmp_path, checkpoint, reference)

    assert "error" not in record, record
    assert record["passed"] is False
    assert "reference_shares_training" in record["failures"], record[
        "failures"
    ]
    assert record["disjointness"]["training"]["groups"] == [], record[
        "disjointness"
    ]
    assert len(record["disjointness"]["training"]["source_digests"]) == 1, (
        record["disjointness"]
    )


def test_a_reference_frame_in_a_training_group_fails_even_when_its_pixels_are_its_own(
    tmp_path,
):
    """A reference frame in a training group fails disjointness on the group alone.

    A holdout frame ``<member>_5_5`` of its own pixels, grouped by tile prefix with a training
    member, fails with ``reference_shares_training``: one shared group and no shared source
    digest.
    """
    images_dir, drawn, _sel, checkpoint = _trained(tmp_path, "exp-group")
    confirm_count_trait(tmp_path)
    like = _a_train_member(drawn)
    sibling = f"{like}_5_5"
    _add_frame(images_dir, sibling, like=like, same_pixels=False)
    reference = _resided(
        tmp_path,
        images_dir,
        drawn,
        {sibling: "holdout"},
        tmp_path / "reference",
    )

    record = assess(tmp_path, checkpoint, reference)

    assert "error" not in record, record
    assert "reference_shares_training" in record["failures"], record[
        "failures"
    ]
    assert record["disjointness"]["training"]["source_digests"] == [], record[
        "disjointness"
    ]
    assert len(record["disjointness"]["training"]["groups"]) == 1, record[
        "disjointness"
    ]


def test_a_mosaic_reference_band_is_held_out_only_inside_a_non_training_region():
    """A mosaic reference rect is held out only inside a non-training region.

    Against a within-image split's recorded regions, a reference rect is held out when it lies
    wholly inside one non-training region; one straddling the training boundary is not.
    """
    from tcip_mcp.pipelines.operating_point import spatial_disjointness
    from tcip_mcp.pipelines.schemas import SpatialManifest

    regions = {
        "train_region": [(0, 0, 500, 1000)],
        "val_region": [(500, 0, 650, 1000)],
        "calibration_region": [(650, 0, 800, 1000)],
        "holdout_region": [(800, 0, 1000, 1000)],
    }
    spatial = SpatialManifest.model_construct(**regions)

    assert (
        spatial_disjointness(
            spatial,
            [(550, 100, 600, 300), (680, 100, 780, 300), (850, 100, 950, 300)],
        )
        == []
    )
    assert spatial_disjointness(spatial, [(400, 100, 600, 300)]) == [
        "[400, 100, 600, 300]"
    ]
    # A manifest not stating its regions whole refuses where it is read back.
    with pytest.raises(ValueError, match="calibration_region"):
        SpatialManifest.model_validate(
            {k: v for k, v in regions.items() if k != "calibration_region"}
        )


# -- the count and classifier criteria ------------------------------------------------


def _count_entry(**fields):
    return fx.with_fields(fx.COUNT_SPEC, **fields)


def test_an_accurate_reference_passes_the_count_criterion():
    from tcip_mcp.pipelines.operating_point import count_criterion

    cal, hold = good_cal_holdout()
    _conf, _evidence, failures = count_criterion(
        cal,
        hold,
        _count_entry(count_error_tolerance=2.0),
        staged_conf_floor=0.01,
        staged_conf_floor_attribute_path="self",
        **COUNT_TEST,
    )

    assert failures == [], failures


def test_a_known_count_bias_fails_the_outcome_tolerance_with_no_dispersion_at_all():
    """A known count bias with no dispersion fails the outcome tolerance.

    Every held-out image over-counts by the same ten objects: the bias is certain, its spread
    zero, and the count is still wrong.
    """
    from tcip_mcp.pipelines.operating_point import count_criterion

    cal, _ = good_cal_holdout()
    hold = dense_records(
        id_prefix="h", miss_pattern=[0] * 20, fp_pattern=[10] * 20, score=0.9
    )
    _conf, evidence, failures = count_criterion(
        cal,
        hold,
        _count_entry(count_error_tolerance=20.0),
        staged_conf_floor=0.01,
        staged_conf_floor_attribute_path="self",
        **COUNT_TEST,
    )

    assert evidence["holdout_at_conf"]["count_bias_std_present"] == 0.0
    assert "count_bias_exceeds_tolerance" in failures, failures


def _items(image: str, calls: list[tuple[bool, bool]]) -> list[dict]:
    return [
        {"image_id": image, "is_true_positive": t, "is_pred_positive": p}
        for t, p in calls
    ]


def test_an_agreeing_classifier_passes_the_classifier_criterion():
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    calls = [(True, True), (True, True), (False, False), (False, False)]
    hold = _items("a", calls) + _items("b", calls)
    evidence, failures = classifier_criterion(
        hold, hold, _count_entry(), equivalence_z=EQUIVALENCE_Z
    )

    assert failures == [], (failures, evidence)


def test_compensating_classification_errors_cannot_validate_through_pooled_counts():
    """Compensating classification errors fail the classifier agreement floor.

    Each image's two positives called negative and two negatives called positive: its positive
    count is exactly right, and every call is wrong.
    """
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    calls = [(True, False), (True, False), (False, True), (False, True)]
    hold = _items("a", calls) + _items("b", calls)
    evidence, failures = classifier_criterion(
        hold, hold, _count_entry(), equivalence_z=EQUIVALENCE_Z
    )

    assert evidence["positive_count_bias"] == 0.0
    assert "classifier_agreement_below_floor" in failures, (failures, evidence)


# -- the trait revision -----------------------------------------------------------------


def test_an_unauthored_tolerance_refuses_the_revision_with_the_breeders_question(
    tmp_path,
):
    """A count revision leaving a tolerance unauthored is refused with the breeder's question.

    With ``count_bias_tolerance_frac`` unset, the proposal raises ``UnauthoredFieldError`` naming
    the field and its question, and the trait is not recorded.
    """
    from tcip_mcp.traits import QUESTIONS, UnauthoredFieldError, trait_names

    with pytest.raises(UnauthoredFieldError) as refused:
        confirm_count_trait(tmp_path, count_bias_tolerance_frac=None)

    assert "count_bias_tolerance_frac" in str(refused.value)
    assert QUESTIONS["count_bias_tolerance_frac"] in str(refused.value)
    assert fx.COUNT_TRAIT not in trait_names(tmp_path)


def test_an_iou_match_operationalization_is_proposed_only_with_its_jitter_and_margin(
    tmp_path,
):
    """An ``iou_match`` proposal is refused without its jitter and margin and recorded with them.

    The fields an IoU match compares by are part of the criterion a count operationalization
    rests on: a proposal selecting ``iou_match`` without them is refused asking the breeder, and
    one stating them is recorded.
    """
    from tcip_mcp.traits import QUESTIONS, UnauthoredFieldError, trait_names

    with pytest.raises(UnauthoredFieldError) as refused:
        confirm_count_trait(tmp_path, localization="iou_match")
    assert "iou_jitter_px" in str(refused.value) and "iou_margin" in str(
        refused.value
    )
    assert QUESTIONS["iou_jitter_px"] in str(refused.value)
    assert fx.COUNT_TRAIT not in trait_names(tmp_path)

    confirmed = confirm_count_trait(
        tmp_path,
        localization="iou_match",
        iou_jitter_px=fx.IOU_JITTER_PX,
        iou_margin=fx.IOU_MARGIN,
    )
    assert confirmed.confirmed
    assert (confirmed.entry.iou_jitter_px, confirmed.entry.iou_margin) == (
        fx.IOU_JITTER_PX,
        fx.IOU_MARGIN,
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("iou_jitter_px", -3.0),
        ("iou_jitter_px", "nan"),
        ("iou_margin", -0.1),
        ("iou_margin", "inf"),
    ],
)
def test_an_iou_field_that_is_no_finite_nonnegative_value_refuses(
    field, value
):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=field):
        fx.with_fields(fx.COUNT_SPEC, **{field: float(value)})


def test_a_delivery_under_another_revision_than_the_assessments_refuses_naming_both(
    tmp_path,
):
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    chain = run_the_chain(tmp_path, experiment_id="exp-revision")
    confirm_count_trait(tmp_path, count_error_tolerance=3.0)
    out_csv = tmp_path / "counts.csv"

    delivered = deliver_per_image_counts(
        tmp_path,
        str(chain.root),
        chain.bucket,
        str(out_csv),
        trait=fx.COUNT_TRAIT,
    )

    assert "error" in delivered, delivered
    assert (
        "revision 1" in delivered["error"]
        and "revision 2" in delivered["error"]
    ), delivered
    assert not out_csv.exists()


# -- the one gate over the producers ------------------------------------------------------


def test_one_producer_clears_the_gate_and_a_series_of_two_refuses(tmp_path):
    """The gate clears one producer's bucket and refuses a series of two, naming each.

    Two checkpoints publish over one capture: the assessed one's bucket alone validates, and the
    two together refuse as more than one checkpoint, naming both buckets.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.delivery import DeliveryRefusedError, Result, gate
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.tools.inference_tools import run_inference

    nothing = Result((), (), population=())
    chain = run_the_chain(tmp_path, experiment_id="exp-series-a")
    other = train_on(chain.selection_dir, tmp_path, "exp-series-b")
    second = f"other/{DATE}"
    published = run_inference(
        tmp_path,
        checkpoint_path=other,
        images_dir=str(chain.images_dir),
        bucket=second,
        stated=Stated(**SAMPLE_DETECTOR_PASS),
    )
    assert "error" not in published, published
    revision = latest_confirmed(fx.COUNT_TRAIT, tmp_path)
    first = chain.read()

    assert gate(
        tmp_path,
        [first],
        delivery_kind="per_image_count",
        revision=revision,
        result=nothing,
    ).validated
    with pytest.raises(
        DeliveryRefusedError, match="more than one checkpoint"
    ) as refused:
        gate(
            tmp_path,
            [first, read_bucket(chain.root, second)],
            delivery_kind="per_image_count",
            revision=revision,
            result=nothing,
        )
    assert chain.bucket in str(refused.value) and second in str(refused.value)


def test_an_assessment_whose_retained_reference_is_gone_answers_for_nothing(
    tmp_path,
):
    """The gate refuses a bucket whose assessment's retained reference is gone.

    An assessment whose retained reference copies are removed reads as a changed reference, so
    the gate refuses the bucket it validated.
    """
    from tcip_mcp.assessment import REFERENCE_DIR, assessment_dir
    from tcip_mcp.delivery import DeliveryRefusedError, Result, gate
    from tcip_mcp.operationalization import latest_confirmed

    chain = run_the_chain(tmp_path, experiment_id="exp-forged")
    shutil.rmtree(
        assessment_dir(tmp_path, chain.assessment["assessment_id"])
        / REFERENCE_DIR
    )

    with pytest.raises(DeliveryRefusedError, match="changed since"):
        gate(
            tmp_path,
            [chain.read()],
            delivery_kind="per_image_count",
            revision=latest_confirmed(fx.COUNT_TRAIT, tmp_path),
            result=Result((), (), population=()),
        )


def test_a_reference_source_edited_after_the_assessment_answers_for_nothing(
    tmp_path,
):
    """The gate refuses a bucket whose reference source was edited after its assessment.

    The pixels an assessment measured are part of its reference: a source image edited after it
    refuses the bucket it validated, naming the image, which cleared before the edit.
    """
    from PIL import Image

    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.delivery import DeliveryRefusedError, Result, gate
    from tcip_mcp.operationalization import latest_confirmed

    chain = run_the_chain(tmp_path, experiment_id="exp-edited-source")
    nothing = Result((), (), population=())
    revision = latest_confirmed(fx.COUNT_TRAIT, tmp_path)
    bucket = chain.read()
    assert gate(
        tmp_path,
        [bucket],
        delivery_kind="per_image_count",
        revision=revision,
        result=nothing,
    ).validated
    assessment = read_assessment(tmp_path, chain.assessment["assessment_id"])
    source = assessment.reference.samples[0].source
    with Image.open(source) as frame:
        edited = frame.copy()
    edited.putpixel((0, 0), (255, 0, 0))
    edited.save(source)

    with pytest.raises(DeliveryRefusedError, match="changed since") as refused:
        gate(
            tmp_path,
            [bucket],
            delivery_kind="per_image_count",
            revision=revision,
            result=nothing,
        )
    assert source in str(refused.value)


# -- a missing fact refuses by name -----------------------------------------------------


def test_an_evaluation_record_missing_its_ground_truth_field_refuses_by_name():
    from tcip_mcp.pipelines.training.evaluation import gt_objects

    with pytest.raises(KeyError, match="gt"):
        gt_objects({"width": IMG, "height": IMG, "dt": []})


def test_a_scalar_prediction_carrying_no_output_refuses_the_assessment_by_name(
    tmp_path,
):
    from types import SimpleNamespace

    from PIL import Image

    from tcip_mcp.assessment import AssessmentRefusedError, _scalar
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample
    from tcip_mcp.pipelines.execution import Pass, execution_record

    class Silent:
        def predict_batch(self, sources, execution, *, tile_batch_size):
            return [{"image": s, "regression_values": []} for s in sources]

    checkpoint = SimpleNamespace(
        task="regression", payload={}, path="regressor.pt"
    )
    p = Pass(
        checkpoint=checkpoint,
        predictor=Silent(),
        scope=ClassScope(),
        execution=execution_record(checkpoint, Stated(), None, None),
        tile_batch_size=1,
    )
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    (tmp_path / "t.csv").write_text("image,value\na,1.0\n", encoding="utf-8")
    sample = Sample(
        member="a",
        source=str(tmp_path / "a.png"),
        ground_truth=str(tmp_path / "t.csv"),
        group="a",
        side="holdout",
        row_key="a",
    )
    entry = fx.COUNT_SPEC.model_copy(
        update={"regression_criterion": "r_squared"}
    )

    with pytest.raises(AssessmentRefusedError, match="_values"):
        _scalar(p, [sample], entry, {sample.location: "digest"})

    class Short:
        def predict_batch(self, sources, execution, *, tile_batch_size):
            return []

    p.predictor = Short()
    with pytest.raises(
        AssessmentRefusedError, match="0 predictions for 1 holdout samples"
    ):
        _scalar(p, [sample], entry, {sample.location: "digest"})


def test_a_polygon_that_fails_to_rasterize_refuses_rather_than_training_an_empty_mask(
    tmp_path, monkeypatch
):
    from PIL import ImageDraw

    from tests._producer_fixtures import dataset_over, label_image, write_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    write_image(images_dir / "a.png", (IMG, IMG))
    label_image(
        images_dir / "a.png",
        [
            Annotation(
                subject=SUBJECT,
                geometry=Polygon(rings=[[(2, 2), (20, 2), (20, 20)]]),
            )
        ],
        IMG,
        IMG,
    )
    dataset = dataset_over("instance_seg", images_dir, subject=SUBJECT)

    def refuse(self, *args, **kwargs):
        raise ValueError("the ring would not draw")

    monkeypatch.setattr(ImageDraw.ImageDraw, "polygon", refuse)
    with pytest.raises(ValueError, match="would not draw"):
        dataset[0]


def test_an_instance_segmentation_assessment_reads_its_reference_through_polygons(
    tmp_path,
):
    """An instance segmentation count fit reads polygons and refuses a box-only reference.

    An instance-segmentation checkpoint's count fit reads its reference through its own task's
    loader: a reference holding the subject as polygons is fitted, and one holding it only as
    boxes refuses by name.
    """
    from tcip_mcp import assessment
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.selection import source_digests
    from tcip_mcp.pipelines.execution import prepare
    from tests._chain_fixtures import (
        BESPOKE_INSTANCE_SEG,
        BESPOKE_MODELS,
        FRAME_STATISTICS,
    )
    from tests._producer_fixtures import samples_over
    from tests._verified_checkpoint_fixtures import (
        detection_images,
        registered_checkpoint,
    )

    checkpoint = load_registered_checkpoint(
        registered_checkpoint(
            tmp_path,
            model_source={
                "builder": BESPOKE_INSTANCE_SEG,
                "source_files": [BESPOKE_MODELS],
                "task": "instance_seg",
                "normalization": FRAME_STATISTICS,
            },
        ),
        project=tmp_path,
    )

    def fit(polygons: bool):
        reference = detection_images(
            tmp_path.parent / f"reference-{polygons}",
            {"subject": SUBJECT},
            n=4,
            polygons=polygons,
        )
        samples = samples_over(reference["images_dir"], subject=SUBJECT)
        return assessment._detection(
            prepare(
                checkpoint,
                collection_pass(
                    tile=False,
                    masks=True,
                    mask_cut=MASK_CUT,
                ),
                device="cpu",
            ),
            samples[:2],
            samples[2:],
            fx.COUNT_SPEC,
            "per_image_count",
            source_digests(samples),
            **COUNT_TEST,
        )

    _pass, measured, _failures = fit(polygons=True)
    assert "count" in measured
    with pytest.raises(ValueError, match="instance_seg loader does not read"):
        fit(polygons=False)


def test_an_assessment_records_the_test_settings_and_the_reason_it_states(
    tmp_path,
):
    """An assessment records the test settings and the reason it states, no door defaulting them.

    The equivalence multiplier and the dispersion quantile a count assessment states are the
    ones its record carries beside its reason, and neither door holds a value for any of them.
    """
    import inspect

    from tcip_mcp import assessment
    from tcip_mcp.tools import calibration_tools

    chain = run_the_chain(tmp_path, experiment_id="exp-stated-test")
    recorded = assessment.read_assessment(
        tmp_path, chain.assessment["assessment_id"]
    )

    count = recorded.criterion["count"]
    assert count["equivalence_z"] == EQUIVALENCE_Z
    assert (
        count["calibration_curve"]["dispersion_quantile"]
        == DISPERSION_QUANTILE
    )
    assert recorded.rationale == RATIONALE
    for door in (assessment.assess, calibration_tools.assess_checkpoint):
        parameters = inspect.signature(door).parameters
        for name in ("equivalence_z", "dispersion_quantile", "rationale"):
            assert parameters[name].default is inspect.Parameter.empty, (
                door,
                name,
            )


def test_a_detector_assessment_collects_at_the_conf_it_states_and_records_it_as_the_floor(
    tmp_path, monkeypatch
):
    """A detector assessment collects at the conf it states and records it as the floor.

    Every prediction a detector assessment collects runs at the conf its statement carries, that
    conf is the floor its evidence records, and a statement carrying no conf refuses naming the
    field.
    """
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.tools.calibration_tools import assess_checkpoint

    _images, _drawn, selection_dir, checkpoint = _trained(
        tmp_path, "exp-floor"
    )
    confirm_count_trait(tmp_path)
    stated_conf = 0.05
    collected: set[float] = set()
    real = GenericPredictor.predict_batch

    def recording(self, sources, execution, **kwargs):
        collected.add(execution.conf)
        return real(self, sources, execution, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_batch", recording)

    record = assess(tmp_path, checkpoint, selection_dir, conf=stated_conf)

    assert "error" not in record, record
    assert collected == {stated_conf}
    assert record["criterion"]["count"]["staged_conf_floor"] == stated_conf
    assert record["criterion"]["count"]["observed_min_score"] >= stated_conf
    unstated = assess_checkpoint(
        tmp_path,
        checkpoint_path=checkpoint,
        trait=fx.COUNT_TRAIT,
        delivery_kind="per_image_count",
        selection_dir=str(selection_dir),
        equivalence_z=EQUIVALENCE_Z,
        dispersion_quantile=DISPERSION_QUANTILE,
        rationale=RATIONALE,
        stated=Stated(),
    )
    assert "states no conf" in unstated["error"], unstated


@pytest.mark.parametrize(
    ("tool", "bands"),
    [
        ("assess_checkpoint", {"selection_dir": "s"}),
        ("assess_reserved_regions", {"k_cal": 3, "k_test": 3}),
    ],
)
def test_the_assess_tools_take_a_dispersion_quantile_within_zero_and_one(
    tool, bands
):
    """The assess tools refuse a dispersion quantile of -0.1 or 1.5 by name, and not 0.0 or 1.0.

    Called through the MCP server, the answer at -0.1 and 1.5 names ``dispersion_quantile``, and
    the answer at 0.0 and 1.0 does not.
    """
    from tests._server_calls import called_through_the_server

    def called(quantile: float) -> str:
        return called_through_the_server(
            tool,
            {
                "checkpoint_path": "absent.pt",
                "trait": fx.COUNT_TRAIT,
                "delivery_kind": "per_image_count",
                **bands,
                "equivalence_z": EQUIVALENCE_Z,
                "dispersion_quantile": quantile,
                "rationale": RATIONALE,
                "stated": {},
            },
        )

    for outside in (-0.1, 1.5):
        assert "dispersion_quantile" in called(outside), outside
    for end in (0.0, 1.0):
        assert "dispersion_quantile" not in called(end), end


# -- a restored execution record runs exactly as recorded ---------------------------------


def test_a_restored_record_runs_as_recorded_and_a_changed_overlap_or_merge_refuses_by_name(
    tmp_path,
):
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, prepare

    _images, _drawn, _sel, checkpoint_path = _trained(tmp_path, "exp-restore")
    checkpoint = load_registered_checkpoint(checkpoint_path, project=tmp_path)

    def tiled(**merge) -> Stated:
        return Stated(
            tile=True, overlap=0.2, **{**SAMPLE_DETECTOR_PASS, **merge}
        )

    recorded = prepare(checkpoint, tiled()).runnable().execution
    assert recorded.tiled, recorded
    assert recorded.postprocess == "nms"
    stated_merge = (
        prepare(checkpoint, tiled(postprocess="nmm")).runnable().execution
    )
    assert stated_merge.postprocess == "nmm"

    assert (
        prepare(
            checkpoint,
            Stated(overlap=0.2, rationale="a restored pass's own reason"),
            restored=recorded,
        )
        .runnable()
        .execution
        == recorded
    )
    with pytest.raises(
        ExecutionRefusedError, match="overlap stated 0.5, recorded 0.2"
    ):
        prepare(
            checkpoint,
            Stated(overlap=0.5, rationale=RATIONALE),
            restored=recorded,
        ).runnable()
    with pytest.raises(
        ExecutionRefusedError, match="postprocess stated 'nmm', recorded 'nms'"
    ):
        prepare(
            checkpoint,
            Stated(postprocess="nmm", rationale=RATIONALE),
            restored=recorded,
        ).runnable()
    # A blank merge is a name the vocabulary refuses.
    with pytest.raises(ValueError, match="Unknown cross-tile merge ''.*nms"):
        prepare(checkpoint, tiled(postprocess="")).runnable()
