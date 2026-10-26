"""Detection measurement integrity, from val-loss over negatives to a pass's derived tile geometry.

The cases cover val-loss over negatives, standard+operating mAP, tiled evaluation, and the tile
geometry a pass derives from its checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._producer_fixtures import checkpoint_admission

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from PIL import Image
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset

    from tcip_annotation.state import Annotation
    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample
    from tcip_mcp.pipelines.execution import Pass
    from tcip_mcp.pipelines.model_build import SourceLayout
    from tcip_mcp.pipelines.schemas import (
        ModelSourceSchema,
        TilingSpec,
        TrainGeometry,
    )
    from tcip_mcp.traits import TraitEntry

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tcip_mcp.pipelines.model_build import CONFIG_KEY  # noqa: E402
from tcip_mcp.pipelines.schemas import ScoreWeights  # noqa: E402
from tcip_mcp.pipelines.training.evaluation import evaluate  # noqa: E402
from tests._predictor_fixtures import StubPredictor, install  # noqa: E402
from tests._producer_fixtures import (  # noqa: E402
    label_image,
    seed_bud_images,
    write_image,
)
from tests._training_values import (
    IOU_THRESHOLD,
    SCORE_WEIGHTS,
    VALIDATION_CONF,
    drawn_split,
    tiling_block,
)  # noqa: E402
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    SAMPLE_DETECTOR_PASS,
    SCOPED_DATA,
    project_checkpoint,
    verified_checkpoint,
)

WEIGHTS = ScoreWeights.model_validate(SCORE_WEIGHTS)
_DIMS = {"in_chans": 3, "num_classes": 1}
"""What a one-subject detector over three-band sources is built at."""
DETECTOR_PASS = Stated(**SAMPLE_DETECTOR_PASS)
"""The execution values an evaluation of a detector here states."""


# ── detection val-loss includes all-negative images ──────────────────────


class _StubDetector:
    """A minimal detector recording each loss pass's image count and box count."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def train(self) -> None:  # noqa: D401
        pass

    def eval(self) -> None:  # noqa: D401
        pass

    def modules(self) -> list[nn.Module]:
        return []

    def __call__(
        self, images: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        self.calls.append(
            (len(images), sum(int(t["boxes"].numel()) for t in targets))
        )
        return {"loss": torch.tensor(2.5)}


class _StubModel:
    """Composed-model stand-in exposing ``.detector`` and both operating-point knobs.

    Its forward returns one empty prediction per image.
    """

    def __init__(self, detector: _StubDetector) -> None:
        self.detector = detector
        self.score_thresh = 0.0
        self.detections_per_img = 100

    def eval(self) -> None:  # noqa: D401
        pass

    def __call__(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        return [
            {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long),
            }
            for _ in images
        ]


def _det_batch(
    box_counts: list[int],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    """Return one detection batch of image tensors and targets, one per count in ``box_counts``."""
    images, targets = [], []
    for n in box_counts:
        images.append(torch.zeros(3, 32, 32))
        boxes = torch.tensor(
            [[1.0, 1.0, 5.0, 5.0]] * n, dtype=torch.float32
        ).reshape(-1, 4)
        targets.append(
            {
                "boxes": boxes,
                "labels": torch.ones((n,), dtype=torch.long),
                "iscrowd": torch.zeros((n,), dtype=torch.long),
            }
        )
    return images, targets


def test_val_loss_forwards_all_negative_images() -> None:
    stub = _StubDetector()
    model = _StubModel(stub)
    loader = [
        _det_batch([1, 0]),
        _det_batch([0]),
    ]  # mixed batch, then all-negative batch
    evaluate(
        model,
        loader,
        torch.device("cpu"),
        "detection",
        dims=_DIMS,
        conf_threshold=VALIDATION_CONF,
        iou_threshold=IOU_THRESHOLD,
        score_weights=WEIGHTS,
        trait=None,
        density=1.0,
        mask_cut=None,
    )
    # Both batches forwarded through the detector (full batch incl. negatives), not just foreground.
    assert stub.calls == [(2, 4), (1, 0)]


def test_all_negative_only_loader_is_not_skipped() -> None:
    stub = _StubDetector()
    model = _StubModel(stub)
    loader = [_det_batch([0, 0])]  # nothing but negatives
    result = evaluate(
        model,
        loader,
        torch.device("cpu"),
        "detection",
        dims=_DIMS,
        conf_threshold=VALIDATION_CONF,
        iou_threshold=IOU_THRESHOLD,
        score_weights=WEIGHTS,
        trait=None,
        density=1.0,
        mask_cut=None,
    )
    assert stub.calls == [(2, 0)]  # forwarded, not skipped
    assert result["loss"] == pytest.approx(
        2.5
    )  # finite, non-zero: negatives contribute loss


# ── tiled evaluation regimes ──────────────────────────────────────────────


def _capture_run_test_evaluation(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch run_test_evaluation to record the built dataset and tiling, returning a stub result."""
    import tcip_mcp.pipelines.training.eval_runners as runners

    captured: dict = {}

    def _fake(
        pass_: Pass,
        loader: DataLoader,
        **kw: float | TilingSpec | TraitEntry | None,
    ) -> dict:
        captured["ds"] = loader.dataset
        captured["tiling"] = kw.get("tiling")
        return {"eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)
    return captured


def test_run_id_reuses_training_tiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import finished_run

    images_dir = seed_bud_images(image_dir(tmp_path, UNDATED_BUCKET))
    data = {
        **SCOPED_DATA,
        "tiling": tiling_block(tile_size=64, sliver_frac=0.5),
    }
    run_dir = finished_run(
        tmp_path, experiment_id="det-measure-tiled", data=data
    )

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(
        tmp_path,
        run_dir.name,
        str(images_dir),
        stated=DETECTOR_PASS,
        iou_threshold=IOU_THRESHOLD,
    )
    assert isinstance(captured["ds"], TiledDetectionDataset)
    assert (
        captured["ds"].num_samples > 3
    )  # more tiles than the 3 source images
    from tcip_mcp.experiments import run_resolution

    assert (
        captured["tiling"]
        == run_resolution(run_dir.name, project=tmp_path).data.tiling
    )
    assert captured["tiling"].tile_size == 64


def test_evaluating_a_run_leaves_its_directory_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluating a completed run by its id leaves its directory byte-identical.

    The run's files before and after are the same names holding the same bytes, the project's
    audit log gains no row, and the result carries no error.
    """
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._audit_fixtures import audit_rows
    from tests._verified_checkpoint_fixtures import finished_run

    images_dir = seed_bud_images(image_dir(tmp_path, UNDATED_BUCKET))
    run_dir = finished_run(tmp_path, experiment_id="det-evaluated")

    def snapshot() -> dict:
        return {
            str(p.relative_to(run_dir)): p.read_bytes()
            for p in sorted(run_dir.rglob("*"))
            if p.is_file()
        }

    before = snapshot()
    audit_before = audit_rows(tmp_path)
    result = evaluate_model(
        tmp_path,
        run_dir.name,
        str(images_dir),
        stated=DETECTOR_PASS,
        iou_threshold=IOU_THRESHOLD,
    )

    assert "error" not in result, result
    assert snapshot() == before
    assert audit_rows(tmp_path) == audit_before


def test_explicit_checkpoint_stays_untiled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.pipelines.data.datasets import (
        DetectionDataset,
        TiledDetectionDataset,
    )
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = seed_bud_images(image_dir(tmp_path, UNDATED_BUCKET))
    ckpt = registered_checkpoint(tmp_path)

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(
        tmp_path,
        ckpt,
        str(images_dir),
        stated=DETECTOR_PASS,
        iou_threshold=IOU_THRESHOLD,
    )
    assert isinstance(captured["ds"], DetectionDataset)
    assert not isinstance(captured["ds"], TiledDetectionDataset)
    assert captured["tiling"] is None


def test_evaluate_model_reads_its_loader_at_the_checkpoints_own_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``evaluate_model`` reads its loader at the checkpoint's own width.

    Over three-band sources, a checkpoint recording one band builds a one-band loader, and a
    checkpoint recording no band count is refused as having none.
    """
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        registered_checkpoint,
    )

    images_dir = seed_bud_images(
        image_dir(tmp_path, UNDATED_BUCKET)
    )  # three-band sources
    scope = {"subject": "bud"}
    ckpt = registered_checkpoint(
        tmp_path,
        model_source=BUILT_DETECTOR,
        data={"num_channels": 1, "scope": scope},
    )

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(
        tmp_path,
        ckpt,
        str(images_dir),
        stated=DETECTOR_PASS,
        iou_threshold=IOU_THRESHOLD,
    )
    assert captured["ds"].expected_channels == 1

    from tcip_mcp.tools.model_tools import register_model

    # A checkpoint whose own data section records no band count, written past the producer.
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    del payload[CONFIG_KEY]["data"]["num_channels"]
    unstated = tmp_path / "unstated.pt"
    torch.save(payload, str(unstated))
    assert "error" not in register_model(
        tmp_path,
        name="unstated-width",
        checkpoint_path=str(unstated),
        actor=None,
    )

    r = evaluate_model(
        tmp_path,
        str(unstated),
        str(images_dir),
        stated=DETECTOR_PASS,
        iou_threshold=IOU_THRESHOLD,
    )

    assert "no band count" in r["error"], r


def _detections_as_ground_truth(
    checkpoint: VerifiedCheckpoint,
    images_dir: Path,
    *,
    subject: str,
    limit: int = 20,
) -> None:
    """Write up to ``limit`` of each png image's own detections back as its ground truth.

    The registered ``checkpoint``'s model, built from its own spec, is read with its score floor
    removed, and only boxes wider and taller than 1 px are written.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor
    from tcip_mcp.pipelines.model_build import (
        STATE_DICT_KEY,
        build_from_model_source,
        recorded_model_dims,
    )
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    model = build_from_model_source(
        checkpoint.spec.model_source,
        checkpoint.layout,
        recorded_model_dims(checkpoint.spec),
        checkpoint.geometry,
    )
    model.load_state_dict(checkpoint.payload[STATE_DICT_KEY])
    model.eval()
    set_detector_operating_point(
        model, score_thresh=0.0, detections_per_img=limit
    )
    for image in sorted(Path(images_dir).glob("*.png")):
        tensor = pil_to_tensor(load_image(image, 3))
        with torch.no_grad():
            boxes = model([tensor])[0]["boxes"].tolist()
        w, h = int(tensor.shape[-1]), int(tensor.shape[-2])
        label_image(
            image,
            [
                Annotation(
                    subject=subject, geometry=BBox(*(float(v) for v in box))
                )
                for box in boxes
                if box[2] - box[0] > 1 and box[3] - box[1] > 1
            ],
            w,
            h,
        )


def test_the_scored_model_is_the_one_the_door_already_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An evaluation builds its model once and reports the execution that model ran under.

    A separately built copy of the checkpoint, counted at the reported record's conf under the
    cap it gives each frame, answers with the measured numbers over the measured loader; counted
    at conf 0.6, the copy answers a different recall.
    """
    from typing import cast

    from pydantic import TypeAdapter

    import tcip_mcp.pipelines.inference.generic_predictor as generic_predictor
    import tcip_mcp.pipelines.model_build as model_build
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Execution
    from tcip_mcp.pipelines.model_build import STATE_DICT_KEY
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    for i in range(2):
        write_image(images_dir / f"img{i}.png", (128, 128), (120, 120, 120))
    # This seed's weights score just above 0.5, so a count at 0.6 keeps none of its detections.
    ckpt = registered_checkpoint(tmp_path, seed=0)
    verified = load_registered_checkpoint(ckpt, project=tmp_path)
    _detections_as_ground_truth(verified, images_dir, subject="bud")

    build = model_build.build_from_model_source
    builds = []

    def _spy(
        source: ModelSourceSchema,
        layout: SourceLayout,
        dims: Mapping[str, object],
        geometry: TrainGeometry,
    ) -> nn.Module:
        builds.append(source)
        return build(source, layout, dims, geometry)

    monkeypatch.setattr(model_build, "build_from_model_source", _spy)
    monkeypatch.setattr(generic_predictor, "build_from_model_source", _spy)

    recorded: dict = {}
    run_test_evaluation = runners.run_test_evaluation

    def _record(
        pass_: Pass,
        loader: DataLoader,
        **kw: float | TilingSpec | TraitEntry | None,
    ) -> dict:
        recorded.update(
            loader=loader,
            device=pass_.predictor.device,
            task=pass_.checkpoint.task,
            kw=kw,
        )
        return run_test_evaluation(pass_, loader, **kw)

    monkeypatch.setattr(runners, "run_test_evaluation", _record)

    # Both routes run from one seed: the loss pass samples proposals, so two unseeded passes over
    # the same weights differ in that one metric by more than float noise.
    torch.manual_seed(777)
    measured = evaluate_model(
        tmp_path,
        ckpt,
        str(images_dir),
        stated=DETECTOR_PASS,
        iou_threshold=IOU_THRESHOLD,
    )
    assert "error" not in measured, measured
    assert len(builds) == 1

    torch.manual_seed(777)
    dims = model_build.recorded_model_dims(verified.spec)
    independent_model = build(
        verified.spec.model_source, verified.layout, dims, verified.geometry
    )
    independent_model.load_state_dict(verified.payload[STATE_DICT_KEY])
    independent_model.to(recorded["device"])
    kw = recorded["kw"]
    execution = TypeAdapter(Execution).validate_python(measured["execution"])

    def _independent(conf: float) -> dict:
        """Return the evaluation of the independent copy counted at ``conf``.

        Each image is capped at the record's density, its objective weighted by the checkpoint's
        recorded weights.
        """
        return evaluate(
            independent_model,
            recorded["loader"],
            recorded["device"],
            recorded["task"],
            dims=dims,
            conf_threshold=conf,
            iou_threshold=kw["iou_threshold"],
            score_weights=verified.spec.evaluation.score_weights,
            trait=kw["trait"],
            density=execution.density,
            mask_cut=execution.mask_cut,
        )

    independent = _independent(cast(float, execution.conf))
    assert independent
    for key, value in independent.items():
        # Two forward passes over the same weights are equal to float noise, not bit for bit.
        assert measured[key] == (
            pytest.approx(value, rel=1e-4)
            if isinstance(value, (int, float))
            else value
        ), key

    # What a count at a conf above every score reports instead, so the equality above is evidence
    # about this fixture rather than a comparison nothing could separate.
    assert _independent(0.6)["recall"] != measured["recall"]


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_the_three_evaluation_loaders_over_one_truth_and_one_predictor_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three evaluation doors build their loaders alike over one truth and one predictor.

    The assessment's reference, the full-frame evaluation and the tile-level diagnostic each
    build their loader over one ground truth for one predictor at the same width and the same
    untiled block: every factory build each door makes is recorded and the three doors' records
    are compared with each other. The predictor reads one band of three-band sources, and every
    door's builds carry one band.
    """
    import tcip_mcp.pipelines.data.datasets as datasets
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.schemas import run_tiling
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._chain_fixtures import assess, confirm_count_trait
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        registered_checkpoint,
    )

    root = tmp_path / "ds"
    images_dir = image_dir(root, UNDATED_BUCKET)
    for i, size in enumerate(range(48, 112, 8)):
        write_image(images_dir / f"img{i}.png", (size, size), (100, 100, 100))
        label_image(
            images_dir / f"img{i}.png",
            [Annotation(subject="bud", geometry=BBox(8, 8, 24, 24))],
            size,
            size,
        )
    drawn = draw_splits(
        tmp_path,
        str(root),
        output_path=str(tmp_path / "selection"),
        subject="bud",
        **drawn_split(
            seed=2,
            val_ratio=0.25,
            calibration_ratio=0.25,
            holdout_ratio=0.25,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in drawn, drawn
    confirm_count_trait(tmp_path)
    ckpt = registered_checkpoint(
        tmp_path,
        model_source=BUILT_DETECTOR,
        data={"num_channels": 1, "scope": {"subject": "bud"}},
    )

    built: list[tuple] = []
    real = datasets.build_dataset

    def recording(
        task: str,
        **kwargs: Sequence[Sample]
        | Mapping[str, int]
        | ClassScope
        | Callable[[Image.Image, dict], tuple[Tensor, dict]]
        | TilingSpec
        | None,
    ) -> Dataset:
        built.append(
            (
                task,
                dict(kwargs["sizes"]),
                run_tiling(task, kwargs.get("tiling")),
            )
        )
        return real(task, **kwargs)

    monkeypatch.setattr(datasets, "build_dataset", recording)
    by_door: dict[str, set] = {}
    for door, act in (
        (
            "assessment",
            lambda: assess(
                tmp_path,
                ckpt,
                tmp_path / "selection",
                device="cpu",
                tile=False,
            ),
        ),
        (
            "full_frame",
            lambda: evaluate_model(
                tmp_path,
                ckpt,
                str(images_dir),
                stated=DETECTOR_PASS,
                use_tiled_inference=True,
                iou_threshold=IOU_THRESHOLD,
            ),
        ),
        (
            "diagnostic",
            lambda: evaluate_model(
                tmp_path,
                ckpt,
                str(images_dir),
                stated=DETECTOR_PASS,
                iou_threshold=IOU_THRESHOLD,
            ),
        ),
    ):
        built.clear()
        act()
        by_door[door] = set(map(repr, built))
        assert by_door[door], door

    assert (
        by_door["assessment"] == by_door["full_frame"] == by_door["diagnostic"]
    ), by_door
    assert by_door["assessment"] == {
        repr(("detection", {"num_channels": 1}, None))
    }


def test_explicit_tiling_override_on_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset
    from tcip_mcp.pipelines.schemas import TilingSpec
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = seed_bud_images(image_dir(tmp_path, UNDATED_BUCKET))
    ckpt = registered_checkpoint(tmp_path)

    captured = _capture_run_test_evaluation(monkeypatch)
    evaluate_model(
        tmp_path,
        ckpt,
        str(images_dir),
        stated=DETECTOR_PASS,
        tiling=TilingSpec.model_validate(
            tiling_block(tile_size=64, sliver_frac=0.5)
        ),
        iou_threshold=IOU_THRESHOLD,
    )
    assert isinstance(captured["ds"], TiledDetectionDataset)


def _sliced_stub(
    boxes: Sequence[Sequence[float]], **training: int | float | None
) -> StubPredictor:
    """Return a three-band stub detector answering ``boxes`` at 0.9 on a 128px frame.

    ``training`` sets the stub's recorded training geometry.
    """
    return StubPredictor(
        task="detection",
        in_chans=3,
        width=128,
        height=128,
        boxes=boxes,
        scores=(0.9,) * len(boxes),
        **training,
    )


def _full_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub: StubPredictor,
    annotations: list[Annotation],
    *,
    checkpoint: VerifiedCheckpoint | None = None,
    **stated: int | float,
) -> dict:
    """Run ``run_full_frame_evaluation`` of one 128px frame through ``stub`` and return it.

    The frame is labeled with ``annotations``, and the pass is stated at the sample detector
    values, ``stated`` overriding any of them.
    """
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation,
    )
    from tests._producer_fixtures import blank_image
    from tests._verified_checkpoint_fixtures import SAMPLE_DETECTOR_PASS

    image = blank_image(tmp_path, "a.png", (128, 128))
    label_image(image, annotations, 128, 128, keep_empty=True)
    images_dir = image.parent
    install(monkeypatch, stub)
    checkpoint = checkpoint or verified_checkpoint(tmp_path)
    return run_full_frame_evaluation(
        checkpoint,
        checkpoint_admission(checkpoint, images_dir),
        stated=Stated(**{**SAMPLE_DETECTOR_PASS, **stated}),
        iou_threshold=IOU_THRESHOLD,
        trait=None,
    )


def test_full_frame_counts_straddling_object_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_annotation.state import Annotation, BBox

    # An object straddling the x=64 tile seam; this stub carries no persisted geometry, so the
    # caller states it.
    r = _full_frame(
        tmp_path,
        monkeypatch,
        _sliced_stub([[54, 54, 74, 74]]),
        [Annotation(subject="bud", geometry=BBox(54, 54, 74, 74))],
        tile_size=64,
        overlap=0.2,
    )

    assert r["eval_regime"] == "full-frame-tiled-inference"
    # counted once against un-fragmented full-frame GT (tile-level would split/duplicate it)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0


def test_full_frame_scores_a_detection_in_a_crowd_region_as_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full-frame gate scores a detection inside a crowd region as neither TP nor FP.

    With one object and one crowd region labeled and a detection on each, the gate counts one
    true positive, no false positive and no miss.
    """
    from tcip_annotation.state import Annotation, BBox

    r = _full_frame(
        tmp_path,
        monkeypatch,
        _sliced_stub([[10, 10, 30, 30], [60, 60, 120, 120]]),
        [
            Annotation(subject="bud", geometry=BBox(10, 10, 30, 30)),
            Annotation(
                subject="bud", geometry=BBox(60, 60, 120, 120), iscrowd=True
            ),
        ],
        tile_size=64,
        overlap=0.2,
    )

    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)


def test_full_frame_reads_each_ground_truth_box_on_the_stored_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full-frame gate reads each ground-truth box on the stored grid.

    A box labeled and detected at (10.1, 10.1, 40.3, 30.3) enters the scored record, as ground
    truth and as detection, as the xywh box [10.1, 10.1, 30.2, 20.2].
    """
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_annotation.state import Annotation, BBox

    scored: list = []
    real_record = evaluation.prediction_record

    def recording(
        dt: list[dict],
        gt: list[dict],
        **kw: int | str | tuple[int, int, int, int] | None,
    ) -> dict:
        record = real_record(dt, gt, **kw)
        scored.append((record["gt"], record["dt"]))
        return record

    monkeypatch.setattr(evaluation, "prediction_record", recording)
    bur = {"num_channels": 3, "scope": {"subject": "bur"}}
    _full_frame(
        tmp_path,
        monkeypatch,
        _sliced_stub([[10.1, 10.1, 40.3, 30.3]]),
        [Annotation(subject="bur", geometry=BBox(10.1, 10.1, 40.3, 30.3))],
        checkpoint=verified_checkpoint(tmp_path, data=bur),
        tile_size=64,
        overlap=0.2,
    )

    assert [g["bbox"] for gt, _ in scored for g in gt] == [
        [10.1, 10.1, 30.2, 20.2]
    ]
    assert [d["bbox"] for _, dt in scored for d in dt] == [
        [10.1, 10.1, 30.2, 20.2]
    ]


# ── the tile geometry a pass derives from its checkpoint ──────────────────


def _geometry_stub(
    *, train_tile_size: int | None = None, train_overlap: float | None = None
) -> StubPredictor:
    """Return a predictor carrying the given persisted training geometry.

    It answers empty results.
    """
    return StubPredictor(
        boxes=(),
        scores=(),
        task="detection",
        in_chans=3,
        train_tile_size=train_tile_size,
        train_overlap=train_overlap,
    )


def _pass_over(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub: StubPredictor,
    **stated: bool | int,
) -> tuple[Pass, list[dict]]:
    """Return the pass a registered checkpoint prepares over one image with ``stub`` predicting."""
    from tests._verified_checkpoint_fixtures import predicted_over

    images_dir = tmp_path / "one"
    write_image(images_dir / "a.png", (100, 100))
    install(monkeypatch, stub)
    return predicted_over(
        tmp_path,
        project_checkpoint(tmp_path),
        str(images_dir),
        device="cpu",
        **stated,
    )


def test_a_tiled_pass_derives_its_tile_edge_from_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _geometry_stub(train_tile_size=224, train_overlap=0.1)

    p, _results = _pass_over(tmp_path, monkeypatch, stub, tile=True)

    assert (p.execution.tile_size, p.execution.sources["tile_size"]) == (
        224,
        "derived",
    )
    assert p.execution.overlap == pytest.approx(0.1)
    assert stub.executions == [p.execution]


def test_a_tiled_pass_with_no_basis_for_its_edge_refuses_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiled pass with no basis for its edge refuses naming it.

    A checkpoint with no persisted training tile geometry has no basis to tile at: a tiled pass
    with no stated edge refuses, naming ``tile_size``.
    """
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    with pytest.raises(ExecutionRefusedError, match="tile_size"):
        _pass_over(tmp_path, monkeypatch, _geometry_stub(), tile=True)


def test_a_stated_edge_agreeing_with_the_checkpoint_is_recorded_as_stated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, _results = _pass_over(
        tmp_path,
        monkeypatch,
        _geometry_stub(train_tile_size=224, train_overlap=0.1),
        tile=True,
        tile_size=224,
    )

    assert (p.execution.tile_size, p.execution.sources["tile_size"]) == (
        224,
        "explicit",
    )


def test_a_stated_edge_contradicting_the_checkpoint_refuses_naming_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    with pytest.raises(ExecutionRefusedError) as exc_info:
        _pass_over(
            tmp_path,
            monkeypatch,
            _geometry_stub(train_tile_size=224),
            tile=True,
            tile_size=512,
        )
    assert "512" in str(exc_info.value) and "224" in str(exc_info.value)


# ── the delivery-grade evaluation resolves tile geometry the same way ──────


def test_the_evaluation_derives_an_unrecorded_tile_geometry_from_its_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluation derives an unrecorded tile geometry from its ground truth.

    Over one 20 px object, a checkpoint with no persisted tiling and no stated edge records a
    100 px edge and its overlap, each under its derivation; a stated 64 px edge is recorded
    ``explicit`` beside the overlap derived for it.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.derivations import (
        TILE_EDGE_DERIVATION,
        TILE_OVERLAP_DERIVATION,
    )
    from tcip_mcp.pipelines.slicing import overlap_ratio

    one_object = [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))]
    derived = _full_frame(tmp_path, monkeypatch, _sliced_stub([]), one_object)[
        "execution"
    ]
    edge_stated = _full_frame(
        tmp_path / "stated",
        monkeypatch,
        _sliced_stub([]),
        one_object,
        tile_size=64,
    )["execution"]

    # One 20 px object: the edge five of it, the overlap the lattice lays 20 px apart at.
    assert (derived["tile_size"], derived["overlap"]) == (
        100,
        pytest.approx(overlap_ratio(100, 20)),
    )
    assert (
        derived["sources"]["tile_size"],
        derived["sources"]["overlap"],
    ) == (TILE_EDGE_DERIVATION, TILE_OVERLAP_DERIVATION)
    assert (edge_stated["tile_size"], edge_stated["overlap"]) == (
        64,
        pytest.approx(overlap_ratio(64, 20)),
    )
    assert (
        edge_stated["sources"]["tile_size"],
        edge_stated["sources"]["overlap"],
    ) == ("explicit", TILE_OVERLAP_DERIVATION)


def test_the_evaluation_derives_tile_geometry_from_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluation's pass takes the checkpoint's persisted 224 px tile and 0.1 overlap."""
    from tcip_annotation.state import Annotation, BBox

    stub = _sliced_stub([], train_tile_size=224, train_overlap=0.1)

    r = _full_frame(
        tmp_path,
        monkeypatch,
        stub,
        [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))],
    )

    assert [(e.tile_size, e.overlap) for e in stub.executions] == [
        (224, pytest.approx(0.1))
    ]
    assert (
        r["execution"]["tile_size"],
        r["execution"]["sources"]["tile_size"],
    ) == (224, "derived")


def test_launch_training_persists_effective_tile_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``launch_training`` persists the effective tile geometry in the run's launch record.

    The effective tiling geometry and object density are in the run's launch record when
    launch_training returns.

    With ``generic_trainer.train`` patched in this process to raise if called, the run reports a
    process id other than this one and completes.
    """
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    import os

    import tcip_mcp.pipelines.training.generic_trainer as gt
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines.slicing import overlap_ratio
    from tcip_mcp.tools import training_tools
    from tests._chain_fixtures import BLOB_BUILDER
    from tests._producer_fixtures import (
        no_tensorboard,
        small_detection_config,
    )
    from tests._verified_checkpoint_fixtures import run_to_end

    def _poison_train(*a: object, **k: object) -> NoReturn:
        raise AssertionError(
            "generic_trainer.train ran inside the launching process: subprocess isolation broken"
        )

    monkeypatch.setattr(gt, "train", _poison_train)

    images_dir = seed_bud_images(image_dir(tmp_path, UNDATED_BUCKET), n=2)

    no_tensorboard(monkeypatch)

    # no tile_size: the lattice the ground truth derives must be persisted
    cfg = small_detection_config(
        images_dir,
        BLOB_BUILDER,
        project=tmp_path,
        tiling=tiling_block(sliver_frac=0.5),
    )
    res = training_tools.launch_training(tmp_path, cfg, actor=None)
    assert isinstance(res, training_tools.RunLaunched), res
    assert res.pid != os.getpid()  # a different OS process, not this one
    eid = res.experiment_id

    resolution = run_resolution(eid, project=tmp_path)
    tiling = resolution.data.tiling
    assert tiling is not None
    # Every 30 px box derives one extent: the edge five of them, the overlap that extent's.
    assert tiling.tile_size == 150
    assert tiling.overlap == pytest.approx(overlap_ratio(150, 30))
    assert resolution.geometry.object_density is not None

    # "completed", not any terminal state: a child run inside this process would hit _poison_train.
    assert run_to_end(tmp_path, eid)["state"] == "completed"
