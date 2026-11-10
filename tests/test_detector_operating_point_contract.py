"""A detection model's two operating-point knobs, on the model itself and honored by its forward.

The knobs are set on the model, a knob set on a model exposing none refuses naming it, the model
contract refuses a model whose eval output ignores either knob or answers attribute ids of another
dtype or shape than boxes by attributes, and a module exposing both is assessed at its stated
floor.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests._chain_fixtures import (
    BARE_NO_KNOB_DETECTOR,
    BARE_SCORE_THRESH_DETECTOR,
    BESPOKE_MODELS,
    BUILT_DETECTOR,
    FRAME_STATISTICS,
)
from tests._training_values import drawn_split

if TYPE_CHECKING:
    from torch import nn

pytest.importorskip("torch")

import torch  # noqa: E402

from tests.bespoke_models import _KeepsItsOperatingPoint  # noqa: E402

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

IMG = 32
# The width and count a three-band, one-class detector run is built at.
_DIMS = {"in_chans": 3, "num_classes": 1}


def _built(model_source: dict) -> nn.Module:
    """Return the model ``model_source`` builds at :data:`_DIMS` and a 32 px frame's geometry.

    The builder is imported from the staged sources of a run config over one noise frame of its
    own with this repository as its project.
    """
    import tempfile

    from tcip_mcp.pipelines.model_build import (
        build_from_model_source,
        staged_sources,
    )
    from tcip_mcp.pipelines.schemas import TrainGeometry, train_config
    from tests import REPO_ROOT
    from tests._chain_fixtures import (
        detector_geometry,
        frame_statistics,
        training_config,
    )
    from tests._image_fixtures import write_noise_image

    with tempfile.TemporaryDirectory() as frames:
        frame = Path(frames) / "frame.png"
        write_noise_image(frame, IMG)
        spec = train_config(
            training_config(
                Path(frames),
                {
                    **model_source,
                    "normalization": frame_statistics([str(frame)], 3),
                },
                {"images_dir": frames, "split": drawn_split()},
            )
        )
    return build_from_model_source(
        spec.model_source,
        staged_sources(spec, REPO_ROOT).layout,
        _DIMS,
        TrainGeometry(
            native_size=None,
            object_density=None,
            detector=detector_geometry((IMG, IMG)),
        ),
    )


def test_a_built_detector_holds_both_knobs_on_itself_and_its_eval_forward_honors_them() -> (
    None
):
    """A built detector holds both knobs on itself, and its eval forward keeps to them.

    The knobs land on the model itself; at a floor of zero and a cap of one a frame keeps its
    one highest-scoring detection of the two a cap of two keeps, and at a floor above that score
    it keeps none.
    """
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    model = _built(BUILT_DETECTOR).eval()
    frame = [torch.rand(3, IMG, IMG)]

    def kept(score_thresh: float, detections_per_img: int) -> list[float]:
        set_detector_operating_point(
            model,
            score_thresh=score_thresh,
            detections_per_img=detections_per_img,
        )
        with torch.no_grad():
            (out,) = model(frame)
        return out["scores"].tolist()

    two = kept(0.0, 2)
    assert model.score_thresh == 0.0 and model.detections_per_img == 2
    assert len(two) == 2 and two[0] >= two[1], two
    assert kept(0.0, 1) == two[:1]
    assert kept(2 * two[0], 2) == []


@pytest.mark.parametrize(
    "model, missing",
    [
        (SimpleNamespace(unrelated=1), "score_thresh"),
        (SimpleNamespace(score_thresh=0.5), "detections_per_img"),
        (SimpleNamespace(detections_per_img=100), "score_thresh"),
        (
            SimpleNamespace(
                detector=SimpleNamespace(
                    roi_heads=SimpleNamespace(
                        score_thresh=0.5, detections_per_img=100
                    )
                )
            ),
            "score_thresh",
        ),
    ],
    ids=["no knob", "no cap knob", "no score knob", "knobs on a child alone"],
)
def test_set_detector_operating_point_refuses_a_knob_the_model_does_not_expose(
    model: SimpleNamespace, missing: str
) -> None:
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    with pytest.raises(ValueError, match=f"exposes no {missing} "):
        set_detector_operating_point(
            model, score_thresh=0.2, detections_per_img=3
        )


# What the contract smokes below synthesize their batch at, the shape a run resolves for itself.
_SMOKE_DIMS = {"in_chans": 3, "num_classes": 1, "img_size": (64, 64)}


def _contract(model: nn.Module, **dims: object) -> dict:
    """Return ``check_model_contract``'s detection report on ``model`` at :data:`_SMOKE_DIMS`.

    ``dims`` replaces any of its keys.
    """
    from tcip_mcp.pipelines.model_contract import check_model_contract

    return dict(
        check_model_contract(
            model,
            "detection",
            device=torch.device("cpu"),
            dims={**_SMOKE_DIMS, **dims},
        )
    )


class _ThreeScoredBoxes(torch.nn.Module):
    """A detection model offering three boxes per frame, scored 0.9, 0.5 and 0.2 in that order.

    It holds both knobs on itself; its training forward answers a zero-gradient loss of its one
    parameter. :meth:`offered` is each frame's three rows before any knob is applied.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.score_thresh = 0.0
        self.detections_per_img = 100

    def offered(self) -> dict[str, torch.Tensor]:
        return {
            "boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]] * 3),
            "scores": torch.tensor([0.9, 0.5, 0.2]),
            "labels": torch.ones(3, dtype=torch.int64),
        }

    def zero_loss(self) -> dict[str, torch.Tensor]:
        return {"loss": self.weight * 0.0}


class _IgnoresItsFloor(_ThreeScoredBoxes):
    """A :class:`_ThreeScoredBoxes` keeping its first ``detections_per_img`` rows at any score."""

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> dict | list[dict]:
        if self.training:
            return self.zero_loss()
        return [
            {
                k: v[: self.detections_per_img]
                for k, v in self.offered().items()
            }
            for _ in images
        ]


class _IgnoresItsCap(_ThreeScoredBoxes):
    """A :class:`_ThreeScoredBoxes` keeping every row scoring at least ``score_thresh``."""

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> dict | list[dict]:
        if self.training:
            return self.zero_loss()
        out = []
        for _ in images:
            rows = self.offered()
            keep = rows["scores"] >= self.score_thresh
            out.append({k: v[keep] for k, v in rows.items()})
        return out


@pytest.mark.parametrize(
    "model_type, knob, value",
    [
        (_IgnoresItsFloor, "score_thresh", 0.9),
        (_IgnoresItsCap, "detections_per_img", 1),
    ],
    ids=["floor ignored", "cap ignored"],
)
def test_the_contract_refuses_a_detection_model_whose_eval_forward_ignores_a_set_knob(
    model_type: type[_ThreeScoredBoxes], knob: str, value: float
) -> None:
    """The contract refuses a model whose eval output ignores a knob it set, naming the knob.

    A model applying its cap but not its floor is refused naming ``score_thresh`` set above its
    highest score, 0.9; one applying its floor but not its cap is refused naming
    ``detections_per_img`` set to one. Each issue names the count the output kept.
    """
    report = _contract(model_type())

    assert report["ok"] is False
    (issue,) = [i for i in report["issues"] if f"ignores {knob}" in i]
    assert f"set to {value}" in issue and "kept 3" in issue, issue
    assert report["unfiltered_counts"] == [3]


def test_a_built_detector_passes_the_contract_with_detections_its_cap_check_kept_to_one() -> (
    None
):
    """A built detector passes the contract, its unfiltered pass returning detections.

    The report's unfiltered counts show the floor and cap checks ran on detections, not
    vacuously over an empty output.
    """
    from tests import bespoke_models
    from tests._chain_fixtures import detector_values

    report = _contract(
        bespoke_models.build_bespoke_detection(**detector_values())
    )

    assert report["ok"], report["issues"]
    assert all(count >= 1 for count in report["unfiltered_counts"]), report


class _KnobHolder(_KeepsItsOperatingPoint, torch.nn.Module):
    """A module holding both operating-point knobs, at a floor of zero and a cap of 100."""

    def __init__(self) -> None:
        super().__init__()
        self.score_thresh = 0.0
        self.detections_per_img = 100


class _KnobsOnAChild(_ThreeScoredBoxes):
    """A :class:`_ThreeScoredBoxes` whose knobs live only on its ``.detector.roi_heads``.

    Its eval output is kept to that child's knobs (:meth:`_KnobHolder.governed`).
    """

    def __init__(self) -> None:
        super().__init__()
        del self.score_thresh, self.detections_per_img
        self.detector = torch.nn.Module()
        self.detector.roi_heads = _KnobHolder()

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> dict | list[dict]:
        if self.training:
            return self.zero_loss()
        heads = self.detector.roi_heads
        return [heads.governed(self.offered()) for _ in images]


def test_the_contract_refuses_a_detection_model_holding_its_knobs_on_a_child_alone() -> (
    None
):
    """The contract refuses a model whose knobs live on a child module, not on itself.

    Knobs on ``.detector.roi_heads`` that the model's output honors are not knobs the platform
    sets, so the report names the knob the model itself lacks and runs no knob check.
    """
    report = _contract(_KnobsOnAChild())

    assert report["ok"] is False
    assert any(
        "exposes no score_thresh on itself" in i for i in report["issues"]
    ), report["issues"]
    assert report["unfiltered_counts"] is None


class _AnswersAttributes(_KeepsItsOperatingPoint, _ThreeScoredBoxes):
    """A :class:`_ThreeScoredBoxes` kept to both knobs, each box carrying ``attributes``.

    Each kept box's ``attributes`` row is ``width`` zeros as ``dtype``, the last ``short`` rows
    dropped.
    """

    def __init__(self, dtype: torch.dtype, width: int, short: int) -> None:
        super().__init__()
        self.dtype, self.width, self.short = dtype, width, short

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> dict | list[dict]:
        if self.training:
            return self.zero_loss()
        rows = {
            **self.offered(),
            "attributes": torch.zeros((3, self.width), dtype=self.dtype),
        }
        out = []
        for _ in images:
            kept = self.governed(rows)
            kept["attributes"] = kept["attributes"][
                : len(kept["boxes"]) - self.short
            ]
            out.append(kept)
        return out


@pytest.mark.parametrize(
    "dtype, width, short, refused",
    [
        (torch.int64, 2, 0, False),
        (torch.float32, 2, 0, True),
        (torch.int64, 3, 0, True),
        (torch.int64, 2, 1, True),
    ],
    ids=["int64 by two", "float", "a column too many", "a row too few"],
)
def test_the_contract_holds_the_attributes_output_to_int64_one_row_per_box_one_column_per_attribute(
    dtype: torch.dtype, width: int, short: int, refused: bool
) -> None:
    """The contract refuses an ``attributes`` output that is not int64 of shape boxes by attributes.

    Under two declared attributes, int64 rows of two columns, one per box, pass; float rows, a
    third column, or one row fewer than the boxes kept are refused naming the dtype and shape the
    output answered.
    """
    from tcip_mcp import subject_registry

    attributes = (
        subject_registry.Attribute("color", "categorical", ("red", "blue")),
        subject_registry.Attribute("grade", "ordinal", ("low", "mid", "high")),
    )

    report = _contract(
        _AnswersAttributes(dtype, width, short), attributes=attributes
    )

    issues = [i for i in report["issues"] if "attributes" in i]
    if not refused:
        assert report["ok"], report["issues"]
        return
    (issue,) = issues
    assert str(dtype) in issue and f"[{3 - short}, {width}]" in issue, issue
    assert "[len(boxes), 2]" in issue, issue


def _checkpoint(tmp_path: Path, builder: str) -> str:
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    return registered_checkpoint(
        tmp_path,
        model_source={
            "builder": builder,
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        data={
            "num_channels": 3,
            "scope": {"subject": "bud"},
        },
    )


def _assessed(tmp_path: Path, builder: str) -> dict:
    """Assess ``builder``'s checkpoint over a drawn selection and return its record or error.

    The images are each their own size, each labeled at its central box spanning half its edge,
    in the dataset ``ds`` registered at :data:`~tests._mapping_fixtures.COUNT_CROP`.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.data_tools import draw_splits
    from tests._chain_fixtures import assess, confirm_count_trait
    from tests._mapping_fixtures import COUNT_CROP
    from tests._producer_fixtures import (
        label_image,
        register_dataset_at,
        write_image,
    )

    root = tmp_path / "ds"
    images_dir = image_dir(root, "2-11-26")
    for i, size in enumerate(range(32, 96, 8)):
        image = write_image(
            images_dir / f"img{i}.png", (size, size), (100, 100, 100)
        )
        box = BBox(size * 0.25, size * 0.25, size * 0.75, size * 0.75)
        label_image(
            image,
            [Annotation(subject="bud", geometry=box)],
            size,
            size,
        )
    register_dataset_at(tmp_path, root, crop=COUNT_CROP)
    selection_dir = tmp_path / "selection"
    drawn = draw_splits(
        tmp_path,
        str(root),
        output_path=str(selection_dir),
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
    return assess(
        tmp_path,
        _checkpoint(tmp_path, builder),
        selection_dir,
        device="cpu",
        tile=False,
    )


def test_a_bespoke_module_exposing_its_own_knobs_is_assessed_at_its_stated_floor(
    tmp_path: Path,
) -> None:
    """A bespoke module exposing its own knobs is assessed at its stated floor.

    A hand-rolled module exposing both knobs on itself is assessed without error; the count
    evidence records the floor and no place the knobs were found, and the record reports no
    ``conf_censored`` failure.
    """
    record = _assessed(tmp_path, BARE_SCORE_THRESH_DETECTOR)

    assert "error" not in record, record
    count = record["criterion"]["count"]
    assert "staged_conf_floor" in count
    assert "staged_conf_floor_attribute_path" not in count
    assert "conf_censored" not in record["failures"]


def test_a_module_exposing_no_knob_refuses_naming_the_knob_it_lacks(
    tmp_path: Path,
) -> None:
    refused = _assessed(tmp_path, BARE_NO_KNOB_DETECTOR)

    assert "score_thresh" in refused.get("error", ""), refused
