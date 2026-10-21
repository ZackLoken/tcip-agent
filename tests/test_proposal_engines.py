"""The two proposal engines that ship registered, and a box candidate through the proposal doors.

``sam`` and ``grounding_dino`` resolve by name without loading torch or transformers; their
parameters are refused as their models state; a mask and a box become candidates by the module
functions the engines compose; an engine reads its image upright; a ringless candidate stages as
its box; the detector runs through the platform's tiled pass, its lattice derived and its merge
threshold stated. The live tests run each checkpoint in a child whose Hugging Face libraries
import under offline mode, so a checkpoint the cache does not hold whole skips on the library's
own refusal.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError
from sahi.models.base import DetectionModel

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests import test_propose_annotations_region as region
from tests._producer_fixtures import kept

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from sahi.prediction import ObjectPrediction

    from tcip_mcp.pipelines.proposal import Proposal

exif_rotated_source = region.exif_rotated_source

SEGMENTER_CHECKPOINT = "facebook/sam-vit-large"
DETECTOR_CHECKPOINT = "IDEA-Research/grounding-dino-base"


@pytest.mark.parametrize("name", ["sam", "grounding_dino"])
def test_each_shipped_engine_resolves_by_name_to_a_proposer(name: str) -> None:
    from tcip_mcp.pipelines.proposal import Proposer, resolve_proposer

    assert isinstance(resolve_proposer(name), Proposer)


def test_importing_the_seam_and_the_server_loads_neither_torch_nor_transformers() -> (
    None
):
    """A fresh interpreter importing the seam and the server holds no torch or transformers."""
    script = (
        "import sys\n"
        "import tcip_mcp.pipelines.proposal\n"
        "import tcip_mcp.server\n"
        "print(sorted({'torch', 'transformers'} & set(sys.modules)))\n"
    )

    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=120,
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "[]"


def test_the_segmenters_parameters_admit_a_stated_prompt() -> None:
    from tcip_mcp.pipelines.proposal_engines import SegmenterParams

    stated = SegmenterParams.model_validate(
        {"points": [[4, 5]], "multimask_output": False}
    )

    assert stated.points == [(4.0, 5.0)]
    assert stated.checkpoint == SEGMENTER_CHECKPOINT


def test_the_segmenters_parameters_refuse_no_prompt_naming_points_and_boxes() -> (
    None
):
    from tcip_mcp.pipelines.proposal_engines import SegmenterParams

    with pytest.raises(ValidationError, match="points, boxes"):
        SegmenterParams.model_validate({"multimask_output": True})


def test_the_segmenters_parameters_refuse_an_unstated_multimask_output() -> (
    None
):
    from tcip_mcp.pipelines.proposal_engines import SegmenterParams

    with pytest.raises(ValidationError, match="multimask_output"):
        SegmenterParams.model_validate({"boxes": [[1, 2, 3, 4]]})


def test_the_segmenters_parameters_refuse_a_key_they_do_not_declare() -> None:
    from tcip_mcp.pipelines.proposal_engines import SegmenterParams

    with pytest.raises(ValidationError, match="stride"):
        SegmenterParams.model_validate(
            {"points": [[4, 5]], "multimask_output": False, "stride": 2}
        )


DETECTOR_PARAMS: dict[str, object] = {
    "phrase": "a nut",
    "box_threshold": 0.3,
    "text_threshold": 0.3,
    "postprocess": "nms",
    "cross_tile_nms": 0.5,
}
"""A detector call stating every required parameter, its lattice left to derive."""


def test_the_detectors_parameters_admit_a_call_stating_each() -> None:
    from tcip_mcp.pipelines.proposal_engines import DetectorParams

    stated = DetectorParams.model_validate(DETECTOR_PARAMS)

    assert stated.postprocess == "nms"
    assert (stated.tile_size, stated.overlap) == (None, None)
    assert stated.checkpoint == DETECTOR_CHECKPOINT


def test_the_detectors_parameters_refuse_a_blank_phrase() -> None:
    from tcip_mcp.pipelines.proposal_engines import DetectorParams

    with pytest.raises(ValidationError, match="phrase"):
        DetectorParams.model_validate({**DETECTOR_PARAMS, "phrase": "  "})


@pytest.mark.parametrize(
    "field", ["box_threshold", "postprocess", "cross_tile_nms"]
)
def test_the_detectors_parameters_refuse_each_unstated_field(
    field: str,
) -> None:
    from tcip_mcp.pipelines.proposal_engines import DetectorParams

    with pytest.raises(ValidationError, match=field):
        DetectorParams.model_validate(
            {k: v for k, v in DETECTOR_PARAMS.items() if k != field}
        )


def test_the_detectors_parameters_refuse_an_unknown_merge_naming_the_merges() -> (
    None
):
    from tcip_mcp.pipelines.execution import CROSS_TILE_MERGES
    from tcip_mcp.pipelines.proposal_engines import DetectorParams

    with pytest.raises(ValidationError) as refused:
        DetectorParams.model_validate(
            {**DETECTOR_PARAMS, "postprocess": "average"}
        )

    assert all(name in str(refused.value) for name in CROSS_TILE_MERGES)


def test_the_detectors_parameters_refuse_a_key_they_do_not_declare() -> None:
    from tcip_mcp.pipelines.proposal_engines import DetectorParams

    with pytest.raises(ValidationError, match="stride"):
        DetectorParams.model_validate({**DETECTOR_PARAMS, "stride": 2})


def test_the_detectors_defaults_are_its_libraries_own_and_complete_a_phrase() -> (
    None
):
    """The defaults read off transformers' and SAHI's signatures complete a phrase prompt.

    They are compared against the two libraries' documented defaults, read here from the same
    signatures, and the merge name is one the platform's merge table holds.
    """
    import inspect

    from sahi.predict import get_sliced_prediction
    from transformers import GroundingDinoProcessor

    from tcip_mcp.pipelines.execution import CROSS_TILE_MERGES
    from tcip_mcp.pipelines.proposal_engines import (
        DetectorParams,
        detector_defaults,
    )

    defaults = detector_defaults()
    grounded = inspect.signature(
        GroundingDinoProcessor.post_process_grounded_object_detection
    ).parameters
    sliced = inspect.signature(get_sliced_prediction).parameters

    stated = DetectorParams(phrase="open bud", **defaults)
    assert stated.box_threshold == grounded["threshold"].default
    assert stated.text_threshold == grounded["text_threshold"].default
    assert stated.postprocess in CROSS_TILE_MERGES
    assert (
        CROSS_TILE_MERGES[stated.postprocess].sahi_type
        == sliced["postprocess_type"].default
    )
    assert (
        stated.cross_tile_nms == sliced["postprocess_match_threshold"].default
    )


EDGE = 1024
"""An input edge for the window geometry: SAM's ``longest_edge`` in the base checkpoint."""


def test_a_prompt_near_a_corner_takes_an_edge_window_clamped_to_the_frame() -> (
    None
):
    from tcip_mcp.pipelines.proposal_engines import prompt_window

    assert prompt_window((30, 2950, 30, 2950), (3000, 4000), EDGE) == (
        0,
        3000 - EDGE,
        EDGE,
        3000,
    )


def test_a_central_prompt_takes_an_edge_window_centered_on_it() -> None:
    from tcip_mcp.pipelines.proposal_engines import prompt_window

    assert prompt_window((2000, 1500, 2000, 1500), (3000, 4000), EDGE) == (
        2000 - EDGE // 2,
        1500 - EDGE // 2,
        2000 + EDGE // 2,
        1500 + EDGE // 2,
    )


def test_a_box_larger_than_the_edge_takes_its_own_extent() -> None:
    from tcip_mcp.pipelines.proposal_engines import prompt_window

    assert prompt_window((100, 1400, 1700, 1600), (3000, 4000), EDGE) == (
        100,
        1500 - EDGE // 2,
        1700,
        1500 + EDGE // 2,
    )


def test_a_frame_under_the_edge_is_its_own_window() -> None:
    from tcip_mcp.pipelines.proposal_engines import prompt_window

    assert prompt_window((10, 20, 10, 20), (300, 400), EDGE) == (
        0,
        0,
        400,
        300,
    )


def test_a_candidate_from_a_window_moves_by_its_origin() -> None:
    from tcip_mcp.pipelines.proposal_engines import (
        mask_candidate,
        shifted_candidate,
    )

    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 3:9] = True
    candidate = mask_candidate(mask, candidate_id=0, score=0.9, engine_meta={})
    assert candidate is not None

    moved = shifted_candidate(candidate, (100, 40))

    assert moved["bbox"] == [103.0, 42.0, 108.0, 45.0]
    assert moved["rings"] == [
        [(x + 100, y + 40) for x, y in ring] for ring in candidate["rings"]
    ]
    assert moved["area"] == candidate["area"]


def test_two_slices_calls_of_one_box_merge_into_one_at_the_stated_threshold() -> (
    None
):
    from sahi.prediction import ObjectPrediction

    from tcip_mcp.pipelines.execution import proposal_execution
    from tcip_mcp.pipelines.slicing import cross_tile_merge

    calls = [
        ObjectPrediction(
            bbox=box, score=score, category_id=0, category_name="a nut"
        )
        for box, score in (
            ([100.0, 100.0, 160.0, 160.0], 0.8),
            ([102.0, 100.0, 160.0, 161.0], 0.6),
        )
    ]
    execution = proposal_execution(
        tile_size=800,
        overlap=0.1,
        geometry_sources={"tile_size": "explicit", "overlap": "explicit"},
        postprocess="nms",
        cross_tile_nms=0.5,
    )

    [merged] = cross_tile_merge(execution)(calls)

    assert merged.score.value == 0.8
    assert merged.category.name == "a nut"
    assert execution.sources["cross_tile_nms"] == "explicit"


class ChannelSquares(DetectionModel):
    """A stand-in detection model: one box per color channel, around that channel's lit pixels.

    Each image's box for a channel spans its nonzero pixels, half-open, scored 0.9.
    """

    def __init__(self) -> None:
        super().__init__(model=object())

    def set_device(self, device: str | None = None) -> None:
        self.device = "cpu"

    def set_model(self, model: object, **kwargs: object) -> None:
        self.model = model

    def perform_inference(self, image: np.ndarray) -> None:
        self.perform_batch_inference([image])

    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        found = []
        for image in images:
            boxes = []
            for channel in range(image.shape[2]):
                ys, xs = np.nonzero(image[:, :, channel])
                if len(xs):
                    boxes.append(
                        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
                    )
            found.append(boxes)
        self._original_predictions = found

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = None,
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        from sahi.prediction import ObjectPrediction

        assert shift_amount_list is not None and full_shape_list is not None
        self._object_prediction_list_per_image = [
            [
                ObjectPrediction(
                    bbox=[float(v) for v in box],
                    score=0.9,
                    category_id=0,
                    category_name="square",
                    shift_amount=shift,
                    full_shape=full,
                )
                for box in boxes
            ]
            for boxes, shift, full in zip(
                self._original_predictions,
                shift_amount_list,
                full_shape_list,
                strict=True,
            )
        ]


SQUARES = ((680, 100, 760, 180), (720, 140, 800, 220))
"""Two 80 pixel squares that overlap, the first across the seam of a 1600 pixel frame's first
two columns of 800 pixel slices, painted in the red and the green channel."""


def _squares_frame(squares: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    """Return a dark 1600 by 1200 RGB frame with each square lit in a channel of its own."""
    pixels = np.zeros((1200, 1600, 3), dtype=np.uint8)
    for channel, (x1, y1, x2, y2) in enumerate(squares):
        pixels[y1:y2, x1:x2, channel] = 255
    return pixels


def test_the_shared_pass_moves_each_slices_detections_into_the_frame() -> None:
    """Two slices' detections come back in frame pixels, by the pass inference runs."""
    from tcip_mcp.pipelines.slicing import sliced_predictions

    pixels = _squares_frame(SQUARES[:1])
    slices = [(0, (0, 0, 800, 800)), (1, (640, 0, 1440, 800))]

    [(_batch, per_slice)] = sliced_predictions(
        ChannelSquares(),
        lambda x0, y0, x1, y1: pixels[y0:y1, x0:x1],
        slices,
        (1200, 1600),
        2,
    )

    assert [[p.bbox.to_xyxy() for p in found] for found in per_slice] == [
        [[680.0, 100.0, 760.0, 180.0]],
        [[680.0, 100.0, 760.0, 180.0]],
    ]


def _whole(*boxes: list[float]) -> list[ObjectPrediction]:
    """Return a whole-frame pass's predictions of ``boxes``."""
    from sahi.prediction import ObjectPrediction

    return [
        ObjectPrediction(bbox=box, score=0.9, category_id=0) for box in boxes
    ]


def test_a_whole_frame_pass_derives_the_lattice_from_its_objects() -> None:
    """The edge is five times the p99 of the boxes' larger sides, the overlap holding one whole.

    The sides are 100 and 240, whose p99 rounds up to 239, so the edge is 1195, past the 800 px
    model edge.
    """
    from tcip_mcp.pipelines.derivations import (
        TILE_EDGE_DERIVATION,
        TILE_OVERLAP_DERIVATION,
    )
    from tcip_mcp.pipelines.proposal_engines import whole_frame_geometry
    from tcip_mcp.pipelines.slicing import overlap_ratio

    geometry = whole_frame_geometry(
        _whole([0.0, 0.0, 100.0, 40.0], [10.0, 10.0, 60.0, 250.0]),
        (3000, 4000),
        800,
        tile_size=None,
        overlap=None,
    )

    assert geometry == (
        1195,
        overlap_ratio(1195, 239),
        {
            "tile_size": TILE_EDGE_DERIVATION,
            "overlap": TILE_OVERLAP_DERIVATION,
        },
    )


def test_a_derived_edge_under_the_models_edge_is_floored_at_it() -> None:
    """Thirty pixel objects derive a 150 px edge; the lattice takes the 800 px model edge."""
    from tcip_mcp.pipelines.derivations import TILE_OVERLAP_DERIVATION
    from tcip_mcp.pipelines.proposal_engines import whole_frame_geometry
    from tcip_mcp.pipelines.slicing import overlap_ratio

    geometry = whole_frame_geometry(
        _whole([0.0, 0.0, 30.0, 30.0], [100.0, 100.0, 130.0, 130.0]),
        (3000, 4000),
        800,
        tile_size=None,
        overlap=None,
    )

    assert geometry == (
        800,
        overlap_ratio(800, 30),
        {"tile_size": "derived", "overlap": TILE_OVERLAP_DERIVATION},
    )


def test_a_tiled_proposal_records_its_lattice_and_merges_its_seams() -> None:
    """A frame larger than its lattice's edge runs on that lattice, merged at the stated threshold.

    Its record states the floored edge, the derived overlap and the stated threshold, each by its
    source; the square cut by a seam merges, so each square is proposed once.
    """
    from tcip_mcp.pipelines.derivations import TILE_OVERLAP_DERIVATION
    from tcip_mcp.pipelines.proposal_engines import detected_proposal
    from tcip_mcp.pipelines.slicing import overlap_ratio

    proposal = detected_proposal(
        ChannelSquares(),
        _squares_frame(SQUARES),
        800,
        postprocess="nms",
        cross_tile_nms=0.3,
        tile_size=None,
        overlap=None,
    )

    execution = proposal.execution
    assert execution is not None
    assert (
        execution.tile_size,
        execution.overlap,
        execution.cross_tile_nms,
    ) == (800, overlap_ratio(800, 80), 0.3)
    assert execution.sources == {
        "tile_size": "derived",
        "overlap": TILE_OVERLAP_DERIVATION,
        "cross_tile_nms": "explicit",
    }
    assert sorted(c["bbox"] for c in proposal.candidates) == [
        [float(v) for v in square] for square in SQUARES
    ]


def test_a_lattice_reaching_the_frame_runs_whole_recording_its_lattice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 px object derives a 2000 px edge past the frame, so the frame runs whole, recorded."""
    from tcip_mcp.pipelines import slicing
    from tcip_mcp.pipelines.derivations import TILE_EDGE_DERIVATION
    from tcip_mcp.pipelines.proposal_engines import detected_proposal

    lattices: list[tuple[int, int, int, float]] = []
    real = slicing.slice_lattice

    def recorded(
        height: int, width: int, tile_size: int, overlap: float
    ) -> list[tuple[int, int, int, int]]:
        lattices.append((height, width, tile_size, overlap))
        return real(height, width, tile_size, overlap)

    monkeypatch.setattr(slicing, "slice_lattice", recorded)

    proposal = detected_proposal(
        ChannelSquares(),
        _squares_frame([(100, 100, 500, 500)]),
        800,
        postprocess="nms",
        cross_tile_nms=0.5,
        tile_size=None,
        overlap=None,
    )

    assert lattices == []
    assert proposal.execution is not None
    assert proposal.execution.tile_size == 2000
    assert proposal.execution.sources["tile_size"] == TILE_EDGE_DERIVATION
    assert [c["bbox"] for c in proposal.candidates] == [
        [100.0, 100.0, 500.0, 500.0]
    ]


def test_a_large_frame_the_whole_frame_pass_finds_nothing_on_refuses() -> None:
    """With nothing to derive from, a frame past the model edge refuses naming the lattice.

    A frame within the model edge proposes nothing, with no record, and the large frame with the
    lattice stated runs, its record naming both ``explicit``.
    """
    from tcip_mcp.pipelines.execution import ExecutionRefusedError
    from tcip_mcp.pipelines.proposal_engines import detected_proposal

    blank = _squares_frame([])
    with pytest.raises(ExecutionRefusedError, match="tile_size and overlap"):
        detected_proposal(
            ChannelSquares(),
            blank,
            800,
            postprocess="nms",
            cross_tile_nms=0.5,
            tile_size=None,
            overlap=None,
        )

    small = detected_proposal(
        ChannelSquares(),
        blank[:800, :800],
        800,
        postprocess="nms",
        cross_tile_nms=0.5,
        tile_size=None,
        overlap=None,
    )
    stated = detected_proposal(
        ChannelSquares(),
        blank,
        800,
        postprocess="nms",
        cross_tile_nms=0.5,
        tile_size=800,
        overlap=0.2,
    )

    assert small == ([], None)
    assert stated.candidates == []
    assert stated.execution is not None
    assert stated.execution.sources == {
        "tile_size": "explicit",
        "overlap": "explicit",
        "cross_tile_nms": "explicit",
    }


def test_a_frame_within_the_models_edge_runs_whole_recording_its_floored_lattice() -> (
    None
):
    from tcip_mcp.pipelines.proposal_engines import detected_proposal

    proposal = detected_proposal(
        ChannelSquares(),
        _squares_frame(SQUARES)[:800, :800],
        800,
        postprocess="nms",
        cross_tile_nms=0.5,
        tile_size=None,
        overlap=None,
    )

    assert proposal.execution is not None
    assert proposal.execution.tile_size == 800
    assert len(proposal.candidates) == 2


def test_an_engines_parameter_refusal_reaches_the_door_as_its_error(
    tmp_path: Path,
) -> None:
    """``propose_annotations`` answers an engine's parameter refusal as an error naming it."""
    from tcip_mcp.tools.proposal_tools import propose_annotations
    from tests._producer_fixtures import write_image

    image = write_image(tmp_path / "images" / UNDATED_BUCKET / "a.png")

    result = propose_annotations(
        tmp_path,
        image_path=str(image),
        engine="sam",
        engine_params={"points": [[1, 1]]},
        actor=None,
    )

    assert "multimask_output" in result.get("error", ""), result


def test_a_mask_of_two_blobs_is_one_candidate_of_two_rings() -> None:
    """A mask of two disjoint blobs is one candidate: two rings, its pixel count, their box."""
    from tcip_mcp.pipelines.proposal_engines import mask_candidate

    mask = np.zeros((40, 60), dtype=bool)
    mask[5:15, 4:12] = True
    mask[20:31, 30:50] = True

    candidate = mask_candidate(
        mask, candidate_id=3, score=0.7, engine_meta={"prompt": 0, "rank": 0}
    )

    assert candidate is not None
    assert len(candidate["rings"]) == 2
    assert candidate["area"] == 8 * 10 + 20 * 11
    assert candidate["bbox"] == [4.0, 5.0, 49.0, 30.0]
    assert candidate["candidate_id"] == 3


def test_an_empty_mask_is_no_candidate() -> None:
    from tcip_mcp.pipelines.proposal_engines import mask_candidate

    assert (
        mask_candidate(
            np.zeros((8, 8), dtype=bool),
            candidate_id=0,
            score=0.5,
            engine_meta={},
        )
        is None
    )


def test_a_box_is_a_candidate_with_no_rings_and_the_boxs_area() -> None:
    from tcip_mcp.pipelines.proposal_engines import box_candidate

    candidate = box_candidate(
        [2.0, 3.0, 12.0, 8.0],
        candidate_id=0,
        score=0.4,
        engine_meta={"label": "nut"},
    )

    assert candidate["rings"] is None
    assert candidate["area"] == 50.0
    assert candidate["bbox"] == [2.0, 3.0, 12.0, 8.0]


def test_an_engine_reads_an_exif_rotated_image_upright(
    exif_rotated_source: Path,
) -> None:
    from tcip_mcp.pipelines.proposal_engines import upright_pixels

    assert upright_pixels(str(exif_rotated_source)).shape == (
        region.UPRIGHT_H,
        region.UPRIGHT_W,
        3,
    )


BOX = (10.0, 10.0, 80.0, 80.0)
"""The ringless candidate's box, far enough from its center that its outline is clear of the
number drawn there."""


class BoxProposer:
    """A proposer answering one ringless candidate at :data:`BOX`."""

    def propose(self, image_path: str, **params: object) -> Proposal:
        from tcip_mcp.pipelines.proposal import Proposal
        from tcip_mcp.pipelines.proposal_engines import box_candidate

        return Proposal(
            [
                box_candidate(
                    BOX,
                    candidate_id=0,
                    score=0.8,
                    engine_meta={"label": "nut"},
                )
            ],
            None,
        )


@pytest.fixture
def boxed_engine() -> Iterator[str]:
    """Register :class:`BoxProposer` under a test name for the test, and return the name."""
    from tcip_mcp.pipelines import proposal

    proposal.register_proposal_engine("boxed", BoxProposer())
    yield "boxed"
    del proposal._ENGINES["boxed"]


def test_a_ringless_candidate_renders_and_stages_as_its_box(
    tmp_path: Path, boxed_engine: str
) -> None:
    """A ringless candidate renders as its box outline and stages as a box read back as one."""
    from PIL import Image

    from tcip_annotation.viz import COLOR_PALETTE
    from tcip_mcp.tools.annotation_tools import read_annotations
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )
    from tests._producer_fixtures import write_image

    image = write_image(
        tmp_path / "images" / UNDATED_BUCKET / "boxed.png",
        (96, 96),
        (40, 40, 40),
    )

    proposed = propose_annotations(
        tmp_path, image_path=str(image), engine=boxed_engine, actor=None
    )
    assert proposed["staged"] is True, proposed
    with Image.open(proposed["image_path"]) as render:
        assert render.convert("RGB").getpixel((10, 20)) == COLOR_PALETTE[0]

    accepted = stage_proposals(
        tmp_path,
        image_path=str(image),
        assignments=kept(proposed, (0, "nut")),
        actor=None,
    )
    assert "error" not in accepted, accepted

    read = read_annotations(str(image), bucket=accepted["bucket"])
    [staged] = read["predictions"]["annotations"]
    assert staged["bbox"] == list(BOX)
    assert "rings" not in staged


def test_a_proposals_execution_record_reaches_its_staging_and_its_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record a proposal answers is staged with it and published with its bucket."""
    import tcip_store as ts
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines import proposal
    from tcip_mcp.pipelines.execution import proposal_execution
    from tcip_mcp.pipelines.proposal_engines import box_candidate
    from tcip_mcp.tools.proposal_tools import (
        ProposalStaging,
        _staging_key_for,
        propose_annotations,
        stage_proposals,
    )
    from tests._producer_fixtures import write_image

    execution = proposal_execution(
        tile_size=800,
        overlap=0.1,
        geometry_sources={"tile_size": "explicit", "overlap": "explicit"},
        postprocess="nms",
        cross_tile_nms=0.5,
    )

    class Recorded:
        def propose(
            self, image_path: str, **params: object
        ) -> proposal.Proposal:
            return proposal.Proposal(
                [
                    box_candidate(
                        BOX, candidate_id=0, score=0.8, engine_meta={}
                    )
                ],
                execution,
            )

    monkeypatch.setattr(
        proposal, "resolve_proposer", lambda engine: Recorded()
    )
    image = write_image(
        tmp_path / "images" / UNDATED_BUCKET / "recorded.png", (96, 96)
    )

    proposed = propose_annotations(
        tmp_path, image_path=str(image), engine="recorded", actor=None
    )
    assert proposed["staged"] is True, proposed
    staging = ProposalStaging.model_validate(
        ts.read(_staging_key_for(str(image), proposed["staging_id"]))
    )
    accepted = stage_proposals(
        tmp_path,
        image_path=str(image),
        assignments=kept(proposed, (0, "nut")),
        actor=None,
    )

    assert staging.execution == execution
    assert read_bucket(tmp_path, accepted["bucket"]).execution == execution


OFFLINE_PROPOSAL = """
import json
import sys

from huggingface_hub import is_offline_mode

from tcip_mcp.pipelines import proposal_engines, slicing
from tcip_mcp.pipelines.execution import EXECUTION_RECORD
from tcip_mcp.pipelines.proposal import resolve_proposer

engine, image, params = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
answer = {"offline": is_offline_mode(), "lattice": []}
lattice = slicing.slice_lattice
slicing.slice_lattice = lambda *args: (
    answer["lattice"].append(args) or lattice(*args)
)
stated, load = {
    "sam": (proposal_engines.SegmenterParams, proposal_engines._segmenter),
    "grounding_dino": (
        proposal_engines.DetectorParams,
        proposal_engines._detector,
    ),
}[engine]
try:
    load(stated.model_validate(params).checkpoint)
except Exception as error:
    answer["refused"] = f"{type(error).__name__}: {error}"
else:
    proposal = resolve_proposer(engine).propose(image, **params)
    answer["candidates"] = proposal.candidates
    answer["execution"] = (
        None
        if proposal.execution is None
        else EXECUTION_RECORD.dump_python(proposal.execution, mode="json")
    )
print(json.dumps(answer))
"""
"""One engine call in an interpreter whose Hugging Face libraries import under offline mode.

The checkpoint is loaded first, through the engine's own cached loader: a load that fails is
reported as the refusal, its exception named, and the engine is not run; a failure of the
engine's call after a load that succeeded propagates. Otherwise it prints whether offline mode
is in force, each ``slice_lattice`` call's arguments, the candidates and the execution record."""


def _offline_proposal(
    engine: str,
    image: str,
    params: dict[str, object],
    cache: Path | None = None,
) -> dict:
    """Run ``engine`` over ``image`` in a child with ``HF_HUB_OFFLINE`` set before any import.

    ``cache``, when given, stands in for the Hugging Face cache. Returns the child's answer.
    """
    import json

    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if cache is not None:
        env["HF_HUB_CACHE"] = str(cache)
    child = subprocess.run(
        [sys.executable, "-c", OFFLINE_PROPOSAL, engine, image]
        + [json.dumps(params)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert child.returncode == 0, child.stderr
    return json.loads(child.stdout.splitlines()[-1])


def _live(answer: dict) -> dict:
    """Return a live run's answer, skipping when its checkpoint failed to load offline."""
    assert answer["offline"] is True
    if "refused" in answer:
        pytest.skip(answer["refused"])
    return answer


def test_a_checkpoint_the_cache_lacks_is_refused_offline_never_fetched(
    tmp_path: Path, bright_square: str
) -> None:
    """Over an absent cache, a live run is refused by the library offline and nothing is fetched."""
    answer = _offline_proposal(
        "sam",
        bright_square,
        {"points": [[48, 48]], "multimask_output": False},
        cache=tmp_path / "absent",
    )

    assert answer["offline"] is True
    assert "refused" in answer, answer
    assert not (tmp_path / "absent").exists()


WEIGHT_SUFFIXES = (".safetensors", ".bin")
"""The suffixes of a checkpoint's weight files, which the incomplete cache leaves out."""


def _cache_without_weights(repo_id: str, cache: Path) -> Path:
    """Copy ``repo_id``'s cached revision into ``cache`` with its weight files left out.

    The revision's ``refs`` and every snapshot file but the weights are copied into the same
    layout the Hugging Face cache scan reports. Skips when the checkpoint is not cached here.
    """
    import shutil

    from huggingface_hub import scan_cache_dir

    repos = [r for r in scan_cache_dir().repos if r.repo_id == repo_id]
    if not repos:
        pytest.skip(f"{repo_id} is not in the Hugging Face cache")
    [repo] = repos
    target = cache / repo.repo_path.name
    shutil.copytree(repo.repo_path / "refs", target / "refs")
    for revision in repo.revisions:
        for file in revision.files:
            if file.file_name.endswith(WEIGHT_SUFFIXES):
                continue
            copied = target / file.file_path.relative_to(repo.repo_path)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file.blob_path, copied)
    return cache


@pytest.mark.parametrize(
    ("engine", "params"),
    [
        (
            "sam",
            {
                "points": [[48, 48]],
                "multimask_output": False,
                "checkpoint": SEGMENTER_CHECKPOINT,
            },
        ),
        (
            "grounding_dino",
            {**DETECTOR_PARAMS, "checkpoint": DETECTOR_CHECKPOINT},
        ),
    ],
)
def test_a_checkpoint_cached_without_its_weights_is_refused_offline_never_fetched(
    tmp_path: Path, bright_square: str, engine: str, params: dict[str, object]
) -> None:
    """A checkpoint cached with its config and snapshot but no weights is refused offline.

    Nothing is fetched into the cache: no weight file appears in it.
    """
    cache = _cache_without_weights(
        str(params["checkpoint"]), tmp_path / "incomplete"
    )

    answer = _offline_proposal(engine, bright_square, params, cache=cache)

    assert answer["offline"] is True
    assert "refused" in answer, answer
    assert not [
        p for p in cache.rglob("*") if p.name.endswith(WEIGHT_SUFFIXES)
    ]


@pytest.fixture
def bright_square(tmp_path: Path) -> str:
    """Write a 96 by 96 dark image holding one bright square, pixels 32 to 63, and return it."""
    from tests._producer_fixtures import painted_frame

    path = tmp_path / "square.png"
    painted_frame(
        96, 96, (20, 20, 20), [((32, 32, 64, 64), (235, 235, 235))]
    ).save(path)
    return str(path)


def test_the_segmenter_prompted_inside_a_square_proposes_the_square(
    bright_square: str,
) -> None:
    [candidate] = _live(
        _offline_proposal(
            "sam",
            bright_square,
            {
                "points": [[48, 48]],
                "multimask_output": False,
                "checkpoint": SEGMENTER_CHECKPOINT,
            },
        )
    )["candidates"]

    assert candidate["bbox"] == pytest.approx([32, 32, 63, 63], abs=2)


def test_the_detector_prompted_with_a_phrase_proposes_scored_boxes(
    bright_square: str,
) -> None:
    candidates = _live(
        _offline_proposal(
            "grounding_dino",
            bright_square,
            {
                **DETECTOR_PARAMS,
                "phrase": "square",
                "box_threshold": 0.25,
                "text_threshold": 0.25,
                "checkpoint": DETECTOR_CHECKPOINT,
            },
        )
    )["candidates"]

    assert candidates
    assert all(c["rings"] is None for c in candidates)
    assert all(math.isfinite(c["score"]) for c in candidates)


SMALL_SQUARE = (2000, 300, 2024, 324)
"""A 24 pixel square far from the center of a 2400 by 1800 frame, as a half-open box."""


def test_the_segmenter_proposes_a_small_square_of_a_large_frame_at_native_resolution(
    tmp_path: Path,
) -> None:
    """A prompt at a small square's center of a frame larger than the edge proposes the square.

    At the whole frame's resampling the square would span about ten pixels.
    """
    from tests._producer_fixtures import painted_frame

    path = tmp_path / "large.png"
    painted_frame(
        2400, 1800, (20, 20, 20), [(SMALL_SQUARE, (235, 235, 235))]
    ).save(path)
    x1, y1, x2, y2 = SMALL_SQUARE

    [candidate] = _live(
        _offline_proposal(
            "sam",
            str(path),
            {
                "points": [[(x1 + x2) / 2, (y1 + y2) / 2]],
                "multimask_output": False,
                "checkpoint": SEGMENTER_CHECKPOINT,
            },
        )
    )["candidates"]

    assert candidate["bbox"] == pytest.approx([x1, y1, x2 - 1, y2 - 1], abs=2)
    wx0, wy0, wx1, wy1 = candidate["engine_meta"]["window"]
    assert wx1 - wx0 == wy1 - wy0 < 2400


SEAM_SQUARE = (680, 100, 760, 180)
"""A square inside the overlap of the first two columns of slices of a 1600 by 1200 frame."""


def test_the_detector_slices_a_large_frame_and_merges_its_seams(
    tmp_path: Path,
) -> None:
    """A frame larger than the model's edge runs on the lattice its squares derive, merged.

    The record's edge is the squares' derived edge floored at the processor's ``shortest_edge``,
    its overlap derived, its threshold the stated one; the lattice is called at the frame's size,
    that edge and that overlap where the edge falls short of the frame, and not at all where it
    reaches it; the seam square is proposed, and no two candidates match at or above the threshold.
    """
    from transformers import GroundingDinoProcessor

    from tcip_annotation.matching import iou_matrix
    from tcip_mcp.pipelines.derivations import (
        TILE_EDGE_DERIVATION,
        TILE_OVERLAP_DERIVATION,
    )
    from tests._producer_fixtures import painted_frame

    path = tmp_path / "seams.png"
    painted_frame(
        1600,
        1200,
        (20, 20, 20),
        [
            (SEAM_SQUARE, (235, 235, 235)),
            ((1100, 700, 1180, 780), (235, 235, 235)),
        ],
    ).save(path)

    answer = _live(
        _offline_proposal(
            "grounding_dino",
            str(path),
            {
                **DETECTOR_PARAMS,
                "phrase": "square",
                "box_threshold": 0.25,
                "text_threshold": 0.25,
                "checkpoint": DETECTOR_CHECKPOINT,
            },
        )
    )
    edge = GroundingDinoProcessor.from_pretrained(
        DETECTOR_CHECKPOINT, local_files_only=True
    ).image_processor.size["shortest_edge"]

    record = answer["execution"]
    assert record["tile_size"] >= edge
    assert record["sources"] == {
        "tile_size": "derived"
        if record["tile_size"] == edge
        else TILE_EDGE_DERIVATION,
        "overlap": TILE_OVERLAP_DERIVATION,
        "cross_tile_nms": "explicit",
    }
    assert answer["lattice"] == (
        [[1200, 1600, record["tile_size"], record["overlap"]]]
        if record["tile_size"] < 1600
        else []
    )
    boxes = [c["bbox"] for c in answer["candidates"]]
    assert iou_matrix([SEAM_SQUARE], boxes).max() > 0.5
    overlaps = iou_matrix(boxes, boxes)
    np.fill_diagonal(overlaps, 0)
    assert overlaps.max() < record["cross_tile_nms"]
