"""A count assessment's reference holds the floor of ground-truth-bearing samples on each side.

Both count assessment doors, the selection path behind ``assess_checkpoint`` and the
reserved-regions path behind ``assess_reserved_regions``, are driven through their MCP tools over
the platform's own producers: a side short of the floor refuses by name before any run directory
opens, and a side at the floor is assessed. ``draw_splits`` states each reference side's
ground-truth-bearing images and the floor beforehand, and the door counts the same images.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import Selection, Side

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.assessment import ASSESSMENTS_DIR, list_assessments  # noqa: E402
from tcip_mcp.assessment_evidence import INSUFFICIENCY_FAILURES  # noqa: E402
from tests import _trait_fixtures as fx  # noqa: E402
from tests._chain_fixtures import (  # noqa: E402
    IMG,
    STEMS,
    SUBJECT,
    assess,
    object_at,
    trained_detector,
)
from tests._producer_fixtures import label_image, mark_complete  # noqa: E402
from tests._training_values import (  # noqa: E402
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    RATIONALE,
    collection_pass,
    drawn_split,
)
from tests.test_block_calibration import (  # noqa: E402
    BOX_STEP,
    HEIGHT,
    WIDTH,
    _attest_regions_complete,
    _build_experiment,
)

SIDES: tuple[Side, Side] = ("calibration", "holdout")

FLOOR = 2
"""The floor each reference side is held to: a sample standard deviation needs two samples."""

SHORT_SIDES = (("calibration",), ("holdout",), ("calibration", "holdout"))
"""Which reference sides a case leaves short of the floor."""


def _negative(image: Path, project: Path) -> None:
    """Make ``image`` a negative: an empty label document a human marks done with nothing on it."""
    label_image(image, [], IMG, IMG, project=project)
    mark_complete(image, SUBJECT, project=project)


def _selection_of(
    project: Path,
    images_dir: Path,
    drawn: Selection,
    members: dict[str, Side],
    out: Path,
) -> Path:
    """Write the reference selection ``members`` names at ``out``, re-sided by the admission.

    The members are ``drawn``'s, each on the side ``members`` gives it, admitted from
    ``images_dir`` and grouped by the admission's own sample producer under ``drawn``'s policy.
    """
    from tcip_mcp.pipelines.data.label_queries import admit
    from tcip_mcp.pipelines.data.selection import Selection, write_selection
    from tcip_mcp.pipelines.data.splits import recorded_group_key_fn

    admitted = admit(images_dir, scope=drawn.scope)
    samples = admitted.samples(
        members, recorded_group_key_fn(drawn.group_by, date=admitted.date)
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


def _through_the_selection(project: Path, short: tuple[Side, ...]) -> dict:
    """Assess the chain's detector over a reference of two samples per side, ``short`` sides padded.

    Each side takes two of the chain's own drawn reference members; on a ``short`` side the
    second is made a negative, so that side holds one ground-truth-bearing image of two.
    Returns ``assess_checkpoint``'s answer.
    """
    from tcip_mcp.pipelines.data.selection import read_selection

    _root, images_dir, selection_dir, checkpoint = trained_detector(
        project, experiment_id="exp-floor"
    )
    drawn = read_selection(selection_dir, project=project)
    members: dict[str, Side] = {}
    for side in SIDES:
        first, second, *_rest = (s for s in drawn.samples if s.side == side)
        if side in short:
            _negative(Path(second.source), project)
        members |= {first.member: side, second.member: side}
    reference = _selection_of(
        project, images_dir, drawn, members, project / "reference"
    )
    return assess(project, checkpoint, reference)


def _through_the_reserved_regions(
    project: Path, short: tuple[Side, ...]
) -> dict:
    """Assess a mosaic's reserved regions in three bands per side, ``short`` sides thinned.

    On a ``short`` side the mosaic keeps only the boxes centered within its region's first 80
    pixels, all in its first band, so that side holds one ground-truth-bearing band of three.
    Returns ``assess_reserved_regions``' answer.
    """
    from tcip_mcp.tools.calibration_tools import assess_reserved_regions
    from tests._verified_checkpoint_fixtures import SAMPLE_CROSS_TILE_NMS

    fx.seed_confirmed_count(project, measured_subject="bud")
    exp = _build_experiment(project)
    manifest = exp["spatial_manifest"]
    regions = {
        "calibration": manifest.calibration_region[0],
        "holdout": manifest.holdout_region[0],
    }

    def kept(x: int) -> bool:
        center = x + 7.5
        return not any(
            regions[side][0] + 80 <= center < regions[side][2]
            for side in short
        )

    label_image(
        exp["raster_path"],
        [
            Annotation(subject="bud", geometry=BBox(x, 80, x + 15, 110))
            for x in range(10, WIDTH - 20, BOX_STEP)
            if kept(x)
        ],
        WIDTH,
        HEIGHT,
    )
    _attest_regions_complete(
        exp["root"],
        exp["stem"],
        [manifest.calibration_region, manifest.holdout_region],
    )
    return assess_reserved_regions(
        project,
        exp["checkpoint_path"],
        fx.COUNT_TRAIT,
        "per_image_count",
        3 if "calibration" in short else 2,
        3 if "holdout" in short else 2,
        EQUIVALENCE_Z,
        DISPERSION_QUANTILE,
        RATIONALE,
        collection_pass(
            postprocess="nms", cross_tile_nms=SAMPLE_CROSS_TILE_NMS
        ),
        device="cpu",
        actor=None,
    )


DOORS = {
    "selection": _through_the_selection,
    "reserved_regions": _through_the_reserved_regions,
}


def _run_directories(project: Path) -> list[Path]:
    root = project / ASSESSMENTS_DIR
    return sorted(root.iterdir()) if root.is_dir() else []


@pytest.mark.parametrize("short", SHORT_SIDES, ids="+".join)
@pytest.mark.parametrize("door", sorted(DOORS))
def test_a_reference_side_below_the_floor_refuses_by_name_before_any_run_opens(
    tmp_path: Path, door: str, short: tuple[Side, ...]
) -> None:
    """A side holding one ground-truth-bearing sample, padded to two or more, refuses by name.

    The one refusal names every short side, the one sample it holds and the floor, and names
    no side at the floor; no assessment is recorded and no run directory is opened.
    """
    answered = DOORS[door](tmp_path, short)

    refusal = answered.get("error", "")
    named = [side for side in SIDES if f"the {side} side holds 1" in refusal]
    assert (
        named,
        f"fewer than the {FLOOR}" in refusal,
        [d.name for d in _run_directories(tmp_path)],
        list_assessments(tmp_path),
    ) == (list(short), True, [], []), refusal


@pytest.mark.parametrize("door", sorted(DOORS))
def test_a_reference_at_the_floor_on_both_sides_is_assessed_and_recorded(
    tmp_path: Path, door: str
) -> None:
    """Both sides at exactly the floor of ground-truth-bearing samples run end to end.

    The assessment is recorded in its one run directory and names no insufficiency; through the
    selection, whose detector finds every object, it passes.
    """
    answered = DOORS[door](tmp_path, ())

    assert "error" not in answered, answered
    (recorded,) = list_assessments(tmp_path)
    assert [d.name for d in _run_directories(tmp_path)] == [
        recorded.assessment_id
    ]
    assert INSUFFICIENCY_FAILURES.isdisjoint(recorded.failures), (
        recorded.failures
    )
    if door == "selection":
        assert recorded.passed, recorded.failures


def test_the_draw_states_each_reference_sides_bearing_images_the_door_counts(
    tmp_path: Path,
) -> None:
    """``draw_splits`` states the ground-truth-bearing images per reference side the door counts.

    Over a capture of eight images bearing two annotations each among negatives, the draw
    leaves the calibration side one bearing image (two annotations) and the holdout side two
    bearing images beside negatives, and answers so without refusing. The assessment of that
    selection refuses the calibration side by the one image the draw states and does not name
    the holdout side, which holds the floor and a negative.
    """
    from tcip_mcp.tools.data_tools import draw_splits

    root, images_dir, _selection, checkpoint = trained_detector(
        tmp_path, experiment_id="exp-floor-draw"
    )
    for index, stem in enumerate(STEMS):
        image = images_dir / f"{stem}.png"
        if index >= 8:
            _negative(image, tmp_path)
            continue
        x0, y0, size = object_at(index)
        label_image(
            image,
            [
                Annotation(
                    subject=SUBJECT,
                    geometry=BBox(x0, y0, x0 + size, y0 + size),
                ),
                Annotation(subject=SUBJECT, geometry=BBox(0, 0, 3, 3)),
            ],
            IMG,
            IMG,
            project=tmp_path,
        )
    redrawn = tmp_path / "redrawn"
    drawn = draw_splits(
        tmp_path,
        str(root),
        output_path=str(redrawn),
        subject=SUBJECT,
        **drawn_split(
            seed=2,
            val_ratio=0.2,
            calibration_ratio=0.1,
            holdout_ratio=0.3,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in drawn, drawn
    assert drawn["reference_floor"] == FLOOR
    assert drawn["foreground_images"] == {"calibration": 1, "holdout": 2}, (
        drawn
    )
    assert drawn["foreground_annotations"]["calibration"] > 1, drawn
    assert drawn["splits"]["holdout"] > 2, drawn

    answered = assess(tmp_path, checkpoint, redrawn)

    refusal = answered.get("error", "")
    named = [side for side in SIDES if f"the {side} side holds" in refusal]
    assert (named, "the calibration side holds 1 " in refusal) == (
        ["calibration"],
        True,
    ), answered
