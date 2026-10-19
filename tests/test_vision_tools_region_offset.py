"""A region-scoped proposal's geometry, from the engine's crop back to the full frame.

``propose_annotations(grid_cells=...)`` hands the engine a crop and translates what comes back
into the source image's own coordinates: the reported bbox, the cached rings and the mask
``stage_proposals`` stages all sit in the full frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.dataset_layout import UNDATED_BUCKET

#: The frame, deliberately not square so a swapped or unshifted axis cannot look plausible.
FRAME_W, FRAME_H = 300, 160
#: The one bright patch the engine below finds, at its true full-frame location (x1, y1, x2, y2).
PATCH = (140, 70, 200, 100)
#: Cell edge of the reference grid the region is named in: columns at 0/60/120/180/240,
#: rows at 0/60/120. The patch sits inside cells C2 and D2, whose rect starts at (120, 60).
TILE_SIZE = 60
REGION_CELLS = ["C2", "D2"]


class PatchProposer:
    """A proposal engine answering one candidate over the pixels with R > 200, G < 60, B < 60.

    The candidate's bbox and ring are those pixels' extrema in the image it opens; it answers
    no candidate when no pixel passes.
    """

    def propose(self, image_path: str, **params: object) -> list[dict]:
        with Image.open(image_path) as im:
            arr = np.asarray(im.convert("RGB"))
        mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        return [
            {
                "candidate_id": 0,
                "bbox": [x1, y1, x2, y2],
                "area": int(mask.sum()),
                "score": 0.9,
                "engine": "patch",
                "engine_meta": {},
                "rings": [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]],
            }
        ]


@pytest.fixture
def patched_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a dataset image with one red patch at ``PATCH`` and wire in the patch engine."""
    from tcip_mcp.pipelines import proposal
    from tests._producer_fixtures import painted_frame

    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    x1, y1, x2, y2 = PATCH
    path = images / "region.png"
    # The patch engine reports inclusive extents, so the patch covers its far edge too.
    painted_frame(
        FRAME_W,
        FRAME_H,
        (20, 20, 20),
        [((x1, y1, x2 + 1, y2 + 1), (255, 0, 0))],
    ).save(path)

    monkeypatch.setattr(
        proposal, "resolve_proposer", lambda engine: PatchProposer()
    )
    return path


def _propose_over_the_region(project: Path, image: Path) -> dict:
    from tcip_mcp.tools.proposal_tools import propose_annotations

    result = propose_annotations(
        project,
        image_path=str(image),
        engine="patch",
        grid_cells=REGION_CELLS,
        tile_size=TILE_SIZE,
        overlap=0.0,
    )
    assert "error" not in result, result
    assert result["candidate_count"] == 1
    return result


def test_accept_refuses_an_image_no_run_staged_proposals_for(
    patched_frame: Path,
    tmp_path: Path,
) -> None:
    """Staging for an image with no proposals is refused, naming the tool to run first."""
    from tcip_mcp.tools.proposal_tools import stage_proposals

    accepted = stage_proposals(
        tmp_path,
        image_path=str(patched_frame),
        assignments=[{"candidate_id": 0, "subject": "leaf"}],
    )
    assert (
        accepted["error"]
        == "No proposals found for region. Run propose_annotations first."
    )


def test_region_scoped_mask_rings_are_staged_at_their_full_frame_location(
    patched_frame: Path,
    tmp_path: Path,
) -> None:
    """Region-scoped mask rings are staged at their full-frame location.

    The mask a breeder confirms covers the object, not the crop-local coordinates the engine
    saw. The region rect starts well inside the frame on both axes, and the staged ring's
    bounds equal the patch's full-frame extent.
    """
    from tcip_annotation import json_io
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import parse_image_path
    from tcip_mcp.tools.proposal_tools import stage_proposals

    _propose_over_the_region(tmp_path, patched_frame)
    accepted = stage_proposals(
        tmp_path,
        image_path=str(patched_frame),
        assignments=[{"candidate_id": 0, "subject": "leaf"}],
    )
    assert "error" not in accepted, accepted
    assert accepted["proposal_count"] == 1

    bucket = read_bucket(
        parse_image_path(patched_frame)[0], accepted["bucket"]
    )
    staged = json_io.read_label_document(
        bucket.document_key(patched_frame.stem)
    ).annotations
    assert len(staged) == 1
    pts = [p for ring in staged[0].geometry.rings for p in ring]
    assert pts
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert (min(xs), min(ys), max(xs), max(ys)) == PATCH


def test_region_scoped_bbox_and_mask_rings_describe_the_same_place(
    patched_frame: Path,
    tmp_path: Path,
) -> None:
    """A region-scoped candidate's bbox and mask rings describe the same place.

    The bbox the tool reports and the rings it caches are two views of one candidate: over a
    region offset from the origin, the rings' bounds equal the reported bbox.
    """
    import tcip_store as ts
    from tcip_mcp.tools.proposal_tools import (
        ProposalStaging,
        _staging_key_for,
    )

    result = _propose_over_the_region(tmp_path, patched_frame)
    reported = result["candidates"][0]["bbox"]

    staging = ProposalStaging.model_validate(
        ts.read(_staging_key_for(str(patched_frame)))
    )
    rings = staging.candidates[0].rings
    pts = [p for ring in rings for p in ring]
    assert pts
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    assert [min(xs), min(ys), max(xs), max(ys)] == reported
