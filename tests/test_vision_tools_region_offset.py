"""A region-scoped proposal's geometry, from the engine's crop back to the full frame.

`propose_annotations(grid_cells=...)` hands the engine a crop and translates what comes back
into the source image's own coordinates. Every consumer downstream reads that geometry as
full-frame: the overlay the agent looks at, and the mask `accept_proposals` stages for a
breeder to confirm. A bbox that is translated while its mask rings are not writes a polygon
that sits where no object is, under a bbox that looks right.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

#: The frame, deliberately not square so a swapped or unshifted axis cannot look plausible.
FRAME_W, FRAME_H = 300, 160
#: The one bright patch the engine below finds, at its true full-frame location (x1, y1, x2, y2).
PATCH = (140, 70, 200, 100)
#: Cell edge of the reference grid the region is named in: columns at 0/60/120/180/240,
#: rows at 0/60/120. The patch sits inside cells C2 and D2, whose rect starts at (120, 60).
TILE_SIZE = 60
REGION_CELLS = ["C2", "D2"]


class PatchProposer:
    """A proposal engine that reads its one candidate out of whatever pixels it is handed.

    Its coordinates are therefore the crop's own, produced by the region read rather than
    written into the test, which is what makes the offset step observable.
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
        return [{
            "candidate_id": 0,
            "bbox": [x1, y1, x2, y2],
            "area": int(mask.sum()),
            "score": 0.9,
            "engine": "patch",
            "engine_meta": {},
            "rings": [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]],
        }]


@pytest.fixture
def patched_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dataset image with one red patch at ``PATCH``, and the patch engine wired in."""
    from tcip_mcp.pipelines import proposal

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    images = tmp_path / "images"
    images.mkdir()
    arr = np.full((FRAME_H, FRAME_W, 3), 20, dtype=np.uint8)
    x1, y1, x2, y2 = PATCH
    arr[y1:y2 + 1, x1:x2 + 1] = (255, 0, 0)
    path = images / "region.png"
    Image.fromarray(arr, mode="RGB").save(path)

    monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: PatchProposer())
    return path


def _propose_over_the_region(image: Path) -> dict:
    from tcip_mcp.tools.vision_tools import propose_annotations

    result = propose_annotations(image_path=str(image), engine="patch",
                                 grid_cells=REGION_CELLS, tile_size=TILE_SIZE)
    assert "error" not in result, result
    assert result["candidate_count"] == 1
    return result


def test_accept_refuses_an_image_no_run_staged_proposals_for(patched_frame: Path) -> None:
    """Nothing staged is a refusal naming the tool to run, never an empty acceptance."""
    from tcip_mcp.tools.vision_tools import accept_proposals

    accepted = accept_proposals(image_path=str(patched_frame),
                                assignments=[{"candidate_id": 0, "subject": "leaf"}])
    assert accepted["error"] == "No proposals found for region. Run propose_annotations first."


def test_region_scoped_mask_rings_are_staged_at_their_full_frame_location(
    patched_frame: Path, tmp_path: Path,
) -> None:
    """The mask a breeder confirms covers the object, not the crop-local coordinates the engine
    saw. The region rect starts well inside the frame on both axes, so an untranslated ring
    lands somewhere else entirely."""
    from tcip_annotation import json_io
    from tcip_mcp.tools.vision_tools import accept_proposals

    _propose_over_the_region(patched_frame)
    accepted = accept_proposals(image_path=str(patched_frame),
                                assignments=[{"candidate_id": 0, "subject": "leaf"}])
    assert "error" not in accepted, accepted
    assert accepted["proposal_count"] == 1

    staged = json_io.read_annotations(tmp_path / "predictions" / "patch" / "region.json")
    assert len(staged) == 1
    pts = [p for ring in staged[0].geometry.rings for p in ring]
    assert pts
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert (min(xs), min(ys), max(xs), max(ys)) == PATCH


def test_region_scoped_bbox_and_mask_rings_describe_the_same_place(
    patched_frame: Path, tmp_path: Path,
) -> None:
    """The bbox the tool reports and the rings it caches are two views of one candidate: both
    are read as full-frame, so they must not be translated by different origins."""
    import tcip_store as ts
    from tcip_mcp.tools.vision_tools import _staging_key_for

    result = _propose_over_the_region(patched_frame)
    reported = result["candidates"][0]["bbox"]

    envelope = ts.read(_staging_key_for(str(patched_frame)))
    rings = envelope["candidates"][0]["rings"]
    pts = [p for ring in rings for p in ring]
    assert pts
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    assert [min(xs), min(ys), max(xs), max(ys)] == reported
    assert envelope["region"]["rect"][:2] != [0, 0]   # the region really is offset from the origin
