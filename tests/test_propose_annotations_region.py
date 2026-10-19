"""Region-scoped `propose_annotations`: crop-and-offset, entirely on the tcip-mcp side.

`propose_annotations` takes an optional `grid_cells`/`tile_size`/`overlap` region: when given, the
tool crops the source image to the named cells' bounding rect, runs the engine on that crop, and
offsets the returned candidates back to the source image's full-frame coordinates. The engine here
reports a red patch in the pixels it is handed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcip_annotation.state import RING_MIN_POINTS
from tcip_mcp.dataset_layout import UNDATED_BUCKET


class PatchProposer:
    """A proposer that reports the red patch in the pixels it is handed."""

    def propose(self, image_path: str, **params: object) -> list[dict]:
        arr = np.asarray(Image.open(image_path).convert("RGB"))
        mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)
        ys, xs = np.nonzero(mask)
        bx1, by1 = float(xs.min()), float(ys.min())
        bx2, by2 = float(xs.max() + 1), float(ys.max() + 1)
        return [
            {
                "candidate_id": 0,
                "bbox": [bx1, by1, bx2, by2],
                "area": int(mask.sum()),
                "score": 0.9,
                "engine": "patch",
                "engine_meta": {},
                "rings": [[(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]],
            }
        ]


#: The upright (as-viewed) frame this fixture draws the patch against.
UPRIGHT_W, UPRIGHT_H = 240, 120
#: The patch's known full-frame location in the upright frame: x1, y1, x2, y2.
PATCH_BOX = (80, 10, 160, 50)
TILE_SIZE = 60  # cols at x=0/60/120/180 (4), rows at y=0/60 (2); PATCH_BOX spans cols B/C, row 1.


def _upright_frame() -> Image.Image:
    """Return the as-viewed canvas: a dark background with a solid red patch at ``PATCH_BOX``."""
    from tests._producer_fixtures import painted_frame

    return painted_frame(
        UPRIGHT_W, UPRIGHT_H, (20, 20, 20), [(PATCH_BOX, (255, 0, 0))]
    )


@pytest.fixture
def exif_rotated_source(tmp_path: Path) -> Path:
    """Write a JPEG of the upright frame rotated 90 degrees and tagged EXIF orientation 6."""
    upright = _upright_frame()
    raw = upright.rotate(
        90, expand=True
    )  # inverse of orientation 6's "rotate 270" correction

    path = tmp_path / "source_exif6.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation
    raw.save(path, exif=exif.tobytes())
    return path


def test_fixture_rotation_is_actually_reversed_by_exif_orientation(
    exif_rotated_source: Path,
) -> None:
    """The fixture's rotation is reversed by its EXIF orientation.

    Opening the raw file and applying ``auto_orient_image`` gives the upright frame's shape, its
    dark background and its red patch.
    """
    from tcip_annotation.utils import auto_orient_image

    with Image.open(exif_rotated_source) as im:
        corrected = np.asarray(auto_orient_image(im).convert("RGB"))
    assert corrected.shape == (UPRIGHT_H, UPRIGHT_W, 3)
    # JPEG's DCT ringing near a hard edge can swing tens of levels, so sample a background corner
    # far from any edge and the patch interior rather than compare the full array pixel-for-pixel.
    x1, y1, x2, y2 = PATCH_BOX
    assert all(
        abs(int(c) - 20) <= 15 for c in corrected[5, 5]
    )  # far corner, upright's placement
    r, g, b = (int(c) for c in corrected[(y1 + y2) // 2, (x1 + x2) // 2])
    assert r > 200 and g < 50 and b < 50  # patch interior, still red


def test_region_crop_carries_no_exif_orientation_tag(
    exif_rotated_source: Path,
    tmp_path: Path,
) -> None:
    """The region crop `propose_annotations` writes for the engine carries no EXIF orientation tag.

    The source opens at the upright frame's size, and the crop written from it has no orientation
    tag.
    """
    from tcip_mcp.pipelines.raster_source import open_raster
    from tcip_mcp.pipelines.reference_grid import reference_cells
    from tcip_mcp.tools.proposal_tools import (
        _region_rect_from_cells,
        _write_region_crop,
    )

    with open_raster(exif_rotated_source, 3) as src:
        assert (src.width, src.height) == (UPRIGHT_W, UPRIGHT_H)
        cells = reference_cells(
            src.width, src.height, TILE_SIZE, 0.0, clamp=True
        )
        rect = _region_rect_from_cells(cells, ["B1", "C1"])
        pixels, _spec = src.read_region(rect)

    crop_path = _write_region_crop(pixels)
    try:
        with Image.open(crop_path) as im:
            exif = im.getexif()
            assert 0x0112 not in exif, (
                f"crop unexpectedly carries an EXIF orientation tag: {exif.get(0x0112)}"
            )
    finally:
        crop_path.unlink(missing_ok=True)


def test_region_scoped_proposal_lands_at_the_full_frame_coordinates(
    exif_rotated_source: Path,
    tmp_path: Path,
) -> None:
    """A region-scoped proposal lands at the full-frame coordinates.

    An EXIF-rotated source, cropped to a region, proposed against, and offset back, reports one
    candidate at the patch's upright full-frame box, and no temp region crop is left. The same
    call naming no overlap refuses naming it.
    """
    from tcip_mcp.pipelines.proposal import register_proposal_engine
    from tcip_mcp.tools.proposal_tools import propose_annotations

    register_proposal_engine("patch", PatchProposer())

    region = {
        "image_path": str(exif_rotated_source),
        "engine": "patch",
        "grid_cells": ["B1", "C1"],
        "tile_size": TILE_SIZE,
    }
    refused = propose_annotations(tmp_path, **region)
    assert "overlap" in refused.get("error", ""), refused
    result = propose_annotations(tmp_path, **region, overlap=0.0)
    assert "error" not in result, result
    assert result["candidate_count"] == 1
    bbox = result["candidates"][0]["bbox"]
    assert bbox == [float(v) for v in PATCH_BOX]

    # No leftover temp crop file: cleanup runs even on the success path.
    tmp_dir = Path(tempfile_gettempdir())
    leftover = list(tmp_dir.glob("tcip_propose_crop_*"))
    assert leftover == [], f"temp region crop(s) not cleaned up: {leftover}"


def tempfile_gettempdir() -> str:
    import tempfile

    return tempfile.gettempdir()


def test_region_scoped_proposal_cleans_up_temp_crop_on_engine_failure(
    exif_rotated_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The temp region crop is deleted when the engine raises."""
    from tcip_mcp.pipelines import proposal
    from tcip_mcp.tools.proposal_tools import propose_annotations

    class BoomProposer:
        def propose(self, image_path: str, **params: object) -> list[dict]:
            raise RuntimeError("engine exploded")

    monkeypatch.setattr(
        proposal, "resolve_proposer", lambda engine: BoomProposer()
    )

    before = set(Path(tempfile_gettempdir()).glob("tcip_propose_crop_*"))
    with pytest.raises(RuntimeError, match="engine exploded"):
        propose_annotations(
            tmp_path,
            image_path=str(exif_rotated_source),
            engine="boom",
            grid_cells=["B1", "C1"],
            tile_size=TILE_SIZE,
            overlap=0.0,
        )
    after = set(Path(tempfile_gettempdir()).glob("tcip_propose_crop_*"))
    assert after - before == set(), (
        "temp region crop leaked after the engine raised"
    )


def _region_cells() -> list:
    """Return the fixture frame's cells at ``TILE_SIZE``: columns A-D, rows 1-2."""
    from tcip_mcp.pipelines.reference_grid import reference_cells

    return reference_cells(UPRIGHT_W, UPRIGHT_H, TILE_SIZE, 0.0, clamp=True)


class TestRegionCellNamesResolveThroughTheOneLookup:
    """Tests that a region's cell names resolve to the named cells' rect, or refuse."""

    def test_named_cells_bound_their_combined_rect(self) -> None:
        from tcip_mcp.tools.proposal_tools import _region_rect_from_cells

        rect = _region_rect_from_cells(_region_cells(), ["B1", "C1"])
        assert (rect.x0, rect.y0, rect.x1, rect.y1) == (60, 0, 180, 60)

    def test_names_are_case_insensitive_and_whitespace_stripped(self) -> None:
        from tcip_mcp.tools.proposal_tools import _region_rect_from_cells

        cells = _region_cells()
        assert _region_rect_from_cells(
            cells, [" b1 ", "c1"]
        ) == _region_rect_from_cells(cells, ["B1", "C1"])

    def test_a_reference_that_is_not_a_cell_name_is_refused(self) -> None:
        from tcip_mcp.tools.proposal_tools import _region_rect_from_cells

        with pytest.raises(ValueError, match="Invalid cell reference"):
            _region_rect_from_cells(_region_cells(), ["3B"])

    def test_a_name_outside_the_grid_names_the_valid_range(self) -> None:
        from tcip_mcp.tools.proposal_tools import _region_rect_from_cells

        with pytest.raises(ValueError, match="Use A1 through D2"):
            _region_rect_from_cells(_region_cells(), ["Z9"])


def _write_rgb_band_group(
    images_dir: Path, stem: str, frame: np.ndarray
) -> Path:
    """Write a ``.bandgroup`` capture of ``frame``'s R, G and B planes and return its manifest.

    Three sibling single-band TIFFs hold the planes, the manifest declaring them in that order.
    """
    import tifffile

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    bands = {}
    for i, name in enumerate(("Red", "Green", "Blue")):
        band_path = images_dir / f"{stem}_{name}.tif"
        tifffile.imwrite(str(band_path), frame[:, :, i])
        bands[name] = band_path
    return write_band_group_manifest(
        images_dir,
        stem,
        bands,
        central_wavelength_nm=None,
        identity_tolerance={},
    )


class TestRegionFrameIsTheResolvedImageSources:
    """Tests that the region path measures and crops the logical image enumeration resolves."""

    def test_a_grouped_capture_crops_its_own_stacked_frame(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A grouped capture's region cells and crop come from its own stacked frame.

        A ``.bandgroup`` manifest names a capture whose pixels live in sibling files; the patch is
        found at its full-frame place in that stacked frame.
        """
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import propose_annotations

        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        manifest = _write_rgb_band_group(
            images_dir, "capture_001", np.asarray(_upright_frame())
        )

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: PatchProposer()
        )

        result = propose_annotations(
            tmp_path,
            image_path=str(manifest),
            engine="patch",
            grid_cells=["B1", "C1"],
            tile_size=TILE_SIZE,
            overlap=0.0,
        )
        assert "error" not in result, result
        assert result["candidates"][0]["bbox"] == [float(v) for v in PATCH_BOX]

    def test_a_source_with_no_rgb_frame_is_refused_before_the_engine_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A source with no RGB frame is refused before the engine runs.

        A two-band uint16 group refuses saying it reads as 2 band(s) of uint16.
        """
        import tifffile

        from tcip_mcp.pipelines import proposal
        from tcip_mcp.pipelines.data.band_groups import (
            write_band_group_manifest,
        )
        from tcip_mcp.tools.proposal_tools import propose_annotations

        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        bands = {}
        for name, fill in (("Green", 111), ("Red", 222)):
            band_path = images_dir / f"cap_{name}.tif"
            tifffile.imwrite(
                str(band_path),
                np.full((UPRIGHT_H, UPRIGHT_W), fill, dtype=np.uint16),
            )
            bands[name] = band_path
        manifest = write_band_group_manifest(
            images_dir,
            "cap",
            bands,
            central_wavelength_nm=None,
            identity_tolerance={},
        )

        class NeverProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                raise AssertionError(
                    "the engine ran on a source with no RGB frame"
                )

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: NeverProposer()
        )

        result = propose_annotations(
            tmp_path,
            image_path=str(manifest),
            engine="patch",
            grid_cells=["B1"],
            tile_size=TILE_SIZE,
            overlap=0.0,
        )
        assert "2 band(s) of uint16" in result.get("error", "")

    def test_a_crop_is_read_in_the_frame_its_cells_were_measured_in(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A crop is read in the frame its cells were measured in.

        For an RGB TIFF one pixel high and far wider than it has bands, its cells are measured in
        the frame its header's axes describe, the crop is read in that same frame, and the engine
        receives an RGB crop of the named cell.
        """
        import tifffile

        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import propose_annotations

        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        path = images_dir / "rgb_row.tif"
        tifffile.imwrite(
            str(path),
            np.tile(np.array([20, 100, 220], dtype=np.uint8), (1, 200_000, 1)),
            photometric="rgb",
        )

        crops: list[tuple[str, tuple[int, int]]] = []

        class RecordingProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                with Image.open(image_path) as im:
                    crops.append((im.mode, im.size))
                return [
                    {
                        "candidate_id": 0,
                        "bbox": [1.0, 0.0, 3.0, 1.0],
                        "area": 2,
                        "score": 0.9,
                        "engine": "patch",
                        "engine_meta": {},
                        "rings": [[(1, 0), (3, 0), (3, 1), (1, 1)]],
                    }
                ]

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: RecordingProposer()
        )

        result = propose_annotations(
            tmp_path,
            image_path=str(path),
            engine="patch",
            grid_cells=["A1"],
            tile_size=1024,
            overlap=0.0,
        )
        assert "error" not in result, result
        assert crops == [("RGB", (1024, 1))]
        assert result["candidates"][0]["bbox"] == [1.0, 0.0, 3.0, 1.0]


class TestWholeFrameDefaultIsUnaffected:
    """Tests that ``grid_cells=None`` (the default) takes the whole-frame path, no offset step."""

    def test_offset_helper_is_never_called_without_grid_cells(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools import proposal_tools

        called = []

        def recording_offset_candidates(
            candidates: list[dict], origin: tuple[float, float]
        ) -> list[dict]:
            called.append((candidates, origin))
            return candidates

        monkeypatch.setattr(
            proposal_tools, "_offset_candidates", recording_offset_candidates
        )

        class OneBoxProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return [
                    {
                        "candidate_id": 0,
                        "bbox": [10.0, 10.0, 30.0, 30.0],
                        "area": 400,
                        "score": 0.9,
                        "engine": "stub",
                        "engine_meta": {},
                        "rings": [[(10, 10), (30, 10), (30, 30), (10, 30)]],
                    }
                ]

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: OneBoxProposer()
        )

        img_path = tmp_path / "whole_frame.jpg"
        Image.new("RGB", (64, 64), color=(50, 50, 50)).save(img_path)

        result = proposal_tools.propose_annotations(
            tmp_path, image_path=str(img_path), engine="stub"
        )
        assert "error" not in result, result
        assert called == []
        assert result["candidates"][0]["bbox"] == [10.0, 10.0, 30.0, 30.0]

    def test_state_envelope_has_no_region_key_without_grid_cells(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import tcip_store as ts
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools import proposal_tools

        class OneBoxProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return [
                    {
                        "candidate_id": 0,
                        "bbox": [1.0, 1.0, 2.0, 2.0],
                        "area": 1,
                        "score": 0.5,
                        "engine": "stub",
                        "engine_meta": {},
                        "rings": [[(1, 1), (2, 1), (2, 2)]],
                    }
                ]

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: OneBoxProposer()
        )

        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        img_path = images_dir / "no_region.jpg"
        Image.new("RGB", (32, 32), color=(10, 10, 10)).save(img_path)

        result = proposal_tools.propose_annotations(
            tmp_path, image_path=str(img_path), engine="stub"
        )
        assert "error" not in result, result

        envelope = ts.read(proposal_tools._staging_key_for(str(img_path)))
        assert set(envelope) == {"engine", "candidates", "image_identity"}

    def test_a_candidate_whose_rings_are_no_polygon_is_refused_where_it_arrives(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A candidate whose rings are no polygon is refused by name where it arrives.

        An engine's rings are held to the polygon rule where they arrive: a ring too short to be a
        shape refuses the proposal, naming the minimum point count, and nothing is staged.
        """
        import tcip_store as ts
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools import proposal_tools

        class TwoPointProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return [
                    {
                        "candidate_id": 0,
                        "bbox": [1.0, 1.0, 2.0, 2.0],
                        "area": 1,
                        "score": 0.5,
                        "engine": "stub",
                        "engine_meta": {},
                        "rings": [[(1, 1), (2, 2)]],
                    }
                ]

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: TwoPointProposer()
        )
        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        img_path = images_dir / "short_ring.jpg"
        Image.new("RGB", (32, 32), color=(10, 10, 10)).save(img_path)

        result = proposal_tools.propose_annotations(
            tmp_path, image_path=str(img_path), engine="stub"
        )

        assert f"{RING_MIN_POINTS} or more points" in result.get(
            "error", ""
        ), result
        assert (
            ts.read(
                proposal_tools._staging_key_for(str(img_path)), default=None
            )
            is None
        )

    def test_an_ordinary_candidate_is_still_staged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A working engine's ordinary candidate is staged."""
        import tcip_store as ts
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools import proposal_tools

        class OneBoxProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return [
                    {
                        "candidate_id": 0,
                        "bbox": [1.0, 1.0, 2.0, 2.0],
                        "area": 1,
                        "score": 0.5,
                        "engine": "stub",
                        "engine_meta": {},
                        "rings": [[(1, 1), (2, 1), (2, 2)]],
                    }
                ]

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: OneBoxProposer()
        )

        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        img_path = images_dir / "storable.jpg"
        Image.new("RGB", (32, 32), color=(10, 10, 10)).save(img_path)

        result = proposal_tools.propose_annotations(
            tmp_path, image_path=str(img_path), engine="stub"
        )

        assert "error" not in result, result
        envelope = proposal_tools.ProposalStaging.model_validate(
            ts.read(proposal_tools._staging_key_for(str(img_path)))
        )
        assert envelope.candidates[0].bbox == (1.0, 1.0, 2.0, 2.0)

    def test_a_candidate_stating_no_confidence_is_refused_where_it_arrives(
        self,
        tmp_path: Path,
    ) -> None:
        """A candidate stating no confidence is refused by name where it arrives, and not staged."""
        import tcip_store as ts
        from tcip_mcp.pipelines.proposal import register_proposal_engine
        from tcip_mcp.tools import proposal_tools

        class UnscoredProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return [
                    {
                        "candidate_id": 0,
                        "bbox": [1.0, 1.0, 2.0, 2.0],
                        "area": 1,
                        "engine": "unscored",
                        "engine_meta": {},
                        "rings": [[(1, 1), (2, 1), (2, 2)]],
                    }
                ]

        register_proposal_engine("unscored", UnscoredProposer())  # type: ignore[arg-type]
        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        img_path = images_dir / "unscored.jpg"
        Image.new("RGB", (32, 32), color=(10, 10, 10)).save(img_path)

        result = proposal_tools.propose_annotations(
            tmp_path, image_path=str(img_path), engine="unscored"
        )

        assert "0.score" in result.get("error", ""), result
        assert (
            ts.read(
                proposal_tools._staging_key_for(str(img_path)), default=None
            )
            is None
        )
