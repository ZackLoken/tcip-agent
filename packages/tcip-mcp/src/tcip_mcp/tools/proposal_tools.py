"""Proposal-workflow tools: turn a chosen auto-labeling engine's output into predictions for
canvas review.

propose_annotations and segment_prompt each ask an engine (the built-in SAM reference, or a
bespoke 'module:factory' the agent brings) to look at pixels and offer candidates or a prompted
mask. stage_proposals lands either an engine's reviewed candidates or explicit boxes/polygons in
the predictions tree through one verdict-guarded staging door, for a human to accept, reject or
edit on the Review canvas. It never writes ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import tcip_store as ts
from tcip_store.file_backend import RootedFileLocator

from tcip_annotation import Annotation, BBox, Polygon
from tcip_annotation.json_io import stored_box_extent_ok
from tcip_annotation.sam_wrapper import grid_to_rect
from tcip_annotation.viz import render_candidates, render_detections

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.image_utils import (
    BandGroupIncomplete, image_dimensions, resolve_image_source,
)
from tcip_mcp.server import mcp

if TYPE_CHECKING:
    import numpy as np

    from tcip_mcp.pipelines.raster_source import Rect


_PROPOSAL_DOC = RootedFileLocator(prefix=(".tcip", "state", "proposals"), suffix=".json")
"""The proposal envelope one dataset image's run stages, under the dataset's own state tree."""

PROPOSAL_STAGING_STORE = "proposal_staging"
ts.register_store(
    ts.StoreDescriptor(
        name=PROPOSAL_STAGING_STORE,
        kind="record",
        key_fields=("date", "stem"),
        frozen=True,
        codec=ts.RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_PROPOSAL_DOC,
    )
)


def proposal_staging_key(dataset_root: str | Path, date: str | None, stem: str) -> ts.Key:
    """The proposals one run staged for one dataset image, for ``stage_proposals``'s assignments
    regime to read back.

    ``last_writer_wins``: a run writes the whole envelope from the candidates it just
    produced, so a re-run replaces the previous one rather than merging into it. Scoped to the
    dataset root, the same as the labels and predictions the proposals eventually become: a
    same-named image in another dataset, or another date bucket of this one, addresses its own
    record. ``date`` is the image's own capture-date bucket, or ``None`` for a flat dataset's
    undated layout, addressed under ``dataset_layout.UNDATED_BUCKET``: a store key holds no empty
    part, so the missing date needs the same declared token ``ingest_images`` buckets a dateless
    source under, rather than a spelling of its own. A flat-layout image and an image in that
    literal bucket therefore share one key for a given stem, which is never a real collision:
    ``ingest_images`` never produces a flat layout beside a dated one in the same dataset.
    """
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    return ts.Key(PROPOSAL_STAGING_STORE, str(dataset_root), (date or UNDATED_BUCKET, stem))


class StagingAddress(NamedTuple):
    """The :func:`proposal_staging_key` for a dataset image, plus the dataset root and date
    :func:`~tcip_mcp.dataset_layout.parse_image_path` derived to reach it.

    ``stage_proposals`` needs the root and date too, whichever regime it runs, to stage the
    predictions at the same address; carrying them here means that address is derived once, not
    twice.
    """

    key: ts.Key
    root: Path
    date: str | None


def _staging_key_for(image_path: str) -> StagingAddress:
    """The :class:`StagingAddress` for the dataset image at ``image_path``.

    Runs :func:`~tcip_mcp.dataset_layout.parse_image_path` once, so ``propose_annotations`` and
    ``stage_proposals`` never derive two different addresses for the same image. Raises
    ``ValueError``, the resolver's own message, for a path outside any dataset's ``images/`` tree.
    """
    from tcip_mcp.dataset_layout import parse_image_path

    root, date, stem = parse_image_path(image_path)
    return StagingAddress(proposal_staging_key(root, date, stem), root, date)


def _unresolvable_staging_source(img: Path, exc: Exception) -> str:
    """A reason for ``propose_annotations`` to decline staging ``img``, when
    :func:`~tcip_mcp.pipelines.image_utils.resolve_image_source` raised ``exc`` for it (the same
    call ``stage_proposals``'s assignments regime will make on this path).

    A band-group member's own path (``capture_Red.tif`` when ``capture.bandgroup`` claims it)
    resolves to nothing: the resolver's own ``FileNotFoundError`` for it reads the same as one for
    a stem that names no image at all, "no image for stem". This names the manifest that claims
    it instead, so the refusal points at the path to propose on rather than repeating a generic
    not-found. ``BandGroupIncomplete`` (a manifest that resolves but is missing a sibling) already
    carries its own manifest-naming message and is returned unchanged.
    """
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.image_utils import BandGroupIncomplete, list_logical_images

    if isinstance(exc, BandGroupIncomplete):
        return str(exc)
    for source in list_logical_images(img.parent).values():
        if isinstance(source, BandGroupRef) and img in source.bands.values():
            return (f"{img} is one band of the group {source.manifest_path.name!r}; propose "
                    f"on {source.manifest_path} instead.")
    return str(exc)


def _region_rect_from_cells(cells: list, names: list[str]) -> "Rect":
    """The bounding rect, in the grid's native-pixel frame, of the named reference-grid cells.

    A region-scoped proposal pass needs one rectangle to crop, not the point prompts
    ``segment_prompt`` turns grid cells into: each name resolves through the one cell lookup
    (``sam_wrapper.grid_to_rect``, so a malformed or out-of-grid name is refused here exactly as it
    is for a point prompt) and the matched cells union to their combined bounding box.
    """
    from tcip_mcp.pipelines.raster_source import Rect

    matched = [grid_to_rect(name, cells) for name in names]
    return Rect(int(min(r[0] for r in matched)), int(min(r[1] for r in matched)),
                int(max(r[2] for r in matched)), int(max(r[3] for r in matched)))


def _write_region_crop(pixels: "np.ndarray") -> Path:
    """Save an RGB region crop to a fresh temp PNG file; the caller deletes it once the engine has
    read it.

    The crop is taken from the raster layer's already-``auto_orient_image``'d frame (a photographic
    source is EXIF-oriented on decode), so it carries no EXIF orientation tag once saved (a PIL
    ``.save()`` never re-emits an orientation tag it didn't read from a source file):
    ``auto_mask``'s own internal re-orientation call is a no-op against this file, not a second,
    wrong rotation of pixels that are already upright.
    """
    import os
    import tempfile

    from PIL import Image

    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="tcip_propose_crop_")
    os.close(fd)
    Image.fromarray(pixels, mode="RGB").save(tmp)
    return Path(tmp)


def _offset_candidates(candidates: list[dict], origin: tuple[float, float]) -> list[dict]:
    """Candidates proposed against a region crop's own pixels, translated into the source image's
    full-frame native coordinates by the crop's own origin.

    Both consumers downstream (``render_candidates``, and ``stage_proposals``'s assignments
    regime reading the cached envelope back later) expect ``bbox``/``rings`` in the source
    image's native frame, never crop-local pixels, so this runs before either sees the
    candidates.
    """
    ox, oy = origin
    shifted = []
    for c in candidates:
        c = dict(c)
        x1, y1, x2, y2 = c["bbox"]
        c["bbox"] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
        c["rings"] = [[(x + ox, y + oy) for x, y in ring] for ring in c["rings"]]
        shifted.append(c)
    return shifted


@mcp.tool()
@audited(scope_arg="image_path")
def propose_annotations(
    image_path: str,
    engine: str = "sam",
    engine_params: dict | None = None,
    grid_cells: list[str] | None = None,
    tile_size: int | None = None,
    overlap: float = 0.0,
) -> dict:
    """Propose candidate annotations on an image for review, using a chosen auto-labeling engine.

    Runs the engine's whole-image proposal pass, renders the numbered candidates, and returns the
    render path and neutral candidate data. Read the render with your own image-capable read
    tool, then call stage_proposals with subject assignments to stage the accepted ones as
    predictions.

    Each candidate renders as a colored, semi-transparent filled polygon (every ring of an
    occlusion-split candidate drawn, not just the largest) with a large numbered label at its
    centroid, colors cycling through the shared class palette; the candidate id in that number is
    the same id ``stage_proposals``' ``assignments`` parameter names.

    On an image under a dataset's ``images/`` tree, the candidates are staged keyed by the
    dataset, capture date and stem, alongside the content identity of the pixels the engine ran
    on: ``stage_proposals``'s assignments regime reads the record back by that same address and
    refuses if the image's content no longer matches it. On a path outside any dataset's
    ``images/`` tree, or a dataset path that regime would itself fail to resolve (a band-group
    member's own path when its manifest claims it), the engine still runs and the render and
    candidates are returned the same way, but nothing is staged (the response's ``staged`` is
    ``false``, naming why): there is no address the assignments regime could ever read the
    record back by, so such a call cannot later be accepted.

    The engine is a capability, not a fixed method: 'sam' is the built-in SAM2 reference; the agent
    can register another engine (``register_proposal_engine``) or pass a dotted 'module:factory' it
    wrote, then trial and compare engines by how well each one's high-conf proposals survive breeder
    review, and pick the most useful for the task.

    ``grid_cells`` restricts the pass to a region instead of the whole frame: name the reference-
    grid cells the region spans (e.g. ``['B3', 'C3', 'B4', 'C4']``, the same grid
    ``overlay_reference_grid``/``segment_prompt`` use), and the engine proposes only over their
    bounding rect. Useful on a large or crowded frame where a whole-image pass returns too many or
    too coarse candidates to review, or where only part of the frame matters right now. The crop is
    taken and the results offset back to full-frame coordinates entirely on this side of the engine
    seam: the engine is handed an ordinary (if smaller) image and never told a region was involved,
    so a bespoke engine gets region support with no code of its own. The one real caveat: an engine
    that keys behavior off the image path itself (a cache, a sidecar lookup keyed by the original
    file) receives the temp crop's path, which it cannot resolve back to the source image. Omitting
    ``grid_cells`` runs the whole frame, unchanged.

    Args:
        image_path: Absolute path to the image file.
        engine: Proposal engine: 'sam' (built-in) or a dotted 'module:factory' the agent brings.
        engine_params: Engine-specific knobs forwarded to the engine (e.g. SAM's model_type,
            points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area). Omit for
            the engine's own defaults.
        grid_cells: Reference-grid cell names bounding the region to propose over (e.g.
            ['B3', 'D5']); the engine sees the bounding rect of the named cells, not the whole
            frame. Requires ``tile_size``. Omit for the whole frame.
        tile_size: Cell edge, in native pixels, of the grid the cells were read off. Required with
            ``grid_cells``.
        overlap: Overlap fraction of the grid the cells were read off, ``segment_prompt``'s same
            semantics.
    """
    from tcip_mcp.pipelines.proposal import resolve_proposer

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        proposer = resolve_proposer(engine)
    except (ValueError, ImportError) as e:
        return {"error": str(e)}

    # A region is cropped and offset entirely here, before the engine ever sees an image path.
    # grid_cells=None skips this branch, taking the exact whole-frame path below, unchanged.
    propose_path = image_path
    crop_tmp: Path | None = None
    origin = (0.0, 0.0)
    region_info: dict | None = None
    if grid_cells is not None:
        if not grid_cells:
            return {"error": "grid_cells is empty; name at least one cell to scope the region."}
        if tile_size is None:
            return {"error": "grid_cells requires tile_size, the cell edge of the grid the "
                             "cells were read off (overlay_reference_grid echoes it back, with "
                             "overlap). Without it a cell name resolves against a grid nobody "
                             "rendered."}
        from tcip_mcp.pipelines.image_utils import image_dimensions
        from tcip_mcp.pipelines.raster_source import open_raster
        from tcip_mcp.pipelines.reference_grid import reference_cells
        from tcip_mcp.tools.vision_tools import _source_for_path

        # One resolution of the source for both halves: the frame the cells are laid over is the
        # frame the crop is read from, so they can never come from two decisions.
        source = _source_for_path(image_path)
        try:
            w, h = image_dimensions(source)
            cells = reference_cells(w, h, tile_size, overlap, clamp=True)
            rect = _region_rect_from_cells(cells, grid_cells)
            with open_raster(source, 3) as src:
                pixels, _spec = src.read_region(rect)
        except ValueError as e:
            return {"error": str(e)}
        if pixels.dtype != "uint8" or pixels.shape[-1] != 3:
            return {"error": "A region crop is handed to the engine as an RGB image, and "
                             f"{img.name} reads as {pixels.shape[-1]} band(s) of {pixels.dtype}. "
                             "Propose over the whole frame instead, or bring an engine that reads "
                             "this source itself."}
        crop_tmp = _write_region_crop(pixels)
        propose_path = str(crop_tmp)
        origin = (float(rect.x0), float(rect.y0))
        region_info = {"grid_cells": list(grid_cells), "tile_size": tile_size, "overlap": overlap,
                       "rect": [rect.x0, rect.y0, rect.x1, rect.y1]}

    try:
        try:
            candidates = proposer.propose(propose_path, **(engine_params or {}))
        except ImportError as e:
            return {"error": str(e)}
        except FileNotFoundError as e:
            return {"error": str(e)}
    finally:
        if crop_tmp is not None:
            crop_tmp.unlink(missing_ok=True)

    if region_info is not None:
        candidates = _offset_candidates(candidates, origin)

    if not candidates:
        # A prior run's record must not outlive this one finding nothing to propose.
        try:
            stale = _staging_key_for(image_path)
        except ValueError:
            pass
        else:
            ts.delete(stale.key)
        return {
            "image_path": None,
            "engine": engine,
            "summary": f"Engine {engine!r} proposed no candidates",
            "staged": False,
            "candidates": [],
        }

    # A bespoke engine's candidates are its own dicts, so what a segmenter returns natively
    # (an array, a numpy scalar) is named here rather than stored as a repr of itself.
    try:
        ts.check_json_value(candidates, path="candidates")
    except (TypeError, ValueError) as exc:
        return {"error": f"Engine {engine!r} proposed a candidate the store cannot hold: {exc}"}

    from tcip_mcp.tools.vision_tools import _display_for_path

    read = _display_for_path(image_path)
    out = render_candidates(read.pixels, candidates, native_size=read.native_size)

    # The envelope records the engine so stage_proposals's assignments regime stamps the right
    # producer.
    envelope: dict = {"engine": engine, "candidates": candidates}
    if region_info is not None:
        envelope["region"] = region_info

    try:
        address = _staging_key_for(image_path)
    except ValueError as exc:
        staged = False
        stage_note = f" Not staged: {exc}"
    else:
        from tcip_mcp.pipelines import image_utils

        try:
            # The same resolution the assignments regime will make: staging over an unrereadable
            # source would leave a record it can never confirm.
            source = image_utils.resolve_image_source(img.parent, img.stem)
        except (FileNotFoundError, image_utils.BandGroupIncomplete) as exc:
            staged = False
            stage_note = f" Not staged: {_unresolvable_staging_source(img, exc)}"
        else:
            import dataclasses

            from tcip_mcp.pipelines.raster_source import content_identity

            identity = content_identity(source)
            envelope["image_identity"] = dataclasses.asdict(identity)
            envelope["image_path"] = str(img.resolve())
            ts.replace(address.key, envelope)
            staged = True
            stage_note = ""

    region_note = f" (region {grid_cells})" if region_info is not None else ""
    return {
        "image_path": out,
        "engine": engine,
        "summary": f"Engine {engine!r} proposed {len(candidates)} candidates{region_note}."
                   f"{stage_note} Review the numbered overlay, then call stage_proposals "
                   f"with subject assignments.",
        "candidate_count": len(candidates),
        "staged": staged,
        "candidates": [
            {
                "id": c["candidate_id"],
                "area": c["area"],
                "score": round(c["score"], 3),
                "bbox": [round(v, 1) for v in c["bbox"]],
            }
            for c in candidates
        ],
    }


def _stage_assignments_regime(image_path: str, img: Path, address: StagingAddress,
                               assignments: list[dict]) -> dict:
    """The reviewed-candidates regime :func:`stage_proposals` runs when ``assignments`` is given.

    Reads back the record ``propose_annotations`` staged for this exact image (dataset, capture
    date and stem) and refuses if the image's content no longer matches the content identity
    that run recorded: the proposals it staged were candidates over those pixels, not whatever
    now sits at this path. That check decodes sample windows of the image (the bound
    ``CONTENT_IDENTITY_*`` constants in ``raster_source.py`` set how many and how large), never
    the whole frame.
    """
    from tcip_mcp.pipelines.image_utils import (
        BandGroupIncomplete, image_dimensions, resolve_image_source,
    )

    try:
        source = resolve_image_source(img.parent, img.stem)
    except (FileNotFoundError, BandGroupIncomplete) as exc:
        return {"error": str(exc)}

    # Load cached proposals from the same record propose_annotations staged them in.
    envelope = ts.read(address.key, default=None)
    if envelope is None:
        return {"error": f"No proposals found for {img.stem}. Run propose_annotations first."}

    from tcip_mcp.pipelines.raster_source import raster_identity_matches

    try:
        matches = raster_identity_matches(envelope["image_identity"], source)
    except ValueError as exc:
        return {"error": f"Could not verify {image_path} against its staged proposals: {exc}"}

    if not matches:
        return {"error": f"{image_path} does not match the image propose_annotations ran on: "
                          "its content has changed since that run staged these candidates. "
                          "Run propose_annotations again on the current image."}

    engine = envelope.get("engine", "unknown")
    candidates = envelope.get("candidates", [])
    cand_map = {c["candidate_id"]: c for c in candidates}

    w, h = image_dimensions(source)

    # Build name-based predictions (created_by=<engine>, score = the proposal score); each keeps
    # every ring, so an occlusion-split object stays split rather than its largest fragment.
    from datetime import datetime, timezone
    from tcip_annotation.state import Polygon as _Polygon
    staged_at = datetime.now(timezone.utc).isoformat()
    proposals: list[Annotation] = []
    n_poly = 0

    for assign in assignments:
        cid = assign["candidate_id"]
        subject = assign.get("subject")
        cand = cand_map.get(cid)
        if cand is None or not subject:
            continue
        score = float(cand.get("score", 0.0))  # neutral proposal score, in [0, 1]
        rings = [[(float(x), float(y)) for x, y in ring]
                 for ring in cand["rings"] if len(ring) >= 3]
        if rings:
            proposals.append(Annotation(
                subject=str(subject), geometry=_Polygon(rings=rings),
                score=score, created_by=engine, created_at=staged_at))
            n_poly += 1

    # Stage into the predictions tree through the shared verdict-guarded helper: model output for a
    # human to accept on the Review canvas, never written straight to ground truth.
    from tcip_mcp.prediction_buckets import BucketHasVerdicts, stage_prediction_shapes

    try:
        staged = stage_prediction_shapes(
            str(address.root), engine, address.date, img.stem,
            annotations=proposals, img_w=w, img_h=h, overwrite=False,
        )
    except BucketHasVerdicts as exc:
        return {"error": str(exc), "verdict_count": exc.count, "suggested_bucket": exc.suggested}
    except ValueError as exc:
        return {"error": str(exc)}
    bucket = staged["bucket"]

    # Render final result for QA
    from tcip_mcp.tools.vision_tools import _box_dict, _display_for_path, _name_map, _subject_indexer

    idx, index = _subject_indexer()
    read = _display_for_path(image_path)
    out = render_detections(read.pixels, [_box_dict(a, index) for a in proposals],
                            native_size=read.native_size, class_names=_name_map(idx))

    note = (f"Staged {n_poly} proposal(s) from {len(assignments)} {engine!r} candidates as "
            f"predictions (created_by={engine!r}) for review, not ground truth.")
    if staged["redirected"]:
        note = (f"bucket {engine!r} has {staged['verdict_count']} review verdict(s), staged to a fresh "
                f"bucket {bucket!r} instead so the reviewed predictions stay intact. " + note)

    return {
        "image_path": out,
        "engine": engine,
        "bucket": bucket,
        "bucket_redirected": staged["redirected"],
        "summary": note,
        "proposal_count": n_poly,
    }


@mcp.tool()
@audited
def segment_prompt(
    image_path: str,
    points: list[dict] | None = None,
    box: dict | None = None,
    grid_cells: list[str] | None = None,
    tile_size: int | None = None,
    overlap: float = 0.0,
    engine: str = "sam",
    engine_params: dict | None = None,
) -> dict:
    """Turn an interactive prompt (points, a box, or grid cells) into mask polygon rings, via an engine.

    Returns ``rings``, the mask's contours as ``[[{x, y}, ...], ...]``, one ring per connected region.
    An occlusion-split object (a leaf crossed by a stem) segments to more than one region and all of
    them come back; keeping only the largest would report part of an object as the whole of it.

    Provide point prompts, a box prompt, or grid-cell references (e.g. ['B3', 'D5'], converted to
    foreground point prompts). A cell name means nothing without the grid that produced it, so
    ``grid_cells`` requires an explicit ``tile_size``, the geometry the overlay whose cells are
    being named was rendered with (``overlay_reference_grid`` echoes ``tile_size`` and ``overlap``
    back for exactly this). There is no default grid to fall back on: guessing one resolves 'B3' to
    a pixel in a grid nobody looked at. The cells recompute here through the same
    ``reference_grid.reference_cells`` the overlay drew, so the resolved centers are the rendered
    cells' own. The segmentation method is a capability, not a hardcode: 'sam' is the built-in
    SAM2 reference engine; the agent can bring another prompted-segmentation engine behind the same
    seam (a dotted 'module:factory').

    Args:
        image_path: Absolute path to the image file.
        points: List of point prompts, each with x, y, and label (1=fg, 0=bg).
        box: Box prompt with x1, y1, x2, y2 in pixel coordinates.
        grid_cells: List of grid cell references like ['B3', 'D5']. Each is a foreground point.
        tile_size: Cell edge, in native pixels, of the grid the cells were read off. Required
            with ``grid_cells``.
        overlap: Overlap fraction of the grid the cells were read off.
        engine: Segmentation engine, 'sam' (built-in) or a dotted 'module:factory' the agent brings.
        engine_params: Engine-specific knobs forwarded to the engine (e.g. SAM's model_type).
    """
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    if points is None and box is None and grid_cells is None:
        return {"error": "Provide either points, box, or grid_cells prompt"}

    if grid_cells is not None:
        if tile_size is None:
            return {"error": "grid_cells requires tile_size, the cell edge of the grid the "
                             "cells were read off (overlay_reference_grid echoes it back, with "
                             "overlap). Without it a cell name resolves against a grid nobody "
                             "rendered."}
        from tcip_annotation.sam_wrapper import grid_to_pixel

        from tcip_mcp.pipelines.reference_grid import reference_cells
        from tcip_mcp.tools.annotation_tools import _dims_for
        w, h = _dims_for(image_path)
        try:
            cells = reference_cells(w, h, tile_size, overlap, clamp=True)
        except ValueError as e:
            return {"error": str(e)}
        points = []
        for cell in grid_cells:
            try:
                cx, cy = grid_to_pixel(cell, cells)
                points.append({"x": cx, "y": cy, "label": 1})
            except ValueError as e:
                return {"error": f"Invalid grid cell {cell!r}: {e}"}

    from tcip_mcp.pipelines.proposal import resolve_proposer

    try:
        proposer = resolve_proposer(engine)
    except (ValueError, ImportError) as e:
        return {"error": str(e)}

    try:
        rings = proposer.segment(image_path, points=points, box=box, **(engine_params or {}))
    except ImportError as e:
        return {"error": f"segmentation engine dependencies not available: {e}"}
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"segmentation failed: {e}"}

    if not rings:
        return {"error": "engine produced empty mask", "rings": []}

    return {
        "rings": [[{"x": x, "y": y} for x, y in ring] for ring in rings],
        "ring_count": len(rings),
        "vertex_count": sum(len(ring) for ring in rings),
        "engine": engine,
    }


def _stage_explicit_regime(image_path: str, img: Path, address: StagingAddress,
                            model_name: str, boxes: list[dict], polygons: list[dict],
                            overwrite: bool) -> dict:
    """The explicit-shapes regime :func:`stage_proposals` runs when ``boxes``/``polygons`` is
    given: model-/agent-proposed shapes staged to
    ``predictions/<model>/<date>/<stem>.json`` for canvas review, the "show on canvas before
    writing ground truth" guardrail.

    Anything a model produces (a SAM mask, a baseline detection, a shape the agent wants a human to
    vet) goes to the predictions tree, never ``annotations/``, so the human reviews it on the Review
    canvas and accepts/rejects/edits before it becomes GT. Boxes and polygons alike land in the one
    per-image prediction file, each carrying a ``subject`` name. This never writes ground truth.
    """
    from tcip_annotation.json_io import ring_vertex

    from tcip_mcp.prediction_buckets import BucketHasVerdicts, stage_prediction_shapes
    from tcip_mcp.workspace import is_valid_name

    if not is_valid_name(model_name):
        return {"error": f"model_name must be a single safe path segment (no separators/'..'), "
                         f"got {model_name!r}"}

    def _unnormalized(vals) -> bool:
        return any(v < -0.01 or v > 1.5 for v in vals)

    norm_boxes: list[tuple[str, float, float, float, float, float]] = []
    for i, b in enumerate(boxes):
        try:
            cx, cy, w, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
            conf = float(b.get("conf", 1.0))
            subject = str(b["subject"])
        except (KeyError, TypeError, ValueError):
            return {"error": f"box {i} needs a subject and numeric conf, cx, cy, w, h (normalized): {b!r}"}
        if not subject:
            return {"error": f"box {i} needs a non-empty subject"}
        if _unnormalized((cx, cy, w, h)):
            return {"error": f"box {i} coords {(cx, cy, w, h)} look un-normalized; cx/cy/w/h must be in [0,1]"}
        norm_boxes.append((subject, conf, cx, cy, w, h))

    try:
        img_source = resolve_image_source(img.parent, img.stem)
    except (FileNotFoundError, BandGroupIncomplete) as exc:
        return {"error": str(exc)}
    img_w, img_h = image_dimensions(img_source)
    dataset_root, date, stem = str(address.root), address.date, img.stem

    # A rounding-slop margin in pixels, not a fraction of the image size: a fractional margin
    # admits a normalized [0,1] ring at every real image size, the bug this check exists to refuse.
    _PIXEL_MARGIN = 1.0

    def _spans_a_pixel(ring: list[tuple[float, float]]) -> bool:
        xs, ys = [x for x, _ in ring], [y for _, y in ring]
        return (max(xs) - min(xs)) >= 1.0 and (max(ys) - min(ys)) >= 1.0

    def _out_of_pixel_bounds(ring: list[tuple[float, float]]) -> bool:
        return (any(x < -_PIXEL_MARGIN or x > img_w + _PIXEL_MARGIN for x, _ in ring)
                or any(y < -_PIXEL_MARGIN or y > img_h + _PIXEL_MARGIN for _, y in ring))

    # Each polygon carries exactly one of two keys, folded into pixel-space rings as it is parsed;
    # the fold needs img_w/img_h, resolved above.
    resolved_polys: list[tuple[str, float, list[list[tuple[float, float]]]]] = []
    for i, p in enumerate(polygons):
        has_points = "points" in p
        has_rings = "rings" in p
        if has_points == has_rings:
            offered = sorted(k for k in ("points", "rings") if k in p)
            return {"error": f"polygon {i} must carry exactly one of 'points' or 'rings', got "
                             f"{offered}: {p!r}"}
        try:
            conf = float(p.get("conf", 1.0))
            subject = str(p["subject"])
        except (KeyError, TypeError, ValueError):
            return {"error": f"polygon {i} needs a subject and numeric conf: {p!r}"}
        if not subject:
            return {"error": f"polygon {i} needs a non-empty subject"}

        if has_points:
            try:
                pts = [(float(x), float(y)) for x, y in p["points"]]
            except (TypeError, ValueError):
                return {"error": f"polygon {i} points must be [x, y] pairs (normalized): {p!r}"}
            if len(pts) < 3:
                return {"error": f"polygon {i} needs at least 3 points, got {len(pts)}"}
            if _unnormalized([v for xy in pts for v in xy]):
                return {"error": f"polygon {i} points look un-normalized; x/y must be in [0,1]"}
            rings_px = [[(x * img_w, y * img_h) for x, y in pts]]
        else:
            try:
                rings = [[ring_vertex(v) for v in ring] for ring in p["rings"]]
            except (TypeError, ValueError, KeyError):
                return {"error": f"polygon {i} rings must be a list of rings of [x, y] pairs or "
                                 f"{{'x':, 'y':}} mappings (pixel coordinates): {p!r}"}
            short = [j for j, ring in enumerate(rings) if len(ring) < 3]
            if short:
                return {"error": f"polygon {i} ring(s) {short} need at least 3 points"}
            sub_pixel = [j for j, ring in enumerate(rings) if not _spans_a_pixel(ring)]
            if sub_pixel:
                return {"error": f"polygon {i} ring(s) {sub_pixel} span under a pixel in an axis; "
                                 f"rings are pixel coordinates, not normalized ones: {p!r}"}
            if any(_out_of_pixel_bounds(ring) for ring in rings):
                return {"error": f"polygon {i} rings look out of the image's pixel bounds "
                                 f"({img_w}x{img_h}): {rings!r}"}
            rings_px = rings
        resolved_polys.append((subject, conf, rings_px))

    from datetime import datetime, timezone
    created_at = datetime.now(timezone.utc).isoformat()

    # A box of nothing is no detection: dropped rather than staged, so it never reaches the
    # accept branch (which would otherwise hand the persistence boundary a degenerate proposal).
    box_proposals: list[Annotation] = []
    dropped_boxes = 0
    for (subject, conf, cx, cy, w, h) in norm_boxes:
        box = BBox((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                  (cx + w / 2) * img_w, (cy + h / 2) * img_h)
        if not stored_box_extent_ok(box):
            dropped_boxes += 1
            continue
        box_proposals.append(Annotation(subject=subject, geometry=box, score=conf,
                                        created_by=model_name, created_at=created_at))

    proposals: list[Annotation] = box_proposals + [
        Annotation(subject=subject,
                   geometry=Polygon(rings=rings_px),
                   score=conf, created_by=model_name, created_at=created_at)
        for (subject, conf, rings_px) in resolved_polys
    ]

    try:
        staged = stage_prediction_shapes(
            dataset_root, model_name, date, stem,
            annotations=proposals, img_w=img_w, img_h=img_h, overwrite=overwrite,
        )
    except BucketHasVerdicts as exc:
        return {"error": str(exc), "verdict_count": exc.count, "suggested_bucket": exc.suggested}
    except ValueError as exc:
        return {"error": str(exc)}
    bucket = staged["bucket"]

    note = ("staged to predictions/ for canvas review, not committed as ground truth; the human "
            "accepts on the Review tab before it becomes GT (focus_human_attention tab='review' to "
            "send them). It is reviewed through the accept path and is never promoted to a "
            "validation reference.")
    if staged["redirected"]:
        note = (f"bucket {model_name!r} has {staged['verdict_count']} review verdict(s), staged to a "
                f"fresh bucket {bucket!r} instead so the reviewed predictions stay intact; " + note)

    return {
        "staged": len(box_proposals) + len(resolved_polys),
        "n_detect": len(box_proposals), "n_segment": len(resolved_polys),
        "dropped_nonpositive_boxes": dropped_boxes,
        "path": staged["path"],
        "model_name": model_name, "bucket": bucket, "bucket_redirected": staged["redirected"],
        "date": date, "stem": stem, "note": note,
    }


@mcp.tool()
@audited(scope_arg="image_path")
def stage_proposals(
    image_path: str,
    *,
    assignments: list[dict] | None = None,
    boxes: list[dict] | None = None,
    polygons: list[dict] | None = None,
    model_name: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Stage model-/agent-proposed shapes as predictions for canvas review, the "show on canvas
    before writing ground truth" guardrail. Never writes ground truth.

    Exactly one input regime per call:

    - ``assignments``: candidates ``propose_annotations`` staged for this image, reviewed and
      each assigned a subject; a mapping from candidate id to subject, rejected candidates simply
      omitted. Reads back the record staged at this exact image (dataset, capture date and stem)
      and refuses if the image's content no longer matches the content identity that run
      recorded. The masks land under ``predictions/<engine>/<date>/<task>`` with
      ``created_by=<engine>`` and ``score`` = the engine's proposal score; ``model_name`` is
      refused alongside ``assignments``, since the staged record's own engine names the bucket.
    - ``boxes``/``polygons``: explicit shapes an agent or another model already has in hand, with
      no cached record to read back. Land under
      ``predictions/<model_name>/<date>/<stem>.json``. ``model_name`` is required, the real
      producer stamped as ``created_by``. ``boxes`` is ``[{subject, conf, cx, cy, w, h}]`` with
      cx/cy/w/h normalized to [0, 1]; ``polygons`` is ``[{subject, conf, points|rings}]``, exactly
      one of two frames per proposal: ``points``, one ring of ``[x, y]`` pairs normalized to
      [0, 1]; or ``rings``, a list of rings in pixel coordinates, each vertex an ``[x, y]`` pair
      or an ``{"x":, "y":}`` mapping, the frame ``segment_prompt`` returns. Both build the same
      ``Polygon`` through the ground-truth door's own vertex parser. ``overwrite=True`` writes in
      place even into an existing bucket, refused if the bucket has verdicts; the default
      redirects to a fresh run-scoped bucket (``<model_name>@r2``, next free) instead, returned as
      ``bucket``.

    Either regime resolves the dataset root, capture date and stem from ``image_path`` itself
    (the same resolver ``propose_annotations`` uses), so the explicit regime no longer takes
    path fragments a caller must keep consistent with the image. Both write through the one
    verdict-guarded staging door (``prediction_buckets.stage_prediction_shapes``), so a re-run
    never overwrites reviewed predictions or orphans their verdicts. Pair with
    ``focus_human_attention(tab='review')`` to send the human straight to the result.

    A staged record's ``subject`` is whatever ``assignments``/``boxes``/``polygons`` named; the
    platform validates no subject name. A staged bucket carries no ``operating_point.json`` stamp
    and so no recorded scope, so every reader of it below (the Review routes, the calibrator, the
    delivery doors) reads its records under the caller's own statement rather than a proven one.

    Args:
        image_path: Absolute path to the dataset image (same as propose_annotations, for the
            assignments regime).
        assignments: List of dicts, each with 'candidate_id' (int) and 'subject' (name). Refused
            alongside boxes/polygons or model_name.
        boxes: Explicit boxes; see above. Refused alongside assignments.
        polygons: Explicit polygons; see above. Refused alongside assignments.
        model_name: Predictions bucket to stage the explicit regime under. Required with
            boxes/polygons, refused with assignments.
        overwrite: Explicit regime only: write in place even into an existing bucket. Refused if
            the bucket has verdicts.
    """
    if assignments is not None and (boxes or polygons):
        return {"error": "assignments cannot be combined with boxes/polygons: pick one input "
                         "regime per call."}
    if assignments is None and not boxes and not polygons:
        return {"error": "provide assignments (propose_annotations's staged candidates), or "
                         "boxes/polygons (explicit shapes) with model_name."}

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        address = _staging_key_for(image_path)
    except ValueError as exc:
        return {"error": str(exc)}

    if assignments is not None:
        if model_name is not None:
            return {"error": "model_name is refused alongside assignments: the staged record's "
                             "own engine names the bucket."}
        return _stage_assignments_regime(image_path, img, address, assignments)

    if model_name is None:
        return {"error": "model_name is required with boxes/polygons: the real producer, "
                         "stamped as created_by."}
    return _stage_explicit_regime(image_path, img, address, model_name, boxes or [],
                                  polygons or [], overwrite)
