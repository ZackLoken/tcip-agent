"""Proposal-workflow tools: turn a chosen auto-labeling engine's output into predictions for
canvas review.

propose_annotations asks a named engine (one registered, or a 'module:factory' the agent brings)
to look at pixels and offer candidates. stage_proposals publishes either an engine's reviewed
candidates or explicit boxes/polygons once as a proposal bucket under the image's dataset root,
refusing one already published, for a human to accept, reject or correct on the Annotate canvas.
It never writes ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import tcip_store as ts
from pydantic import BaseModel, ConfigDict

from tcip_annotation import Annotation, BBox, Polygon, bbox_of
from tcip_annotation.grid import grid_to_rect
from tcip_annotation.json_io import stored_box_extent_ok
from tcip_annotation.viz import render_candidates, render_detections

from tcip_mcp.audit import now_iso, record_event_or_raise
from tcip_mcp.pipelines.image_utils import (
    BandGroupIncomplete, image_dimensions, resolve_image_path,
)
from tcip_mcp.project_paths import viz_output_path
from tcip_mcp.server import tool

if TYPE_CHECKING:
    import numpy as np

    from tcip_mcp.pipelines.raster_source import Rect


PROPOSAL_STAGING_STORE = "proposal_staging"


class _StatedSubject(BaseModel):
    """One staged input's own ``subject``, required; the entry's other keys are its door's own."""

    model_config = ConfigDict(extra="allow")

    subject: str


def proposal_staging_key(dataset_root: str | Path, capture: str, stem: str) -> ts.Key:
    """The candidates the latest proposal run produced for one dataset image, whole, under the
    dataset root, by the image's capture and stem."""
    return ts.Key(PROPOSAL_STAGING_STORE, str(dataset_root), (capture, stem))


def _staging_key_for(image_path: str) -> ts.Key:
    """The :func:`proposal_staging_key` of the dataset image at ``image_path``; ``ValueError``,
    :func:`~tcip_mcp.dataset_layout.parse_image_path`'s own, for an image of no capture."""
    from tcip_mcp.dataset_layout import parse_image_path

    return proposal_staging_key(*parse_image_path(image_path))


def _region_rect_from_cells(cells: list, names: list[str]) -> "Rect":
    """The bounding rect, in the grid's native-pixel frame, of the named reference-grid cells.

    Each name resolves through ``grid.grid_to_rect``, so a malformed or out-of-grid name is
    refused here, and the matched cells union to their combined bounding box.
    """
    from tcip_mcp.pipelines.raster_source import Rect

    matched = [grid_to_rect(name, cells) for name in names]
    return Rect(int(min(r[0] for r in matched)), int(min(r[1] for r in matched)),
                int(max(r[2] for r in matched)), int(max(r[3] for r in matched)))


def _write_region_crop(pixels: "np.ndarray") -> Path:
    """Save an RGB region crop to a fresh temp PNG file; the caller deletes it once the engine has
    read it.

    The crop is taken from the raster layer's already-``auto_orient_image``'d frame, so it carries
    no EXIF orientation tag once saved.
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


@tool()
def propose_annotations(
    project: Path,
    image_path: str,
    engine: str,
    engine_params: dict | None = None,
    grid_cells: list[str] | None = None,
    tile_size: int | None = None,
    overlap: float = 0.0,
) -> dict:
    """Propose candidate annotations on an image for review, using a named auto-labeling engine.

    Runs the engine's whole-image proposal pass, renders the numbered candidates, and returns the
    render path and neutral candidate data. Read the render with your own image-capable read tool,
    then call stage_proposals with subject assignments to stage the accepted ones as predictions.

    Each candidate renders as a colored, semi-transparent filled polygon (every ring of an
    occlusion-split candidate drawn, not just the largest) with a large numbered label at its
    centroid, colors cycling through the shared class palette; the candidate id in that number is
    the same id ``stage_proposals``' ``assignments`` parameter names.

    On an image under a dataset's ``images/`` tree, the candidates are staged keyed by the dataset,
    capture date and stem, alongside the content identity of the pixels the engine ran on:
    ``stage_proposals``'s assignments regime reads the record back by that same address and refuses
    if the image's content no longer matches it. On a path outside any dataset's ``images/`` tree
    the engine still runs and the render and candidates are returned the same way, but nothing is
    staged (the response's ``staged`` is ``false``, naming why), so such a call cannot later be
    accepted. A path naming no logical image (a band of a grouped capture, which names its
    manifest) refuses (:func:`~tcip_mcp.pipelines.image_utils.resolve_image_path`). Writing or
    clearing the staged record leaves one audit line in the dataset's log; a call that writes
    neither leaves none.

    ``engine`` names one registered (``register_proposal_engine``) or a dotted 'module:factory'.
    An empty or unknown name refuses, naming the registered ones.

    ``grid_cells`` restricts the pass to a region instead of the whole frame: name the
    reference-grid cells the region spans (e.g. ``['B3', 'C3', 'B4', 'C4']``), and the engine
    proposes only over their bounding rect. Useful on a large or crowded frame where a whole-image
    pass returns too many or too coarse candidates to review, or where only part of the frame
    matters right now. The crop is taken and the results offset back to full-frame coordinates on
    this side of the engine seam; an engine that keys behavior off the image path itself (a cache,
    a sidecar lookup keyed by the original file) receives the temp crop's path, which it cannot
    resolve back to the source image. Omitting ``grid_cells`` runs the whole frame.

    Args:
        image_path: Absolute path to the image file.
        engine: Proposal engine: a registered name or a dotted 'module:factory' the agent brings.
        engine_params: Engine-specific knobs forwarded to the engine. Omit for the engine's own
            defaults.
        grid_cells: Reference-grid cell names bounding the region to propose over (e.g. ['B3',
            'D5']); the engine sees the bounding rect of the named cells, not the whole frame.
            Requires ``tile_size``. Omit for the whole frame.
        tile_size: Cell edge, in native pixels, of the grid the cells were read off. Required with
            ``grid_cells``.
        overlap: Overlap fraction of the grid the cells were read off.
    """
    from tcip_mcp.pipelines.proposal import resolve_proposer

    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        source = resolve_image_path(img)
        proposer = resolve_proposer(engine)
    except (FileNotFoundError, BandGroupIncomplete, ValueError, ImportError) as e:
        return {"error": str(e)}

    # A region is cropped and offset entirely here, before the engine ever sees an image path.
    # grid_cells=None skips this branch, taking the whole-frame path below.
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
        from tcip_mcp.pipelines.raster_source import open_raster
        from tcip_mcp.pipelines.reference_grid import reference_cells

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
            if ts.delete(stale):
                record_event_or_raise("propose_annotations",
                                      {"image_path": image_path, "staged": 0},
                                      actor=None, scope=stale.root)
        return {
            "image_path": None,
            "engine": engine,
            "summary": f"Engine {engine!r} proposed no candidates",
            "staged": False,
            "candidates": [],
        }

    # An engine's own candidates are refused here, where they arrive, when unstorable, no polygon
    # or stating no confidence of their own.
    try:
        ts.check_json_value(candidates, path="candidates")
        for candidate in candidates:
            Polygon(rings=candidate["rings"])
            candidate["score"] = float(candidate["score"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"error": f"Engine {engine!r} proposed a candidate this platform cannot hold: {exc}"}

    from tcip_mcp.tools.vision_tools import _read_for_display

    read = _read_for_display(source)
    out = render_candidates(read.pixels, candidates, native_size=read.native_size,
                            output_path=viz_output_path(project, "candidates"))

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
        import dataclasses

        from tcip_mcp.pipelines.raster_source import content_identity

        envelope["image_identity"] = dataclasses.asdict(content_identity(source))
        ts.replace(address, envelope)
        record_event_or_raise("propose_annotations", {
            "image_path": image_path, "staged": len(candidates)}, actor=None,
            scope=address.root)
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


def _stage_assignments_regime(project: Path, image_path: str, img: Path, address: ts.Key,
                               assignments: list[dict]) -> dict:
    """Stage the candidates ``assignments`` names, each with its subject, from the record staged
    at ``address`` for ``img``; refuses when that record is absent or the image's content no
    longer matches the content identity it recorded."""
    try:
        source = resolve_image_path(img)
    except (FileNotFoundError, BandGroupIncomplete, ValueError) as exc:
        return {"error": str(exc)}

    # Load cached proposals from the same record propose_annotations staged them in.
    envelope = ts.read(address, default=None)
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

    engine = envelope["engine"]
    candidates = envelope["candidates"]
    cand_map = {c["candidate_id"]: c for c in candidates}

    w, h = image_dimensions(source)

    # Build name-based predictions (created_by=<engine>, score = the proposal score); each keeps
    # every ring, so an occlusion-split object stays split rather than its largest fragment.
    staged_at = now_iso()
    proposals: list[Annotation] = []
    n_poly = 0

    for i, assign in enumerate(assignments):
        cand = cand_map.get(assign["candidate_id"])
        if cand is None:
            continue
        rings = [[(float(x), float(y)) for x, y in ring] for ring in cand["rings"]]
        try:
            proposals.append(Annotation(
                subject=_StatedSubject.model_validate(assign).subject, geometry=Polygon(rings=rings),
                score=cand["score"], created_by=engine, created_at=staged_at))
        except ValueError as exc:
            return {"error": f"assignment {i}: {exc}"}
        n_poly += 1

    # Model output for a human to accept on the Annotate canvas, never written straight to ground truth.
    try:
        bucket = _stage_document(project, address, engine, img, annotations=proposals,
                                 img_w=w, img_h=h)
    except ValueError as exc:
        return {"error": str(exc)}

    # Render final result for QA
    from tcip_mcp.tools.vision_tools import _box_dict, _name_map, _read_for_display, _subject_indexer

    idx, index = _subject_indexer()
    read = _read_for_display(source)
    out = render_detections(read.pixels, [_box_dict(a, index) for a in proposals],
                            native_size=read.native_size, class_names=_name_map(idx),
                            output_path=viz_output_path(project, "staged"))

    note = (f"Staged {n_poly} proposal(s) from {len(assignments)} {engine!r} candidates as "
            f"predictions (created_by={engine!r}) for review, not ground truth.")

    return {
        "image_path": out,
        "engine": engine,
        "bucket": bucket,
        "summary": note,
        "proposal_count": n_poly,
    }


def _stage_document(project: Path, address: ts.Key, producer: str, image: Path, *,
                    annotations: list[Annotation], img_w: int, img_h: int) -> str | None:
    """Publish ``annotations`` as the bucket of one staged proposal for ``image``, named
    ``<producer>/<capture>/<stem>`` under the image's dataset root (both read off the image's
    staging key ``address``)
    (:func:`~tcip_mcp.buckets.publish`), its record naming ``producer`` as what proposed them, and
    return the bucket's name; no annotations publishes nothing and returns ``None``. A bucket of
    that name already published refuses (:class:`~tcip_mcp.buckets.BucketExists`)."""
    from tcip_annotation import json_io

    from tcip_mcp.buckets import Document, publish
    from tcip_mcp.pipelines.data.selection import ClassScope

    if not annotations:
        return None
    data = json_io.document_payload(annotations, img_w, img_h)
    assert data is not None, "a non-empty proposal encodes a document"
    name = "/".join([producer, *address.parts])
    return publish(project, Path(address.root), name, [Document(str(image), data)],
                   producer={"proposed_by": producer},
                   scope=ClassScope(), execution=None, raster_path=None, raster_identity=None,
                   assessment_id=None, actor=None).name


def _stage_explicit_regime(project: Path, image_path: str, img: Path, address: ts.Key,
                           model_name: str, boxes: list[dict], polygons: list[dict]) -> dict:
    """Stage explicit boxes and polygons, each carrying a ``subject``, as one staged proposal of
    ``model_name`` for the image (:func:`_stage_document`)."""
    from tcip_annotation.json_io import ring_vertex

    from tcip_mcp.workspace import is_valid_name

    if not is_valid_name(model_name):
        return {"error": f"model_name must be a single safe path segment (no separators/'..'), "
                         f"got {model_name!r}"}

    def _unnormalized(vals) -> bool:
        return any(v < -0.01 or v > 1.5 for v in vals)

    norm_boxes: list[tuple[Any, float, float, float, float, float]] = []
    for i, b in enumerate(boxes):
        try:
            cx, cy, w, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
            conf = float(b["conf"])
        except (KeyError, TypeError, ValueError):
            return {"error": f"box {i} needs numeric conf, cx, cy, w, h (normalized): {b!r}"}
        if _unnormalized((cx, cy, w, h)):
            return {"error": f"box {i} coords {(cx, cy, w, h)} look un-normalized; cx/cy/w/h must be in [0,1]"}
        norm_boxes.append((b, conf, cx, cy, w, h))

    try:
        img_source = resolve_image_path(img)
    except (FileNotFoundError, BandGroupIncomplete, ValueError) as exc:
        return {"error": str(exc)}
    img_w, img_h = image_dimensions(img_source)

    # A rounding-slop margin in pixels, not a fraction of the image size: a fractional margin
    # admits a normalized [0,1] ring at every real image size, the bug this check exists to refuse.
    pixel_margin = 1.0

    def _spans_a_pixel(ring: list[tuple[float, float]]) -> bool:
        b = bbox_of(Polygon(rings=[ring]))
        return b.x2 - b.x1 >= 1.0 and b.y2 - b.y1 >= 1.0

    def _out_of_pixel_bounds(ring: list[tuple[float, float]]) -> bool:
        return (any(x < -pixel_margin or x > img_w + pixel_margin for x, _ in ring)
                or any(y < -pixel_margin or y > img_h + pixel_margin for _, y in ring))

    created_at = now_iso()

    # Each polygon carries exactly one of two keys, folded into pixel-space rings as it is parsed;
    # the fold needs img_w/img_h, resolved above.
    polygon_proposals: list[Annotation] = []
    for i, p in enumerate(polygons):
        has_points = "points" in p
        has_rings = "rings" in p
        if has_points == has_rings:
            offered = sorted(k for k in ("points", "rings") if k in p)
            return {"error": f"polygon {i} must carry exactly one of 'points' or 'rings', got "
                             f"{offered}: {p!r}"}
        try:
            conf = float(p["conf"])
        except (KeyError, TypeError, ValueError):
            return {"error": f"polygon {i} needs a numeric conf: {p!r}"}

        if has_points:
            try:
                pts = [(float(x), float(y)) for x, y in p["points"]]
            except (TypeError, ValueError):
                return {"error": f"polygon {i} points must be [x, y] pairs (normalized): {p!r}"}
            if _unnormalized([v for xy in pts for v in xy]):
                return {"error": f"polygon {i} points look un-normalized; x/y must be in [0,1]"}
            rings_px = [[(x * img_w, y * img_h) for x, y in pts]]
        else:
            try:
                rings_px = [[(float(x), float(y)) for x, y in map(ring_vertex, ring)]
                            for ring in p["rings"]]
            except (TypeError, ValueError, KeyError):
                return {"error": f"polygon {i} rings must be a list of rings of [x, y] pairs or "
                                 f"{{'x':, 'y':}} mappings (pixel coordinates): {p!r}"}
        try:
            proposal = Annotation(subject=_StatedSubject.model_validate(p).subject,
                                  geometry=Polygon(rings=rings_px),
                                  score=conf, created_by=model_name, created_at=created_at)
        except ValueError as exc:
            return {"error": f"polygon {i}: {exc}"}
        if has_rings:
            sub_pixel = [j for j, ring in enumerate(rings_px) if not _spans_a_pixel(ring)]
            if sub_pixel:
                return {"error": f"polygon {i} ring(s) {sub_pixel} span under a pixel in an axis; "
                                 f"rings are pixel coordinates, not normalized ones: {p!r}"}
            if any(_out_of_pixel_bounds(ring) for ring in rings_px):
                return {"error": f"polygon {i} rings look out of the image's pixel bounds "
                                 f"({img_w}x{img_h}): {rings_px!r}"}
        polygon_proposals.append(proposal)

    # A box of nothing is no detection: dropped rather than staged, so it never reaches the
    # accept branch (which would otherwise hand the persistence boundary a degenerate proposal).
    box_proposals: list[Annotation] = []
    dropped_boxes = 0
    for i, (stated, conf, cx, cy, w, h) in enumerate(norm_boxes):
        box = BBox.from_normalized_center((cx, cy, w, h), img_w, img_h)
        try:
            proposal = Annotation(subject=_StatedSubject.model_validate(stated).subject,
                                  geometry=box, score=conf,
                                  created_by=model_name, created_at=created_at)
        except ValueError as exc:
            return {"error": f"box {i}: {exc}"}
        if not stored_box_extent_ok(box):
            dropped_boxes += 1
            continue
        box_proposals.append(proposal)

    proposals: list[Annotation] = box_proposals + polygon_proposals

    try:
        bucket = _stage_document(project, address, model_name, img, annotations=proposals,
                                 img_w=img_w, img_h=img_h)
    except ValueError as exc:
        return {"error": str(exc)}

    note = ("staged as a prediction bucket for canvas review, not committed as ground truth; the "
            "human accepts each proposal on the Annotate tab before it becomes GT "
            "(focus_human_attention tab='annotate' to send them). It is never promoted to a "
            "reference.")

    return {
        "staged": len(proposals),
        "n_detect": len(box_proposals), "n_segment": len(polygon_proposals),
        "dropped_boxes": dropped_boxes,
        "bucket": bucket, "model_name": model_name, "date": address.parts[0], "stem": img.stem,
        "note": note,
    }


@tool()
def stage_proposals(
    project: Path,
    image_path: str,
    *,
    assignments: list[dict] | None = None,
    boxes: list[dict] | None = None,
    polygons: list[dict] | None = None,
    model_name: str | None = None,
) -> dict:
    """Stage model-/agent-proposed shapes as predictions for canvas review. Never writes ground
    truth.

    Exactly one input regime per call:

    - ``assignments``: candidates ``propose_annotations`` staged for this image, reviewed and each
      assigned a subject; a mapping from candidate id to subject, rejected candidates simply
      omitted. Reads back the record staged at this exact image (dataset, capture date and stem)
      and refuses if the image's content no longer matches the content identity that run recorded.
      The masks are staged with ``created_by=<engine>`` and ``score`` = the engine's proposal
      score; ``model_name`` is refused alongside ``assignments``.
    - ``boxes``/``polygons``: explicit shapes an agent or another model already has in hand, with
      no cached record to read back. ``model_name`` is required, the real producer stamped as
      ``created_by``. ``boxes`` is
      ``[{subject, conf, cx, cy, w, h}]`` with cx/cy/w/h normalized to [0, 1]; ``polygons`` is
      ``[{subject, conf, points|rings}]``, exactly one of two frames per proposal: ``points``, one
      ring of ``[x, y]`` pairs normalized to [0, 1]; or ``rings``, a list of rings in pixel
      coordinates, each vertex an ``[x, y]`` pair or an ``{"x":, "y":}`` mapping. Both build the
      same ``Polygon`` through the ground-truth door's own vertex parser.

    Either regime resolves the dataset root, capture date and stem from ``image_path`` itself, and
    publishes the image's proposal once as its own bucket named ``<producer>/<date>/<stem>`` under
    that root, returned as ``bucket``, its record naming the engine or ``model_name`` as
    what proposed it and no checkpoint, execution record or class scope: a proposal already staged
    for the image under the same producer refuses, so a re-run never overwrites reviewed
    predictions or orphans their verdicts, and no delivery ships one. Pair with
    ``focus_human_attention(tab='annotate')`` to send the human straight to the result.

    A staged annotation's ``subject`` is whatever ``assignments``/``boxes``/``polygons`` named;
    the platform validates no subject name.

    Args:
        image_path: Absolute path to the dataset image (same as propose_annotations, for the
            assignments regime).
        assignments: List of dicts, each with 'candidate_id' (int) and 'subject' (name). Refused
            alongside boxes/polygons or model_name.
        boxes: Explicit boxes; see above. Refused alongside assignments.
        polygons: Explicit polygons; see above. Refused alongside assignments.
        model_name: The producer the explicit regime stages under. Required with
            boxes/polygons, refused with assignments.
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
        return _stage_assignments_regime(project, image_path, img, address, assignments)

    if model_name is None:
        return {"error": "model_name is required with boxes/polygons: the real producer, "
                         "stamped as created_by."}
    return _stage_explicit_regime(project, image_path, img, address, model_name, boxes or [],
                                  polygons or [])
