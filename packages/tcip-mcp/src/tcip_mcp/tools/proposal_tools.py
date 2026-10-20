"""Proposal-workflow tools: turn an auto-labeling engine's output into staged predictions.

propose_annotations asks a named engine (one registered, or a 'module:factory' the agent brings)
to look at pixels and offer candidates, staging each run under an id of its own. stage_proposals
publishes either one run's reviewed candidates or explicit boxes/polygons once, as a proposal
bucket under the image's dataset root named for that id. It never writes ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

import tcip_store as ts
from tcip_annotation import Annotation, BBox, Polygon, bbox_of
from tcip_annotation.grid import grid_to_rect
from tcip_annotation.json_io import stored_box_extent_ok
from tcip_annotation.state import Rings
from tcip_annotation.viz import render_candidates, render_detections
from tcip_mcp.audit import now_iso, record_event_or_raise
from tcip_mcp.pipelines.execution import Execution, ExecutionRefusedError
from tcip_mcp.pipelines.image_utils import (
    BandGroupIncompleteError,
    resolve_image_path,
)
from tcip_mcp.pipelines.raster_source import RasterIdentity, SourceHeader
from tcip_mcp.project_paths import viz_output_path
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool

if TYPE_CHECKING:
    import numpy as np

    from tcip_mcp.pipelines.raster_source import Rect


PROPOSAL_STAGING_STORE = "proposal_staging"


class _StatedSubject(BaseModel):
    """One staged input's own ``subject``, required; the entry's other keys are allowed as given."""

    model_config = ConfigDict(extra="allow")

    subject: str


class ProposalCandidate(BaseModel):
    """One engine candidate in the neutral schema, coordinates in the source image's frame.

    ``rings`` is a :class:`~tcip_annotation.Polygon`'s, refused as that polygon refuses it, or
    ``None`` for a candidate whose geometry is its ``bbox``. The engine's other keys (its
    ``engine_meta`` signals) are kept as given, each a JSON value.
    """

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue]

    candidate_id: int
    bbox: tuple[float, float, float, float]
    area: float
    rings: (
        Annotated[Rings, AfterValidator(lambda rings: Polygon(rings).rings)]
        | None
    )
    score: float


class ProposalStaging(BaseModel):
    """The candidates one proposal run produced for one dataset image.

    It names the ``engine``, the content identity of the pixels it ran on, and the ``execution``
    record of the inference pass that produced the candidates (``None`` for a proposal no such
    pass made), which the bucket its accepted candidates publish to carries;
    :func:`proposal_staging_key` addresses it by the run's id.
    """

    model_config = ConfigDict(extra="forbid")

    engine: str
    candidates: list[ProposalCandidate]
    image_identity: RasterIdentity
    execution: Execution | None


class Assignment(BaseModel):
    """One staged candidate a reviewer kept, by its id, and the subject it is of."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: int
    subject: str


class StagedAssignments(BaseModel):
    """The candidates of one proposal run a reviewer kept, each assigned a subject.

    ``staging_id`` names the run, as :func:`propose_annotations` answers it; a rejected candidate
    is simply omitted.
    """

    model_config = ConfigDict(extra="forbid")

    staging_id: str
    candidates: list[Assignment]


def proposal_staging_key(
    dataset_root: str | Path, capture: str, stem: str, staging_id: str
) -> ts.Key:
    """Return the key of the candidates proposal run ``staging_id`` produced for a dataset image.

    The record holds the candidates whole, under the dataset root, by the image's capture and
    stem and the run's id.
    """
    return ts.Key(
        PROPOSAL_STAGING_STORE, str(dataset_root), (capture, stem, staging_id)
    )


def _staging_key_for(image_path: str, staging_id: str) -> ts.Key:
    """Return the :func:`proposal_staging_key` of run ``staging_id`` on the image at ``image_path``.

    Raises ``ValueError``, :func:`~tcip_mcp.dataset_layout.parse_image_path`'s own, for an image of
    no capture.
    """
    from tcip_mcp.dataset_layout import parse_image_path

    return proposal_staging_key(*parse_image_path(image_path), staging_id)


def _region_rect_from_cells(cells: list, names: list[str]) -> Rect:
    """Return the bounding rect, in the grid's native-pixel frame, of the named grid cells.

    Each name resolves through ``grid.grid_to_rect``, so a malformed or out-of-grid name is
    refused here, and the matched cells union to their combined bounding box.
    """
    from tcip_mcp.pipelines.raster_source import Rect

    matched = [grid_to_rect(name, cells) for name in names]
    return Rect(
        int(min(r[0] for r in matched)),
        int(min(r[1] for r in matched)),
        int(max(r[2] for r in matched)),
        int(max(r[3] for r in matched)),
    )


def _write_region_crop(pixels: np.ndarray) -> Path:
    """Save an RGB region crop to a fresh temp PNG file and return its path.

    The crop is taken from the raster layer's already-``auto_orient_image``'d frame, so it carries
    no EXIF orientation tag once saved. The returned temporary file is not removed here.
    """
    import os
    import tempfile

    from PIL import Image

    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="tcip_propose_crop_")
    os.close(fd)
    Image.fromarray(pixels, mode="RGB").save(tmp)
    return Path(tmp)


def _offset_candidates(
    candidates: list[ProposalCandidate], origin: tuple[float, float]
) -> list[ProposalCandidate]:
    """Translate candidates from a region crop's own pixels into the source image's full frame.

    Each candidate is moved by the crop's own origin
    (:func:`~tcip_mcp.pipelines.proposal_engines.shifted_candidate`).
    """
    from tcip_mcp.pipelines.proposal_engines import shifted_candidate

    return [
        ProposalCandidate.model_validate(
            shifted_candidate(c.model_dump(), origin)
        )
        for c in candidates
    ]


@tool()
def propose_annotations(
    project: Path,
    image_path: str,
    engine: str,
    engine_params: dict | None = None,
    grid_cells: list[str] | None = None,
    tile_size: int | None = None,
    overlap: float | None = None,
    *,
    actor: str | None,
) -> dict:
    """Propose candidate annotations on an image for review, using a named auto-labeling engine.

    Runs the engine's whole-image proposal pass, renders the numbered candidates, and returns the
    render path, the neutral candidate data and the ``staging_id`` a ``stage_proposals``
    acceptance names.

    Each candidate renders as a colored, semi-transparent filled polygon (every ring of an
    occlusion-split candidate drawn, not just the largest), or a box outline for a candidate whose
    ``rings`` is ``None``, with a large numbered label at the center of its ``bbox``, colors
    cycling through the shared class palette; the candidate id in that number is the same id
    ``stage_proposals``' ``assignments`` names.

    On an image under a dataset's ``images/`` tree, the run's candidates are staged as a record of
    their own, keyed by the dataset, capture date, stem and the run's fresh id
    (:func:`~tcip_mcp.experiments.mint_experiment_id`), which the response answers as
    ``staging_id``, alongside the content identity of the pixels the engine ran on; an earlier
    run's record stays as it was. On a path outside any dataset's ``images/`` tree the engine
    still runs and the render and candidates are returned the same way, but nothing is staged
    (``staged`` is ``false`` and ``staging_id`` ``None``, the summary naming why), and a run
    proposing nothing stages nothing the same way. A path naming no logical image (a band of a
    grouped capture, which names its manifest) refuses
    (:func:`~tcip_mcp.pipelines.image_utils.resolve_image_path`). Writing the staged record leaves
    one audit line in the dataset's log naming its ``staging_id``, by ``actor`` (the person who
    prompted the run, or ``None`` for a call no person makes); a call that writes none leaves none.

    ``engine`` names one registered or a dotted 'module:factory'. Two ship registered: ``sam``, a
    promptable segmenter (``engine_params``: ``points``, a list of ``[x, y]`` pixel points, and
    ``boxes``, a list of ``[x1, y1, x2, y2]`` pixel boxes, each its own prompt, at least one
    stated; ``multimask_output``; optional ``checkpoint``), and ``grounding_dino``, a
    text-prompted detector proposing boxes (``engine_params``: ``phrase``, ``box_threshold``,
    ``text_threshold``, ``postprocess``, ``cross_tile_nms``; optional ``tile_size``, ``overlap``
    and ``checkpoint``). The segmenter runs each prompt on a window cut at its processor's
    ``longest_edge`` around it, a wider box taking its own extent, and the processor scales the
    window's longer side to that edge, so a window whose longer side is the edge reaches the model
    at native resolution, a frame shorter than it upsampled and a wider box downsampled by its
    ratio to the edge; the detector is run by the platform's tiled inference pass on a lattice a
    whole-frame pass derives unless stated (``TILE_EDGE_DERIVATION``,
    ``TILE_OVERLAP_DERIVATION``), its execution record staged with the candidates. Each
    downloads its checkpoint's weights from the Hugging Face host on first use. An agent brings
    its own by ``register_proposal_engine`` or a dotted factory. An empty or unknown name
    refuses, naming the registered ones; ``engine_params`` the engine's own parameter model
    refuses, and a derivation the pass cannot make, come back as the error.

    ``grid_cells`` restricts the pass to a region instead of the whole frame: they are the
    reference-grid cell names (e.g. ``['B3', 'C3', 'B4', 'C4']``) whose bounding rect the engine
    proposes over. The crop is taken and the results offset back to
    full-frame coordinates on this side of the engine seam; an engine that keys behavior off the
    image path itself (a cache, a sidecar lookup keyed by the original file) receives the temp
    crop's path, which it cannot resolve back to the source image. Omitting ``grid_cells`` runs
    the whole frame.

    Args:
        image_path: The image file (:func:`~tcip_mcp.registry_paths.located` against the
            project).
        engine: Proposal engine: a registered name or a dotted 'module:factory' the agent brings.
        engine_params: The engine's own parameters, forwarded to its ``propose``, which
            validates them; the prompts travel here.
        grid_cells: Reference-grid cell names bounding the region to propose over (e.g. ['B3',
            'D5']); the engine sees the bounding rect of the named cells, not the whole frame.
            Requires ``tile_size`` and ``overlap``.
        tile_size: Cell edge, in native pixels, of the grid the cells were read off. Required with
            ``grid_cells``.
        overlap: Overlap fraction of the grid the cells were read off. Required with
            ``grid_cells``.

    """
    from tcip_mcp.pipelines.proposal import resolve_proposer

    image_path = str(located(image_path, project))
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    try:
        source = resolve_image_path(img)
        proposer = resolve_proposer(engine)
        raster = SourceHeader(source).open_at_route_count()
    except (
        FileNotFoundError,
        BandGroupIncompleteError,
        ValueError,
        ImportError,
    ) as e:
        return {"error": str(e)}
    # The region crop, the engine's pass, the render and the staged content identity all read
    # this one reader.
    with raster:
        # A region is cropped and offset entirely here, before the engine ever sees an image path.
        propose_path = image_path
        crop_tmp: Path | None = None
        origin: tuple[float, float] | None = None
        if grid_cells is not None:
            if not grid_cells:
                return {
                    "error": "grid_cells is empty; name at least one cell to scope the region."
                }
            if tile_size is None or overlap is None:
                return {
                    "error": "grid_cells requires tile_size and overlap, the geometry of the "
                    "grid the cells were read off (overlay_reference_grid echoes both "
                    "back). Without them a cell name resolves against a grid nobody "
                    "rendered."
                }
            from tcip_mcp.pipelines.reference_grid import reference_cells

            try:
                cells = reference_cells(
                    raster.width, raster.height, tile_size, overlap, clamp=True
                )
                rect = _region_rect_from_cells(cells, grid_cells)
                pixels, _spec = raster.read_region(rect)
            except ValueError as e:
                return {"error": str(e)}
            if pixels.dtype != "uint8" or pixels.shape[-1] != 3:
                return {
                    "error": "A region crop is handed to the engine as an RGB image, and "
                    f"{img.name} reads as {pixels.shape[-1]} band(s) of "
                    f"{pixels.dtype}. Propose over the whole frame instead, or "
                    "bring an engine that reads this source itself."
                }
            crop_tmp = _write_region_crop(pixels)
            propose_path = str(crop_tmp)
            origin = (float(rect.x0), float(rect.y0))

        try:
            try:
                proposed = proposer.propose(
                    propose_path, **(engine_params or {})
                )
            except (
                ImportError,
                FileNotFoundError,
                ValidationError,
                ExecutionRefusedError,
            ) as e:
                return {"error": str(e)}
        finally:
            if crop_tmp is not None:
                crop_tmp.unlink(missing_ok=True)

        # An engine's own candidates are refused here, where they arrive, when unstorable, rings
        # that are no polygon or stating no confidence of their own.
        try:
            candidates = TypeAdapter(list[ProposalCandidate]).validate_python(
                proposed.candidates
            )
        except ValidationError as exc:
            return {
                "error": f"Engine {engine!r} proposed a candidate this platform cannot "
                f"hold: {exc}"
            }
        if origin is not None:
            candidates = _offset_candidates(candidates, origin)

        if not candidates:
            return {
                "image_path": None,
                "engine": engine,
                "summary": f"Engine {engine!r} proposed no candidates",
                "staged": False,
                "staging_id": None,
                "candidates": [],
            }

        from tcip_mcp.tools.vision_tools import _read_for_display

        read = _read_for_display(raster)
        out = render_candidates(
            read.pixels,
            [c.model_dump() for c in candidates],
            native_size=read.native_size,
            output_path=viz_output_path(project, "candidates"),
        )

        from tcip_mcp.experiments import mint_experiment_id

        minted = mint_experiment_id("staging")
        staging_id: str | None = minted
        try:
            address = _staging_key_for(image_path, minted)
        except ValueError as exc:
            staging_id = None
            stage_note = f" Not staged: {exc}"
        else:
            from tcip_mcp.pipelines.raster_source import content_identity

            # The record names the engine so stage_proposals's assignments regime stamps the
            # right producer.
            ts.replace(
                address,
                ProposalStaging(
                    engine=engine,
                    candidates=candidates,
                    image_identity=content_identity(raster),
                    execution=proposed.execution,
                ).model_dump(mode="json"),
                expect=ts.Version.ABSENT,
            )
            record_event_or_raise(
                "propose_annotations",
                {
                    "image_path": image_path,
                    "staged": len(candidates),
                    "staging_id": staging_id,
                },
                actor=actor,
                scope=address.root,
            )
            stage_note = ""

    region_note = f" (region {grid_cells})" if origin is not None else ""
    return {
        "image_path": out,
        "engine": engine,
        "summary": f"Engine {engine!r} proposed {len(candidates)} candidates{region_note}."
        f"{stage_note} Review the numbered overlay, then call stage_proposals "
        f"with subject assignments.",
        "candidate_count": len(candidates),
        "staged": staging_id is not None,
        "staging_id": staging_id,
        "candidates": [
            {
                "id": c.candidate_id,
                "area": c.area,
                "score": round(c.score, 3),
                "bbox": [round(v, 1) for v in c.bbox],
            }
            for c in candidates
        ],
    }


def _stage_assignments_regime(
    project: Path,
    image_path: str,
    img: Path,
    address: ts.Key,
    assignments: StagedAssignments,
    actor: str | None,
) -> dict:
    """Stage the candidates ``assignments`` names, each with its subject, for ``img``, by ``actor``.

    The candidates come from the record of the run ``assignments`` names, staged at ``address``;
    refuses when that record is absent or the image's content no longer matches the content
    identity it recorded.
    """
    try:
        source = resolve_image_path(img)
    except (FileNotFoundError, BandGroupIncompleteError, ValueError) as exc:
        return {"error": str(exc)}

    stored = ts.read(address, default=None)
    if stored is None:
        return {
            "error": f"No proposal run {assignments.staging_id!r} is staged for {img.stem}. "
            "Run propose_annotations and pass the staging_id it answers."
        }
    staging = ProposalStaging.model_validate(stored)

    from tcip_mcp.pipelines.raster_source import open_as_recorded
    from tcip_mcp.tools.vision_tools import (
        _box_dict,
        _name_map,
        _read_for_display,
        _subject_indexer,
    )

    # One reader, opened at the count the identity recorded, verifies the content and renders it.
    identity = staging.image_identity
    try:
        raster, fresh = open_as_recorded(identity, source)
        with raster:
            if fresh is None:
                return {
                    "error": f"{image_path} does not match the image propose_annotations "
                    "ran on: its content has changed since that run staged these "
                    "candidates. Run propose_annotations again on the current "
                    "image."
                }
            read = _read_for_display(raster)
    except ValueError as exc:
        return {
            "error": f"Could not read {image_path} to verify it against its staged "
            f"proposals: {exc}"
        }

    engine = staging.engine
    cand_map = {c.candidate_id: c for c in staging.candidates}

    # A polygon keeps every ring, so an occlusion-split object stays split; no rings is a box.
    staged_at = now_iso()
    proposals: list[Annotation] = []

    for i, assign in enumerate(assignments.candidates):
        cand = cand_map.get(assign.candidate_id)
        if cand is None:
            continue
        try:
            proposals.append(
                Annotation(
                    subject=assign.subject,
                    geometry=BBox(*cand.bbox)
                    if cand.rings is None
                    else Polygon(rings=cand.rings),
                    score=cand.score,
                    created_by=engine,
                    created_at=staged_at,
                )
            )
        except ValueError as exc:
            return {"error": f"assignment {i}: {exc}"}

    # Model output for a human to accept on the Annotate canvas, never written straight to
    # ground truth.
    try:
        bucket = _stage_document(
            project,
            address,
            engine,
            img,
            annotations=proposals,
            img_w=identity.width,
            img_h=identity.height,
            execution=staging.execution,
            actor=actor,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    idx, index = _subject_indexer()
    out = render_detections(
        read.pixels,
        [_box_dict(a, index) for a in proposals],
        native_size=read.native_size,
        class_names=_name_map(idx),
        output_path=viz_output_path(project, "staged"),
    )

    note = (
        f"Staged {len(proposals)} proposal(s) from {len(assignments.candidates)} {engine!r} "
        "candidates as "
        f"predictions (created_by={engine!r}) for review, not ground truth."
    )

    return {
        "image_path": out,
        "engine": engine,
        "bucket": bucket,
        "summary": note,
        "proposal_count": len(proposals),
    }


def _stage_document(
    project: Path,
    address: ts.Key,
    producer: str,
    image: Path,
    *,
    annotations: list[Annotation],
    img_w: int,
    img_h: int,
    execution: Execution | None,
    actor: str | None,
) -> str | None:
    """Publish ``annotations`` as the bucket of one staged proposal for ``image``; return its name.

    The bucket is named ``<producer>/<capture>/<stem>/<id>`` under the image's dataset root, all
    three read off the staging key ``address`` (:func:`proposal_staging_key`), so the name says
    which staging it publishes and never lands on an earlier one
    (:func:`~tcip_mcp.buckets.publish`); its record names ``producer`` as what proposed them and
    carries ``execution``, the record of the inference pass that produced them, and its
    publication's audit line ``actor`` as who staged them. No annotations publishes nothing and
    returns ``None``.
    """
    from tcip_annotation import json_io
    from tcip_mcp.buckets import Document, Proposed, publish
    from tcip_mcp.pipelines.data.selection import ClassScope

    if not annotations:
        return None
    data = json_io.document_payload(annotations, img_w, img_h)
    assert data is not None, "a non-empty proposal encodes a document"
    name = "/".join([producer, *address.parts])
    return publish(
        project,
        Path(address.root),
        name,
        [Document(str(image), data)],
        producer=Proposed(proposed_by=producer),
        scope=ClassScope(),
        execution=execution,
        raster_path=None,
        raster_identity=None,
        assessment_id=None,
        actor=actor,
    ).name


def _stage_explicit_regime(
    project: Path,
    image_path: str,
    img: Path,
    address: ts.Key,
    model_name: str,
    boxes: list[dict],
    polygons: list[dict],
    actor: str | None,
) -> dict:
    """Stage explicit boxes and polygons, each carrying a ``subject``, as one staged proposal.

    The proposal is ``model_name``'s for the image, staged by ``actor`` (:func:`_stage_document`).
    """
    from tcip_annotation.json_io import ring_vertex
    from tcip_mcp.workspace import is_valid_name

    if not is_valid_name(model_name):
        return {
            "error": f"model_name must be a single safe path segment (no separators/'..'), "
            f"got {model_name!r}"
        }

    def _unnormalized(vals: Iterable[float]) -> bool:
        return any(v < -0.01 or v > 1.5 for v in vals)

    norm_boxes: list[tuple[Any, float, float, float, float, float]] = []
    for i, b in enumerate(boxes):
        try:
            cx, cy, w, h = (
                float(b["cx"]),
                float(b["cy"]),
                float(b["w"]),
                float(b["h"]),
            )
            conf = float(b["conf"])
        except (KeyError, TypeError, ValueError):
            return {
                "error": f"box {i} needs numeric conf, cx, cy, w, h (normalized): {b!r}"
            }
        if _unnormalized((cx, cy, w, h)):
            return {
                "error": f"box {i} coords {(cx, cy, w, h)} look un-normalized; "
                "cx/cy/w/h must be in [0,1]"
            }
        norm_boxes.append((b, conf, cx, cy, w, h))

    try:
        img_source = resolve_image_path(img)
    except (FileNotFoundError, BandGroupIncompleteError, ValueError) as exc:
        return {"error": str(exc)}
    img_w, img_h = SourceHeader(img_source).display_frame

    # A rounding-slop margin in pixels, not a fraction of the image size: a fractional margin
    # admits a normalized [0,1] ring at every real image size, the bug this check exists to refuse.
    pixel_margin = 1.0

    def _spans_a_pixel(ring: list[tuple[float, float]]) -> bool:
        b = bbox_of(Polygon(rings=[ring]))
        return b.x2 - b.x1 >= 1.0 and b.y2 - b.y1 >= 1.0

    def _out_of_pixel_bounds(ring: list[tuple[float, float]]) -> bool:
        return any(
            x < -pixel_margin or x > img_w + pixel_margin for x, _ in ring
        ) or any(
            y < -pixel_margin or y > img_h + pixel_margin for _, y in ring
        )

    created_at = now_iso()

    # Each polygon carries exactly one of two keys, folded into pixel-space rings as it is parsed;
    # the fold needs img_w/img_h, resolved above.
    polygon_proposals: list[Annotation] = []
    for i, p in enumerate(polygons):
        has_points = "points" in p
        has_rings = "rings" in p
        if has_points == has_rings:
            offered = sorted(k for k in ("points", "rings") if k in p)
            return {
                "error": f"polygon {i} must carry exactly one of 'points' or 'rings', got "
                f"{offered}: {p!r}"
            }
        try:
            conf = float(p["conf"])
        except (KeyError, TypeError, ValueError):
            return {"error": f"polygon {i} needs a numeric conf: {p!r}"}

        if has_points:
            try:
                pts = [(float(x), float(y)) for x, y in p["points"]]
            except (TypeError, ValueError):
                return {
                    "error": f"polygon {i} points must be [x, y] pairs (normalized): {p!r}"
                }
            if _unnormalized([v for xy in pts for v in xy]):
                return {
                    "error": f"polygon {i} points look un-normalized; x/y must be in [0,1]"
                }
            rings_px = [[(x * img_w, y * img_h) for x, y in pts]]
        else:
            try:
                rings_px = [
                    [(float(x), float(y)) for x, y in map(ring_vertex, ring)]
                    for ring in p["rings"]
                ]
            except (TypeError, ValueError, KeyError):
                return {
                    "error": f"polygon {i} rings must be a list of rings of [x, y] pairs or "
                    f"{{'x':, 'y':}} mappings (pixel coordinates): {p!r}"
                }
        try:
            proposal = Annotation(
                subject=_StatedSubject.model_validate(p).subject,
                geometry=Polygon(rings=rings_px),
                score=conf,
                created_by=model_name,
                created_at=created_at,
            )
        except ValueError as exc:
            return {"error": f"polygon {i}: {exc}"}
        if has_rings:
            sub_pixel = [
                j
                for j, ring in enumerate(rings_px)
                if not _spans_a_pixel(ring)
            ]
            if sub_pixel:
                return {
                    "error": f"polygon {i} ring(s) {sub_pixel} span under a pixel in an axis; "
                    f"rings are pixel coordinates, not normalized ones: {p!r}"
                }
            if any(_out_of_pixel_bounds(ring) for ring in rings_px):
                return {
                    "error": f"polygon {i} rings look out of the image's pixel bounds "
                    f"({img_w}x{img_h}): {rings_px!r}"
                }
        polygon_proposals.append(proposal)

    # A box of nothing is no detection: dropped rather than staged, so it never reaches the
    # accept branch (which would otherwise hand the persistence boundary a degenerate proposal).
    box_proposals: list[Annotation] = []
    dropped_boxes = 0
    for i, (stated, conf, cx, cy, w, h) in enumerate(norm_boxes):
        box = BBox.from_normalized_center((cx, cy, w, h), img_w, img_h)
        try:
            proposal = Annotation(
                subject=_StatedSubject.model_validate(stated).subject,
                geometry=box,
                score=conf,
                created_by=model_name,
                created_at=created_at,
            )
        except ValueError as exc:
            return {"error": f"box {i}: {exc}"}
        if not stored_box_extent_ok(box):
            dropped_boxes += 1
            continue
        box_proposals.append(proposal)

    proposals: list[Annotation] = box_proposals + polygon_proposals

    try:
        bucket = _stage_document(
            project,
            address,
            model_name,
            img,
            annotations=proposals,
            img_w=img_w,
            img_h=img_h,
            execution=None,
            actor=actor,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    note = (
        "staged as a prediction bucket for canvas review, not committed as ground truth; the "
        "human accepts each proposal on the Annotate tab before it becomes GT "
        "(focus_human_attention tab='annotate' to send them). It is never promoted to a "
        "reference."
    )

    return {
        "staged": len(proposals),
        "n_detect": len(box_proposals),
        "n_segment": len(polygon_proposals),
        "dropped_boxes": dropped_boxes,
        "bucket": bucket,
        "model_name": model_name,
        "date": address.parts[0],
        "stem": img.stem,
        "note": note,
    }


@tool()
def stage_proposals(
    project: Path,
    image_path: str,
    *,
    assignments: StagedAssignments | None = None,
    boxes: list[dict] | None = None,
    polygons: list[dict] | None = None,
    model_name: str | None = None,
    actor: str | None,
) -> dict:
    """Stage model-/agent-proposed shapes as an image's proposal bucket of predictions.

    Never writes ground truth.

    Exactly one input regime per call:

    - ``assignments``: the candidates one ``propose_annotations`` run staged for this image, named
      by the run's ``staging_id``, reviewed and each assigned a subject
      (``{staging_id, candidates: [{candidate_id, subject}]}``), rejected candidates simply
      omitted. Reads back that run's record for this exact image (dataset, capture date, stem and
      id) and refuses if the image's content no longer matches the content identity that run
      recorded. Each is staged as its polygon, or its box when it has no rings, with
      ``created_by=<engine>`` and ``score`` = the engine's proposal score; ``model_name`` is
      refused alongside ``assignments``.
    - ``boxes``/``polygons``: explicit shapes an agent or another model already has in hand, with
      no cached record to read back. ``model_name`` is required, the real producer stamped as
      ``created_by``. ``boxes`` is
      ``[{subject, conf, cx, cy, w, h}]`` with cx/cy/w/h normalized to [0, 1]; ``polygons`` is
      ``[{subject, conf, points|rings}]``, exactly one of two frames per proposal: ``points``, one
      ring of ``[x, y]`` pairs normalized to [0, 1]; or ``rings``, a list of rings in pixel
      coordinates, each vertex an ``[x, y]`` pair or an ``{"x":, "y":}`` mapping. Both build the
      same ``Polygon`` through ``json_io.ring_vertex``.

    Either regime resolves the dataset root, capture date and stem from ``image_path`` itself, and
    publishes the image's proposal as a new bucket named ``<producer>/<date>/<stem>/<id>`` under
    that root, returned as ``bucket``: ``id`` is the run's ``staging_id`` for ``assignments`` and
    a fresh one (:func:`~tcip_mcp.experiments.mint_experiment_id`) for explicit shapes. Its record
    names the engine or ``model_name`` as what proposed it, carries the execution record of the
    tiled pass the proposal ran through, ``None`` where none ran (a prompt window, explicit
    shapes), and names no checkpoint or class scope. A bucket is published once: a second stage
    of the same run refuses (:class:`~tcip_mcp.buckets.BucketExistsError`), and a new run or new
    explicit shapes publish a bucket of their own, so nothing overwrites reviewed predictions or
    orphans their verdicts. The publication's audit line names ``actor``, the person who staged
    it, or ``None`` for a call no person makes.

    A staged annotation's ``subject`` is whatever ``assignments``/``boxes``/``polygons`` named;
    the platform validates no subject name.

    Args:
        image_path: The dataset image, as propose_annotations takes it (the same one, for the
            assignments regime).
        assignments: The run's ``staging_id`` and its kept ``candidates``, each a
            ``candidate_id`` (int) and a ``subject`` (name). Refused alongside boxes/polygons or
            model_name.
        boxes: Explicit boxes; see above. Refused alongside assignments.
        polygons: Explicit polygons; see above. Refused alongside assignments.
        model_name: The producer the explicit regime stages under. Required with
            boxes/polygons, refused with assignments.

    """
    if assignments is not None and (boxes or polygons):
        return {
            "error": "assignments cannot be combined with boxes/polygons: pick one input "
            "regime per call."
        }
    if assignments is None and not boxes and not polygons:
        return {
            "error": "provide assignments (propose_annotations's staged candidates), or "
            "boxes/polygons (explicit shapes) with model_name."
        }

    image_path = str(located(image_path, project))
    img = Path(image_path)
    if not img.is_file():
        return {"error": f"Image not found: {image_path}"}

    from tcip_mcp.experiments import mint_experiment_id

    try:
        address = _staging_key_for(
            image_path,
            assignments.staging_id
            if assignments is not None
            else mint_experiment_id("staging"),
        )
    except ValueError as exc:
        return {"error": str(exc)}

    if assignments is not None:
        if model_name is not None:
            return {
                "error": "model_name is refused alongside assignments: the staged record's "
                "own engine names the bucket."
            }
        return _stage_assignments_regime(
            project, image_path, img, address, assignments, actor
        )

    if model_name is None:
        return {
            "error": "model_name is required with boxes/polygons: the real producer, "
            "stamped as created_by."
        }
    return _stage_explicit_regime(
        project,
        image_path,
        img,
        address,
        model_name,
        boxes or [],
        polygons or [],
        actor,
    )
