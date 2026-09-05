"""Block-aware calibration/holdout: validate a detection operating point directly against a
mosaic's own reserved calibration/test bands (see ``split_construction.spatial_single_source_split``'s
four-way split, ``reserve_calibration_fraction``), for a raster training source too large or too
singular to hold whole images out from.

Ties together the region-completeness gate (:mod:`region_completeness`), the halo mechanism
(:func:`~tcip_mcp.pipelines.data.tiling.region_halo`), a recursive
:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split` sub-banding of each reserved region,
and per-band record building (shaped like ``pipelines.calibration.calibrate_operating_point``'s own
per-image records) into :func:`~tcip_mcp.pipelines.operating_point.resolve_operating_point`, the
same gate every other calibration path resolves through -- never a second, parallel validation
mechanism.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Enough bands to measure a per-band bias spread (resolve_operating_point's own equivalence gate
# needs n >= 2 present images per side) without fragmenting a modest region into slivers.
DEFAULT_K_CAL = 3
DEFAULT_K_TEST = 3


class BlockCalibrationRefused(ValueError):
    """A named block-calibration refusal: completeness, feasibility, or a resolution precondition
    (no experiment_id, no spatial-strip split, no reserved calibration region). Each message
    states exactly what's missing, never a generic gate failure."""


def _reserved_spatial_regions(split: dict) -> dict | None:
    """The reserved calibration/test regions of a spatial-strip split, or ``None``.

    ``None`` when the manifest is not a spatial-strip split at all, or when it reserved no
    calibration or no test region: the one predicate behind both the cheap precheck and the
    resolver's own refusals, so the two cannot disagree about whether a run qualifies.
    """
    if split.get("group_by") != "spatial_strip":
        return None
    spatial = split.get("spatial") or {}
    if not spatial.get("calibration_region") or not spatial.get("test_region"):
        return None
    return spatial


def reserved_calibration_region_available(experiment_id: str) -> bool:
    """Whether ``experiment_id``'s own persisted split is a spatial-strip split with a non-empty
    reserved ``calibration`` region: the cheap, dataset-free precondition check
    ``run_inference`` runs before deciding whether a ``trait`` + ``raster_path`` export may
    proceed into block calibration at all, rather than the unconditional refusal it used to be.
    """
    from tcip_mcp.experiments import read_split_manifest

    return _reserved_spatial_regions(read_split_manifest(experiment_id)) is not None


def _band_rects(
    region_rect: tuple[int, int, int, int], k: int, tile_size: int, overlap: float,
    buffer_px: int, seed: int, name_prefix: str,
) -> dict[str, tuple[int, int, int, int]]:
    """``k`` non-overlapping, buffered bands over ``region_rect``, in full-mosaic coordinates.

    Recurses :func:`spatial_strip_split` over the region's own local extent (its lattice starts at
    local ``(0, 0)``), then translates every returned rect back by the region's own origin
    (``+ x0, + y0``), the clean, lattice-phase-safe translation confirmed by design review.
    """
    from tcip_mcp.pipelines.data.splits import spatial_strip_split

    x0, y0, x1, y1 = region_rect
    width, height = x1 - x0, y1 - y0
    names = tuple(f"{name_prefix}_{i}" for i in range(k))
    fractions = tuple(1.0 / k for _ in range(k))
    spatial = spatial_strip_split(
        width, height, tile_size, overlap, fractions=fractions, split_names=names,
        seed=seed, buffer=buffer_px,
    )
    out: dict[str, tuple[int, int, int, int]] = {}
    for name in names:
        rects = spatial.regions.get(name) or []
        if not rects:
            continue
        # stripes_per_split defaults to 1 (never overridden here): one contiguous rect per band.
        lx0, ly0, lx1, ly1 = rects[0]
        out[name] = (lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0)
    return out


def _select_gt_for_band(
    gt_boxes_xyxy: np.ndarray, gt_labels: np.ndarray, band_rect: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """This band's own GT: boxes whose center lands in ``band_rect``, kept at their full extent
    (never clipped) and translated to the band's own local (inner-rect-relative) pixel space.

    Mirrors the DT side's own inclusion rule exactly (:func:`_band_records`'s center-in-inner-rect
    keep test) rather than clipping GT to the rect: a straddling object must be included or
    excluded consistently on both sides of the curve derivation, or the reference is biased at every band
    boundary (a clipped-but-kept GT box paired against a dropped, off-center detection reads as a
    false negative that a real delivered pass would never report).
    """
    if len(gt_boxes_xyxy) == 0:
        return gt_boxes_xyxy, gt_labels
    x0, y0, x1, y1 = band_rect
    cx = (gt_boxes_xyxy[:, 0] + gt_boxes_xyxy[:, 2]) / 2.0
    cy = (gt_boxes_xyxy[:, 1] + gt_boxes_xyxy[:, 3]) / 2.0
    keep = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
    kept = gt_boxes_xyxy[keep].astype(np.float64, copy=True)
    kept[:, [0, 2]] -= x0
    kept[:, [1, 3]] -= y0
    return kept, gt_labels[keep]


def _check_completeness(
    dataset_root: str, subject: str, stem: str, rects: dict[str, tuple[int, int, int, int]],
) -> None:
    """Refuse by name (the incomplete/stale cells and the subject) unless every rect in ``rects``
    is fully covered by an attested-complete, non-stale region-completeness record. Called against
    the whole reserved calibration/test regions, before any sub-banding: every band is a sub-rect
    of its own region, so a region-level pass covers every band by construction, and checking here
    (rather than per band, after sub-banding might itself fail) keeps this the first thing that
    can refuse, never masked by a later geometry/density error."""
    from tcip_mcp.pipelines.region_completeness import incomplete_cells_for_rect

    problems: list[str] = []
    for name, rect in sorted(rects.items()):
        missing = incomplete_cells_for_rect(dataset_root, subject, stem, rect)
        if missing is None:
            problems.append(f"{name}: no region-completeness record exists for subject {subject!r}")
        elif missing:
            problems.append(f"{name}: cells not attested complete for subject {subject!r}: {missing}")
    if problems:
        raise BlockCalibrationRefused(
            "block calibration refused: the reserved calibration/test regions are not fully "
            f"attested complete for subject {subject!r} -- {'; '.join(problems)}. Attest every "
            "listed cell complete (the Annotate canvas's Attest control) before block "
            "calibration can treat this region's GT as trustworthy."
        )


def _check_feasibility(gt_counts: dict[str, int], *, side: str, min_present: int = 2) -> None:
    """Refuse by name when fewer than ``min_present`` bands on this side carry any GT at all:
    :func:`~tcip_mcp.pipelines.operating_point.resolve_operating_point`'s own equivalence gate
    already refuses a side with under 2 present images, checked here first (before paying for any
    tiled inference) so a doomed layout fails cheaply and with a block-scoped explanation, not a
    generic downstream holdout-gate message."""
    n_present = sum(1 for c in gt_counts.values() if c > 0)
    if n_present < min_present:
        raise BlockCalibrationRefused(
            f"block calibration refused: the resolved {side} band layout "
            f"({len(gt_counts)} band(s)) leaves only {n_present} band(s) with any GT, fewer than "
            f"the {min_present} an equivalence check needs; reduce k_{side}, widen the reserved "
            f"fraction, or check region completeness/annotation density for this mosaic."
        )


def _density_uniformity_flags(gt_counts: dict[str, int], *, factor: float = 3.0) -> list[str]:
    """Band names whose GT count is a stark outlier (more than ``factor``x the median, or less
    than ``1/factor``x it) relative to its siblings: a cheap smoke check for a plausible
    attestation error (a region marked complete after only part of it was actually annotated),
    flagged in provenance, never gating on its own."""
    import statistics

    counts = [c for c in gt_counts.values() if c > 0]
    if len(counts) < 3:
        return []
    median = statistics.median(counts)
    if median <= 0:
        return []
    return sorted(
        name for name, c in gt_counts.items()
        if c > 0 and (c > factor * median or c < median / factor)
    )


def resolve_block_calibration_records(
    predictor: Any, *, trait_name: str, experiment_id: str | None,
    global_nms_iou: float, export_tile_size: int, tile_batch_size: int = 96, postprocess: str = "nms",
    k_cal: int = DEFAULT_K_CAL, k_test: int = DEFAULT_K_TEST, seed: int = 0,
) -> tuple[Any, dict, dict]:
    """Resolve a detection operating point directly against a mosaic's own reserved
    calibration/test regions. Returns ``(bundle, provenance, evidence)``: ``bundle`` is the same
    :class:`~tcip_mcp.pipelines.resolution.ResolvedBundle` shape every calibration path returns;
    ``provenance`` carries the resolved ``dataset_root``/``stem``/``training_raster_path``/
    ``spatial_manifest`` (for the caller's own claim-scope check against the actual export target)
    plus ``density_uniformity_flags``; ``evidence`` is the resolver this ran, the arguments it ran
    over and the reserved regions they came from, which is what an export door earning a validation
    record for a validated block-calibrated count reopens the gate with. The arguments are the same
    dict passed to ``resolve_operating_point`` here, never a second assembly of them; the trait and
    the producing run are left out because they are ``open_validation``'s own arguments.

    ``experiment_id`` unresolved (``None``, or given but not found) refuses outright, unlike the
    ordinary image-set calibration path's legitimate "foreign checkpoint, skip the disjointness
    check" allowance (:func:`~tcip_mcp.pipelines.operating_point._train_disjointness`): this path's
    whole premise is one specific checkpoint validated against one specific mosaic's own reserved
    regions, so an unresolvable experiment_id means there is no mosaic to validate against at all,
    not a foreign-but-harmless case to skip past.

    The block scale (:func:`~tcip_mcp.pipelines.derivations.derive_block_scale_px`) prefers a
    real planting-grid pitch over the GT-object-spacing fallback when the training experiment's
    own ``config.json`` (``data.plant_csv_paths``, a list of plant-locations CSV paths) resolves
    at least two georeferenced plants and the training raster's own pixel size is resolvable
    (:func:`~tcip_mcp.pipelines.pixel_size.resolve_pixel_size`); a ``BandGroupRef`` source (no
    single file to read tags from) always falls back to GT-spacing.

    ``export_tile_size`` (required, no default) is the edge the caller's own whole-mosaic export
    pass resolved to run at; refused when it differs from the split manifest's own ``tile_size``,
    naming both, so the reserved-region claim this resolves and the bucket the export pass writes
    are tiled at one regime by construction, never two values nothing holds equal.
    """
    from tcip_store import store

    from tcip_mcp.experiments import config_key, read_split_manifest

    if experiment_id is None:
        raise BlockCalibrationRefused(
            "block calibration refused: this checkpoint's training experiment_id could not be "
            "resolved (no stamped experiment_id, and the registry entries naming this checkpoint "
            "bind no run: register it in experiment mode to bind one); block calibration validates "
            "against one specific mosaic's own reserved regions and has no meaning without "
            "knowing which training run's split produced them."
        )

    split = read_split_manifest(experiment_id)
    if split.get("group_by") != "spatial_strip":
        raise BlockCalibrationRefused(
            f"block calibration refused: experiment {experiment_id!r} has no spatial-strip split "
            "manifest (split.json's group_by != 'spatial_strip'); block calibration only applies "
            "to a single-mosaic training run with a reserved calibration region."
        )
    spatial = _reserved_spatial_regions(split)
    if spatial is None:
        raise BlockCalibrationRefused(
            f"block calibration refused: experiment {experiment_id!r}'s spatial split reserved no "
            "calibration region (train it with data.split.reserve_calibration_fraction set) or no "
            "test region."
        )
    cal_region = spatial["calibration_region"]
    test_region = spatial["test_region"]
    stem = spatial.get("stem")
    if not stem:
        raise BlockCalibrationRefused(
            f"block calibration refused: experiment {experiment_id!r}'s spatial manifest carries "
            "no stem to resolve the training mosaic's own image/label files from."
        )

    config = store.read(config_key(experiment_id), default={})
    data_cfg = (config.get("data") if isinstance(config, dict) else None) or {}
    labels_dir, images_dir = data_cfg.get("labels_dir"), data_cfg.get("images_dir")
    subject, attribute = data_cfg.get("subject"), data_cfg.get("attribute")
    if not labels_dir or not images_dir:
        raise BlockCalibrationRefused(
            f"block calibration refused: experiment {experiment_id!r}'s config.json carries no "
            "data.labels_dir/data.images_dir to resolve the training mosaic's own files from."
        )
    if not subject:
        raise BlockCalibrationRefused(
            f"block calibration refused: experiment {experiment_id!r}'s training config carries "
            "no registered subject; block calibration reads ground truth through the registry the "
            "same way every other calibration path does."
        )

    from tcip_mcp.dataset_layout import dataset_root_of

    dataset_root = dataset_root_of(labels_dir)
    if dataset_root is None:
        raise BlockCalibrationRefused(
            f"block calibration refused: {labels_dir!r} is not under a recognized dataset root; "
            "the region-completeness store resolves from the dataset root the mosaic's own label "
            "file lives under."
        )

    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.image_utils import resolve_image_source
    from tcip_mcp.tools.inference_tools import resolve_decode_id_map

    id_map = resolve_decode_id_map(predictor, labels_dir, scope=(subject, attribute))
    if id_map is None:
        raise BlockCalibrationRefused(
            "block calibration refused: this checkpoint records no name->id map and none could be "
            f"derived from a registry for {labels_dir!r}; the mosaic's ground truth is decoded "
            "through that map, so there is nothing to read it with."
        )

    gt_path = str(Path(labels_dir) / f"{stem}.json")
    gt_boxes, gt_labels, n_unlabeled = json_det_targets(gt_path, subject, attribute, id_map)
    if n_unlabeled:
        raise BlockCalibrationRefused(
            f"block calibration refused: {n_unlabeled} instance(s) in {stem!r} are unlabeled for "
            f"attribute {attribute!r}. The ordinary calibration path drops a whole image with any "
            "unlabeled instance rather than score against partial GT; a block calibration has only "
            "one image (the mosaic), so there is no partial-image exclusion available here, and "
            "scoring the model's real detections of these instances as false positives would "
            "silently bias the calibrated confidence. Label every instance for this attribute in "
            "the reserved regions, or calibrate a trait with no attribute scope."
        )
    gt_boxes_arr = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    gt_labels_arr = np.asarray(gt_labels, dtype=np.int64)

    tile_size, overlap = int(spatial["tile_size"]), float(spatial["overlap"])
    if tile_size != int(export_tile_size):
        raise BlockCalibrationRefused(
            f"block calibration refused: the split manifest's reserved regions were tiled at "
            f"{tile_size}px, but this export is resolved to run the whole-mosaic pass at "
            f"{int(export_tile_size)}px; the reserved-region claim and the exported bucket must be "
            "tiled at one regime, or the claim says nothing about the counts the export actually "
            "produces."
        )
    mosaic_w, mosaic_h = int(spatial["width"]), int(spatial["height"])

    def _region_rect(region: list) -> tuple[int, int, int, int]:
        # stripes_per_split's default (1) guarantees exactly one contiguous rect per side.
        return tuple(region[0])

    cal_rect, test_rect = _region_rect(cal_region), _region_rect(test_region)

    # Completeness first, against the whole reserved regions, before any sub-banding: checking
    # only after band geometry resolves could misreport an unattested region as a geometry error.
    _check_completeness(
        str(dataset_root), subject, stem, {"calibration_region": cal_rect, "test_region": test_rect})

    from tcip_mcp.pipelines.derivations import derive_block_scale_px

    def _in_regions(cx: float, cy: float, regions: list) -> bool:
        return any(rx0 <= cx < rx1 and ry0 <= cy < ry1 for rx0, ry0, rx1, ry1 in regions)

    centers = (gt_boxes_arr[:, :2] + gt_boxes_arr[:, 2:]) / 2.0 if len(gt_boxes_arr) else gt_boxes_arr
    # One real spatial scale, pooled across both reserved regions' own GT.
    reserved_mask = np.array([
        _in_regions(cx, cy, cal_region) or _in_regions(cx, cy, test_region) for cx, cy in centers
    ], dtype=bool) if len(centers) else np.zeros((0,), dtype=bool)
    reserved_boxes_xywh = [
        [x1, y1, x2 - x1, y2 - y1]
        for (x1, y1, x2, y2) in gt_boxes_arr[reserved_mask].tolist()
    ]
    training_source = resolve_image_source(images_dir, stem)

    plants = None
    plant_csv_paths = data_cfg.get("plant_csv_paths")
    if plant_csv_paths:
        from tcip_mcp.pipelines.postprocessing.plant_mapping import read_plant_csvs

        plants = read_plant_csvs([Path(p) for p in plant_csv_paths]) or None

    from tcip_mcp.pipelines.raster_source import BandGroupRef

    raster_path_for_scale = (
        training_source if not isinstance(training_source, BandGroupRef) else None)

    try:
        buffer_px, scale_source = derive_block_scale_px(
            tile_size=tile_size, gt_boxes_per_image=[reserved_boxes_xywh],
            plants=plants, raster_path=raster_path_for_scale)
    except ValueError as exc:
        raise BlockCalibrationRefused(
            f"block calibration refused: no block scale is derivable for the reserved "
            f"regions' own GT ({exc}); the reserved calibration/test regions need at least two "
            "GT objects between them to derive a spatial scale from."
        ) from exc

    try:
        cal_bands = _band_rects(cal_rect, k_cal, tile_size, overlap, buffer_px, seed, "cal")
        test_bands = _band_rects(test_rect, k_test, tile_size, overlap, buffer_px, seed + 1, "test")
    except ValueError as exc:
        raise BlockCalibrationRefused(
            f"block calibration refused: the resolved band layout (k_cal={k_cal}, k_test={k_test}, "
            f"block scale {buffer_px}px from {scale_source}) is infeasible for the reserved "
            f"regions' own extent ({exc}); reduce k_cal/k_test or widen the reserved fraction."
        ) from exc

    def _band_gt_counts(bands: dict[str, tuple[int, int, int, int]]) -> dict[str, int]:
        counts = {}
        for name, rect in bands.items():
            b, _l = _select_gt_for_band(gt_boxes_arr, gt_labels_arr, rect)
            counts[name] = len(b)
        return counts

    cal_gt_counts, test_gt_counts = _band_gt_counts(cal_bands), _band_gt_counts(test_bands)
    _check_feasibility(cal_gt_counts, side="cal")
    _check_feasibility(test_gt_counts, side="test")
    density_flags = (
        _density_uniformity_flags(cal_gt_counts) + _density_uniformity_flags(test_gt_counts))
    if density_flags:
        logger.warning("block calibration: GT-density outlier band(s) relative to siblings: %s",
                       density_flags)

    from tcip_mcp.pipelines.operating_point import (
        derive_max_dets_from_counts, set_detector_operating_point,
    )

    density_cap = derive_max_dets_from_counts(
        list(cal_gt_counts.values()) + list(test_gt_counts.values()))

    applied, applied_attribute_path = set_detector_operating_point(
        predictor.model, score_thresh=0.01, detections_per_img=density_cap)
    predictor.score_threshold = applied.get("score_thresh", 0.01)
    # predict_tiled's separate post-merge full-band cap is never reset by construction alone.
    predictor.max_dets = density_cap

    from tcip_mcp.pipelines.raster_source import open_raster

    with open_raster(training_source, predictor.in_chans) as reader:
        if (reader.width, reader.height) != (mosaic_w, mosaic_h):
            raise BlockCalibrationRefused(
                f"block calibration refused: the split manifest's recorded mosaic dimensions "
                f"({mosaic_w}x{mosaic_h}) do not match the actual raster {training_source}'s "
                f"current dimensions ({reader.width}x{reader.height}); the raster was likely "
                f"replaced or truncated since training. Retrain or re-split against the current "
                f"file before block calibration can trust the reserved regions' geometry."
            )
        cal_records, cal_rects = _band_records(
            reader, cal_bands, mosaic_w, mosaic_h, tile_size, overlap, predictor,
            tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, postprocess=postprocess,
            gt_boxes=gt_boxes_arr, gt_labels=gt_labels_arr,
            stem=stem,
        )
        test_records, test_rects = _band_records(
            reader, test_bands, mosaic_w, mosaic_h, tile_size, overlap, predictor,
            tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou, postprocess=postprocess,
            gt_boxes=gt_boxes_arr, gt_labels=gt_labels_arr,
            stem=stem,
        )

    from tcip_mcp.dataset_layout import annotation_date
    from tcip_mcp.pipelines.operating_point import (
        attach_spatial_split_kind_provenance, resolve_operating_point,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash

    dh = dataset_hash(labels_dir)
    # Explicit dict[str, Any] so the **resolver_inputs splat below checks against each of
    # resolve_operating_point's differently-typed keyword parameters.
    resolver_inputs: dict[str, Any] = {
        "dataset_hash": dh, "calibration_records": cal_records, "holdout_records": test_records,
        "tiled": True, "tiled_source": "default", "cross_tile_nms": None, "max_dets": density_cap,
        "max_dets_derived_from": (
            "~1.5x p99 GT objects/image, pooled across all calibration+test bands"),
        "staged_conf_floor": applied.get("score_thresh"),
        "staged_conf_floor_attribute_path": applied_attribute_path,
        "cal_rects": cal_rects, "hold_rects": test_rects,
        # No manifest on this route; not-applicable regardless via the record's spatial_strip.
        "split_manifest_dir": None, "calibration_date": annotation_date(labels_dir),
    }
    bundle = resolve_operating_point(trait_name, experiment_id=experiment_id, **resolver_inputs)
    attach_spatial_split_kind_provenance(bundle, spatial)

    provenance = {
        "experiment_id": experiment_id, "dataset_root": str(dataset_root), "stem": stem,
        "training_raster_path": str(training_source), "spatial_manifest": spatial,
        "density_uniformity_flags": density_flags, "block_scale_px": buffer_px,
        "block_scale_source": scale_source, "k_cal": k_cal, "k_test": k_test,
        "cal_gt_counts": cal_gt_counts, "test_gt_counts": test_gt_counts,
    }
    evidence = {
        "resolver": "resolve_operating_point",
        "inputs": resolver_inputs,
        "reference_inputs": {
            "label_dirs": {"reserved_regions": str(labels_dir)},
            "scope_roots": {"training_mosaic": str(dataset_root)},
            "stated_values": {"stem": stem, "calibration_region": list(cal_rect),
                              "test_region": list(test_rect), "block_scale_px": buffer_px,
                              "block_scale_source": scale_source, "k_cal": k_cal, "k_test": k_test},
        },
    }
    return bundle, provenance, evidence


def _band_records(
    reader: Any, bands: dict[str, tuple[int, int, int, int]], mosaic_w: int, mosaic_h: int,
    tile_size: int, overlap: float, predictor: Any, *, tile_batch_size: int, global_nms_iou: float,
    postprocess: str, gt_boxes: np.ndarray, gt_labels: np.ndarray, stem: str,
) -> tuple[list[dict], dict[str, tuple[int, int, int, int]]]:
    """Per-band COCO-shaped records (:func:`~tcip_mcp.pipelines.training.evaluation.
    build_coco_image_record`, the exact model ``calibrate_operating_point._records`` builds) plus
    the band rects keyed by a globally-unique image_id, for ``resolve_operating_point``'s
    ``cal_rects``/``hold_rects`` geometric disjointness check.

    Runs the unmodified :meth:`~tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor.
    predict_tiled` over a haloed :class:`~tcip_mcp.pipelines.raster_source._RegionView` of
    ``reader``, keeps only detections whose center lands in the un-haloed inner rect (kept at full
    extent, never clipped), and selects GT the same way (:func:`_select_gt_for_band`) so both
    sides of the curve derivation apply one inclusion rule at every band boundary.
    """
    from tcip_mcp.pipelines.data.tiling import region_halo
    from tcip_mcp.pipelines.raster_source import Rect, _RegionView
    from tcip_mcp.pipelines.training.evaluation import build_coco_image_record

    records: list[dict] = []
    rects_by_id: dict[str, tuple[int, int, int, int]] = {}
    for name, band_rect in sorted(bands.items()):
        haloed, inner = region_halo(band_rect, mosaic_w, mosaic_h, tile_size, overlap)
        hx0, hy0, hx1, hy1 = haloed
        ix0, iy0, ix1, iy1 = inner
        view = _RegionView(reader, Rect(hx0, hy0, hx1, hy1))
        image_id = f"{stem}::block_{name}"
        result = predictor.predict_tiled(
            view, tile_size=tile_size, overlap=overlap, tile_batch_size=tile_batch_size,
            global_nms_iou=global_nms_iou, postprocess=postprocess, require_masks=False,
            source_label=image_id,
        )
        cap_hit = result.get("cap_hit", False)
        dt: list[dict] = []
        for (bx1, by1, bx2, by2), score, label in zip(
            result["boxes"], result["scores"], result["labels"],
        ):
            cx, cy = (bx1 + bx2) / 2.0 + hx0, (by1 + by2) / 2.0 + hy0
            if ix0 <= cx < ix1 and iy0 <= cy < iy1:
                dt.append({
                    "category_id": int(label), "score": float(score),
                    "bbox": [bx1 + hx0 - ix0, by1 + hy0 - iy0, bx2 - bx1, by2 - by1],
                })
        selected_boxes, selected_labels = _select_gt_for_band(gt_boxes, gt_labels, inner)
        gt = [
            {"category_id": int(lab), "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0}
            for (x1, y1, x2, y2), lab in zip(selected_boxes.tolist(), selected_labels.tolist())
        ]
        rec = build_coco_image_record(ix1 - ix0, iy1 - iy0, gt, dt, image_id=image_id)
        rec["cap_hit"] = cap_hit
        records.append(rec)
        rects_by_id[image_id] = inner
    return records, rects_by_id
