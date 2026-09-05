import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type Konva from "konva";

import { api } from "@/api/client";
import { classesApi, subjectColor } from "@/api/classes";
import { resultsApi, type RegisteredModel } from "@/api/inference";
import type { ActionPayload } from "@/api/types.generated";
import { BandPicker } from "@/components/BandPicker";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
import { DisclosureChevron } from "@/components/CollapsibleSection";
import { ColorPickerModal } from "@/components/ColorPickerModal";
import { TabHeading } from "@/components/TabHeading";
import { EditShapeOverlay } from "@/components/review/EditShapeOverlay";
import { FilterChip } from "@/components/review/FilterChip";
import { ReviewLegend } from "@/components/review/ReviewLegend";
import { ReviewOverlays } from "@/components/review/ReviewOverlays";
import { useBandSelection } from "@/hooks/useBandSelection";
import { useCoverageGrid } from "@/hooks/useCoverageGrid";
import { useDisclosure } from "@/hooks/useDisclosure";
import { useImageBands } from "@/hooks/useImageBands";
import { useImageNav } from "@/hooks/useImageNav";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { usePrefetchAdjacentImages } from "@/hooks/usePrefetchAdjacentImages";
import { useRegionServes } from "@/hooks/useRegionServes";
import {
  applyEditDrag,
  clampShapeToImage,
  hitTestEdit,
  MIN_BOX_SIDE,
  seedEditShape,
  type EditDrag,
  type EditShape,
} from "@/lib/editGeometry";
import {
  buildReviewShapes,
  computeViewport,
  createCanvasPusher,
  measureCanvasHost,
  onCanvasStateRequest,
  type CanvasStateBody,
} from "@/lib/canvasSync";
import { useReviewColors, type ReviewColors } from "@/lib/reviewColors";
import type { LoadedImage } from "@/lib/imageLoader";
import { currentImage, labelPath, predictionPath } from "@/lib/paths";
import { zoomToRect } from "@/lib/viewGeometry";
import { compositeParams, showsBandPicker } from "@/lib/bandSelection";
import { datasetKey, loadDatasetVisibility, saveDatasetVisibility } from "@/lib/datasetUiState";
import { detOutcomeGeometry } from "@/lib/reviewGeometry";
import { useSubjectColors } from "@/lib/subjectColors";
import { useStore } from "@/store";
import type { MatchesResponse, ReviewImageStatus, ReviewStatusFilter } from "@/store/types";

// Plain-language labels for a breeder audience: the TP/FP/FN tag stays as a short code next
// to it, not as the primary label a non-CV user has to decode.
const COLOR_LABELS: { key: keyof ReviewColors; label: string; tag: string; dashed?: boolean }[] = [
  { key: "tp", label: "Matches ground truth", tag: "TP" },
  { key: "fp", label: "Extra detection", tag: "FP" },
  { key: "fn", label: "Missed by the model", tag: "FN" },
  { key: "active", label: "Under review", tag: "active", dashed: true },
];
const HANDLE_HIT_PX = 10; // screen-px hit radius for edit handles

const TYPE_ORDER: ("tp" | "fp" | "fn")[] = ["tp", "fp", "fn"];

const IMAGE_STATUS_LABEL: Record<MatchesResponse["image_status"], string> = {
  not_started: "not started",
  started: "in progress",
  completed: "reviewed",
};
const IMAGE_STATUS_CLASS: Record<MatchesResponse["image_status"], string> = {
  not_started: "bg-tcip-border text-tcip-muted",
  started: "bg-tcip-fn/20 text-tcip-fn",
  completed: "bg-tcip-tp/20 text-tcip-tp",
};

export function ReviewTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const patchGui = useStore((s) => s.patchGui);
  // Narrow subscriptions: pan/zoom mutates gui.view (the whole gui object is replaced by
  // setView), so subscribing to the whole gui slice re-rendered this tab (and its overlays)
  // on every tick. Take only view (needed for the canvas-push heartbeat) and the review filters.
  const view = useStore((s) => s.gui.view);
  const filters = useStore((s) => s.gui.review);
  const setView = useStore((s) => s.setView);
  const matches = useStore((s) => s.review.matches);
  // A review_focus command bumps this to force a matches refetch even when image + paths are
  // unchanged (the recompute effect otherwise skips identical-path rebuilds).
  const refetchNonce = useStore((s) => s.review.refetchNonce);
  const setMatches = useStore((s) => s.setMatches);
  const setLoading = useStore((s) => s.setReviewLoading);
  const setDetectionIdx = useStore((s) => s.setReviewDetectionIdx);
  const markDetReviewed = useStore((s) => s.markDetectionReviewed);
  // Shared annotation status (coloring, Complete lock), synced when a verdict authors GT.
  const setStoreImageStatus = useStore((s) => s.setImageStatus);
  // Image-level review status (its own store slice) drives Review navigation: which images are
  // Reviewed vs Unreviewed, and which have anything to review at all.
  const reviewStatus = useStore((s) => s.reviewStatus);
  const setReviewImageStatuses = useStore((s) => s.setReviewImageStatuses);
  const setReviewImageStatus = useStore((s) => s.setReviewImageStatus);
  const setReviewStatusFilter = useStore((s) => s.setReviewStatusFilter);

  const detectionIdx = filters.detection_idx;
  const { path: imgPath, name: imgName } = currentImage(dataset);
  const paths = useMemo(
    () => ({ gt: labelPath(dataset, imgName), pred: predictionPath(dataset, imgName) }),
    [dataset, imgName],
  );

  const bandsInfo = useImageBands(imgPath);
  const [bandSelection, setBandSelection] = useBandSelection(bandsInfo);

  // Every request for this view (the canvas' own and the prefetcher's warm-up) carries one set of
  // band params, so the two never warm and read different renders of the same image.
  const composite = compositeParams(bandsInfo, bandSelection);

  // One GT dir + one prediction dir now (the detect/segment split is gone); a unified label file
  // holds every subject's box and polygon annotations together.
  const reviewDirs = useMemo(
    () => ({ gtDir: dataset.annotations_dir, predDir: dataset.predictions_dir }),
    [dataset.annotations_dir, dataset.predictions_dir],
  );

  // Review navigation config: bucket each image into Reviewed/Unreviewed (completed vs not) so the
  // shared status-filter walk applies, and skip images with nothing to review.
  const reviewNavByImage = useMemo(() => {
    const m: Record<string, "reviewed" | "unreviewed"> = {};
    for (const name of dataset.image_list) {
      m[name] = reviewStatus.byImage[name] === "completed" ? "reviewed" : "unreviewed";
    }
    return m;
  }, [dataset.image_list, reviewStatus.byImage]);
  const hasDetections = reviewStatus.hasDetections;
  const isNavigable = useCallback(
    (name: string) => hasDetections[name] ?? true, // reviewable until the batch fetch says otherwise
    [hasDetections],
  );
  // Shared filtered navigation, scoped to review status + non-empty images (same traversal
  // machinery as the arrow keys + TopBar Prev/Next, different filter source).
  // nav (useImageNav) is declared further below, once priorityOrder is computed; it needs
  // that value to feed useImageNav's `order` option.

  const [showGT, setShowGT] = useState(true);
  const [showPred, setShowPred] = useState(true);
  // GT/Pred visibility is remembered per (project, date, subject/model): restore on dataset change,
  // save on toggle. (Position + filters are persisted centrally in the open path.)
  const visKey = datasetKey(dataset);
  useEffect(() => {
    const v = visKey ? loadDatasetVisibility(visKey) : null;
    setShowGT(v ? v.showGT : true);
    setShowPred(v ? v.showPred : true);
  }, [visKey]);
  const updateShowGT = (v: boolean) => {
    setShowGT(v);
    if (visKey) saveDatasetVisibility(visKey, { showGT: v, showPred });
  };
  const updateShowPred = (v: boolean) => {
    setShowPred(v);
    if (visKey) saveDatasetVisibility(visKey, { showGT, showPred: v });
  };
  // The filter shelf is collapsed by default and remembers the last state across sessions.
  const { open: filtersOpen, toggle: toggleFilters } = useDisclosure("tcip.review.filtersOpen");
  const [counterDraft, setCounterDraft] = useState<string | null>(null);
  const counterRef = useRef<HTMLInputElement | null>(null);
  const [imageStatus, setImageStatus] = useState<MatchesResponse["image_status"]>("not_started");
  // A reviewed (completed) image is locked: no verdicts/edits until it's reopened.
  const reviewLocked = imageStatus === "completed";
  // Every review route is scoped to the dataset root whose store holds the verdicts.
  const canReview = !!dataset.dataset_root;
  const needsDatasetSelection = !dataset.dataset_root && !!dataset.project_root;
  // Result of "use this review as a validation reference" (dataset-level, so it clears on selection).
  const [validating, setValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<{
    validated: boolean;
    reason: string;
  } | null>(null);
  useEffect(() => {
    setValidationResult(null);
  }, [visKey]);
  // The trait a validation-reference promotion is computed for, resolved from this project's own
  // registered traits (mirrors ResultsTab, never assumed from dataset.subject, which names an
  // object class, not necessarily a registered trait): auto-selected when there is exactly one,
  // left blank (with an explicit error, not a silent guess) when there are zero, offered as a
  // choice when there are several.
  const [availableTraits, setAvailableTraits] = useState<string[]>([]);
  const [trait, setTrait] = useState("");
  const [traitError, setTraitError] = useState<string | null>(null);
  useEffect(() => {
    if (!dataset.project_root) return;
    setTrait("");
    setTraitError(null);
    void resultsApi
      .traits(dataset.project_root)
      .then((res) => {
        setAvailableTraits(res.traits);
        if (res.traits.length === 0) {
          setTraitError("No trait is registered for this project yet.");
        } else if (res.traits.length === 1) {
          setTrait(res.traits[0]);
        }
      })
      .catch((e) => {
        setAvailableTraits([]);
        setTraitError(
          `Could not load this project's registered traits: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [dataset.project_root]);
  // The bucket's own generation confidence, fetched once per prediction dir (read-only, no
  // gate run) so the "Conf ≥" filter can warn live (see the filter shelf below).
  const [generationConf, setGenerationConf] = useState<number | null>(null);
  useEffect(() => {
    setGenerationConf(null);
    if (!dataset.predictions_dir) return;
    let cancelled = false;
    void api.review.generationConf(dataset.predictions_dir).then(
      (res) => {
        if (!cancelled) setGenerationConf(res.generation_conf);
      },
      () => {
        // Fetch failed: stay null. A missing generation_conf reads the same as a missing sidecar
        // (a foreign checkpoint, a not-yet-staged bucket), and the backend's own _conf_censored
        // treats a None staged_conf_floor as always censored, not as nothing to check, so this
        // warns too, rather than going quiet.
      },
    );
    return () => {
      cancelled = true;
    };
  }, [dataset.predictions_dir]);
  // Raising this filter above the predictions' own generation confidence hides low-confidence
  // detections from review; any verdict then recorded under it raises review_conf_threshold past
  // generation_conf, which validate_reference's identical gate reads as conf_censored, the same
  // signal, surfaced here before a review is even complete. A bucket with no recorded
  // generation_conf warns too: the backend's own None staged_conf_floor branch is always-censored,
  // never "no evidence, so nothing to warn about," so going quiet here would be silent in exactly
  // the case the real gate refuses hardest.
  const confFilterCensoring =
    !!dataset.predictions_dir &&
    (generationConf === null || filters.conf_threshold > generationConf);

  // ── Active-learning priority queue ──────────────────────────────────
  // prioritize_review_queue's ranking otherwise never reaches the breeder; the only path was the
  // agent manually steering focus_human_attention() one image at a time. Session-local (like generationConf/
  // validationResult above): nothing else in the app needs to know the computed order.
  const [pqModels, setPqModels] = useState<RegisteredModel[]>([]);
  const [pqModelPath, setPqModelPath] = useState("");
  const [pqJobId, setPqJobId] = useState<string | null>(null);
  const [pqStatus, setPqStatus] = useState<"idle" | "running" | "completed" | "failed">("idle");
  const [pqQueue, setPqQueue] = useState<
    { image: string; score: number; calibration_member?: boolean }[] | null
  >(null);
  const [pqError, setPqError] = useState<string | null>(null);
  // Set when the calibration side couldn't be marked (unreadable split record/manifest, or a
  // root mismatch); no queue entry then carries calibration_member, and this names why.
  const [pqMarksUnresolved, setPqMarksUnresolved] = useState<string | null>(null);
  // Auto-enabled once a queue completes (that's clearly what computing one was for); the breeder
  // can turn it back off to browse in the ordinary (positional) order without discarding the queue.
  const [pqUseOrder, setPqUseOrder] = useState(false);

  useEffect(() => {
    setPqModels([]);
    setPqModelPath("");
    if (!dataset.project_root) return;
    void resultsApi
      .registeredModels(dataset.project_root)
      .then((r) => setPqModels(r.models ?? []))
      .catch(() => setPqModels([]));
  }, [dataset.project_root]);

  // A new dataset/date selection invalidates whatever queue was computed for the previous one.
  useEffect(() => {
    setPqJobId(null);
    setPqStatus("idle");
    setPqQueue(null);
    setPqError(null);
    setPqUseOrder(false);
    setPqMarksUnresolved(null);
  }, [visKey]);

  useEffect(() => {
    if (!pqJobId || pqStatus !== "running") return;
    let cancelled = false;
    const poll = async () => {
      try {
        const body = await api.review.priorityQueueJob(pqJobId);
        if (cancelled) return;
        if (body.status === "completed") {
          setPqStatus("completed");
          setPqQueue(body.queue);
          setPqUseOrder(true);
          setPqMarksUnresolved(body.marks_unresolved ?? null);
        } else if (body.status === "failed") {
          setPqStatus("failed");
          setPqError(body.error ?? "The priority queue could not be computed.");
        }
      } catch {
        // A transient poll failure just tries again on the next tick.
      }
    };
    void poll();
    const t = setInterval(() => void poll(), 1000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [pqJobId, pqStatus]);

  async function computePriorityQueue() {
    if (!dataset.dataset_root || !dataset.images_dir || !pqModelPath) return;
    setPqStatus("running");
    setPqError(null);
    setPqQueue(null);
    setPqMarksUnresolved(null);
    try {
      const res = await api.review.launchPriorityQueue({
        dataset_root: dataset.dataset_root,
        checkpoint_path: pqModelPath,
        images_dir: dataset.images_dir,
      });
      setPqJobId(res.job_id);
    } catch (e) {
      setPqStatus("failed");
      setPqError(e instanceof Error ? e.message : String(e));
    }
  }

  // Map the computed queue (image names, ranked) onto image_list indices for useImageNav; an
  // image the queue named that no longer appears in image_list (deleted/renamed since scoring) is
  // dropped rather than crashing the traversal.
  const priorityOrder = useMemo(() => {
    if (!pqUseOrder || !pqQueue) return undefined;
    const indexByName = new Map(dataset.image_list.map((name, i) => [name, i]));
    return pqQueue
      .map((q) => indexByName.get(q.image.split(/[/\\]/).pop() ?? q.image))
      .filter((i): i is number => i !== undefined);
  }, [pqUseOrder, pqQueue, dataset.image_list]);

  // calibration_member, keyed the same way priorityOrder maps queue names onto image_list.
  const pqCalibrationByImage = useMemo(() => {
    if (!pqQueue) return null;
    const m = new Map<string, boolean>();
    for (const q of pqQueue) {
      if (q.calibration_member === undefined) continue;
      m.set(q.image.split(/[/\\]/).pop() ?? q.image, q.calibration_member);
    }
    return m;
  }, [pqQueue]);

  const nav = useImageNav({
    byImage: reviewNavByImage,
    activeFilter: reviewStatus.activeFilter,
    isNavigable,
    order: priorityOrder,
  });
  usePrefetchAdjacentImages(composite.bands, composite.stretch);

  // Region serves for large rasters: when zoom passes the base bitmap's resolution, the
  // viewport's coverage cells stream in above it (display only; no accumulation on this tab).
  const [baseFacts, setBaseFacts] = useState<LoadedImage | null>(null);
  const reviewImgW = matches?.img_width ?? 0;
  const reviewImgH = matches?.img_height ?? 0;
  const reviewServed = baseFacts?.ok ? baseFacts.servedSize : null;
  // No chrome on this tab, so the grid is worth fetching only once genuinely below native.
  const reviewBelowNative =
    !!reviewServed &&
    reviewImgW > 0 &&
    reviewImgH > 0 &&
    (reviewServed.w < reviewImgW || reviewServed.h < reviewImgH);
  const regionGrid = useCoverageGrid({
    imagePath: reviewBelowNative ? imgPath : null,
    subject: null,
    date: null,
    datasetRoot: null,
  });
  const regions = useRegionServes({
    imagePath: imgPath,
    imgW: matches?.img_width ?? 0,
    imgH: matches?.img_height ?? 0,
    view,
    servingCells: regionGrid.servingCells,
    servingTileSize: regionGrid.serving?.tile_size ?? null,
    coverageCells: [],
    baseFacts,
    composite,
  });
  // User-tunable symbology colours (persisted + shared with the status bar); legend swatches
  // open a picker. Changing TP here recolours the TP count in the bottom toolbar too.
  const [reviewColors, setReviewColors] = useReviewColors();
  const registry = useStore((s) => s.registry.subjects);
  const colorTick = useSubjectColors(); // bumps on a recolour, so swatches recompute below
  const subjectSwatches = useMemo(() => {
    void colorTick; // read only to force recompute; subjectColor() itself needs no argument for it
    return Object.keys(registry).map((name) => ({ name, color: subjectColor(name) }));
  }, [registry, colorTick]);

  // ── Live canvas push (agent visibility: capture_live_canvas) ──────────────
  // Which image the installed matches belong to: identity beats the loading flag (a failed or
  // superseded reload leaves stale matches with loading=false; identity still blocks the push).
  const matchesImageRef = useRef<string | null>(null);
  const buildCanvasBodyRef = useRef<() => CanvasStateBody | null>(() => null);
  buildCanvasBodyRef.current = () => {
    if (!imgPath || !dataset.project_root || !matches) return null;
    // Mid-transition guards: the store must hold this image's matches, and not be mid-reload;
    // otherwise the previous image's shapes would push under the new image_path (a false canvas).
    if (matchesImageRef.current !== imgName) return null;
    if (useStore.getState().review.loading) return null;
    // Binding-presence gate: a dataset without an adopted generation must stop pushing.
    const generation = useStore.getState().bindingGeneration;
    useStore.getState().setCanvasBindingMissing(generation == null);
    if (generation == null) return null;
    const host = measureCanvasHost();
    return {
      binding_generation: generation,
      tab: "review",
      image_path: imgPath,
      image: imgName ?? "",
      img_width: matches.img_width,
      img_height: matches.img_height,
      viewport: host ? computeViewport(view, host, matches.img_width, matches.img_height) : null,
      user: useStore.getState().user || undefined,
      classes: subjectSwatches,
      legend: {
        tp: reviewColors.tp,
        fp: reviewColors.fp,
        fn: reviewColors.fn,
        active: reviewColors.active,
      },
      counts: { tp: matches.n_tp, fp: matches.n_fp, fn: matches.n_fn },
      shapes: buildReviewShapes(matches, reviewColors, detectionIdx, {
        showGT,
        showPred,
      }),
    };
  };
  const canvasPusherRef = useRef(createCanvasPusher((b) => api.canvas.pushState(b)));
  useEffect(() => () => canvasPusherRef.current.dispose(), []);
  // Verdicts / detection focus / visibility change the drawn shapes → full push; pan/zoom → heartbeat.
  useEffect(() => {
    canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), true);
  }, [matches, reviewColors, detectionIdx, showGT, showPred, imgPath]);
  useEffect(() => {
    canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), false);
  }, [view, subjectSwatches]);
  useEffect(
    () =>
      onCanvasStateRequest(() => {
        canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), true);
        canvasPusherRef.current.flush();
      }),
    [],
  );
  const [colorEditKey, setColorEditKey] = useState<keyof ReviewColors | null>(null);
  const [edit, setEdit] = useState<EditShape | null>(null);
  const editDrag = useRef<EditDrag | null>(null);
  // "Mark missed object": draw a brand-new GT box with no detection selected, for a
  // previously-unlabeled image the walkable TP/FP/FN list has nothing to seed from.
  // `drawingMiss` = armed (next canvas drag draws the box); once drawn, the box moves into the
  // same `edit` state the existing shape editor already handles (drag handles, Save/Cancel), with
  // `pendingMiss` marking that a Save should submit a brand-new missed-object verdict, not an edit
  // of `current`.
  const [drawingMiss, setDrawingMiss] = useState(false);
  const [missDraft, setMissDraft] = useState<[number, number, number, number] | null>(null);
  const missDraftStart = useRef<[number, number] | null>(null);
  const [pendingMiss, setPendingMiss] = useState(false);
  // One GT-mutating request at a time: key auto-repeat / double-clicks must not append
  // or delete twice, and no verdict may land while indices are stale mid-reload.
  const actionPending = useRef(false);
  // Label-dir sets whose .original/ baseline is already captured this session (see ensureBackup).
  const backedUpKeys = useRef<Set<string>>(new Set());

  // Install a matches payload (from /matches or a verdict's /action response) onto the canvas: set
  // state, then land on the hinted detection (else the first unreviewed) and zoom to it. A pending
  // `review_focus` index (the agent asked to center on detection N) wins for one install.
  function applyMatches(res: MatchesResponse, indexHint?: number) {
    setMatches(res);
    matchesImageRef.current = imgName; // which image these matches belong to (canvas-push guard)
    setImageStatus(res.image_status);
    if (imgName) setReviewImageStatus(imgName, res.image_status); // keep the nav filter live

    const focusIdx = useStore.getState().review.focusDetectionIdx;
    const effectiveHint = indexHint ?? focusIdx ?? undefined;
    if (focusIdx !== null && focusIdx !== undefined) useStore.getState().setReviewFocusIdx(null);
    if (effectiveHint === undefined) {
      const firstUnreviewed = res.detections.findIndex((d) => !d.reviewed);
      const target = firstUnreviewed >= 0 ? firstUnreviewed : 0;
      setDetectionIdx(target);
      zoomToDetection(res.detections[target]?.bbox);
    } else {
      const clamped = Math.max(0, Math.min(res.detections.length - 1, effectiveHint));
      setDetectionIdx(clamped);
      zoomToDetection(res.detections[clamped]?.bbox);
    }
  }

  async function reloadMatches(indexHint?: number, signal?: AbortSignal) {
    if (!dataset.dataset_root || !imgPath || !imgName) return;
    setLoading(true);
    try {
      // One unified label file per image holds every subject's box and polygon annotations, so a
      // single gt/pred path drives the match: compute_matches keys IoU on each annotation's bbox
      // and the overlay renders each by its own geometry.
      const res = await api.review.matches(
        {
          dataset_root: dataset.dataset_root,
          image_name: imgName,
          image_path: imgPath,
          gt_path: paths.gt,
          pred_path: paths.pred,
          iou_threshold: filters.iou_threshold,
          conf_threshold: filters.conf_threshold,
          filter_type: filters.filter_type,
          filter_class: filters.filter_class,
        },
        signal,
      );
      // Identity check: if the user navigated while this was in flight, installing the
      // response would put another image's matches under the current image.
      const now = useStore.getState().gui.dataset;
      if ((now.image_list[now.current_image_index] ?? null) !== imgName) return;
      applyMatches(res, indexHint);
    } catch (e) {
      // A superseded (aborted) request is expected during slider drags; ignore it.
      if (signal?.aborted || (e instanceof DOMException && e.name === "AbortError")) return;
      useStore
        .getState()
        .pushToast(`Could not load review matches: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }

  function zoomToDetection(bbox: [number, number, number, number] | undefined) {
    if (!bbox) return;
    const [x1, y1, x2, y2] = bbox;
    // The detection's own width/height as the per-axis pad reproduces the 3x fit; the offset
    // clamp zoomToRect applies is a deliberate change (an edge detection now pins to the edge).
    const dw = Math.max(1, x2 - x1);
    const dh = Math.max(1, y2 - y1);
    // Dims come from the store's freshest matches: the render closure's copy is one image stale
    // at this moment (applyMatches installs, then zooms).
    const m = useStore.getState().review.matches;
    const host = measureCanvasHost() ?? { w: 1200, h: 800 };
    setView(
      zoomToRect(
        { x0: x1, y0: y1, x1: x2, y1: y2 },
        { host, imgW: m?.img_width ?? 0, imgH: m?.img_height ?? 0, padX: dw, padY: dh },
      ),
    );
  }

  useEffect(() => {
    // Debounce so dragging the IoU/Conf sliders doesn't fire a /matches recompute per
    // tick, and abort the in-flight request so a slow earlier response can't clobber a
    // newer one (out-of-order responses previously won).
    const ac = new AbortController();
    const t = setTimeout(() => void reloadMatches(undefined, ac.signal), 180);
    return () => {
      clearTimeout(t);
      ac.abort();
    };
    // The path strings (not the `paths` object: mergeSnapshot rebuilds the dataset
    // object on every WS snapshot, which would spuriously re-fire this and reset the
    // detection index/zoom) so a backend-adopted change of the prediction dir (e.g. the
    // agent re-selects the dataset with a different model) refreshes the matches instead
    // of silently showing the previous model's TP/FP/FN.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    imgPath,
    paths.gt,
    paths.pred,
    filters.iou_threshold,
    filters.conf_threshold,
    filters.filter_type,
    filters.filter_class,
    refetchNonce,
  ]);

  // Batch-fetch image-level review status + detection presence for the whole (subject, date), so nav
  // can filter Reviewed/Unreviewed and skip images with nothing to review. Re-runs when the dataset
  // or its reviewed-kind dirs change; live per-image updates ride on verdicts (setReviewImageStatus).
  useEffect(() => {
    const datasetRoot = dataset.dataset_root;
    const imageList = dataset.image_list;
    if (!datasetRoot || imageList.length === 0) return;
    let cancelled = false;
    void (async () => {
      try {
        const res = await api.review.imageStatuses({
          dataset_root: datasetRoot,
          gt_dir: reviewDirs.gtDir,
          pred_dir: reviewDirs.predDir,
        });
        if (cancelled) return;
        const stems = new Set(res.detection_stems);
        // res.unreadable names label document paths, not stems; a stem this names stays
        // navigable (the breeder can still reach and see it), never silently skipped.
        const unreadableStems = new Set(
          res.unreadable.map((p) => (p.split(/[/\\]/).pop() ?? p).replace(/\.[^.]+$/, "")),
        );
        const byImage: Record<string, ReviewImageStatus> = {};
        const has: Record<string, boolean> = {};
        for (const name of imageList) {
          byImage[name] = res.statuses[name] ?? "not_started";
          const stem = name.replace(/\.[^.]+$/, "");
          has[name] = unreadableStems.has(stem) || stems.has(stem);
        }
        setReviewImageStatuses(byImage, has, res.unreadable);
      } catch (e) {
        if (cancelled) return;
        // A prior dataset's facts must never gate navigation in this one: clear rather than
        // leave them in place, and name the failure so the breeder knows nav is unfiltered.
        setReviewImageStatuses({}, {}, []);
        useStore
          .getState()
          .pushToast(`Could not load review status: ${e instanceof Error ? e.message : String(e)}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    dataset.dataset_root,
    dataset.image_list,
    reviewDirs.gtDir,
    reviewDirs.predDir,
    setReviewImageStatuses,
  ]);

  function stepImage(delta: number) {
    nav.stepImage(delta);
  }

  function stepDetection(delta: number) {
    if (!matches) return;
    const next = Math.max(0, Math.min(matches.detections.length - 1, detectionIdx + delta));
    setDetectionIdx(next);
    zoomToDetection(matches.detections[next]?.bbox);
  }

  /**
   * Find the next unreviewed detection: if the active type filter has no more
   * unreviewed detections, try the next type, and if nothing remains on
   * the image at all, advance to the next image.
   */
  function advanceToNextUnreviewed() {
    if (!matches) return;
    const dets = matches.detections;
    // Same filter, next unreviewed
    for (let i = 0; i < dets.length; i++) {
      const j = (detectionIdx + 1 + i) % dets.length;
      if (!dets[j].reviewed) {
        setDetectionIdx(j);
        zoomToDetection(dets[j].bbox);
        return;
      }
    }
    // Filter type exhausted, try other types under "all" filter context
    if (filters.filter_type !== "all") {
      const otherTypes = TYPE_ORDER.filter((t) => t !== filters.filter_type);
      for (const t of otherTypes) {
        // The current `dets` list reflects the current filter; we can use the
        // counts from matches (n_tp/n_fp/n_fn) only as a hint that there's
        // more to look at. The cleanest path is to relax the type filter to
        // "all" and reload; reloadMatches will jump to the first unreviewed.
        if (
          (t === "tp" && matches.n_tp > 0) ||
          (t === "fp" && matches.n_fp > 0) ||
          (t === "fn" && matches.n_fn > 0)
        ) {
          patchGui({ review: { ...filters, filter_type: "all" } });
          return; // reload will fire via the deps effect
        }
      }
    }
    // No more on this image, go to next image.
    stepImage(1);
  }

  const current = matches?.detections[detectionIdx] ?? null;

  // Under a classified scope (matches.attribute set) a verdict judges an object's confirmed
  // value, never its presence; class_name on every detection is that value.
  const classifiedScope = !!matches?.attribute;

  // The class filter's own options: matches.attribute's values under a classified scope, else
  // every subject present (matches.detections is already narrowed by the active filters).
  const availableClasses = useMemo(() => {
    if (!matches) return [];
    const names = new Set<string>();
    if (matches.attribute) {
      const attr = matches.attribute;
      for (const a of matches.gt)
        if (a.attributes[attr] !== undefined) names.add(a.attributes[attr]);
      for (const a of matches.preds)
        if (a.attributes[attr] !== undefined) names.add(a.attributes[attr]);
    } else {
      for (const a of matches.gt) names.add(a.subject);
      for (const a of matches.preds) names.add(a.subject);
    }
    return Array.from(names).sort();
  }, [matches]);

  async function recordAction(
    action: Exclude<ActionPayload["action"], "swept">,
    edited?: { box?: [number, number, number, number]; polygon?: number[][] },
  ): Promise<boolean> {
    if (actionPending.current) return false;
    if (reviewLocked) return false; // a completed/reviewed image is locked until reopened
    if (!current || !dataset.dataset_root || !imgPath || !imgName) return false;
    // Reject on a detection that has ground truth (TP/FN) deletes that GT box: a destructive,
    // irreversible action (CLAUDE.md "confirm before destructive actions"). Reject on an FP is
    // safe (discards a prediction, GT unchanged) and needs no confirmation.
    if (action === "rejected" && current.det_type !== "fp") {
      const confirmed = window.confirm(
        "This deletes the existing ground-truth box for this object. This cannot be undone. Continue?",
      );
      if (!confirmed) return false;
    }
    actionPending.current = true;
    try {
      // The .original snapshot must exist before the first GT write: awaited, and a
      // failure aborts the verdict rather than mutating labels with no pristine baseline.
      if (!(await ensureBackup())) return false;
      const res = await api.review.action({
        dataset_root: dataset.dataset_root,
        image_name: imgName,
        image_path: imgPath,
        gt_path: paths.gt,
        pred_path: paths.pred,
        det_type: current.det_type,
        class_name: current.class_name,
        conf: current.conf,
        iou: current.iou,
        gt_idx: current.gt_idx,
        pred_idx: current.pred_idx,
        bbox: current.bbox,
        action,
        edited_box: edited?.box ?? null,
        edited_points: edited?.polygon ?? null,
        iou_threshold: filters.iou_threshold,
        conf_threshold: filters.conf_threshold,
        filter_type: filters.filter_type,
        filter_class: filters.filter_class,
        user: useStore.getState().user,
      });
      setImageStatus(res.image_status);
      if (imgName) setReviewImageStatus(imgName, res.image_status);
      markDetReviewed(detectionIdx, action);
      advanceToNextUnreviewed();
      if (res.annotation_status) {
        // GT changed: the verdict already recomputed matches server-side (gt_idx/pred_idx rebuilt
        // from the written files), so install them directly, no second /matches round-trip.
        setStoreImageStatus(imgName, res.annotation_status);
        applyMatches(res.matches, useStore.getState().gui.review.detection_idx);
      }
      return true;
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not record review action: ${e instanceof Error ? e.message : String(e)}`);
      return false;
    } finally {
      actionPending.current = false;
    }
  }

  async function ensureBackup(): Promise<boolean> {
    if (!dataset.dataset_root) return false;
    const dirs = [dataset.annotations_dir].filter(Boolean) as string[];
    if (!dirs.length) return true;
    // backup_original_labels captures every original file in the dir in one pass, so it only needs
    // running once per label-dir set: skip the (whole-dir-scanning) call once it's done this session.
    const key = `${dataset.dataset_root}/${dirs.join("/")}`;
    if (backedUpKeys.current.has(key)) return true;
    try {
      await api.review.backupLabels(dataset.dataset_root, dirs);
      backedUpKeys.current.add(key);
      return true;
    } catch {
      useStore.getState().pushToast("Backup failed. Retry the action.");
      return false;
    }
  }

  // Submit a brand-new "missed object" box directly through the existing /api/review/action
  // endpoint (no new backend route): det_type "fn", gt_idx/pred_idx null, so record_action's
  // _apply_gt_mutation appends a new GT annotation and records a proper gt-only verdict entry.
  async function recordMissedObject(box: [number, number, number, number]): Promise<boolean> {
    if (actionPending.current) return false;
    if (reviewLocked) return false;
    if (!dataset.dataset_root || !imgPath || !imgName) return false;
    if (!dataset.subject) {
      useStore
        .getState()
        .pushToast("Can't record a missed object: no subject configured for this dataset.");
      return false;
    }
    actionPending.current = true;
    try {
      if (!(await ensureBackup())) return false;
      const res = await api.review.action({
        dataset_root: dataset.dataset_root,
        image_name: imgName,
        image_path: imgPath,
        gt_path: paths.gt,
        pred_path: paths.pred,
        det_type: "fn",
        class_name: dataset.subject,
        conf: null,
        iou: null,
        gt_idx: null,
        pred_idx: null,
        bbox: box,
        action: "edited",
        edited_box: box,
        edited_points: null,
        iou_threshold: filters.iou_threshold,
        conf_threshold: filters.conf_threshold,
        filter_type: filters.filter_type,
        filter_class: filters.filter_class,
        user: useStore.getState().user,
      });
      setImageStatus(res.image_status);
      if (imgName) setReviewImageStatus(imgName, res.image_status);
      if (res.annotation_status) {
        setStoreImageStatus(imgName, res.annotation_status);
        applyMatches(res.matches, useStore.getState().gui.review.detection_idx);
      }
      return true;
    } catch (e) {
      useStore
        .getState()
        .pushToast(
          `Could not record the missed object: ${e instanceof Error ? e.message : String(e)}`,
        );
      return false;
    } finally {
      actionPending.current = false;
    }
  }

  // Record an explicit "I checked this image for missed objects and found none" attestation: no
  // geometry, no gt/pred index, so record_action's _apply_gt_mutation no-ops on GT. Distinct from
  // recordMissedObject (which always writes a new GT box): this is the case where the sweep itself
  // is the whole verdict. Backs adjudication coverage the same way a genuine missed-object find
  // already does (record_detection_action stamps missed_object_attested from having neither index
  // set, review_calibration.review_to_records folds it into the image's adjudication-covered fact).
  async function recordSweepAttested(): Promise<boolean> {
    if (actionPending.current) return false;
    if (reviewLocked) return false;
    if (!dataset.dataset_root || !imgPath || !imgName) return false;
    actionPending.current = true;
    try {
      const res = await api.review.action({
        dataset_root: dataset.dataset_root,
        image_name: imgName,
        image_path: imgPath,
        gt_path: paths.gt,
        pred_path: paths.pred,
        det_type: "sweep",
        class_name: "",
        conf: null,
        iou: null,
        gt_idx: null,
        pred_idx: null,
        bbox: [0, 0, matches?.img_width ?? 0, matches?.img_height ?? 0],
        action: "swept",
        edited_box: null,
        edited_points: null,
        iou_threshold: filters.iou_threshold,
        conf_threshold: filters.conf_threshold,
        filter_type: filters.filter_type,
        filter_class: filters.filter_class,
        user: useStore.getState().user,
      });
      setImageStatus(res.image_status);
      if (imgName) setReviewImageStatus(imgName, res.image_status);
      useStore.getState().pushToast("Recorded: no missed objects found on this image.", "info");
      return true;
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not record the sweep: ${e instanceof Error ? e.message : String(e)}`);
      return false;
    } finally {
      actionPending.current = false;
    }
  }

  async function markImageComplete(completed: boolean) {
    if (!dataset.dataset_root || !imgName) return;
    try {
      const res = await api.review.markComplete({
        dataset_root: dataset.dataset_root,
        image_name: imgName,
        gt_path: paths.gt,
        pred_dir: dataset.predictions_dir,
        completed,
        subject: dataset.subject,
      });
      setImageStatus(res.image_status); // local review badge
      if (imgName) setReviewImageStatus(imgName, res.image_status); // keep the nav filter live
      // The annotation status comes from the server (GT files on disk), scoped to the confirmed
      // subject; null when no subject was named, so nothing to sync or mirror.
      if (res.annotation_status) {
        setStoreImageStatus(imgName, res.annotation_status);
        // The registry this status is mirrored into is the project's, not the review store's.
        if (dataset.project_root) {
          void classesApi
            .setImageStatus(
              dataset.project_root,
              imgName,
              res.annotation_status,
              dataset.subject,
              dataset.date,
              dataset.dataset_root,
              dataset.annotations_dir,
              useStore.getState().user || undefined,
            )
            .catch(() => {});
        }
      }
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not mark image reviewed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // Promote the current dataset's completed review into a validation reference. Runs the platform's
  // own validation gate server-side; the honest validated / not-yet result is surfaced (never forced).
  async function promoteReviewToValidationReference() {
    if (!dataset.dataset_root) {
      useStore.getState().pushToast("Select a dataset first.");
      return;
    }
    if (!trait) {
      useStore.getState().pushToast(traitError ?? "Pick a trait before validating.");
      return;
    }
    if (!dataset.predictions_dir) {
      useStore
        .getState()
        .pushToast("No predictions to validate. Select a model with predictions first.");
      return;
    }
    if (!dataset.subject) {
      useStore.getState().pushToast("Pick a subject before validating.");
      return;
    }
    setValidating(true);
    try {
      const res = await api.review.validateReference({
        dataset_root: dataset.dataset_root,
        trait,
        pred_dir: dataset.predictions_dir,
        subject: dataset.subject,
      });
      setValidationResult({ validated: res.validated, reason: res.reason });
      useStore.getState().pushToast(res.reason);
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not check the review: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setValidating(false);
    }
  }

  // ── In-place edit: pick the shape up on this canvas, adjust, save to GT ──

  function startEdit() {
    if (reviewLocked) return;
    if (!current || !matches) return;
    const geom = detOutcomeGeometry(current, matches);
    if (!geom) {
      useStore.getState().pushToast("This detection has no shape to adjust.");
      return;
    }
    if (geom.kind === "point") {
      // The canvas editor adjusts an outline, and /review/action carries a box or one contour. A
      // point has neither, and inventing a small box around it would write a fabricated extent into
      // ground truth. The verdict keys (accept/reject) still apply to it.
      useStore
        .getState()
        .pushToast(
          "A point marks a location, not an outline, so it can't be resized. Accept or reject it " +
            "here, or move it in Annotate.",
          "info",
        );
      return;
    }
    if (geom.kind === "polygon" && geom.rings.length > 1) {
      // Editing by hand commits one contour (edited_points), so seeding part 1 of an
      // occlusion-split shape would save that part as the entire object: a quietly wrong
      // measurement. Refuse instead, and say what the reviewer can do.
      useStore
        .getState()
        .pushToast(
          `This shape covers ${geom.rings.length} separate parts of one object; hand-adjustment ` +
            `only works on a single outline. Accept, reject, or redraw it in Annotate.`,
          "info",
        );
      return;
    }
    setEdit(clampShapeToImage(seedEditShape(geom), matches.img_width, matches.img_height));
  }

  function cancelEdit() {
    setEdit(null);
    editDrag.current = null;
    setPendingMiss(false);
  }

  async function commitEdit() {
    if (!edit) return;
    if (edit.kind === "box") {
      const [x1, y1, x2, y2] = edit.box;
      if (x2 - x1 < MIN_BOX_SIDE || y2 - y1 < MIN_BOX_SIDE) {
        useStore.getState().pushToast("Box too small to save. Drag a corner to enlarge it.");
        return;
      }
      const ok = pendingMiss
        ? await recordMissedObject(edit.box)
        : await recordAction("edited", { box: edit.box });
      if (ok) cancelEdit();
      return;
    }
    if (edit.points.length < 3) {
      useStore.getState().pushToast("Polygon needs at least 3 points to save.");
      return;
    }
    if (await recordAction("edited", { polygon: edit.points.map((p) => [p[0], p[1]]) }))
      cancelEdit();
  }

  // ── Mark missed object: draw a brand-new box with no detection selected ──

  function startMarkMissedObject() {
    if (reviewLocked || edit) return;
    setDrawingMiss(true);
  }

  function cancelMarkMissedObject() {
    setDrawingMiss(false);
    setMissDraft(null);
    missDraftStart.current = null;
  }

  function onMissDown(x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) {
    if (ev.evt.button !== 0) return;
    missDraftStart.current = [x, y];
    setMissDraft([x, y, x, y]);
  }

  function onMissMove(x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) {
    if (!missDraftStart.current || ev.evt.buttons === 0) return;
    const [sx, sy] = missDraftStart.current;
    setMissDraft([Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]);
  }

  function onMissUp() {
    const box = missDraft;
    missDraftStart.current = null;
    setMissDraft(null);
    setDrawingMiss(false);
    if (!box) return;
    const [x1, y1, x2, y2] = box;
    if (x2 - x1 < MIN_BOX_SIDE || y2 - y1 < MIN_BOX_SIDE) {
      useStore.getState().pushToast("Box too small. Drag out a bigger area for the missed object.");
      return;
    }
    setPendingMiss(true);
    setEdit(
      clampShapeToImage({ kind: "box", box }, matches?.img_width ?? 0, matches?.img_height ?? 0),
    );
  }

  // An edit belongs to one detection on one matches snapshot; leaving either discards it.
  useEffect(() => {
    setEdit(null);
    editDrag.current = null;
    setPendingMiss(false);
    setDrawingMiss(false);
    setMissDraft(null);
    missDraftStart.current = null;
  }, [detectionIdx, imgName, matches]);

  function onEditDown(x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) {
    editDrag.current = null; // a miss (or a stale drag from an off-canvas release) grabs nothing
    if (!edit || ev.evt.button !== 0) return;
    const tol = HANDLE_HIT_PX / (useStore.getState().gui.view.scale || 1);
    editDrag.current = hitTestEdit(edit, x, y, tol);
  }

  function onEditMove(x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) {
    const drag = editDrag.current;
    if (!edit || !drag || !matches) return;
    // The button was released outside the canvas; Konva never delivered the mouseup.
    if (ev.evt.buttons === 0) {
      editDrag.current = null;
      return;
    }
    const r = applyEditDrag(edit, drag, x, y, matches.img_width, matches.img_height);
    editDrag.current = r.drag;
    if (r.shape !== edit) setEdit(r.shape);
  }

  function onEditUp() {
    editDrag.current = null;
  }

  useKeyboardShortcuts([
    {
      keys: "a",
      action: (e) => {
        if (!e.repeat) void recordAction("accepted");
      },
      when: () => !!current && !edit && !reviewLocked && canReview,
    },
    {
      keys: "r",
      action: (e) => {
        if (!e.repeat) void recordAction("rejected");
      },
      // Same guard the Reject button is disabled on, so a classified tp/fn raises no dialog.
      when: () => !!current && !edit && !rejectDisabled,
    },
    { keys: "e", action: () => startEdit(), when: () => !!current && !edit && canReview },
    {
      keys: "enter",
      action: (e) => {
        // A focused button owns Enter (its native click already fired or will).
        if (!e.repeat && !(document.activeElement instanceof HTMLButtonElement)) void commitEdit();
      },
      when: () => !!edit,
    },
    { keys: "escape", action: () => cancelEdit(), when: () => !!edit },
    { keys: "arrowleft", action: () => stepDetection(-1), when: () => !edit },
    { keys: "arrowright", action: () => stepDetection(1), when: () => !edit },
    // Image flips ignore held-key auto-repeat; each one costs a full image render.
    {
      keys: "arrowup",
      action: (e) => {
        if (!e.repeat) stepImage(-1);
      },
      when: () => !edit,
    },
    {
      keys: "arrowdown",
      action: (e) => {
        if (!e.repeat) stepImage(1);
      },
      when: () => !edit,
    },
  ]);

  const imageUrl = imgPath ? api.images.url(imgPath, composite) : null;
  const imgW = matches?.img_width ?? 0;
  const imgH = matches?.img_height ?? 0;

  // Verdicts author ground truth: accept an FP adds the prediction to GT; reject a detection
  // with GT (TP/FN) deletes that box; accept keeps existing GT.
  const acceptLabel = "Accept";
  const rejectLabel = "Reject";
  // Under a classified scope a verdict judges the confirmed value instead; removing the object
  // is out of scope, so Reject on a tp/fn refuses and is disabled.
  const acceptTitle = classifiedScope
    ? current?.det_type === "fp"
      ? "Confirm this predicted value for the object (A)"
      : "Keep this confirmed value (A)"
    : current?.det_type === "fp"
      ? "Add this prediction to ground truth (A)"
      : "Keep this ground-truth object (A)";
  const rejectRefusedTitle =
    "Removing this object is a detector-scope action; add or remove it in the Annotate tab, " +
    "then review its value here.";
  const rejectTitle = classifiedScope
    ? current?.det_type === "fp"
      ? "Discard this predicted value; the confirmed value stays (R)"
      : rejectRefusedTitle
    : current?.det_type === "fp"
      ? "Discard this prediction; ground truth unchanged (R)"
      : "Delete this ground-truth object (R)";
  const rejectDisabled =
    reviewLocked || !canReview || (classifiedScope && current?.det_type !== "fp");

  return (
    <div className="flex-1 flex flex-col relative min-h-0">
      <TabHeading tab="review" />
      <div className="relative border-b border-tcip-border bg-tcip-panel">
        {/* Row 1: filter shelf toggle + live summary + legend, then image / detection navigation */}
        <div className="flex items-center gap-2 px-3 py-1.5 text-[11px]">
          <button
            className="tcip-btn inline-flex items-center gap-1.5"
            onClick={toggleFilters}
            aria-expanded={filtersOpen}
            disabled={!!edit}
            title="Show or hide the review filters"
          >
            <DisclosureChevron open={filtersOpen} />
            Filters
          </button>
          {/* Live summary: always shows every filter, so the shelf can stay collapsed. */}
          <span className="flex items-center gap-1.5 tabular-nums">
            <FilterChip>IoU ≥ {filters.iou_threshold.toFixed(2)}</FilterChip>
            <FilterChip
              warn={confFilterCensoring}
              title={
                generationConf === null
                  ? "This bucket has no recorded generation confidence, so it always reads as conf-censored for validation, regardless of this filter."
                  : "Raising this above the predictions' own generation confidence hides low-confidence detections from review; verdicts recorded from here on will read as conf-censored for validation."
              }
            >
              {confFilterCensoring ? "⚠ " : ""}Conf ≥ {filters.conf_threshold.toFixed(2)}
            </FilterChip>
            {/* Not tooltip-only: a breeder who raises this filter needs to see why it matters
                without hovering. */}
            {confFilterCensoring && (
              <span className="text-tcip-warn max-w-[280px]">
                {generationConf === null
                  ? "no recorded generation confidence for this bucket, always conf-censored for validation"
                  : `above this bucket's own generation confidence (${generationConf.toFixed(2)}), new verdicts will be conf-censored for validation`}
              </span>
            )}
            <FilterChip>
              {filters.filter_type === "all" ? "All types" : filters.filter_type.toUpperCase()}
            </FilterChip>
            <FilterChip>
              {filters.filter_class === "all" ? "All classes" : filters.filter_class}
            </FilterChip>
            <FilterChip>
              {reviewStatus.activeFilter === "all"
                ? "All images"
                : reviewStatus.activeFilter === "reviewed"
                  ? "Reviewed"
                  : "Unreviewed"}
            </FilterChip>
            <FilterChip>
              {showGT && showPred
                ? "GT + Pred"
                : showGT
                  ? "GT only"
                  : showPred
                    ? "Pred only"
                    : "Hidden"}
            </FilterChip>
          </span>

          <span aria-hidden className="mx-1 h-4 w-px bg-tcip-border" />
          {needsDatasetSelection && (
            <span className="text-tcip-warn max-w-[360px]">
              No dataset is selected, so review verdicts have nowhere to be recorded. Reopen this
              project from the top bar to select its dataset.
            </span>
          )}
          {reviewStatus.unreadable.length > 0 && (
            <span
              className="text-tcip-warn max-w-[360px] truncate"
              title={reviewStatus.unreadable.join(", ")}
            >
              {reviewStatus.unreadable.length} label document
              {reviewStatus.unreadable.length === 1 ? "" : "s"} could not be read:{" "}
              {reviewStatus.unreadable.join(", ")}
            </span>
          )}
          {/* Which registered trait a validation-reference promotion is computed for (see
              ResultsTab's identical picker): only shown when the project has more than one. */}
          {availableTraits.length > 1 && (
            <span className="flex items-center gap-1.5">
              <label className="tcip-label">Trait</label>
              <select
                className="tcip-input w-auto"
                value={trait}
                onChange={(e) => setTrait(e.target.value)}
              >
                <option value="" disabled>
                  Choose a trait…
                </option>
                {availableTraits.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </span>
          )}
          {/* Dataset-level: promote this review into a validation reference the results can trust.
              The backend runs the same validation check and answers validated / not-yet honestly. */}
          <button
            className="tcip-btn"
            onClick={() => void promoteReviewToValidationReference()}
            disabled={validating || !!edit || !trait || !canReview}
            title={
              traitError ??
              "Check whether this review confirms the model's counts well enough to trust them for results. Runs the platform's own validation check; it will tell you if it isn't enough yet."
            }
          >
            {validating ? "Checking…" : "Use review as validation reference"}
          </button>
          {validationResult && (
            <span className="flex items-center gap-1.5">
              <span
                className={`tcip-badge ${
                  validationResult.validated
                    ? "bg-tcip-tp/20 text-tcip-tp"
                    : "bg-tcip-fn/20 text-tcip-fn"
                }`}
                title={validationResult.reason}
              >
                {validationResult.validated ? "Validated" : "Not yet"}
              </span>
              {/* The reason was tooltip-only, with no visible next step: a breeder who hits
                  "Not yet" needs to see why without hovering, and what to try. */}
              {!validationResult.validated && (
                <span className="text-[11px] text-tcip-muted max-w-[360px] whitespace-pre-wrap">
                  {validationResult.reason}
                </span>
              )}
            </span>
          )}

          {/* Attest a missed object on any image, even one with no existing detections to select;
              draws a brand-new box, submitted through the same /api/review/action endpoint as an
              edited FN verdict. */}
          <button
            className="tcip-btn"
            onClick={() => (drawingMiss ? cancelMarkMissedObject() : startMarkMissedObject())}
            disabled={
              !imgName ||
              reviewLocked ||
              !!edit ||
              !dataset.subject ||
              !canReview ||
              classifiedScope
            }
            title={
              classifiedScope
                ? "Adding a new object here would fabricate a state nobody assessed; add it " +
                  "through the Annotate tab, then review its value here."
                : drawingMiss
                  ? "Cancel drawing a missed object"
                  : "Draw a box around an object the model missed, even on an image with no existing detections"
            }
          >
            {drawingMiss ? "Cancel drawing" : "＋ Mark missed object"}
          </button>

          {/* Record the sweep itself as the verdict, for an image where nothing was missed: no box
              to draw, so "Mark missed object" has nothing to submit. */}
          <button
            className="tcip-btn"
            onClick={() => void recordSweepAttested()}
            disabled={!imgName || reviewLocked || !!edit || drawingMiss || !canReview}
            title="Record that you checked this image for missed objects and found none"
          >
            ✓ Confirm: nothing missed
          </button>

          <span className="flex-1" />

          <span className={`tcip-badge ${IMAGE_STATUS_CLASS[imageStatus]}`}>
            {IMAGE_STATUS_LABEL[imageStatus]}
          </span>

          {/* Image navigation: same function + layout/order as the Annotate tab */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-wide text-tcip-muted">
              Image
            </span>
            {imgName && (
              <span className="max-w-[150px] truncate font-mono text-tcip-fg" title={imgName}>
                {imgName}
              </span>
            )}
            {imgName && pqCalibrationByImage?.get(imgName) && (
              <span
                className="tcip-badge bg-tcip-warn/20 text-tcip-warn"
                title="This image is on the bound run's calibration side; reviewing it edits a label inside that run's calibration universe, which its validation will disclose as moved"
              >
                Calibration
              </span>
            )}
            {imgName && !pqCalibrationByImage?.get(imgName) && pqQueue && pqMarksUnresolved && (
              <span className="tcip-badge bg-tcip-warn/20 text-tcip-warn" title={pqMarksUnresolved}>
                Calibration unknown
              </span>
            )}
            <button
              className="tcip-btn"
              onClick={() => stepImage(-1)}
              disabled={!nav.canPrev || !!edit}
              aria-label="Previous image"
            >
              ◀
            </button>
            <input
              ref={counterRef}
              className="tcip-input w-10 text-center font-mono"
              value={counterDraft ?? (nav.position > 0 ? String(nav.position) : "")}
              onChange={(e) => setCounterDraft(e.target.value.replace(/[^0-9]/g, ""))}
              onFocus={() => setCounterDraft(String(nav.position || 1))}
              onBlur={() => setCounterDraft(null)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  const num = parseInt(counterDraft ?? "", 10);
                  if (!Number.isNaN(num)) nav.jumpToPosition(num);
                  setCounterDraft(null);
                  counterRef.current?.blur();
                } else if (e.key === "Escape") {
                  setCounterDraft(null);
                  counterRef.current?.blur();
                }
              }}
            />
            <span className="tabular-nums text-tcip-muted">/ {nav.total}</span>
            <button
              className="tcip-btn"
              onClick={() => stepImage(1)}
              disabled={!nav.canNext || !!edit}
              aria-label="Next image"
            >
              ▶
            </button>
          </div>

          {/* Reviewed: same position as Annotate's Complete; a reversible confirm. */}
          <label className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={imageStatus === "completed"}
              onChange={(e) => void markImageComplete(e.target.checked)}
              disabled={!imgName || !!edit || !dataset.subject || !canReview}
            />
            Reviewed
          </label>
        </div>

        {/* Row 2: the filter controls, collapsed by default and remembered across sessions */}
        {filtersOpen && (
          <div className="flex flex-wrap items-center gap-2 px-3 py-1.5 border-t border-tcip-border text-[11px]">
            <span className="text-tcip-muted">IoU ≥</span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={filters.iou_threshold * 100}
              disabled={!!edit}
              onChange={(e) =>
                patchGui({
                  review: { ...filters, iou_threshold: Number(e.target.value) / 100 },
                })
              }
            />
            <span className="tabular-nums w-10">{filters.iou_threshold.toFixed(2)}</span>

            <span className="ml-3 text-tcip-muted">Conf ≥</span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={filters.conf_threshold * 100}
              disabled={!!edit}
              onChange={(e) =>
                patchGui({
                  review: { ...filters, conf_threshold: Number(e.target.value) / 100 },
                })
              }
            />
            <span className="tabular-nums w-10">{filters.conf_threshold.toFixed(2)}</span>

            <span aria-hidden className="mx-2 h-4 w-px bg-tcip-border" />
            <select
              className="tcip-select"
              value={filters.filter_type}
              disabled={!!edit}
              onChange={(e) =>
                patchGui({ review: { ...filters, filter_type: e.target.value as never } })
              }
            >
              <option value="all">All</option>
              <option value="tp">TP</option>
              <option value="fp">FP</option>
              <option value="fn">FN</option>
            </select>
            <select
              className="tcip-select"
              aria-label="Class filter"
              value={filters.filter_class}
              disabled={!!edit}
              title="Show only detections of one class, or all classes"
              onChange={(e) => patchGui({ review: { ...filters, filter_class: e.target.value } })}
            >
              <option value="all">All classes</option>
              {availableClasses.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <select
              className="tcip-select"
              value={reviewStatus.activeFilter}
              disabled={!!edit}
              title="Show all images, or only those whose review is complete / incomplete"
              onChange={(e) => setReviewStatusFilter(e.target.value as ReviewStatusFilter)}
            >
              <option value="all">All images</option>
              <option value="unreviewed">Unreviewed</option>
              <option value="reviewed">Reviewed</option>
            </select>

            <span aria-hidden className="mx-2 h-4 w-px bg-tcip-border" />
            <span className="text-[10px] font-semibold uppercase tracking-wide text-tcip-muted">
              Visibility
            </span>
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={showGT}
                onChange={(e) => updateShowGT(e.target.checked)}
              />
              Ground truth
            </label>
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={showPred}
                onChange={(e) => updateShowPred(e.target.checked)}
              />
              Predictions
            </label>

            <span aria-hidden className="mx-2 h-4 w-px bg-tcip-border" />
            {/* prioritize_review_queue's ranking, otherwise reachable only one image at a
                time via the agent's own focus_human_attention() calls; surfaced here as a real browsable order. */}
            <span
              className="text-[10px] font-semibold uppercase tracking-wide text-tcip-muted"
              title="Rank unreviewed images by how much the model would learn from your input on them, so you look at the most useful ones first"
            >
              Priority order
            </span>
            <select
              className="tcip-select"
              aria-label="Priority-order model"
              value={pqModelPath}
              disabled={pqStatus === "running"}
              onChange={(e) => setPqModelPath(e.target.value)}
            >
              <option value="">Choose a model…</option>
              {pqModels.map((m) => (
                <option key={m.checkpoint_path} value={m.checkpoint_path}>
                  {m.name}
                </option>
              ))}
            </select>
            <button
              className="tcip-btn"
              disabled={!pqModelPath || pqStatus === "running" || !canReview}
              onClick={() => void computePriorityQueue()}
              title="Rank this date's images by how useful reviewing them would be"
            >
              {pqStatus === "running" ? "Ranking…" : "Rank images"}
            </button>
            {pqStatus === "completed" && pqQueue && (
              <label className="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={pqUseOrder}
                  onChange={(e) => setPqUseOrder(e.target.checked)}
                />
                Browse in priority order ({pqQueue.length} ranked)
              </label>
            )}
            {pqStatus === "failed" && pqError && (
              <span className="text-tcip-fp max-w-[280px]">{pqError}</span>
            )}

            {bandsInfo && bandSelection && showsBandPicker(bandsInfo, bandSelection) && (
              <>
                <span aria-hidden className="mx-2 h-4 w-px bg-tcip-border" />
                <BandPicker
                  bandCount={bandsInfo.band_count}
                  bands={bandsInfo.bands}
                  selection={bandSelection}
                  onChange={setBandSelection}
                  sampled={bandsInfo.sampled}
                  pixelFraction={bandsInfo.pixel_fraction}
                  overviewScale={bandsInfo.overview_scale}
                />
              </>
            )}
          </div>
        )}
      </div>

      <div className="relative flex-1 flex flex-col min-h-0">
        <CanvasStage
          imageUrl={imageUrl}
          imagePath={imgPath}
          autoFit={false}
          imgWidth={imgW}
          imgHeight={imgH}
          regions={regions}
          onBaseFacts={setBaseFacts}
          onPixelDown={edit ? onEditDown : drawingMiss ? onMissDown : undefined}
          onPixelMove={edit ? onEditMove : drawingMiss ? onMissMove : undefined}
          onPixelUp={edit ? onEditUp : drawingMiss ? onMissUp : undefined}
          overlay={
            edit ? (
              <EditShapeOverlay edit={edit} color={reviewColors.active} />
            ) : missDraft ? (
              <EditShapeOverlay edit={{ kind: "box", box: missDraft }} color={reviewColors.fn} />
            ) : undefined
          }
        >
          {matches && (
            <ReviewOverlays
              matches={matches}
              focusedIdx={detectionIdx}
              showGT={showGT}
              showPred={showPred}
              colors={reviewColors}
              suppressFocusedGt={!!edit && current?.det_type !== "fp"}
              suppressFocusedPred={!!edit && current?.det_type === "fp"}
            />
          )}
        </CanvasStage>
        {/* Screen-fixed detection-type badge: in image coords it was illegible at fit
            zoom and canvas-blanketing when zoomed to a detection. */}
        {current && (
          <span
            className="absolute top-2 right-3 tcip-badge border bg-tcip-panel/90 pointer-events-none font-bold"
            style={{
              color: reviewColors[current.det_type],
              borderColor: reviewColors[current.det_type],
            }}
          >
            {current.det_type.toUpperCase()}
            <span className="mx-1 text-tcip-border">|</span>
            <span className="font-normal text-tcip-muted">
              {edit ? "editing" : current.reviewed ? current.reviewed_action : "reviewing"}
            </span>
          </span>
        )}

        <ReviewLegend colors={reviewColors} items={COLOR_LABELS} onEdit={setColorEditKey} />
      </div>

      {/* Empty-state card: tells the reviewer why there is nothing to step through,
          "no predictions configured" vs "filters exclude everything". Non-opaque and
          pointer-transparent so still-rendered GT overlays stay visible behind it. */}
      {matches && matches.detections.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="max-w-md rounded-lg border border-tcip-border bg-tcip-panel/90 px-5 py-4 text-center">
            <p className="text-sm font-semibold text-tcip-fg">No detections to review</p>
            <p className="mt-1 text-xs text-tcip-muted">
              {!dataset.predictions_dir
                ? "No predictions directory configured; run inference or select a model with predictions for this dataset."
                : `No detections on this image under the current filters (IoU ≥ ${filters.iou_threshold.toFixed(
                    2,
                  )}, Conf ≥ ${filters.conf_threshold.toFixed(2)}, type ${
                    filters.filter_type
                  }); relax filters to see more.`}
            </p>
          </div>
        </div>
      )}

      <div className="flex items-center gap-2 px-3 py-1.5 border-t border-tcip-border bg-tcip-panel text-[11px]">
        <span className="text-tcip-muted">Detection</span>
        <button
          className="tcip-btn"
          onClick={() => stepDetection(-1)}
          disabled={!matches || matches.detections.length === 0 || detectionIdx <= 0 || !!edit}
          title="Previous detection (←)"
        >
          ◀
        </button>
        <span className="tabular-nums">
          {matches && matches.detections.length > 0
            ? `${detectionIdx + 1} / ${matches.detections.length}`
            : "0 / 0"}
        </span>
        <button
          className="tcip-btn"
          onClick={() => stepDetection(1)}
          disabled={
            !matches ||
            matches.detections.length === 0 ||
            detectionIdx >= matches.detections.length - 1 ||
            !!edit
          }
          title="Next detection (→)"
        >
          ▶
        </button>
        {current && (
          <>
            <span className="text-tcip-muted">
              {current.class_name}
              {current.conf !== null && (
                <>
                  <span className="mx-1.5 text-tcip-border">|</span>conf {current.conf.toFixed(2)}
                </>
              )}
              {current.iou !== null && (
                <>
                  <span className="mx-1.5 text-tcip-border">|</span>IoU {current.iou.toFixed(2)}
                </>
              )}
            </span>
          </>
        )}

        <span className="flex-1" />

        {current && !edit && (
          <>
            {reviewLocked && <span className="text-tcip-muted">Reviewed; uncheck to edit</span>}
            <button
              className="tcip-btn-primary"
              onClick={() => void recordAction("accepted")}
              disabled={reviewLocked || !canReview}
              title={acceptTitle}
            >
              ✓&nbsp;&nbsp;{acceptLabel}
            </button>
            <button
              className="tcip-btn"
              onClick={startEdit}
              disabled={reviewLocked || !canReview}
              title="Adjust this shape on the canvas (E)"
            >
              ✎&nbsp;&nbsp;Edit
            </button>
            <button
              className="tcip-btn-danger"
              onClick={() => void recordAction("rejected")}
              disabled={rejectDisabled}
              title={rejectTitle}
            >
              ✕&nbsp;&nbsp;{rejectLabel}
            </button>
          </>
        )}
        {edit && (current || pendingMiss) && (
          <>
            <span className="tcip-badge bg-transparent border border-tcip-pred text-tcip-pred">
              {pendingMiss ? "Marking missed object" : "Editing"}
            </span>
            <button
              className="tcip-btn-primary"
              onClick={() => void commitEdit()}
              title={
                pendingMiss
                  ? "Save this missed object to ground truth (Enter)"
                  : current?.det_type === "fp"
                    ? "Write this shape to ground truth (Enter)"
                    : "Replace the ground-truth shape with this one (Enter)"
              }
            >
              ✓&nbsp;&nbsp;Save {pendingMiss ? "missed object" : "edit"}
            </button>
            <button
              className="tcip-btn"
              onClick={cancelEdit}
              title="Discard this adjustment; ground truth unchanged (Esc)"
            >
              Cancel
            </button>
          </>
        )}
      </div>

      {colorEditKey && (
        <ColorPickerModal
          title={`${COLOR_LABELS.find((c) => c.key === colorEditKey)?.label ?? "Colour"}`}
          initialColor={reviewColors[colorEditKey]}
          onSubmit={(c) => {
            setReviewColors((prev) => ({ ...prev, [colorEditKey]: c }));
            setColorEditKey(null);
          }}
          onCancel={() => setColorEditKey(null)}
        />
      )}
    </div>
  );
}
