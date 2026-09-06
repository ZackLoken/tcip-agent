import { useEffect, useMemo, useRef, useState } from "react";
import { Rect } from "react-konva";
import Konva from "konva";

import { api } from "@/api/client";
import { classesApi, subjectColor, type ImageStatus } from "@/api/classes";
import { sessionsApi } from "@/api/sessions";
import { AnnotateLegend } from "@/components/annotate/AnnotateLegend";
import { AnnotationShapes } from "@/components/annotate/AnnotationShapes";
import { AttributePanel } from "@/components/annotate/AttributePanel";
import { InProgressPolygon } from "@/components/annotate/InProgressPolygon";
import { SnapIndicator } from "@/components/annotate/SnapIndicator";
import { AnnotateToolbar } from "@/components/AnnotateToolbar";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
import { CoverageChrome } from "@/components/Canvas/CoverageChrome";
import { CoverageOverlay } from "@/components/Canvas/CoverageOverlay";
import { TabHeading } from "@/components/TabHeading";
import { useBandSelection } from "@/hooks/useBandSelection";
import { useCoverageGrid } from "@/hooks/useCoverageGrid";
import { useCoverageTracking } from "@/hooks/useCoverageTracking";
import { useDisclosure } from "@/hooks/useDisclosure";
import { useImageBands } from "@/hooks/useImageBands";
import { useImageNav } from "@/hooks/useImageNav";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { usePrefetchAdjacentImages } from "@/hooks/usePrefetchAdjacentImages";
import { useRegionCompleteness } from "@/hooks/useRegionCompleteness";
import { useRegionServes } from "@/hooks/useRegionServes";
import { compositeParams } from "@/lib/bandSelection";
import { cellAt, currentCoverageCell, stepUnsweptCell, type GridCell } from "@/lib/coverage";
import type { LoadedImage } from "@/lib/imageLoader";
import { canvasHoldsSubject } from "@/lib/imageStatus";
import { currentImage, labelPath } from "@/lib/paths";
import { fitView, zoomToRect } from "@/lib/viewGeometry";
import {
  buildAnnotateShapes,
  computeViewport,
  createCanvasPusher,
  measureCanvasHost,
  onCanvasStateRequest,
  type CanvasStateBody,
} from "@/lib/canvasSync";
import { canvasToAnnotations } from "@/lib/labelSerde";
import {
  computePolygonBboxes,
  cutRing,
  findHitPoint,
  findHoveredPolygon,
  keyboardCutRefusal,
  pointInRings,
  pointToSegmentDist,
  ringsBbox,
  withRing,
} from "@/lib/polygonGeometry";
import { applyEditDrag, hitTestEdit, MIN_BOX_SIDE, type EditDrag } from "@/lib/editGeometry";
import { useSubjectColors } from "@/lib/subjectColors";
import { nextMode } from "@/lib/toolMode";
import { useStore } from "@/store";
import type { Box, PolygonShape } from "@/store/types";

const SNAP_RADIUS_CANVAS = 15;
const VERTEX_HANDLE_RADIUS = 4;
const EDGE_INSERT_THRESHOLD = 6;
const STREAM_MIN_DIST_CANVAS = 6; // screen px between vertices laid down in Stream (freehand) mode
// Screen-px grab radius for a placed point: the whole mark is its own handle, so this matches the
// mark's outer reach (see the tick geometry in PointOverlay) rather than a hidden smaller target.
const POINT_HIT_CANVAS = 11;

export function AnnotateTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const view = useStore((s) => s.gui.view);
  const setView = useStore((s) => s.setView);
  const mode = useStore((s) => s.gui.mode);
  const activeSubject = useStore((s) => s.gui.active_subject);
  // The subject registry (subject -> {description?, attributes?}); drives colours (name-derived,
  // GUI-local) and the per-instance attribute editor.
  const registry = useStore((s) => s.registry.subjects);

  const canvas = useStore((s) => s.canvas);
  const loadLabels = useStore((s) => s.loadLabelsIntoCanvas);
  const addBox = useStore((s) => s.addBox);
  const dragBox = useStore((s) => s.dragBox);
  const deleteBox = useStore((s) => s.deleteBox);
  const deletePolygon = useStore((s) => s.deletePolygon);
  const splitPolygon = useStore((s) => s.splitPolygon);
  const updatePolygon = useStore((s) => s.updatePolygon);
  const dragVertex = useStore((s) => s.dragVertex);
  const addPoint = useStore((s) => s.addPoint);
  const dragPoint = useStore((s) => s.dragPoint);
  const deletePoint = useStore((s) => s.deletePoint);
  const selectPoint = useStore((s) => s.selectPoint);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const setCurrentPolygon = useStore((s) => s.setCurrentPolygon);
  const commitCurrentPolygon = useStore((s) => s.commitCurrentPolygon);
  const selectPolygon = useStore((s) => s.selectPolygon);
  const markClean = useStore((s) => s.markClean);
  const pushUndo = useStore((s) => s.pushUndo);
  const setActiveSubject = useStore((s) => s.setActiveSubject);

  const annotateUi = useStore((s) => s.annotateUi);
  const setHoveredPolygon = useStore((s) => s.setHoveredPolygon);
  const imageStatus = useStore((s) => s.imageStatus);
  const setImageStatus = useStore((s) => s.setImageStatus);
  const startImageSessionTracking = useStore((s) => s.startImageSessionTracking);
  const incrementAnnotationsAdded = useStore((s) => s.incrementAnnotationsAdded);
  const markSessionFlushed = useStore((s) => s.markSessionFlushed);
  const clearSessionTracking = useStore((s) => s.clearSessionTracking);

  const [drawing, setDrawing] = useState<Box | null>(null);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  // The cut tool's pending first click: never canvas.currentPolygon, which undo, the mirror and
  // the stream/vertex-placement branches all read as an open polygon in progress.
  const [cutStart, setCutStart] = useState<{
    point: [number, number];
    polygonIdx: number;
    polygon: PolygonShape;
  } | null>(null);
  const stageRef = useRef<Konva.Stage | null>(null);
  // Box editing (mirrors polygon vertex editing): a selected box shows handles; a press on
  // one starts a corner-resize / move drag. selectedBoxIdx is cleared on image change below.
  const [selectedBoxIdx, setSelectedBoxIdx] = useState<number | null>(null);
  // Hover indices for hover-only labels, one per shape kind the canvas draws; each is written
  // only on a transition (see processMoveRef), so a plain move re-renders nothing.
  const [hoveredBoxIdx, setHoveredBoxIdx] = useState<number | null>(null);
  const [hoveredDerivedIdx, setHoveredDerivedIdx] = useState<number | null>(null);
  const [hoveredPointIdx, setHoveredPointIdx] = useState<number | null>(null);
  const boxDragRef = useRef<{ idx: number; drag: EditDrag } | null>(null);
  // Index of the point being dragged. A point has no vertices, so repositioning it is the whole
  // edit: one undo snapshot is taken when the drag starts (see onDown), like a box/vertex drag.
  const pointDragRef = useRef<number | null>(null);

  // I/O safety. The canvas belongs to exactly the image last loaded from disk:
  //  - loadedPathsRef: the (image, label) the current shapes came from. save() writes there,
  //    never to a path recomputed from a since-changed dataset, which is how the old code could
  //    write one image's shapes onto another's file.
  //  - loadedKeyRef: gates reloads to a genuine image-identity change, so unrelated store updates
  //    (a WS snapshot, a mode/subject toggle) don't re-read disk and clobber unsaved edits.
  //  - saveBlocked: set when a load failed, so a blank canvas can't overwrite the labels on disk.
  const loadedKeyRef = useRef<string | null>(null);
  const loadedPathsRef = useRef<{
    image: string;
    label: string | null;
    mtime: string | null;
  } | null>(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const [saveBlocked, setSaveBlocked] = useState(false);
  // True when a save/reload conflict is showing (file changed underneath us);
  // the banner then offers a Reload button.
  const [conflict, setConflict] = useState(false);
  const agentActivity = useStore((s) => s.agentActivity);

  const imgPath = currentImage(dataset).path;
  const currentImageName = dataset.image_list[dataset.current_image_index] ?? null;
  // A confirmed negative is a completed review (empty): lock it like "complete".
  const currentStatus = currentImageName ? imageStatus.byImage[currentImageName] : undefined;
  const isLocked = currentStatus === "complete" || currentStatus === "negative";
  const saveDisabled = !imgPath || isLocked || saveBlocked;

  const bandsInfo = useImageBands(imgPath);
  const [bandSelection, setBandSelection] = useBandSelection(bandsInfo);

  // Every request for this view (the canvas' own and the prefetcher's warm-up) carries one set
  // of band params, so the two never warm and read different renders of the same image.
  const composite = compositeParams(bandsInfo, bandSelection);

  // Image navigation (shared with TopBar + Review; honors the status filter).
  const nav = useImageNav();
  usePrefetchAdjacentImages(composite.bands, composite.stretch);

  // ── Coverage: lattice, region serves, session accumulation ────────────────
  const [baseFacts, setBaseFacts] = useState<LoadedImage | null>(null);
  const coverageGrid = useCoverageGrid({
    imagePath: imgPath,
    subject: activeSubject,
    date: dataset.date,
    datasetRoot: dataset.dataset_root,
  });
  const coverageViewing = useMemo(
    () => ({
      bands: composite.bands,
      stretch: composite.stretch,
      stats_source: baseFacts?.statsSource ?? null,
      display_bounds: baseFacts?.displayBounds ?? null,
      base_served_size: baseFacts?.servedSizeRaw ?? null,
    }),
    [composite.bands, composite.stretch, baseFacts],
  );
  const completeness = useRegionCompleteness({
    imagePath: imgPath,
    datasetRoot: dataset.dataset_root,
    subject: activeSubject,
    grid: coverageGrid.grid,
  });
  const coverage = useCoverageTracking({
    imagePath: imgPath,
    datasetRoot: dataset.dataset_root,
    subject: activeSubject,
    date: dataset.date,
    grid: coverageGrid.grid,
    cells: coverageGrid.cells,
    view,
    imgW: canvas.imgWidth,
    imgH: canvas.imgHeight,
    viewing: coverageViewing,
    workingScale: completeness.workingScale,
  });
  const regions = useRegionServes({
    imagePath: imgPath,
    imgW: canvas.imgWidth,
    imgH: canvas.imgHeight,
    view,
    servingCells: coverageGrid.servingCells,
    servingTileSize: coverageGrid.serving?.tile_size ?? null,
    coverageCells: coverageGrid.cells,
    baseFacts,
    composite,
    onCellServedAtNative: coverage.noteServedAtNative,
  });
  const coverageMultiCell = coverageGrid.cells.length > 1;
  const { open: coverageOverlayOn, toggle: toggleCoverageOverlay } = useDisclosure(
    "tcip.annotate.coverageGridOverlayOpen",
  );
  // The cell a Map click just opened (currentCoverageCell names it while it stays in view);
  // cleared on an image change so a stale selection cannot outlive its own raster.
  const [mapSelectedCell, setMapSelectedCell] = useState<GridCell | null>(null);
  useEffect(() => {
    setMapSelectedCell(null);
  }, [imgPath]);

  // A tool must always be shown active: when the Map tool is withdrawn and settled (not merely
  // a grid still loading), fall back to a drawing tool rather than leaving the canvas inert.
  useEffect(() => {
    if (mode === "map" && !coverageMultiCell && coverageGrid.settled) {
      useStore.getState().setMode("box");
    }
  }, [mode, coverageMultiCell, coverageGrid.settled]);

  function jumpToCell(cell: GridCell) {
    const host = measureCanvasHost();
    if (!host || !coverageGrid.grid) return;
    // Half the cell stride per axis; the lattice pins overlap to 0, so stride = tile size.
    const pad = coverageGrid.grid.tile_size / 2;
    setView(
      zoomToRect(
        { x0: cell.x0, y0: cell.y0, x1: cell.x1, y1: cell.y1 },
        { host, imgW: canvas.imgWidth, imgH: canvas.imgHeight, padX: pad, padY: pad },
      ),
    );
  }

  function coverageViewportRect() {
    const host = measureCanvasHost();
    if (!host) return null;
    const vp = computeViewport(view, host, canvas.imgWidth, canvas.imgHeight);
    return vp ? { x0: vp.x, y0: vp.y, x1: vp.x + vp.w, y1: vp.y + vp.h } : null;
  }

  function imageFitScale(): number | null {
    const host = measureCanvasHost();
    if (!host || canvas.imgWidth <= 0 || canvas.imgHeight <= 0) return null;
    return fitView(host, canvas.imgWidth, canvas.imgHeight).scale;
  }

  function overview() {
    const host = measureCanvasHost();
    if (!host || canvas.imgWidth <= 0 || canvas.imgHeight <= 0) return;
    setMapSelectedCell(null);
    setView(fitView(host, canvas.imgWidth, canvas.imgHeight));
  }

  function stepCoverageCell(delta: 1 | -1) {
    if (!coverageMultiCell) return;
    const host = measureCanvasHost();
    if (!host) return;
    const v = useStore.getState().gui.view;
    const viewport = computeViewport(v, host, canvas.imgWidth, canvas.imgHeight);
    const center = viewport
      ? { x: viewport.x + viewport.w / 2, y: viewport.y + viewport.h / 2 }
      : { x: canvas.imgWidth / 2, y: canvas.imgHeight / 2 };
    const target = stepUnsweptCell(coverageGrid.cells, coverage.swept, center, delta);
    if (!target) {
      useStore
        .getState()
        .pushToast(
          `All ${coverageGrid.cells.length} grid cells are swept at the working zoom.`,
          "info",
        );
      return;
    }
    jumpToCell(target);
  }

  // A box selection belongs to one image; leaving it drops the selection + any drag (and ends a
  // live freehand stream so it can't bleed vertices onto the next image).
  useEffect(() => {
    setSelectedBoxIdx(null);
    setHoveredBoxIdx(null);
    setHoveredDerivedIdx(null);
    setHoveredPointIdx(null);
    boxDragRef.current = null;
    pointDragRef.current = null;
    streamingRef.current = false;
    setCutStart(null);
    useStore.getState().setCut(false);
  }, [currentImageName]);

  // Leaving polygon mode clears a pending cut and its flag: the gesture and the flag are only
  // meaningful there, and a lingering armed cut would surprise the next mode's own clicks.
  useEffect(() => {
    if (mode !== "polygon") {
      setCutStart(null);
      useStore.getState().setCut(false);
    }
  }, [mode]);

  // Disarming the cut flag, by the toolbar button or the x key, must not leave a pending start
  // with a rubber band pointing at a tool no longer armed; neither caller has cutStart in scope.
  useEffect(() => {
    if (!annotateUi.cut) setCutStart(null);
  }, [annotateUi.cut]);

  // Confirming the image (Complete/Negative) locks editing: a pending cut and its flag must not
  // survive into a state where every click on it is refused.
  useEffect(() => {
    if (isLocked) {
      setCutStart(null);
      useStore.getState().setCut(false);
    }
  }, [isLocked]);

  // ── Live canvas push (agent visibility: capture_live_canvas) ──────────────
  // The ref always holds the freshest closure so the debounced pusher never reads stale state.
  const colorTick = useSubjectColors(); // bumps on a recolour, so swatches recompute below
  const subjectSwatches = useMemo(() => {
    void colorTick; // read only to force recompute; subjectColor() itself needs no argument for it
    return Object.keys(registry).map((name) => ({ name, color: subjectColor(name) }));
  }, [registry, colorTick]);
  const buildCanvasBodyRef = useRef<() => CanvasStateBody | null>(() => null);
  buildCanvasBodyRef.current = () => {
    if (!imgPath || !dataset.project_root) return null;
    // Never push mid-transition: attaching the previous image's still-live shapes to the new
    // image_path would show the agent a false canvas; wait until the loaded identity matches.
    if (loadedPathsRef.current?.image !== imgPath) return null;
    // Binding-presence gate: a dataset without an adopted generation (pre-feature data, a
    // deleted binding, an export/adopt pass) must stop pushing rather than fabricate one.
    const generation = useStore.getState().bindingGeneration;
    useStore.getState().setCanvasBindingMissing(generation == null);
    if (generation == null) return null;
    const host = measureCanvasHost();
    return {
      binding_generation: generation,
      tab: "annotate",
      image_path: imgPath,
      image: currentImageName ?? "",
      img_width: canvas.imgWidth,
      img_height: canvas.imgHeight,
      viewport: host ? computeViewport(view, host, canvas.imgWidth, canvas.imgHeight) : null,
      mode,
      active_subject: activeSubject ?? undefined,
      cut_armed: annotateUi.cut,
      dirty: canvas.dirty,
      user: useStore.getState().user || undefined,
      classes: subjectSwatches,
      counts: {
        boxes: canvas.boxes.length,
        polygons: canvas.polygons.length,
        points: canvas.points.length,
        image_ratings: canvas.imageAnnotations.length,
        drawing_points: canvas.currentPolygon.length,
      },
      shapes: buildAnnotateShapes({
        boxes: canvas.boxes,
        polygons: canvas.polygons,
        points: canvas.points,
        currentPolygon: canvas.currentPolygon,
        drawingBox: drawing,
        selectedPolygonIdx: canvas.selectedPolygonIdx,
        selectedBoxIdx,
        selectedPointIdx: canvas.selectedPointIdx,
        mode,
        activeSubject: activeSubject ?? "",
        visible: annotateUi.visible,
        colorFor: subjectColor,
        cutStart: cutStart
          ? { point: cutStart.point, color: subjectColor(cutStart.polygon.subject) }
          : null,
        cursor,
      }),
    };
  };
  const canvasPusherRef = useRef(createCanvasPusher((b) => api.canvas.pushState(b)));
  useEffect(() => () => canvasPusherRef.current.dispose(), []);
  // Anything that changes which shapes the canvas draws → full push (geometry travels), except
  // mid-drag/stream where committed geometry re-serializing per tick would jank dense images:
  // those downgrade to heartbeats and the release (drag ref clearing, commit) sends the full.
  useEffect(() => {
    const interacting =
      !!annotateUi.draggingVertex ||
      streamingRef.current ||
      !!drawing ||
      !!boxDragRef.current ||
      pointDragRef.current !== null;
    canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), !interacting);
  }, [
    canvas.boxes,
    canvas.polygons,
    canvas.points,
    canvas.currentPolygon,
    canvas.selectedPolygonIdx,
    canvas.selectedPointIdx,
    imgPath,
    mode,
    activeSubject,
    selectedBoxIdx,
    annotateUi.visible,
    annotateUi.cut,
    annotateUi.draggingVertex,
    drawing,
    cutStart,
  ]);
  useEffect(() => {
    canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), false);
  }, [view, subjectSwatches, canvas.dirty]);
  useEffect(
    () =>
      onCanvasStateRequest(() => {
        canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), true);
        canvasPusherRef.current.flush();
      }),
    [],
  );

  // Leaving Stream mode ends any live stream (the in-progress polygon is left for the user).
  useEffect(() => {
    if (!annotateUi.stream) streamingRef.current = false;
  }, [annotateUi.stream]);

  // ── Label load + save ───────────────────────────────────────────────

  // Save the current canvas to the path it was actually loaded from. Reads the live store + refs
  // (not render closures), so it stays correct even when called from an effect while the app is
  // mid-transition to another image.
  async function save(opts?: { interactive?: boolean }) {
    // interactive=false is the auto-flush on navigate/unmount: it can't show the Reload
    // banner (the user is on another image), but a dropped save must never be silent:
    // it surfaces as a toast naming the image whose edits were lost.
    const interactive = opts?.interactive ?? true;
    const paths = loadedPathsRef.current;
    if (!paths) return; // no confirmed load → refuse to overwrite on-disk labels
    const c = useStore.getState().canvas;
    if (!c.dirty) return;
    if (!paths.label) {
      // No annotations directory is set for this dataset, so there is nowhere to write; a
      // server round trip would only come back with the same refusal.
      if (interactive)
        setIoError("This dataset has no annotations directory set; select one to save.");
      return;
    }
    const projectRoot = useStore.getState().gui.dataset.project_root;
    const imgFileName = paths.image.split(/[/\\]/).pop() ?? "image";

    let result;
    try {
      result = await api.annotate.save({
        image_path: paths.image,
        label_path: paths.label,
        annotations: canvasToAnnotations({
          boxes: c.boxes,
          polygons: c.polygons,
          points: c.points,
          imageAnnotations: c.imageAnnotations,
        }),
        project_root: projectRoot,
        base_mtime: paths.mtime,
        user: useStore.getState().user,
      });
    } catch {
      // Identity check: a stale failure for a since-left image must not raise a
      // banner over the image now on screen.
      if (interactive && loadedPathsRef.current === paths) {
        setIoError(
          "Could not save annotations. Your edits are kept in the editor; press Save to retry.",
        );
      } else {
        useStore.getState().pushToast(`Save failed: ${imgFileName}'s edits were not written.`);
      }
      return;
    }

    if (result.status === "conflict") {
      // Someone else (agent or another tab) wrote this file since we loaded it.
      // Never clobber their work; keep the canvas dirty. The banner belongs to the
      // image on screen; after navigating away, the loss is reported as a toast.
      if (interactive && loadedPathsRef.current === paths) {
        setConflict(true);
        setIoError(
          "These labels changed elsewhere (the agent or another tab). Reload to load the latest (discards your unsaved edits), or keep editing.",
        );
      } else {
        useStore
          .getState()
          .pushToast(
            `Save failed: ${imgFileName}'s labels changed elsewhere (agent or another tab) first.`,
          );
      }
      return;
    }

    // Heals an unconfirmed status from the saved content; a confirmed name (complete or
    // negative) is a human mark, rewritten only through the toolbar's re-confirm action.
    const name = paths.image.split(/[/\\]/).pop() ?? "";
    if (projectRoot && name) {
      const current = useStore.getState().imageStatus.byImage[name];
      const confirmed = current === "complete" || current === "negative";
      if (!confirmed) {
        const hasContent = canvasHoldsSubject(
          {
            boxes: c.boxes,
            polygons: c.polygons,
            points: c.points,
            imageAnnotations: c.imageAnnotations,
          },
          dataset.subject,
        );
        const newStatus: ImageStatus = hasContent ? "partial" : "unannotated";
        if (current !== newStatus) {
          setImageStatus(name, newStatus);
          if (dataset.subject) {
            // Best-effort status write; the labels are already saved.
            void classesApi
              .setImageStatus(
                projectRoot,
                name,
                newStatus,
                dataset.subject,
                dataset.date,
                dataset.dataset_root,
                dataset.annotations_dir,
                useStore.getState().user || undefined,
              )
              .catch(() => {});
          }
        }
      }
    }

    // Staleness guard: flushLeaving() fires this save without awaiting it, so by
    // the time the POST resolves the load effect may already have loaded the next
    // image and repointed loadedPathsRef. Rewinding the ref here would make every
    // later save write the new image's shapes onto the old image's label file
    // (with an echoed mtime that matches it, so the backend's 409 guard can't catch
    // it), and markClean() would silently drop edits already made on the new image.
    if (loadedPathsRef.current !== paths) return;

    loadedPathsRef.current = { ...paths, mtime: result.base_mtime };
    markClean();
    setIoError(null);
    setConflict(false);
    // The saved content may have made an attested cell stale or changed a saved count; the
    // store is the only source of truth, so the open image's completeness read runs again.
    completeness.reload();
  }

  // Re-fetch the current image's labels from disk, discarding local edits. Used to
  // resolve a conflict (409) or to pick up an agent write on a clean canvas.
  async function reloadCurrent() {
    const paths = loadedPathsRef.current;
    if (!paths) return;
    try {
      const labels = await api.annotate.load(paths.image, paths.label);
      loadLabels(labels);
      loadedPathsRef.current = { ...paths, mtime: labels.base_mtime };
      setIoError(null);
      setConflict(false);
      setSaveBlocked(false);
    } catch {
      setIoError("Reload failed. Check the connection and try again.");
    }
  }

  // Flush telemetry + any unsaved edits for the image being left, using the path
  // that canvas belongs to. Called before loading a different image and on unmount.
  function flushLeaving() {
    const leaving = useStore.getState().sessionTracking.currentImageName;
    if (leaving) emitImageSessionEvent(leaving);
    void save({ interactive: false });
  }

  // React to the agent writing labels (panel event). If it touched the file we're
  // viewing: reload on a clean canvas, or offer a Reload conflict prompt if dirty.
  // (Different file → the StatusBar indicator already shows the activity.)
  useEffect(() => {
    if (
      !agentActivity ||
      agentActivity.panel !== "annotate" ||
      agentActivity.eventType !== "labels_written"
    )
      return;
    const paths = loadedPathsRef.current;
    if (!paths) return;
    const written = Array.isArray(agentActivity.data.written)
      ? (agentActivity.data.written as string[])
      : [];
    const norm = (p: string | null) => (p ? p.replace(/\\/g, "/") : "");
    const current = norm(paths.label);
    if (!current || !written.some((w) => norm(w) === current)) return;
    if (useStore.getState().canvas.dirty) {
      setConflict(true);
      setIoError(
        "The agent just updated this image's labels. Reload to load them (discards your unsaved edits), or keep editing.",
      );
    } else {
      void reloadCurrent();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentActivity?.seq]);

  useEffect(() => {
    if (!imgPath || !currentImageName) return;
    const label = labelPath(dataset, currentImageName);
    const key = `${imgPath}\0${label ?? ""}`;

    // Already displaying this exact image + label target. Ignore: this is what
    // stops an unrelated store change (a WS state snapshot, a mode/subject toggle,
    // any patchGui that swaps the dataset object) from re-reading disk and
    // discarding unsaved canvas edits.
    if (loadedKeyRef.current === key) return;

    // Switching images: flush the previous image's work first (to the path it
    // belongs to), then load the new one.
    flushLeaving();

    let cancelled = false;
    void (async () => {
      try {
        const labels = await api.annotate.load(imgPath, label);
        if (cancelled) return;
        loadLabels(labels);
        loadedKeyRef.current = key;
        loadedPathsRef.current = { image: imgPath, label, mtime: labels.base_mtime };
        setSaveBlocked(false);
        setIoError(null);
        setConflict(false);
        startImageSessionTracking(currentImageName);
      } catch {
        if (cancelled) return;
        // Show a blank canvas but block saving so a transient load failure can't let an
        // empty canvas overwrite the labels still on disk. image_path stays empty so the
        // Complete checkbox won't derive a status from this blank canvas either.
        loadLabels({
          image_path: "",
          img_width: 0,
          img_height: 0,
          boxes: [],
          polygons: [],
          points: [],
          imageAnnotations: [],
        });
        loadedKeyRef.current = key;
        loadedPathsRef.current = null;
        setSaveBlocked(true);
        setConflict(false);
        setIoError(
          "Could not load this image's labels. Saving is disabled to avoid overwriting the labels on disk.",
        );
        startImageSessionTracking(currentImageName);
      }
    })();
    return () => {
      cancelled = true;
    };
    // Keyed on image identity + label dir only (see loadedKeyRef guard); save /
    // loadLabels / tracking actions are stable or ref-based.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imgPath, currentImageName, dataset.annotations_dir]);

  function emitImageSessionEvent(imageName: string) {
    const state = useStore.getState();
    const tracking = state.sessionTracking;
    const projectRoot = state.gui.dataset.project_root;
    if (!projectRoot) return;
    if (tracking.currentImageName !== imageName || tracking.imageEnterTimeMs === null) return;

    const c = state.canvas;
    const finalAnnotationCount =
      c.boxes.length + c.polygons.length + c.points.length + c.imageAnnotations.length;
    const elapsedSeconds = Math.max(0, (Date.now() - tracking.imageEnterTimeMs) / 1000);
    const key = `${imageName}|${tracking.imageEnterTimeMs}|${tracking.annotationsAddedDelta}|${finalAnnotationCount}`;
    if (tracking.lastFlushedKey === key) return;

    markSessionFlushed(key);
    clearSessionTracking();
    void sessionsApi
      .imageEvent({
        project_root: projectRoot,
        image_name: imageName,
        session_seconds_delta: Number(elapsedSeconds.toFixed(2)),
        annotations_added_delta: tracking.annotationsAddedDelta,
        final_annotation_count: finalAnnotationCount,
        dataset_root: state.gui.dataset.dataset_root,
        subject: state.gui.dataset.subject,
        date: state.gui.dataset.date,
      })
      .catch(() => {
        // Best-effort telemetry; annotation flow should never block on this.
      });
  }

  function commitPolygonAndTrack() {
    if (isLocked) return;
    // Closing always ends a live stream: a double-click's leading clicks re-arm streaming,
    // and a stale flag would immediately stream a fresh polygon from the next mouse move.
    streamingRef.current = false;
    if (commitCurrentPolygon()) {
      incrementAnnotationsAdded(1);
    }
  }

  // Flush telemetry + any unsaved edits when the tab unmounts (e.g. switching to
  // another tab). Image-to-image flushing is handled by the load effect above.
  useEffect(() => {
    return () => {
      flushLeaving();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stepImage(delta: number) {
    // Shared filtered traversal; the label-load effect flushes the outgoing image
    // (save + telemetry) when the index changes, so no explicit save is needed here.
    nav.stepImage(delta);
    selectPolygon(null);
  }

  function selectSubjectByIndex(idx: number) {
    // Number keys pick the Nth declared subject (0-based), mirroring the old class-number keys.
    const names = Object.keys(useStore.getState().registry.subjects);
    if (names[idx]) setActiveSubject(names[idx]);
  }

  // No subject selected → an authored shape has nowhere to attach and the backend save rejects
  // it. Refuse to start a drawing and say so, once (not per click).
  const noSubjectNoticeRef = useRef<string | null>(null);
  function requireSubject(): boolean {
    if (activeSubject) return true;
    if (noSubjectNoticeRef.current !== currentImageName) {
      noSubjectNoticeRef.current = currentImageName;
      useStore.getState().pushToast("Select a subject before drawing (use the subject picker).");
    }
    return false;
  }

  // No keyboard path selects a polygon, so this names the click as the precondition for both
  // paths. Channelled "cut", so a repeat replaces the standing toast with a count.
  function requireCutSelection(): void {
    useStore
      .getState()
      .pushToast(
        "Select a polygon first by clicking it, then click two points on either side of it or " +
          "press Shift+H / Shift+V.",
        "error",
        "cut",
      );
  }

  // Shared by the mouse and keyboard cut paths, so the two never drift onto separate wordings.
  function multiRingCutRefusal(ringCount: number): string {
    return (
      `This shape covers ${ringCount} separate parts of one object; the cut applies to a ` +
      "single outline. Cut a part in polygon mode after the others are removed, or redraw it."
    );
  }

  // Shift+H / Shift+V: two endpoints one bounding-box width (or height) outside each side of
  // the centre line, then the same splitPolygon/incrementAnnotationsAdded path a mouse cut takes.
  function runAxisCut(axis: "horizontal" | "vertical"): void {
    if (cutStart) {
      // A pending click-cut start is the mouse's; the keys leave it alone but say why nothing
      // happened rather than staying silent.
      useStore
        .getState()
        .pushToast(
          "A cut is half-placed. Press Esc to clear it, then x to re-arm.",
          "error",
          "cut",
        );
      return;
    }
    const idx = canvas.selectedPolygonIdx;
    if (idx === null) {
      requireCutSelection();
      return;
    }
    const polygon = canvas.polygons[idx];
    if (polygon.rings.length > 1) {
      useStore.getState().pushToast(multiRingCutRefusal(polygon.rings.length), "error", "cut");
      return;
    }
    const [minX, minY, maxX, maxY] = ringsBbox(polygon.rings);
    const width = maxX - minX;
    const height = maxY - minY;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const [a, b]: [[number, number], [number, number]] =
      axis === "horizontal"
        ? [
            [minX - width, cy],
            [maxX + width, cy],
          ]
        : [
            [cx, minY - height],
            [cx, maxY + height],
          ];
    const result = cutRing(polygon.rings[0], a, b);
    if ("reason" in result) {
      useStore.getState().pushToast(keyboardCutRefusal(result.reason), "error", "cut");
      return;
    }
    splitPolygon(idx, result.rings);
    incrementAnnotationsAdded(1);
  }

  useKeyboardShortcuts([
    { keys: "ctrl+z", action: () => undo(), when: () => !isLocked },
    { keys: "ctrl+shift+z", action: () => redo(), when: () => !isLocked },
    { keys: "ctrl+y", action: () => redo(), when: () => !isLocked },
    { keys: "ctrl+s", action: () => void save(), when: () => !isLocked },
    { keys: "m", action: () => useStore.getState().setMode(nextMode(mode)) },
    {
      keys: "v",
      action: () => useStore.getState().setStream(!annotateUi.stream),
      when: () => mode === "polygon",
    },
    {
      keys: "s",
      action: () => useStore.getState().setSnap(!annotateUi.snap),
      when: () => mode === "polygon",
    },
    {
      keys: "x",
      action: () => useStore.getState().setCut(!annotateUi.cut),
      when: () => mode === "polygon" && !isLocked,
    },
    {
      keys: "shift+h",
      action: () => runAxisCut("horizontal"),
      when: () => mode === "polygon" && !isLocked && annotateUi.cut,
      whileFocused: true,
    },
    {
      keys: "shift+v",
      action: () => runAxisCut("vertical"),
      when: () => mode === "polygon" && !isLocked && annotateUi.cut,
      whileFocused: true,
    },
    {
      keys: "delete",
      action: () => {
        if (canvas.selectedPolygonIdx !== null) deletePolygon(canvas.selectedPolygonIdx);
        else if (selectedBoxIdx !== null) {
          deleteBox(selectedBoxIdx);
          setSelectedBoxIdx(null);
        } else if (canvas.selectedPointIdx !== null) deletePoint(canvas.selectedPointIdx);
      },
      when: () => !isLocked,
    },
    {
      keys: "escape",
      action: () => {
        setCurrentPolygon([]);
        setDrawing(null);
        selectPolygon(null);
        setSelectedBoxIdx(null);
        selectPoint(null);
        setCutStart(null);
        useStore.getState().setCut(false);
      },
    },
    // Held-key auto-repeat (~30/s) would queue a full image render per tick: one flip per press.
    {
      keys: "arrowleft",
      action: (e) => {
        if (!e.repeat) stepImage(-1);
      },
    },
    {
      keys: "arrowright",
      action: (e) => {
        if (!e.repeat) stepImage(1);
      },
    },
    {
      keys: "enter",
      action: () => commitPolygonAndTrack(),
      when: () => !isLocked && mode === "polygon" && canvas.currentPolygon.length >= 3,
    },
    // Coverage cell navigation, multi-cell grids only: previous/next unswept cell.
    { keys: "[", action: () => stepCoverageCell(-1), when: () => coverageMultiCell },
    { keys: "]", action: () => stepCoverageCell(1), when: () => coverageMultiCell },
    { keys: "0", action: () => selectSubjectByIndex(0) },
    { keys: "1", action: () => selectSubjectByIndex(1) },
    { keys: "2", action: () => selectSubjectByIndex(2) },
    { keys: "3", action: () => selectSubjectByIndex(3) },
    { keys: "4", action: () => selectSubjectByIndex(4) },
    { keys: "5", action: () => selectSubjectByIndex(5) },
    { keys: "6", action: () => selectSubjectByIndex(6) },
    { keys: "7", action: () => selectSubjectByIndex(7) },
    { keys: "8", action: () => selectSubjectByIndex(8) },
    { keys: "9", action: () => selectSubjectByIndex(9) },
  ]);

  // ── Snap helper (image-space) ───────────────────────────────────────

  function snapImagePoint(
    ix: number,
    iy: number,
    excludeVertex?: [number, number, number],
  ): [number, number] {
    if (!annotateUi.snap) return [ix, iy];
    const sc = view.scale || 1;
    const thr = SNAP_RADIUS_CANVAS / sc; // image-space radius
    let best: [number, number] | null = null;
    let bestD = thr;
    canvas.polygons.forEach((poly, pi) => {
      poly.rings.forEach((ring, ri) => {
        ring.forEach(([px, py], vi) => {
          if (
            excludeVertex &&
            excludeVertex[0] === pi &&
            excludeVertex[1] === ri &&
            excludeVertex[2] === vi
          )
            return;
          const d = Math.hypot(px - ix, py - iy);
          if (d < bestD) {
            bestD = d;
            best = [px, py];
          }
        });
      });
    });
    return best ?? [ix, iy];
  }

  // ── Mouse handlers ──────────────────────────────────────────────────

  // Set when a press starts a vertex drag / edge insert, so the trailing click of the
  // drag release can't place a vertex or deselect (one gesture, one meaning).
  const didDragRef = useRef(false);
  // True between the start/stop clicks of a freehand (Stream mode) polygon.
  const streamingRef = useRef(false);

  // Presses/clicks outside the image extent are inert for every tool: they author nothing
  // and change no selection. In-progress gestures still clamp to the edge on move/release.
  const outsideImage = (ix: number, iy: number) =>
    canvas.imgWidth > 0 &&
    canvas.imgHeight > 0 &&
    (ix < 0 || iy < 0 || ix > canvas.imgWidth || iy > canvas.imgHeight);

  const onDown = (ix: number, iy: number, ev: Konva.KonvaEventObject<MouseEvent>) => {
    if (isLocked) return;
    if (ev.evt.button !== 0) return; // right-button drags must not fabricate boxes
    // A fresh press starts a new gesture: clear the drag flag first. A completed vertex drag
    // fires no trailing click, so without this the stale flag would swallow the next click
    // (e.g. an outside click meant to deselect), forcing a second click.
    didDragRef.current = false;
    if (outsideImage(ix, iy)) return;
    if (mode === "map") return; // navigation only; the click (onClick) does the jump
    if (mode === "point") {
      // A press on an existing point selects it and picks it up; the whole mark is the handle.
      // Missing every point does nothing here: the click (see onClick) places a new one, so a
      // single click both authors and a press-drag repositions without a mode or modifier.
      const hit = findHitPoint([ix, iy], canvas.points, POINT_HIT_CANVAS / (view.scale || 1));
      if (hit !== null) {
        selectPoint(hit);
        pushUndo(); // one snapshot per drag; dragPoint itself pushes none
        pointDragRef.current = hit;
        didDragRef.current = true;
      }
      return;
    }
    if (mode === "box") {
      const sc = view.scale || 1;
      // Grab a handle / body of the already-selected box to resize or move it.
      if (selectedBoxIdx !== null && canvas.boxes[selectedBoxIdx]) {
        const b = canvas.boxes[selectedBoxIdx];
        const drag = hitTestEdit({ kind: "box", box: [b.x1, b.y1, b.x2, b.y2] }, ix, iy, 8 / sc);
        if (drag) {
          pushUndo(); // one snapshot per drag; the moves themselves don't push
          boxDragRef.current = { idx: selectedBoxIdx, drag };
          didDragRef.current = true;
          return;
        }
      }
      // Otherwise a press inside an existing (active-subject) box selects it; empty space
      // deselects and starts a new box.
      for (let i = canvas.boxes.length - 1; i >= 0; i--) {
        const b = canvas.boxes[i];
        if (b.subject === activeSubject && ix >= b.x1 && ix <= b.x2 && iy >= b.y1 && iy <= b.y2) {
          setSelectedBoxIdx(i);
          return;
        }
      }
      setSelectedBoxIdx(null);
      if (!requireSubject()) return;
      const cx = Math.max(0, Math.min(canvas.imgWidth || ix, ix));
      const cy = Math.max(0, Math.min(canvas.imgHeight || iy, iy));
      setDrawing({ x1: cx, y1: cy, x2: cx, y2: cy, subject: activeSubject!, attributes: {} });
      return;
    }
    // Polygon: button press starts either a vertex drag (if clicked within
    // handle radius of a vertex on the selected polygon), an edge insert,
    // or a new vertex add.
    if (canvas.currentPolygon.length === 0 && canvas.selectedPolygonIdx !== null) {
      if (annotateUi.cut) return; // a cut click is never a vertex grab or an edge insert
      const pi = canvas.selectedPolygonIdx;
      const poly = canvas.polygons[pi];
      if (!poly) return;
      const sc = view.scale || 1;
      const vertThr = 8 / sc;
      // Try vertex grab, on any ring of the selected annotation.
      for (let ri = 0; ri < poly.rings.length; ri++) {
        const ring = poly.rings[ri];
        for (let vi = 0; vi < ring.length; vi++) {
          const [px, py] = ring[vi];
          if (Math.hypot(px - ix, py - iy) < vertThr) {
            // Capture undo once at drag start; the drag itself uses dragVertex (no
            // per-mousemove undo push, which would otherwise flood the 30-entry stack).
            pushUndo();
            useStore.getState().setDraggingVertex([pi, ri, vi]);
            didDragRef.current = true;
            return;
          }
        }
      }
      // Try edge insert, the nearest edge across every ring; the new vertex joins that ring.
      const edgeThr = EDGE_INSERT_THRESHOLD / sc;
      let bestRing = -1;
      let bestEdge = -1;
      let bestDist = edgeThr;
      let bestProj: [number, number] | null = null;
      for (let ri = 0; ri < poly.rings.length; ri++) {
        const ring = poly.rings[ri];
        for (let ei = 0; ei < ring.length; ei++) {
          const [ax, ay] = ring[ei];
          const [bx, by] = ring[(ei + 1) % ring.length];
          const { dist, proj } = pointToSegmentDist(ix, iy, ax, ay, bx, by);
          if (dist < bestDist) {
            bestDist = dist;
            bestRing = ri;
            bestEdge = ei;
            bestProj = proj;
          }
        }
      }
      if (bestRing >= 0 && bestEdge >= 0 && bestProj) {
        const newPts = poly.rings[bestRing].slice();
        newPts.splice(bestEdge + 1, 0, bestProj);
        updatePolygon(pi, withRing(poly, bestRing, newPts));
        useStore.getState().setDraggingVertex([pi, bestRing, bestEdge + 1]);
        didDragRef.current = true;
        return;
      }
      // A miss keeps the selection; the click event deselects (one click = one action,
      // never deselect-and-start-a-polygon from the same press).
    }
  };

  // One bbox per polygon (recomputed only when the polygon list changes) lets the hover
  // scan reject most polygons with four comparisons before the O(vertices) ray-cast.
  const polygonBboxes = useMemo(() => computePolygonBboxes(canvas.polygons), [canvas.polygons]);

  // rAF-throttle mouse moves: coalesce a burst of pointer events into one update per frame.
  // The ref always holds the freshest closure, so a re-render between scheduling and the
  // frame firing means the callback runs on current, never stale, state.
  const pendingMoveRef = useRef<[number, number] | null>(null);
  const moveRafRef = useRef<number | null>(null);
  const processMoveRef = useRef<(ix: number, iy: number) => void>(() => {});
  processMoveRef.current = (ix: number, iy: number) => {
    setCursor([ix, iy]);
    if (isLocked || mode === "map") return; // Map mode authors nothing on hover either

    // Point drag (repositioning a placed point)
    const pDrag = pointDragRef.current;
    if (pDrag !== null) {
      dragPoint(
        pDrag,
        Math.max(0, Math.min(canvas.imgWidth || ix, ix)),
        Math.max(0, Math.min(canvas.imgHeight || iy, iy)),
      );
      return;
    }

    // Vertex drag
    const dragging = annotateUi.draggingVertex;
    if (dragging) {
      const [pi, ri, vi] = dragging;
      const poly = canvas.polygons[pi];
      if (poly) {
        const [sx, sy] = snapImagePoint(ix, iy, [pi, ri, vi]);
        const clamped: [number, number] = [
          Math.max(0, Math.min(canvas.imgWidth || sx, sx)),
          Math.max(0, Math.min(canvas.imgHeight || sy, sy)),
        ];
        dragVertex(pi, ri, vi, clamped); // no per-move undo push (see onDown drag start)
      }
      return;
    }

    // Streaming (freehand): between the two clicks, drop a vertex each time the pointer has
    // moved far enough, no button held.
    if (streamingRef.current && annotateUi.stream && mode === "polygon") {
      const pts = canvas.currentPolygon;
      const last = pts[pts.length - 1];
      const minD = STREAM_MIN_DIST_CANVAS / (view.scale || 1);
      if (!last || Math.hypot(ix - last[0], iy - last[1]) >= minD) {
        const [sx, sy] = snapImagePoint(ix, iy);
        setCurrentPolygon([
          ...pts,
          [
            Math.max(0, Math.min(canvas.imgWidth || sx, sx)),
            Math.max(0, Math.min(canvas.imgHeight || sy, sy)),
          ],
        ]);
      }
      return;
    }

    // Resizing / moving a selected box
    const bDrag = boxDragRef.current;
    if (bDrag && mode === "box") {
      const b = canvas.boxes[bDrag.idx];
      if (b) {
        const r = applyEditDrag(
          { kind: "box", box: [b.x1, b.y1, b.x2, b.y2] },
          bDrag.drag,
          ix,
          iy,
          canvas.imgWidth || ix,
          canvas.imgHeight || iy,
        );
        boxDragRef.current = { idx: bDrag.idx, drag: r.drag };
        if (r.shape.kind === "box") {
          const [x1, y1, x2, y2] = r.shape.box;
          dragBox(bDrag.idx, { ...b, x1, y1, x2, y2 }); // undo captured on down; spread keeps subject/attrs
        }
      }
      return;
    }

    // Box drag (rubber-band stops at the image edge; polygons already clamp)
    if (drawing) {
      const cx = Math.max(0, Math.min(canvas.imgWidth || ix, ix));
      const cy = Math.max(0, Math.min(canvas.imgHeight || iy, iy));
      setDrawing({ ...drawing, x2: cx, y2: cy });
      return;
    }

    // Polygon hover detection (bbox-prefiltered)
    if (mode === "polygon" && canvas.currentPolygon.length === 0) {
      const hover = findHoveredPolygon([ix, iy], canvas.polygons, polygonBboxes);
      if (hover !== annotateUi.hoveredPolygonIdx) setHoveredPolygon(hover);
    }

    // Box and derived-box hover join the same pass, with the box itself as the prefilter;
    // state writes only on transitions, so a plain move re-renders nothing.
    if (mode === "box") {
      let hoverBox: number | null = null;
      for (let i = canvas.boxes.length - 1; i >= 0; i--) {
        const b = canvas.boxes[i];
        if (b.subject === activeSubject && ix >= b.x1 && ix <= b.x2 && iy >= b.y1 && iy <= b.y2) {
          hoverBox = i;
          break;
        }
      }
      let hoverDerived: number | null = null;
      if (hoverBox === null) {
        for (let i = canvas.polygons.length - 1; i >= 0; i--) {
          if (canvas.polygons[i].subject !== activeSubject) continue;
          const bb = polygonBboxes[i];
          if (bb && ix >= bb[0] && iy >= bb[1] && ix <= bb[2] && iy <= bb[3]) {
            hoverDerived = i;
            break;
          }
        }
      }
      if (hoverBox !== hoveredBoxIdx) setHoveredBoxIdx(hoverBox);
      if (hoverDerived !== hoveredDerivedIdx) setHoveredDerivedIdx(hoverDerived);
    }

    // Point hover: the proximity check is the whole hit test, same radius as the grab target.
    if (mode === "point") {
      const hover = findHitPoint([ix, iy], canvas.points, POINT_HIT_CANVAS / (view.scale || 1));
      if (hover !== hoveredPointIdx) setHoveredPointIdx(hover);
    }
  };

  const onMove = (ix: number, iy: number) => {
    pendingMoveRef.current = [ix, iy];
    if (moveRafRef.current != null) return;
    moveRafRef.current = requestAnimationFrame(() => {
      moveRafRef.current = null;
      const p = pendingMoveRef.current;
      if (p) processMoveRef.current(p[0], p[1]);
    });
  };

  useEffect(() => {
    return () => {
      if (moveRafRef.current != null) cancelAnimationFrame(moveRafRef.current);
    };
  }, []);

  const onUp = (ix: number, iy: number) => {
    if (isLocked || mode === "map") return;
    if (pointDragRef.current !== null) {
      pointDragRef.current = null;
      // didDragRef stays set: the trailing click of this release must not place a second point
      // on top of the one just moved (onClick consumes and clears the flag).
      useStore.getState().recomputeDirty(); // the drag flagged dirty per tick without comparing
      canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), true);
      return;
    }
    if (boxDragRef.current) {
      const draggedIdx = boxDragRef.current.idx;
      boxDragRef.current = null;
      didDragRef.current = false;
      const resized = useStore.getState().canvas.boxes[draggedIdx];
      if (
        resized &&
        (resized.x2 - resized.x1 < MIN_BOX_SIDE || resized.y2 - resized.y1 < MIN_BOX_SIDE)
      ) {
        undo();
        useStore.getState().pushToast("Box too small to keep; the resize was undone.");
        return;
      }
      useStore.getState().recomputeDirty();
      // The drag suppressed full pushes; the settled geometry ships now.
      canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), true);
      return;
    }
    if (annotateUi.draggingVertex) {
      useStore.getState().setDraggingVertex(null);
      useStore.getState().recomputeDirty();
      return;
    }
    if (mode === "box" && drawing) {
      const cx = Math.max(0, Math.min(canvas.imgWidth || ix, ix));
      const cy = Math.max(0, Math.min(canvas.imgHeight || iy, iy));
      const box: Box = {
        x1: Math.min(drawing.x1, cx),
        y1: Math.min(drawing.y1, cy),
        x2: Math.max(drawing.x1, cx),
        y2: Math.max(drawing.y1, cy),
        subject: drawing.subject,
        attributes: {},
      };
      if (box.x2 - box.x1 < MIN_BOX_SIDE || box.y2 - box.y1 < MIN_BOX_SIDE) {
        useStore.getState().pushToast("Box too small to keep. Drag out a bigger area.");
      } else {
        addBox(box);
        incrementAnnotationsAdded(1);
      }
      setDrawing(null);
    }
  };

  const onClick = (ix: number, iy: number, ev: Konva.KonvaEventObject<MouseEvent>) => {
    if (ev.evt.button !== 0) return;
    if (outsideImage(ix, iy)) return;
    if (mode === "map") {
      // Navigation only: a click opens the cell's tile, no annotation handler runs, and this
      // is offered even while the image is locked (viewing coverage is not an edit).
      const cell = cellAt(coverageGrid.cells, ix, iy);
      if (cell) {
        jumpToCell(cell);
        setMapSelectedCell(cell);
      }
      return;
    }
    if (isLocked) return;
    if (mode === "point") {
      if (didDragRef.current) {
        didDragRef.current = false; // the trailing click of a point select/drag release
        return;
      }
      // One click = one action: an existing selection is dropped first, so a click never both
      // deselects and authors a point (the same rule polygon mode follows).
      if (canvas.selectedPointIdx !== null) {
        selectPoint(null);
        return;
      }
      if (!requireSubject()) return;
      // One click commits it: a point has nothing to drag out and no second vertex to wait for.
      addPoint({
        x: Math.max(0, Math.min(canvas.imgWidth || ix, ix)),
        y: Math.max(0, Math.min(canvas.imgHeight || iy, iy)),
        subject: activeSubject!,
        attributes: {},
      });
      incrementAnnotationsAdded(1);
      return;
    }
    if (mode !== "polygon") return;
    if (annotateUi.draggingVertex) return;
    if (didDragRef.current) {
      didDragRef.current = false; // the trailing click of a vertex-drag release
      return;
    }

    // Cut: an endpoint is outside the selected ring by definition, so without this branch the
    // click would miss the polygon hit test below and deselect the very shape being cut.
    if (annotateUi.cut) {
      if (!cutStart) {
        // No start pending: a click inside any polygon (re)selects it (an endpoint must fall
        // outside the ring); only a click outside every polygon places the start.
        let hitIdx: number | null = null;
        for (let pi = 0; pi < canvas.polygons.length; pi++) {
          if (pointInRings([ix, iy], canvas.polygons[pi].rings)) {
            hitIdx = pi;
            break;
          }
        }
        if (hitIdx !== null) {
          selectPolygon(hitIdx);
          return;
        }
        if (canvas.selectedPolygonIdx === null) {
          requireCutSelection();
          return;
        }
        const polygonIdx = canvas.selectedPolygonIdx;
        setCutStart({ point: [ix, iy], polygonIdx, polygon: canvas.polygons[polygonIdx] });
        return;
      }
      // Compared by index and rings reference, not object identity: an attribute edit keeps the
      // same rings array and must not cancel the cut; a geometry edit replaces it and must.
      const idx = canvas.selectedPolygonIdx;
      const current = idx !== null ? canvas.polygons[idx] : null;
      if (
        idx === null ||
        idx !== cutStart.polygonIdx ||
        !current ||
        current.rings !== cutStart.polygon.rings
      ) {
        setCutStart(null);
        useStore
          .getState()
          .pushToast(
            "The polygon changed since the first click; the cut was cancelled. Select it and " +
              "place both points again.",
            "error",
            "cut",
          );
        return;
      }
      if (current.rings.length > 1) {
        setCutStart(null);
        useStore.getState().pushToast(multiRingCutRefusal(current.rings.length), "error", "cut");
        return;
      }
      const result = cutRing(current.rings[0], cutStart.point, [ix, iy]);
      if ("reason" in result) {
        setCutStart(null);
        useStore.getState().pushToast(result.reason, "error", "cut");
        return;
      }
      splitPolygon(idx, result.rings);
      incrementAnnotationsAdded(1);
      setCutStart(null);
      return;
    }

    // Stream (freehand): click starts laying vertices, click again pauses (the polygon stays
    // open, resume with another click), and double-click closes it, exactly like non-stream
    // drawing. The button is never held; right-click (onContextMenu) cancels outright.
    if (annotateUi.stream) {
      if (streamingRef.current) {
        streamingRef.current = false; // pause: closing is double-click's job, same as always
        return;
      }
      // Selection parity with Stream off: when no polygon is in progress, a click on an
      // existing polygon selects it, and empty space deselects before anything streams.
      if (canvas.currentPolygon.length === 0) {
        for (let pi = 0; pi < canvas.polygons.length; pi++) {
          if (pointInRings([ix, iy], canvas.polygons[pi].rings)) {
            selectPolygon(pi);
            return;
          }
        }
        if (canvas.selectedPolygonIdx !== null) {
          selectPolygon(null); // one click = one action: deselect first, stream on the next click
          return;
        }
      }
      if (canvas.currentPolygon.length === 0 && !requireSubject()) return;
      const [sx, sy] = snapImagePoint(ix, iy);
      if (canvas.currentPolygon.length === 0) {
        pushUndo();
        setCurrentPolygon([[sx, sy]]);
      } else {
        setCurrentPolygon([...canvas.currentPolygon, [sx, sy]]); // resume the open polygon
      }
      streamingRef.current = true;
      return;
    }

    // Placing vertices into a new polygon
    if (canvas.currentPolygon.length > 0) {
      const [sx, sy] = snapImagePoint(ix, iy);
      setCurrentPolygon([...canvas.currentPolygon, [sx, sy]]);
      return;
    }

    // Not currently drawing: clicking on any part of a polygon selects the whole annotation
    for (let pi = 0; pi < canvas.polygons.length; pi++) {
      if (pointInRings([ix, iy], canvas.polygons[pi].rings)) {
        selectPolygon(pi);
        return;
      }
    }
    // Clicked empty space with nothing selected: start a new polygon
    if (canvas.selectedPolygonIdx === null) {
      if (!requireSubject()) return;
      const [sx, sy] = snapImagePoint(ix, iy);
      setCurrentPolygon([[sx, sy]]);
    } else {
      // Already had a selection and click didn't land on a polygon → deselect
      selectPolygon(null);
    }
  };

  const onDoubleClick = (ix: number, iy: number) => {
    if (isLocked) return;
    if (outsideImage(ix, iy)) return;
    if (mode !== "polygon") return;
    streamingRef.current = false; // a double-click ends laying even when too short to close
    if (canvas.currentPolygon.length >= 3) {
      commitPolygonAndTrack();
    }
  };

  const onContextMenu = (ix: number, iy: number, ev: Konva.KonvaEventObject<MouseEvent>) => {
    ev.evt.preventDefault();
    if (isLocked) return;
    if (outsideImage(ix, iy)) return;
    // Point mode: right-click deletes the point under the cursor (a box's right-click delete,
    // scoped to one coordinate). Nothing under the cursor just clears the selection.
    if (mode === "point") {
      const hit = findHitPoint([ix, iy], canvas.points, POINT_HIT_CANVAS / (view.scale || 1));
      if (hit !== null) deletePoint(hit);
      else selectPoint(null);
      return;
    }
    // A pending cut has no open polygon, so without this branch a right-click inside the
    // selected polygon reaches its delete below and the cancel gesture deletes the parent.
    if (cutStart) {
      setCutStart(null);
      return;
    }
    // Right-click cancels an in-progress / streaming polygon first
    if (mode === "polygon" && (canvas.currentPolygon.length > 0 || streamingRef.current)) {
      streamingRef.current = false;
      setCurrentPolygon([]);
      return;
    }
    // Polygon mode + selected polygon: try vertex delete, then polygon delete
    if (mode === "polygon" && canvas.selectedPolygonIdx !== null) {
      const pi = canvas.selectedPolygonIdx;
      const poly = canvas.polygons[pi];
      if (poly) {
        const sc = view.scale || 1;
        const vertThr = 10 / sc;
        for (let ri = 0; ri < poly.rings.length; ri++) {
          const ring = poly.rings[ri];
          for (let vi = 0; vi < ring.length; vi++) {
            const [px, py] = ring[vi];
            if (Math.hypot(px - ix, py - iy) < vertThr) {
              pushUndo();
              if (ring.length > 3) {
                const newPts = ring.slice();
                newPts.splice(vi, 1);
                updatePolygon(pi, withRing(poly, ri, newPts));
              } else if (poly.rings.length > 1) {
                // Below a triangle the ring is no longer a contour: drop that part, keep the rest
                // of the annotation (only the last remaining part takes the whole shape with it).
                updatePolygon(pi, { ...poly, rings: poly.rings.filter((_, i) => i !== ri) });
              } else {
                deletePolygon(pi);
              }
              return;
            }
          }
        }
        if (pointInRings([ix, iy], poly.rings)) {
          deletePolygon(pi);
          return;
        }
      }
      selectPolygon(null);
      return;
    }
    // Box right-click delete, box mode only.
    if (mode === "box") {
      for (let i = 0; i < canvas.boxes.length; i++) {
        const b = canvas.boxes[i];
        if (ix >= b.x1 && ix <= b.x2 && iy >= b.y1 && iy <= b.y2) {
          deleteBox(i);
          return;
        }
      }
    }
    // Non-selected polygon right-click delete (polygon mode)
    if (mode === "polygon") {
      for (let pi = 0; pi < canvas.polygons.length; pi++) {
        if (pointInRings([ix, iy], canvas.polygons[pi].rings)) {
          deletePolygon(pi);
          return;
        }
      }
    }
  };

  // ── Symbology (scale-dependent) ─────────────────────────────────────

  const s = view.scale || 1;
  const scaleLineW = 1 / s;
  const boxStroke = Math.max(1, Math.min(2 + s * 0.5, 6)) * scaleLineW;
  const polyStroke = Math.max(1, Math.min(2.5 + s * 0.5, 7)) * scaleLineW;
  // Both radii resolve in screen px, then compensate by 1/s exactly once: mixing the
  // spaces here once made selected handles grow 1/s² and blanket the frame at fit zoom.
  const vertScreen = Math.max(3, Math.min(VERTEX_HANDLE_RADIUS * (1.6 - s * 0.2), 12));
  const selScreen = Math.max(vertScreen + 1, Math.min(VERTEX_HANDLE_RADIUS * (2.2 - s * 0.2), 16));
  const vertR = vertScreen * scaleLineW;
  const selVertR = selScreen * scaleLineW;
  const labelSize = Math.max(8, Math.min(Math.round(9 * (0.6 + s * 0.4)), 18)) * scaleLineW;
  // A point's mark is fixed in screen px (resolved once, then 1/s like every other radius here): the
  // annotation asserts a location and no extent, so its glyph must not grow with the image and imply
  // one. The ticks reach past the core to POINT_HIT_CANVAS: the mark is the grab target.
  const pointCoreR = 3.5 * scaleLineW;
  const pointSelCoreR = 5 * scaleLineW;
  const pointTickInner = 6.5 * scaleLineW;
  const pointTickOuter = POINT_HIT_CANVAS * scaleLineW;

  if (!imgPath || !currentImageName) {
    return (
      <div className="flex-1 flex flex-col min-h-0">
        <AnnotateToolbar
          onSave={() => void save()}
          saveDisabled={saveDisabled}
          dirty={canvas.dirty}
          isLocked={isLocked}
        />
        <div className="flex-1 flex items-center justify-center bg-tcip-canvas px-4">
          <div className="max-w-lg rounded-lg border border-tcip-border bg-tcip-panel px-5 py-4 text-center">
            <p className="text-sm font-semibold text-tcip-fg">No image loaded</p>
            <p className="mt-1 text-xs text-tcip-muted">
              Pick a dataset/date with images, then use Prev/Next and the image counter to navigate.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const imageUrl = imgPath ? api.images.url(imgPath, composite) : null;

  const renderLabels = annotateUi.visible;
  const hoveredIdx = annotateUi.hoveredPolygonIdx;
  const draggingIdx = annotateUi.draggingVertex?.[0];
  const coverageViewport = coverageViewportRect();
  const activeCoverageCell = currentCoverageCell(
    coverageGrid.cells,
    coverageViewport,
    mapSelectedCell,
  );
  // Every raster the grid route serves a lattice for, single-cell rasters included, so a
  // one-cell image's own attestation stays reachable; only the Map tool stays multi-cell only.
  const showCoverageChrome =
    !!coverageGrid.grid ||
    !!coverageGrid.error ||
    !!completeness.error ||
    (coverageGrid.settled && !!coverageGrid.reason);

  function setGridZoom(zoom: number) {
    if (!activeSubject || !dataset.dataset_root) return;
    void api.coverage
      .setGridZoom({
        subject: activeSubject,
        zoom,
        dataset_root: dataset.dataset_root,
        user: useStore.getState().user,
      })
      .then(
        () => coverageGrid.refetch(),
        (err: unknown) => {
          const detail = err instanceof Error ? err.message : String(err);
          useStore.getState().pushToast(`Could not set the grid zoom: ${detail}`);
        },
      );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <TabHeading tab="annotate" />
      <AnnotateToolbar
        onSave={() => void save()}
        saveDisabled={saveDisabled}
        dirty={canvas.dirty}
        isLocked={isLocked}
        bandsInfo={bandsInfo}
        bandSelection={bandSelection}
        onBandSelectionChange={setBandSelection}
        completeWarning={coverage.completeWarning}
        workingScaleReason={completeness.workingScaleReason}
        workingScaleSubject={activeSubject}
        coverageMultiCell={coverageMultiCell}
        replaceRequired={coverage.replaceRequired}
      />
      <div className="relative flex-1 flex flex-col min-h-0">
        <CanvasStage
          imageUrl={imageUrl}
          imagePath={imgPath}
          imgWidth={canvas.imgWidth}
          imgHeight={canvas.imgHeight}
          regions={regions}
          onBaseFacts={setBaseFacts}
          onStageRef={(st) => (stageRef.current = st)}
          onPixelDown={onDown}
          onPixelMove={onMove}
          onPixelUp={onUp}
          onPixelClick={onClick}
          onPixelDoubleClick={onDoubleClick}
          onPixelContextMenu={onContextMenu}
          overlay={
            <>
              {/* In-progress polygon (rubber-bands to the cursor) */}
              {mode === "polygon" && canvas.currentPolygon.length > 0 && (
                <InProgressPolygon
                  points={canvas.currentPolygon}
                  cursor={cursor}
                  stroke={activeSubject ? subjectColor(activeSubject) : "#FFE7B1"}
                  strokeW={polyStroke}
                  vertR={vertR}
                />
              )}

              {/* Pending cut: its start plus a dashed segment to the cursor */}
              {mode === "polygon" && cutStart && (
                <InProgressPolygon
                  points={[cutStart.point]}
                  cursor={cursor}
                  stroke={subjectColor(cutStart.polygon.subject)}
                  strokeW={polyStroke}
                  vertR={vertR}
                />
              )}

              {/* Box draft */}
              {drawing && (
                <Rect
                  x={Math.min(drawing.x1, drawing.x2)}
                  y={Math.min(drawing.y1, drawing.y2)}
                  width={Math.abs(drawing.x2 - drawing.x1)}
                  height={Math.abs(drawing.y2 - drawing.y1)}
                  stroke={subjectColor(drawing.subject)}
                  strokeWidth={boxStroke}
                  dash={[6 * scaleLineW, 4 * scaleLineW]}
                />
              )}

              {/* Snap indicator */}
              {annotateUi.snap && cursor && mode === "polygon" && (
                <SnapIndicator
                  cursor={cursor}
                  polygons={canvas.polygons}
                  scale={s}
                  radius={SNAP_RADIUS_CANVAS / s}
                />
              )}
            </>
          }
        >
          {coverageMultiCell && coverageOverlayOn && coverageGrid.grid && (
            <CoverageOverlay
              cells={coverageGrid.cells}
              viewport={coverageViewport}
              scale={s}
              swept={coverage.swept}
              pending={coverage.pending}
              activeComplete={completeness.activeComplete}
              activeStale={completeness.activeStale}
              otherComplete={completeness.otherComplete}
              annotationCounts={completeness.annotationCounts}
            />
          )}
          {/* Committed shapes: memoized, cursor-independent (see AnnotationShapes) */}
          <AnnotationShapes
            boxes={canvas.boxes}
            polygons={canvas.polygons}
            points={canvas.points}
            mode={mode}
            activeSubject={activeSubject}
            selectedPolygonIdx={canvas.selectedPolygonIdx}
            selectedBoxIdx={selectedBoxIdx}
            selectedPointIdx={canvas.selectedPointIdx}
            hoveredIdx={hoveredIdx}
            hoveredBoxIdx={hoveredBoxIdx}
            hoveredDerivedIdx={hoveredDerivedIdx}
            hoveredPointIdx={hoveredPointIdx}
            draggingIdx={draggingIdx}
            renderLabels={renderLabels}
            boxStroke={boxStroke}
            polyStroke={polyStroke}
            vertR={vertR}
            selVertR={selVertR}
            labelSize={labelSize}
            pointCoreR={pointCoreR}
            pointSelCoreR={pointSelCoreR}
            pointTickInner={pointTickInner}
            pointTickOuter={pointTickOuter}
            scaleLineW={scaleLineW}
          />
        </CanvasStage>

        {ioError && (
          <div className="absolute top-12 left-3 right-3 z-30 flex items-center gap-2 rounded-md border border-tcip-fp/50 bg-tcip-panel/95 px-3 py-1.5 text-[11px] text-tcip-fp">
            <span className="flex-1">{ioError}</span>
            {conflict && (
              <button className="tcip-btn text-[11px]" onClick={() => void reloadCurrent()}>
                Reload
              </button>
            )}
            <button
              type="button"
              className="text-tcip-fp/70 hover:text-tcip-fp"
              aria-label="Dismiss"
              title="Dismiss"
              onClick={() => {
                setIoError(null);
                setConflict(false);
              }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Floating canvas chrome, DOM order following the reading order (top-left, top-right,
            bottom-left, bottom-right) so tab order matches what a sighted user reads first. */}
        <button
          type="button"
          onClick={overview}
          title="Overview: fit the whole image to the canvas"
          className="absolute top-3 left-3 z-20 rounded-full border border-tcip-border bg-tcip-panel/90 px-2.5 py-1 text-[11px] text-tcip-muted backdrop-blur hover:border-tcip-border-hover hover:text-tcip-fg"
        >
          Overview
        </button>

        <AttributePanel selectedBoxIdx={mode === "box" ? selectedBoxIdx : null} locked={isLocked} />

        <AnnotateLegend />

        {showCoverageChrome && (
          <CoverageChrome
            subject={activeSubject}
            derivation={coverageGrid.derivation ?? ""}
            reason={coverageGrid.reason}
            settled={coverageGrid.settled}
            freshDerivationDiffers={coverageGrid.freshDerivationDiffers}
            onRederiveLattice={coverageGrid.rederiveLattice}
            onSetGridZoom={setGridZoom}
            gridFetchError={coverageGrid.error}
            readError={completeness.error}
            countsError={completeness.countsError}
            canOverlay={coverageMultiCell && !!coverageGrid.grid}
            overlayOn={coverageOverlayOn}
            onToggleOverlay={toggleCoverageOverlay}
            currentCellName={activeCoverageCell?.name ?? null}
            currentCellComplete={
              !!activeCoverageCell &&
              (completeness.activeComplete.has(activeCoverageCell.name) ||
                completeness.activeStale.has(activeCoverageCell.name))
            }
            currentCellStale={
              !!activeCoverageCell && completeness.activeStale.has(activeCoverageCell.name)
            }
            otherLattice={completeness.otherLattice}
            replaceRequired={coverage.replaceRequired}
            onArmReplace={coverage.armReplace}
            swept={coverage.swept}
            pending={coverage.pending}
            coarserCount={coverage.coarserCount}
            workingScale={completeness.workingScale}
            workingScaleReason={completeness.workingScaleReason}
            fitScale={imageFitScale()}
            activeComplete={completeness.activeComplete}
            activeStale={completeness.activeStale}
            activeCellsAttestedView={completeness.activeCellsAttestedView}
            otherComplete={completeness.otherCompleteBySubject}
            annotationCounts={completeness.annotationCounts}
            onAttest={(complete) => {
              if (activeCoverageCell && coverageGrid.grid) {
                completeness.write(
                  activeCoverageCell.name,
                  coverageGrid.grid,
                  complete,
                  view.scale,
                );
              }
            }}
          />
        )}
      </div>
    </div>
  );
}
