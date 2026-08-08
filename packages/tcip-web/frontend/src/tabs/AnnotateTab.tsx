import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Circle, Line, Rect, Text } from "react-konva";
import Konva from "konva";

import { api } from "@/api/client";
import { classesApi, subjectColor, type AttributeDef } from "@/api/classes";
import { sessionsApi } from "@/api/sessions";
import { AnnotateToolbar } from "@/components/AnnotateToolbar";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
import { CoverageMinimap } from "@/components/Canvas/CoverageMinimap";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import { useCoverageGrid } from "@/hooks/useCoverageGrid";
import { useCoverageTracking } from "@/hooks/useCoverageTracking";
import { useImageBands } from "@/hooks/useImageBands";
import { useImageNav } from "@/hooks/useImageNav";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { usePrefetchAdjacentImages } from "@/hooks/usePrefetchAdjacentImages";
import { useRegionServes } from "@/hooks/useRegionServes";
import {
  compositeParams,
  defaultBandSelection,
  isPlainColourFrame,
  type BandSelection,
} from "@/lib/bandSelection";
import { stepUnsweptCell, type GridCell } from "@/lib/coverage";
import type { LoadedImage } from "@/lib/imageLoader";
import { zoomToRect } from "@/lib/viewGeometry";
import {
  buildAnnotateShapes,
  computeViewport,
  createCanvasPusher,
  measureCanvasHost,
  onCanvasStateRequest,
  pointShapeVisible,
  type CanvasStateBody,
} from "@/lib/canvasSync";
import { canvasToAnnotations } from "@/lib/labelSerde";
import {
  computePolygonBboxes,
  findHitPoint,
  findHoveredPolygon,
  pointInRings,
  ringsBbox,
} from "@/lib/polygonGeometry";
import { applyEditDrag, hitTestEdit, type EditDrag } from "@/lib/reviewEditGeometry";
import { useStore } from "@/store";
import type {
  Box,
  DatasetSelection,
  Mode,
  PointShape,
  PolygonShape,
  PredictionReference,
} from "@/store/types";

const SNAP_RADIUS_CANVAS = 15;
const VERTEX_HANDLE_RADIUS = 4;
const EDGE_INSERT_THRESHOLD = 6;
const STREAM_MIN_DIST_CANVAS = 6; // screen px between vertices laid down in Stream (freehand) mode
const MIN_BOX_SIDE = 3;
// Screen-px grab radius for a placed point: the whole mark is its own handle, so this matches the
// mark's outer reach (see the tick geometry in PointOverlay) rather than a hidden smaller target.
const POINT_HIT_CANVAS = 11;

function currentImagePath(dataset: DatasetSelection): string | null {
  if (!dataset.dataset_root || !dataset.date) return null;
  const name = dataset.image_list[dataset.current_image_index];
  if (!name) return null;
  return `${dataset.dataset_root}/images/${dataset.date}/${name}`;
}

/** A polygon's read-only derived box (the axis-aligned bounds of every ring), for box-mode display
 *  only. Reuses ringsBbox (the same min/max the loader and COCO export re-derive), so it can't
 *  drift. */
function derivedBoxFromPolygon(p: PolygonShape): Box {
  const [x1, y1, x2, y2] = ringsBbox(p.rings);
  return { x1, y1, x2, y2, subject: p.subject, attributes: {} };
}

/** One ring replaced, the rest of the annotation untouched (an edit belongs to one contour). */
function withRing(p: PolygonShape, ringIdx: number, ring: [number, number][]): PolygonShape {
  const rings = p.rings.slice();
  rings[ringIdx] = ring;
  return { ...p, rings };
}

/** The mode `m` advances to: Point -> Box -> Polygon -> Point, the toolbar's left-to-right order
 *  (the array below is a rotation of the same cycle, so the transitions are unchanged). */
function nextMode(mode: Mode): Mode {
  const order: Mode[] = ["box", "polygon", "point"];
  return order[(order.indexOf(mode) + 1) % order.length];
}

function pointToSegmentDist(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): { dist: number; t: number; proj: [number, number] } {
  const dx = bx - ax;
  const dy = by - ay;
  const len_sq = dx * dx + dy * dy;
  if (len_sq === 0) {
    const d = Math.hypot(px - ax, py - ay);
    return { dist: d, t: 0, proj: [ax, ay] };
  }
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len_sq));
  const proj: [number, number] = [ax + t * dx, ay + t * dy];
  const d = Math.hypot(px - proj[0], py - proj[1]);
  return { dist: d, t, proj };
}

/**
 * The committed boxes + polygons (content layer). Memoized, and crucially, the mouse
 * cursor is not one of its props, so a mouse move (which only updates cursor-following
 * overlays) does not re-render/reconcile these hundreds–thousands of Konva nodes. It
 * re-renders only when the shapes, selection/hover, active subject, or zoom-derived stroke
 * sizes actually change.
 */
interface AnnotationShapesProps {
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  mode: Mode;
  activeSubject: string | null;
  selectedPolygonIdx: number | null;
  selectedBoxIdx: number | null;
  selectedPointIdx: number | null;
  hoveredIdx: number | null;
  hoveredBoxIdx: number | null;
  hoveredDerivedIdx: number | null;
  hoveredPointIdx: number | null;
  draggingIdx: number | undefined;
  renderLabels: boolean;
  boxStroke: number;
  polyStroke: number;
  vertR: number;
  selVertR: number;
  labelSize: number;
  pointCoreR: number;
  pointSelCoreR: number;
  pointTickInner: number;
  pointTickOuter: number;
  scaleLineW: number;
}

const AnnotationShapes = memo(function AnnotationShapes({
  boxes,
  polygons,
  points,
  mode,
  activeSubject,
  selectedPolygonIdx,
  selectedBoxIdx,
  selectedPointIdx,
  hoveredIdx,
  hoveredBoxIdx,
  hoveredDerivedIdx,
  hoveredPointIdx,
  draggingIdx,
  renderLabels,
  boxStroke,
  polyStroke,
  vertR,
  selVertR,
  labelSize,
  pointCoreR,
  pointSelCoreR,
  pointTickInner,
  pointTickOuter,
  scaleLineW,
}: AnnotationShapesProps) {
  if (!renderLabels) return null;
  return (
    <>
      {/* Boxes (only the active subject in box mode). The legend carries the standing
          symbology; a shape is named on the canvas only while selected or hovered. */}
      {mode === "box" &&
        boxes.map((b, i) =>
          b.subject === activeSubject ? (
            <BoxOverlay
              key={`box-${i}`}
              box={b}
              stroke={i === selectedBoxIdx ? "#00BFFF" : subjectColor(b.subject)}
              width={boxStroke}
              labelSize={labelSize}
              label={b.subject}
              showLabel={i === selectedBoxIdx || i === hoveredBoxIdx}
              selected={i === selectedBoxIdx}
              handleR={selVertR}
            />
          ) : null,
        )}

      {/* Read-only derived boxes: each active-subject polygon's bounding box, shown in box mode so a
          polygon's detection footprint is visible while boxing. Render-only, derived from
          polygonBbox here and never added to canvas.boxes, so it can't be selected/edited/deleted or
          saved (handle-less). Dashed marks it as read-only, distinct from a real editable box
          (solid), the same convention in-progress/under-review shapes already use. */}
      {mode === "box" &&
        polygons.map((p, i) =>
          p.subject === activeSubject ? (
            <BoxOverlay
              key={`derived-${i}`}
              box={derivedBoxFromPolygon(p)}
              stroke={subjectColor(p.subject)}
              width={boxStroke}
              labelSize={labelSize}
              label={p.subject}
              showLabel={i === hoveredDerivedIdx}
              dashed
            />
          ) : null,
        )}

      {/* Polygons */}
      {polygons.map((p, i) => {
        const selected = selectedPolygonIdx === i;
        const hovered = hoveredIdx === i;
        const dragging = draggingIdx === i;
        // Outside polygon mode only the selected polygon shows (the shape being inspected)
        if (mode !== "polygon" && !selected) return null;
        // In polygon mode filter to the active subject unless selected
        if (mode === "polygon" && !selected && p.subject !== activeSubject) return null;
        const showVerts = selected || hovered || dragging;
        return (
          <PolygonOverlay
            key={`poly-${i}`}
            polygon={p}
            stroke={selected ? "#00BFFF" : subjectColor(p.subject)}
            width={polyStroke}
            vertexRadius={selected ? selVertR : vertR}
            showVertices={showVerts}
            labelSize={labelSize}
            label={p.subject}
            showLabel={selected || hovered}
          />
        );
      })}

      {/* Points: the same visibility rule the agent's mirror uses (pointShapeVisible) */}
      {points.map((p, i) => {
        const selected = selectedPointIdx === i;
        if (
          !pointShapeVisible({
            mode,
            subject: p.subject,
            activeSubject: activeSubject ?? "",
            selected,
          })
        )
          return null;
        return (
          <PointOverlay
            key={`point-${i}`}
            point={p}
            stroke={selected ? "#00BFFF" : subjectColor(p.subject)}
            coreR={selected ? pointSelCoreR : pointCoreR}
            tickInner={pointTickInner}
            tickOuter={pointTickOuter}
            lineW={scaleLineW * 1.6}
            labelSize={labelSize}
            label={p.subject}
            showLabel={selected || i === hoveredPointIdx}
          />
        );
      })}
    </>
  );
});

export function AnnotateTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const view = useStore((s) => s.gui.view);
  const setView = useStore((s) => s.setView);
  const mode = useStore((s) => s.gui.mode);
  const activeSubject = useStore((s) => s.gui.active_subject);
  const predRef = useStore((s) => s.gui.pred_reference);
  // The subject registry (subject -> {description?, attributes?}); drives colours (name-derived,
  // GUI-local) and the per-instance attribute editor.
  const registry = useStore((s) => s.registry.subjects);

  const canvas = useStore((s) => s.canvas);
  const loadLabels = useStore((s) => s.loadLabelsIntoCanvas);
  const addBox = useStore((s) => s.addBox);
  const dragBox = useStore((s) => s.dragBox);
  const deleteBox = useStore((s) => s.deleteBox);
  const deletePolygon = useStore((s) => s.deletePolygon);
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
  const setPredReference = useStore((s) => s.setPredReference);
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

  const imgPath = currentImagePath(dataset);
  const currentImageName = dataset.image_list[dataset.current_image_index] ?? null;
  // A confirmed negative is a completed review (empty): lock it like "complete".
  const currentStatus = currentImageName ? imageStatus.byImage[currentImageName] : undefined;
  const isLocked = currentStatus === "complete" || currentStatus === "negative";
  const saveDisabled = !imgPath || isLocked || saveBlocked;

  // Band-composite picker (multispectral only). bandsInfo drives conditional visibility
  // (band_count > 3); the selection is seeded from the reported bands and otherwise left to
  // the breeder, carried across image navigation until the dataset's own band set changes.
  const bandsInfo = useImageBands(imgPath);
  const [bandSelection, setBandSelection] = useState<BandSelection | null>(null);
  useEffect(() => {
    // An ordinary RGBA frame has four bands and no band choice to make: it displays as its own
    // pixels, so no selection is seeded and the picker stays out of the way.
    if (bandsInfo && bandsInfo.band_count > 3 && !isPlainColourFrame(bandsInfo)) {
      setBandSelection((prev) => prev ?? defaultBandSelection(bandsInfo.bands));
    } else {
      setBandSelection(null);
    }
  }, [bandsInfo]);

  // Every request for this view (the canvas' own and the prefetcher's warm-up) carries one set
  // of band params, so the two never warm and read different renders of the same image.
  const composite = compositeParams(bandsInfo, bandSelection);

  // Image navigation (shared with TopBar + Review; honors the status filter).
  const nav = useImageNav();
  usePrefetchAdjacentImages(composite.bands, composite.stretch);

  // ── Coverage: lattice, region serves, session accumulation ────────────────
  // Gated on the base serve's read facts: a base already at native needs none of it.
  const [baseFacts, setBaseFacts] = useState<LoadedImage | null>(null);
  const coverageGrid = useCoverageGrid(imgPath, baseFacts, canvas.imgWidth, canvas.imgHeight);
  const coverageViewing = useMemo(
    () => ({
      bands: composite.bands,
      stretch: composite.stretch,
      stats_source: baseFacts?.statsSource ?? null,
      display_bounds: baseFacts?.displayBounds ?? null,
      base_served_size: baseFacts?.servedSize
        ? `${baseFacts.servedSize.w}x${baseFacts.servedSize.h}`
        : null,
    }),
    [composite.bands, composite.stretch, baseFacts],
  );
  const coverage = useCoverageTracking({
    imagePath: imgPath,
    datasetRoot: dataset.dataset_root,
    subject: dataset.subject,
    date: dataset.date,
    grid: coverageGrid.grid,
    cells: coverageGrid.cells,
    view,
    imgW: canvas.imgWidth,
    imgH: canvas.imgHeight,
    viewing: coverageViewing,
  });
  const regions = useRegionServes({
    imagePath: imgPath,
    imgW: canvas.imgWidth,
    imgH: canvas.imgHeight,
    view,
    cells: coverageGrid.cells,
    tileSize: coverageGrid.grid?.tile_size ?? null,
    baseFacts,
    composite,
    onCellServedAtNative: coverage.noteServedAtNative,
  });
  const coverageMultiCell = coverageGrid.cells.length > 1;

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
  }, [currentImageName]);

  // ── Live canvas push (agent visibility: capture_live_canvas) ──────────────
  // The ref always holds the freshest closure so the debounced pusher never reads stale state.
  const subjectSwatches = useMemo(
    () => Object.keys(registry).map((name) => ({ name, color: subjectColor(name) })),
    [registry],
  );
  const buildCanvasBodyRef = useRef<() => CanvasStateBody | null>(() => null);
  buildCanvasBodyRef.current = () => {
    if (!imgPath || !dataset.project_root) return null;
    // Never push mid-transition: after an image change the canvas briefly still holds the
    // previous image's shapes: attaching them to the new image_path would show the agent a
    // false canvas. Wait until the loaded-labels identity matches (the post-load push covers it).
    if (loadedPathsRef.current?.image !== imgPath) return null;
    const host = measureCanvasHost();
    return {
      schema_version: 1,
      project_root: dataset.project_root,
      tab: "annotate",
      image_path: imgPath,
      image: currentImageName ?? "",
      img_width: canvas.imgWidth,
      img_height: canvas.imgHeight,
      viewport: host ? computeViewport(view, host, canvas.imgWidth, canvas.imgHeight) : null,
      mode,
      active_subject: activeSubject ?? undefined,
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
    annotateUi.draggingVertex,
    drawing,
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

    // Update per-image status unless it is pinned Complete. Runs before the staleness guard
    // below on the captured paths + canvas snapshot, so it stays correct for the image that
    // was actually saved even after navigating away.
    const name = paths.image.split(/[/\\]/).pop() ?? "";
    if (projectRoot && name) {
      const current = useStore.getState().imageStatus.byImage[name];
      if (current !== "complete") {
        const hasContent =
          c.boxes.length + c.polygons.length + c.points.length + c.imageAnnotations.length > 0;
        // content -> partial; an empty save keeps a prior confirmed negative, else unannotated
        // (a negative needs an explicit Complete, not just an empty file).
        const newStatus = hasContent
          ? "partial"
          : current === "negative"
            ? "negative"
            : "unannotated";
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
    const stem = currentImageName.replace(/\.[^.]+$/, "");
    const label = dataset.annotations_dir ? `${dataset.annotations_dir}/${stem}.json` : null;
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
        startImageSessionTracking(
          currentImageName,
          labels.boxes.length +
            labels.polygons.length +
            labels.points.length +
            labels.imageAnnotations.length,
        );
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
        startImageSessionTracking(currentImageName, 0);
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
        loaded_annotation_count: tracking.loadedAnnotationCount,
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
      coverage.noteAuthoringCommit();
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
    setPredReference(null);
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
      action: () => {
        if (mode === "polygon" && canvas.currentPolygon.length >= 3) commitPolygonAndTrack();
      },
      when: () => !isLocked,
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
    if (isLocked) return;

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
    if (isLocked) return;
    if (pointDragRef.current !== null) {
      pointDragRef.current = null;
      // didDragRef stays set: the trailing click of this release must not place a second point
      // on top of the one just moved (onClick consumes and clears the flag).
      useStore.getState().recomputeDirty(); // the drag flagged dirty per tick without comparing
      canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), true);
      return;
    }
    if (boxDragRef.current) {
      boxDragRef.current = null;
      didDragRef.current = false;
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
      if (box.x2 - box.x1 > MIN_BOX_SIDE && box.y2 - box.y1 > MIN_BOX_SIDE) {
        addBox(box);
        incrementAnnotationsAdded(1);
        coverage.noteAuthoringCommit();
      }
      setDrawing(null);
    }
  };

  const onClick = (ix: number, iy: number, ev: Konva.KonvaEventObject<MouseEvent>) => {
    if (isLocked) return;
    if (ev.evt.button !== 0) return;
    if (outsideImage(ix, iy)) return;
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
      coverage.noteAuthoringCommit();
      return;
    }
    if (mode !== "polygon") return;
    if (annotateUi.draggingVertex) return;
    if (didDragRef.current) {
      didDragRef.current = false; // the trailing click of a vertex-drag release
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

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <AnnotateToolbar
        onSave={() => void save()}
        saveDisabled={saveDisabled}
        dirty={canvas.dirty}
        isLocked={isLocked}
        bandsInfo={bandsInfo}
        bandSelection={bandSelection}
        onBandSelectionChange={setBandSelection}
        completeWarning={coverage.completeWarning}
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
                <SnapIndicator cursor={cursor} polygons={canvas.polygons} scale={s} />
              )}
            </>
          }
        >
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

          {/* Prediction reference (static per navigation) */}
          {predRef && <PredReferenceOverlay pred={predRef} lineW={scaleLineW} />}
        </CanvasStage>

        {ioError && (
          <div className="absolute top-3 left-3 right-3 flex items-center gap-2 rounded-md border border-tcip-fp/50 bg-tcip-panel/95 px-3 py-1.5 text-[11px] text-tcip-fp">
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

        {!isLocked && <AttributePanel selectedBoxIdx={mode === "box" ? selectedBoxIdx : null} />}

        <AnnotateLegend />

        {coverageMultiCell && coverageGrid.grid && (
          <CoverageMinimap
            imagePath={imgPath}
            composite={composite}
            grid={coverageGrid.grid}
            cells={coverageGrid.cells}
            swept={coverage.swept}
            sweptVersion={coverage.version}
            viewport={(() => {
              const host = measureCanvasHost();
              const vp = host
                ? computeViewport(view, host, canvas.imgWidth, canvas.imgHeight)
                : null;
              return vp ? { x0: vp.x, y0: vp.y, x1: vp.x + vp.w, y1: vp.y + vp.h } : null;
            })()}
            onJump={jumpToCell}
          />
        )}
      </div>
    </div>
  );
}

/** Per-instance attribute editing + a geometry-less (image/plant-level) rating entry. Minimal but
 *  functional (the polished editor is a later slice): the selected shape's attributes, plus the
 *  image-level ratings that ride in the same label file with no box. */
function AttributePanel({ selectedBoxIdx }: { selectedBoxIdx: number | null }) {
  const activeSubject = useStore((s) => s.gui.active_subject);
  const registry = useStore((s) => s.registry.subjects);
  const boxes = useStore((s) => s.canvas.boxes);
  const polygons = useStore((s) => s.canvas.polygons);
  const points = useStore((s) => s.canvas.points);
  const selectedPolygonIdx = useStore((s) => s.canvas.selectedPolygonIdx);
  const selectedPointIdx = useStore((s) => s.canvas.selectedPointIdx);
  const imageAnnotations = useStore((s) => s.canvas.imageAnnotations);
  const updateBox = useStore((s) => s.updateBox);
  const updatePolygon = useStore((s) => s.updatePolygon);
  const updatePoint = useStore((s) => s.updatePoint);
  const addImageAnnotation = useStore((s) => s.addImageAnnotation);
  const updateImageAnnotation = useStore((s) => s.updateImageAnnotation);
  const deleteImageAnnotation = useStore((s) => s.deleteImageAnnotation);

  const selected =
    selectedBoxIdx != null && boxes[selectedBoxIdx]
      ? ({ kind: "box", idx: selectedBoxIdx, shape: boxes[selectedBoxIdx] } as const)
      : selectedPolygonIdx != null && polygons[selectedPolygonIdx]
        ? ({
            kind: "polygon",
            idx: selectedPolygonIdx,
            shape: polygons[selectedPolygonIdx],
          } as const)
        : selectedPointIdx != null && points[selectedPointIdx]
          ? ({ kind: "point", idx: selectedPointIdx, shape: points[selectedPointIdx] } as const)
          : null;

  const withAttr = (attrs: Record<string, string>, attr: string, value: string) => {
    const next = { ...attrs };
    if (value) next[attr] = value;
    else delete next[attr];
    return next;
  };

  const setInstanceAttr = (attr: string, value: string) => {
    if (!selected) return;
    if (selected.kind === "box") {
      const b = boxes[selected.idx];
      updateBox(selected.idx, { ...b, attributes: withAttr(b.attributes, attr, value) });
    } else if (selected.kind === "polygon") {
      const p = polygons[selected.idx];
      updatePolygon(selected.idx, { ...p, attributes: withAttr(p.attributes, attr, value) });
    } else {
      const p = points[selected.idx];
      updatePoint(selected.idx, { ...p, attributes: withAttr(p.attributes, attr, value) });
    }
  };

  // Open until the user collapses it, and never re-keyed on the active image: for many traits this
  // is the only way to record a rating, so it must not be hidden when an image loads.
  const [ratingsOpen, setRatingsOpen] = useState(true);

  // Manually dismissible (unlike the legends, this panel holds real inputs, so hover-to-reveal
  // would fight the user reaching into it). Re-opens on a fresh selection so a stale dismiss can't
  // hide the one shape you're now trying to edit.
  const [dismissed, setDismissed] = useState(false);
  useEffect(() => {
    if (selected) setDismissed(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.kind, selected?.idx]);

  const hasContent = !!selected || imageAnnotations.length > 0;
  if (dismissed || !hasContent) {
    return (
      <button
        type="button"
        onClick={() => setDismissed(false)}
        className="absolute top-3 right-3 z-20 flex items-center gap-1.5 rounded-full border border-tcip-border bg-tcip-panel/90 px-2.5 py-1 text-[11px] text-tcip-muted backdrop-blur hover:border-tcip-border-hover hover:text-tcip-fg"
      >
        Attributes
      </button>
    );
  }

  return (
    <div className="absolute top-3 right-3 z-20 w-60 rounded-md border border-tcip-border bg-tcip-panel/95 p-3 text-[11px] shadow-lg backdrop-blur">
      <div className="mb-1 flex items-center justify-between">
        <h4 className="text-[11px] font-semibold tracking-wide text-tcip-fg">Attributes</h4>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          aria-label="Close attributes panel"
          title="Close"
          className="text-tcip-muted hover:text-tcip-fg"
        >
          ✕
        </button>
      </div>
      {selected ? (
        <div className="mb-2">
          <div className="mb-1 text-tcip-muted">
            Selected <span className="font-semibold text-tcip-fg">{selected.shape.subject}</span>
          </div>
          <AttributeEditors
            subject={selected.shape.subject}
            attributes={selected.shape.attributes}
            registry={registry}
            onChange={setInstanceAttr}
          />
        </div>
      ) : (
        <p className="mb-2 text-tcip-muted">Select a shape to set its attributes.</p>
      )}

      <CollapsibleSection
        className="mt-2 rounded border border-tcip-border bg-tcip-bg/60 p-2"
        title="Ratings for this whole image"
        caption="Applies to the whole image, not to any shape."
        open={ratingsOpen}
        onToggle={() => setRatingsOpen((o) => !o)}
      >
        {imageAnnotations.length === 0 && (
          <p className="mb-1 text-tcip-muted">None on this image.</p>
        )}
        {imageAnnotations.map((a, i) => (
          <div key={i} className="mb-1.5 rounded border border-tcip-border p-1.5">
            <div className="mb-1 flex items-center gap-1">
              <span className="font-semibold text-tcip-fg">{a.subject}</span>
              <button
                type="button"
                className="ml-auto text-tcip-muted hover:text-tcip-fp"
                title="Remove this rating"
                onClick={() => deleteImageAnnotation(i)}
              >
                ✕
              </button>
            </div>
            <AttributeEditors
              subject={a.subject}
              attributes={a.attributes}
              registry={registry}
              onChange={(attr, value) =>
                updateImageAnnotation(i, { ...a, attributes: withAttr(a.attributes, attr, value) })
              }
            />
          </div>
        ))}
        <button
          type="button"
          className="tcip-btn mt-1 w-full text-[11px]"
          disabled={!activeSubject}
          onClick={() => activeSubject && addImageAnnotation(activeSubject)}
        >
          + Rating for {activeSubject ?? "…"}
        </button>
      </CollapsibleSection>
    </div>
  );
}

/** One `<select>` per declared attribute of the subject; empty resets the value. */
function AttributeEditors({
  subject,
  attributes,
  registry,
  onChange,
}: {
  subject: string;
  attributes: Record<string, string>;
  registry: Record<string, { attributes?: Record<string, AttributeDef> }>;
  onChange: (attr: string, value: string) => void;
}) {
  const defs = registry[subject]?.attributes ?? {};
  const entries = Object.entries(defs);
  if (entries.length === 0) {
    return <p className="text-tcip-muted">No attributes declared for {subject}.</p>;
  }
  return (
    <>
      {entries.map(([name, def]) => (
        <label key={name} className="mb-1 flex items-center gap-1.5">
          <span className="w-20 shrink-0 truncate text-tcip-muted" title={name}>
            {name}
          </span>
          <select
            className="tcip-select flex-1 text-[11px]"
            value={attributes[name] ?? ""}
            onChange={(e) => onChange(name, e.target.value)}
          >
            <option value="">—</option>
            {def.values.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
      ))}
    </>
  );
}

/** Hover-triggered legend, anchored lower-left of the canvas. Lists the dataset's subjects
 *  (outline colour = subject, GUI-local) plus the selected-shape blue, the same grammar as
 *  Review. In box mode, an extra row explains the dashed boxes: a polygon's own read-only bounds,
 *  not a second editable annotation. */
function AnnotateLegend() {
  const registry = useStore((s) => s.registry.subjects);
  const mode = useStore((s) => s.gui.mode);
  const names = Object.keys(registry);
  return (
    <div className="group absolute bottom-3 left-3 z-20">
      <div className="pointer-events-none absolute bottom-full left-0 mb-2 w-max min-w-[8rem] translate-y-1 whitespace-nowrap rounded-md border border-tcip-border-hover bg-tcip-panel p-3 opacity-0 shadow-lg transition-all group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100">
        <h4 className="mb-2 text-[11px] font-semibold tracking-wide text-tcip-fg">
          Annotate Legend
        </h4>
        <ul className="space-y-1.5">
          {names.map((name) => (
            <li key={name} className="flex items-center gap-2.5 text-[12px]">
              <span
                className="inline-block h-[13px] w-[18px] shrink-0 rounded-[2px] border-[2.5px]"
                style={{ borderColor: subjectColor(name) }}
              />
              <span className="text-tcip-fg">{name}</span>
            </li>
          ))}
          <li className="flex items-center gap-2.5 text-[12px]">
            <span
              className="inline-block h-[13px] w-[18px] shrink-0 rounded-[2px] border-[2.5px]"
              style={{ borderColor: "#00BFFF" }}
            />
            <span className="text-tcip-fg">Selected</span>
          </li>
          {mode === "box" && (
            <li className="flex items-center gap-2.5 text-[12px]">
              <span
                className="inline-block h-[13px] w-[18px] shrink-0 rounded-[2px] border-[2.5px] border-dashed"
                style={{ borderColor: "currentColor" }}
              />
              <span className="text-tcip-muted">Dashed = polygon&apos;s box (read-only)</span>
            </li>
          )}
        </ul>
      </div>
      <button
        type="button"
        className="flex items-center gap-1.5 rounded-full border border-tcip-border bg-tcip-panel/90 px-2.5 py-1 text-[11px] text-tcip-muted backdrop-blur hover:border-tcip-border-hover hover:text-tcip-fg"
      >
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.4" />
          <path
            d="M8 7.2v3.4M8 5.2v.05"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
        Legend
      </button>
    </div>
  );
}

/* ── Subcomponents ───────────────────────────────────────────────────── */

// Per-shape memo: dragVertex/dragBox replace the whole polygons/boxes array on each RAF
// tick (slice() keeps the unchanged elements' identity), so an unrelated shape's props are
// referentially equal and it skips re-render, containing a one-shape drag to that shape.
const BoxOverlay = memo(function BoxOverlay({
  box,
  stroke,
  width,
  labelSize,
  label,
  showLabel,
  selected,
  handleR,
  dashed,
}: {
  box: Box;
  stroke: string;
  width: number;
  labelSize: number;
  label: string;
  /** Labels are hover/selection-only; the legend is the standing symbology reference. */
  showLabel?: boolean;
  selected?: boolean;
  handleR?: number;
  dashed?: boolean;
}) {
  const corners: [number, number][] = [
    [box.x1, box.y1],
    [box.x2, box.y1],
    [box.x2, box.y2],
    [box.x1, box.y2],
  ];
  return (
    <>
      <Rect
        x={box.x1}
        y={box.y1}
        width={box.x2 - box.x1}
        height={box.y2 - box.y1}
        stroke={stroke}
        strokeWidth={width}
        dash={dashed ? [6 * width, 4 * width] : undefined}
      />
      {selected &&
        handleR &&
        corners.map(([cx, cy], i) => (
          <Rect
            key={`h-${i}`}
            x={cx - handleR}
            y={cy - handleR}
            width={handleR * 2}
            height={handleR * 2}
            fill="#ffffff"
            stroke={stroke}
            strokeWidth={width * 0.6}
          />
        ))}
      {showLabel && <HaloLabel x={box.x1} y={box.y1} text={label} fill={stroke} size={labelSize} />}
    </>
  );
});

const PolygonOverlay = memo(function PolygonOverlay({
  polygon,
  stroke,
  width,
  vertexRadius,
  showVertices,
  labelSize,
  label,
  showLabel,
}: {
  polygon: PolygonShape;
  stroke: string;
  width: number;
  vertexRadius: number;
  showVertices: boolean;
  labelSize: number;
  label: string;
  showLabel?: boolean;
}) {
  // Every ring of the annotation draws, in the instance's own stroke: the shape a reviewer confirms
  // is all of it, not the first contour. Selection/hover styling is shared, so touching any part
  // lights up all of them: that shared highlight is what reads as "these are one object".
  const rings = polygon.rings.filter((ring) => ring.length >= 2);
  if (!rings.length) return null;
  const [x0, y0] = rings[0][0];
  return (
    <>
      {rings.map((ring, ri) => (
        <Line key={`r-${ri}`} points={ring.flat()} closed stroke={stroke} strokeWidth={width} />
      ))}
      {showVertices &&
        rings.map((ring, ri) =>
          ring.map(([x, y], i) => (
            <Circle
              key={`v-${ri}-${i}`}
              x={x}
              y={y}
              radius={vertexRadius}
              fill={stroke}
              stroke="#ffffff"
              strokeWidth={width * 0.5}
            />
          )),
        )}
      {/* One label per annotation, not per ring: a two-part shape is one annotation. */}
      {showLabel && <HaloLabel x={x0} y={y0} text={label} fill={stroke} size={labelSize} />}
    </>
  );
});

/**
 * A placed point: four short ticks converging on the coordinate, plus a filled core in the
 * subject's colour with a white keyline. The ticks are the point of the mark: they say "this exact
 * location" the way an instrument's reticle does, and they are what separates a point from the two
 * things it could otherwise be mistaken for on this canvas: a very small box or a collapsed polygon
 * (both hollow outlines) and a polygon vertex handle (a bare filled dot). Selection uses the same
 * highlighter blue as every other selected shape, and the label is the subject name, so a point
 * joins the canvas' existing grammar instead of inventing a second one.
 */
const PointOverlay = memo(function PointOverlay({
  point,
  stroke,
  coreR,
  tickInner,
  tickOuter,
  lineW,
  labelSize,
  label,
  showLabel,
}: {
  point: PointShape;
  stroke: string;
  coreR: number;
  tickInner: number;
  tickOuter: number;
  lineW: number;
  labelSize: number;
  label: string;
  showLabel?: boolean;
}) {
  const { x, y } = point;
  const ticks: [number, number, number, number][] = [
    [x, y - tickInner, x, y - tickOuter],
    [x, y + tickInner, x, y + tickOuter],
    [x - tickInner, y, x - tickOuter, y],
    [x + tickInner, y, x + tickOuter, y],
  ];
  return (
    <>
      {ticks.map(([x1, y1, x2, y2], i) => (
        <Line key={`t-${i}`} points={[x1, y1, x2, y2]} stroke={stroke} strokeWidth={lineW} />
      ))}
      <Circle x={x} y={y} radius={coreR} fill={stroke} stroke="#ffffff" strokeWidth={lineW * 0.6} />
      {showLabel && (
        <HaloLabel x={x + tickOuter} y={y} text={label} fill={stroke} size={labelSize} />
      )}
    </>
  );
});

function InProgressPolygon({
  points,
  cursor,
  stroke,
  strokeW,
  vertR,
}: {
  points: [number, number][];
  cursor: [number, number] | null;
  stroke: string;
  strokeW: number;
  vertR: number;
}) {
  const dash = [strokeW * 4, strokeW * 4];
  return (
    <>
      <Line points={points.flat()} stroke={stroke} strokeWidth={strokeW} dash={dash} />
      {cursor && (
        <Line
          points={[...points[points.length - 1], ...cursor]}
          stroke={stroke}
          strokeWidth={strokeW * 0.6}
          dash={dash}
        />
      )}
      {points.map(([x, y], i) => (
        <Circle
          key={`cp-${i}`}
          x={x}
          y={y}
          radius={vertR}
          fill={stroke}
          stroke="#ffffff"
          strokeWidth={strokeW * 0.5}
        />
      ))}
    </>
  );
}

function SnapIndicator({
  cursor,
  polygons,
  scale,
}: {
  cursor: [number, number];
  polygons: PolygonShape[];
  scale: number;
}) {
  const thr = SNAP_RADIUS_CANVAS / scale;
  let best: [number, number] | null = null;
  let bestD = thr;
  for (const poly of polygons) {
    for (const ring of poly.rings) {
      for (const [x, y] of ring) {
        const d = Math.hypot(x - cursor[0], y - cursor[1]);
        if (d < bestD) {
          bestD = d;
          best = [x, y];
        }
      }
    }
  }
  if (!best) return null;
  const r = 7 / scale; // ring the snap target just outside the vertex handle (~vertex-sized, not 2×)
  return (
    <Circle
      x={best[0]}
      y={best[1]}
      radius={r}
      stroke="#FFE7B1"
      strokeWidth={1.5 / scale}
      dash={[2.5 / scale, 2.5 / scale]}
    />
  );
}

function HaloLabel({
  x,
  y,
  text,
  fill,
  size,
}: {
  x: number;
  y: number;
  text: string;
  fill: string;
  size: number;
}) {
  return (
    <>
      <Text
        x={x + 2}
        y={y - size - 2}
        text={text}
        fill="#000000"
        fontSize={size}
        fontStyle="bold"
        shadowColor="#000000"
        shadowBlur={size * 0.2}
        shadowOffset={{ x: 0, y: 0 }}
        shadowOpacity={0.9}
      />
      <Text x={x + 2} y={y - size - 2} text={text} fill={fill} fontSize={size} fontStyle="bold" />
    </>
  );
}

function PredReferenceOverlay({ pred, lineW }: { pred: PredictionReference; lineW: number }) {
  if (pred.type === "box") {
    const [x1, y1, x2, y2] = pred.coords as number[];
    return (
      <Rect
        x={x1}
        y={y1}
        width={x2 - x1}
        height={y2 - y1}
        stroke="#00BFFF"
        strokeWidth={2 * lineW}
        dash={[8 * lineW, 4 * lineW]}
      />
    );
  }
  return (
    <Line
      points={(pred.coords as number[][]).flat()}
      closed
      stroke="#00BFFF"
      strokeWidth={2 * lineW}
      dash={[8 * lineW, 4 * lineW]}
    />
  );
}
