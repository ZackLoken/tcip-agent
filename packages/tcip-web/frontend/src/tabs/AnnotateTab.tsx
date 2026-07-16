import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Circle, Line, Rect, Text } from "react-konva";
import Konva from "konva";

import { api, IMAGE_MAX_WIDTH } from "@/api/client";
import type { Mtimes } from "@/api/client";
import { classesApi, type ClassEntry } from "@/api/classes";
import { sessionsApi } from "@/api/sessions";
import { AnnotateToolbar } from "@/components/AnnotateToolbar";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
import { useImageNav } from "@/hooks/useImageNav";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { usePrefetchAdjacentImages } from "@/hooks/usePrefetchAdjacentImages";
import {
  buildAnnotateShapes,
  computeViewport,
  createCanvasPusher,
  measureCanvasHost,
  onCanvasStateRequest,
  type CanvasStateBody,
} from "@/lib/canvasSync";
import {
  computePolygonBboxes,
  findHoveredPolygon,
  pointInPolygon,
  polygonBbox,
} from "@/lib/polygonGeometry";
import { applyEditDrag, hitTestEdit, type EditDrag } from "@/lib/reviewEditGeometry";
import { useStore } from "@/store";
import type { Box, DatasetSelection, PolygonShape, PredictionReference } from "@/store/types";

const SNAP_RADIUS_CANVAS = 15;
const VERTEX_HANDLE_RADIUS = 4;
const EDGE_INSERT_THRESHOLD = 6;
const STREAM_MIN_DIST_CANVAS = 6; // screen px between vertices laid down in Stream (freehand) mode
const MIN_BOX_SIDE = 3;

function currentImagePath(dataset: DatasetSelection): string | null {
  if (!dataset.dataset_root || !dataset.date) return null;
  const name = dataset.image_list[dataset.current_image_index];
  if (!name) return null;
  return `${dataset.dataset_root}/images/${dataset.date}/${name}`;
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
 * The committed boxes + polygons (content layer). Memoized and — crucially — the mouse
 * cursor is NOT one of its props, so a mouse move (which only updates cursor-following
 * overlays) does not re-render/reconcile these hundreds–thousands of Konva nodes. It
 * re-renders only when the shapes, selection/hover, active class, or zoom-derived stroke
 * sizes actually change.
 */
interface AnnotationShapesProps {
  boxes: Box[];
  polygons: PolygonShape[];
  mode: "box" | "polygon";
  activeClass: number;
  selectedPolygonIdx: number | null;
  selectedBoxIdx: number | null;
  hoveredIdx: number | null;
  draggingIdx: number | undefined;
  renderLabels: boolean;
  /** Class registry entries. Not read directly — classColor/className are stable
   *  store methods that read live state via get() — but the memo compares props,
   *  so this is what invalidates it (and re-renders the tab, which subscribes to
   *  the same list) when a class's color or name changes. Without it the canvas
   *  keeps stale colors/names until an unrelated re-render. */
  classes: ClassEntry[];
  classColor: (id: number) => string;
  className: (id: number) => string;
  boxStroke: number;
  polyStroke: number;
  vertR: number;
  selVertR: number;
  labelSize: number;
}

const AnnotationShapes = memo(function AnnotationShapes({
  boxes,
  polygons,
  mode,
  activeClass,
  selectedPolygonIdx,
  selectedBoxIdx,
  hoveredIdx,
  draggingIdx,
  renderLabels,
  classColor,
  className,
  boxStroke,
  polyStroke,
  vertR,
  selVertR,
  labelSize,
}: AnnotationShapesProps) {
  if (!renderLabels) return null;
  return (
    <>
      {/* Boxes (only active class in box mode, like yolo-annotator) */}
      {mode === "box" &&
        boxes.map((b, i) =>
          b.class_id === activeClass ? (
            <BoxOverlay
              key={`box-${i}`}
              box={b}
              stroke={i === selectedBoxIdx ? "#00BFFF" : classColor(b.class_id)}
              width={boxStroke}
              labelSize={labelSize}
              label={`${b.class_id}: ${className(b.class_id)}`}
              selected={i === selectedBoxIdx}
              handleR={selVertR}
            />
          ) : null,
        )}

      {/* Polygons */}
      {polygons.map((p, i) => {
        const selected = selectedPolygonIdx === i;
        const hovered = hoveredIdx === i;
        const dragging = draggingIdx === i;
        // In box mode only show the selected polygon; in polygon mode show all
        if (mode === "box" && !selected) return null;
        // In polygon mode filter to active class unless selected
        if (mode === "polygon" && !selected && p.class_id !== activeClass) return null;
        const showVerts = selected || hovered || dragging;
        return (
          <PolygonOverlay
            key={`poly-${i}`}
            polygon={p}
            stroke={selected ? "#00BFFF" : classColor(p.class_id)}
            width={polyStroke}
            vertexRadius={selected ? selVertR : vertR}
            showVertices={showVerts}
            labelSize={labelSize}
            label={`${p.class_id}: ${className(p.class_id)}`}
          />
        );
      })}
    </>
  );
});

export function AnnotateTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const view = useStore((s) => s.gui.view);
  const mode = useStore((s) => s.gui.mode);
  const activeClass = useStore((s) => s.gui.active_class);
  const predRef = useStore((s) => s.gui.pred_reference);
  const classColor = useStore((s) => s.classColor);
  const className = useStore((s) => s.className);
  // Subscribe to the class registry itself: classColor/className are stable
  // function refs, so without this an upsertClass (color/name edit) would never
  // re-render the tab or invalidate the AnnotationShapes memo.
  const classList = useStore((s) => s.classes.list);

  const canvas = useStore((s) => s.canvas);
  const loadLabels = useStore((s) => s.loadLabelsIntoCanvas);
  const addBox = useStore((s) => s.addBox);
  const dragBox = useStore((s) => s.dragBox);
  const deleteBox = useStore((s) => s.deleteBox);
  const deletePolygon = useStore((s) => s.deletePolygon);
  const updatePolygon = useStore((s) => s.updatePolygon);
  const dragVertex = useStore((s) => s.dragVertex);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const setCurrentPolygon = useStore((s) => s.setCurrentPolygon);
  const commitCurrentPolygon = useStore((s) => s.commitCurrentPolygon);
  const selectPolygon = useStore((s) => s.selectPolygon);
  const markClean = useStore((s) => s.markClean);
  const setPredReference = useStore((s) => s.setPredReference);
  const pushUndo = useStore((s) => s.pushUndo);
  const setActiveClass = useStore((s) => s.setActiveClass);

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
  const boxDragRef = useRef<{ idx: number; drag: EditDrag } | null>(null);

  // I/O safety. The canvas belongs to exactly the image last loaded from disk:
  //  - loadedPathsRef: the (image, det, seg) the current boxes/polygons came from.
  //    save() writes THERE — never to paths recomputed from a since-changed dataset,
  //    which is how the old code could write one image's boxes onto another's file.
  //  - loadedKeyRef: gates reloads to a genuine image-identity change, so unrelated
  //    store updates (a WS snapshot, a mode/class toggle) don't re-read disk and
  //    clobber unsaved edits.
  //  - saveBlocked: set when a load failed, so a blank canvas can't overwrite the
  //    labels still on disk.
  const loadedKeyRef = useRef<string | null>(null);
  const loadedPathsRef = useRef<{
    image: string;
    det: string | null;
    seg: string | null;
    mtimes: Mtimes;
  } | null>(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const [saveBlocked, setSaveBlocked] = useState(false);
  // True when a save/reload conflict is showing (file changed underneath us) —
  // the banner then offers a Reload button.
  const [conflict, setConflict] = useState(false);
  const agentActivity = useStore((s) => s.agentActivity);

  const imgPath = currentImagePath(dataset);
  const currentImageName = dataset.image_list[dataset.current_image_index] ?? null;
  // A confirmed negative is a completed review (empty) — lock it like "complete".
  const currentStatus = currentImageName ? imageStatus.byImage[currentImageName] : undefined;
  const isLocked = currentStatus === "complete" || currentStatus === "negative";
  const saveDisabled = !imgPath || isLocked || saveBlocked;

  // Image navigation (shared with TopBar + Review; honors the status filter).
  const nav = useImageNav();
  usePrefetchAdjacentImages();

  // A box selection belongs to one image; leaving it drops the selection + any drag (and ends a
  // live freehand stream so it can't bleed vertices onto the next image).
  useEffect(() => {
    setSelectedBoxIdx(null);
    boxDragRef.current = null;
    streamingRef.current = false;
  }, [currentImageName]);

  // ── Live canvas push (agent visibility: visualize_canvas) ──────────────
  // The ref always holds the freshest closure so the debounced pusher never reads stale state.
  const buildCanvasBodyRef = useRef<() => CanvasStateBody | null>(() => null);
  buildCanvasBodyRef.current = () => {
    if (!imgPath || !dataset.project_root) return null;
    // Never push mid-transition: after an image change the canvas briefly still holds the
    // previous image's shapes — attaching them to the new image_path would show the agent a
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
      active_class: activeClass,
      dirty: canvas.dirty,
      user: useStore.getState().user || undefined,
      classes: classList,
      counts: {
        boxes: canvas.boxes.length,
        polygons: canvas.polygons.length,
        drawing_points: canvas.currentPolygon.length,
      },
      shapes: buildAnnotateShapes({
        boxes: canvas.boxes,
        polygons: canvas.polygons,
        currentPolygon: canvas.currentPolygon,
        drawingBox: drawing,
        selectedPolygonIdx: canvas.selectedPolygonIdx,
        selectedBoxIdx,
        mode,
        activeClass,
        visible: annotateUi.visible,
        colorFor: classColor,
        nameFor: className,
      }),
    };
  };
  const canvasPusherRef = useRef(createCanvasPusher((b) => api.canvas.pushState(b)));
  useEffect(() => () => canvasPusherRef.current.dispose(), []);
  // Anything that changes WHICH shapes the canvas draws → full push (geometry travels), except
  // mid-drag/stream where committed geometry re-serializing per tick would jank dense images —
  // those downgrade to heartbeats and the release (drag ref clearing, commit) sends the full.
  useEffect(() => {
    const interacting =
      !!annotateUi.draggingVertex || streamingRef.current || !!drawing || !!boxDragRef.current;
    canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), !interacting);
  }, [
    canvas.boxes,
    canvas.polygons,
    canvas.currentPolygon,
    canvas.selectedPolygonIdx,
    imgPath,
    mode,
    activeClass,
    selectedBoxIdx,
    annotateUi.visible,
    annotateUi.draggingVertex,
    drawing,
  ]);
  useEffect(() => {
    canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), false);
  }, [view, classList, canvas.dirty]);
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

  // Save the current canvas to the paths it was actually loaded from. Reads the
  // live store + refs (not render closures), so it stays correct even when called
  // from an effect while the app is mid-transition to another image.
  async function save(opts?: { interactive?: boolean }) {
    // interactive=false is the auto-flush on navigate/unmount: it can't show the Reload
    // banner (the user is on another image), but a dropped save must never be silent —
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
        detect_path: paths.det,
        segment_path: paths.seg,
        boxes: c.boxes,
        polygons: c.polygons,
        project_root: projectRoot,
        base_mtimes: paths.mtimes,
        user: useStore.getState().user,
      });
    } catch {
      // Identity check: a stale failure for a since-left image must not raise a
      // banner over the image now on screen.
      if (interactive && loadedPathsRef.current === paths) {
        setIoError(
          "Could not save annotations — your edits are kept in the editor; press Save to retry.",
        );
      } else {
        useStore
          .getState()
          .pushToast(`Could not save ${imgFileName} — edits made there were not written.`);
      }
      return;
    }

    if (result.status === "conflict") {
      // Someone else (agent or another tab) wrote this file since we loaded it.
      // Never clobber their work; keep the canvas dirty. The banner belongs to the
      // image on screen — after navigating away, the loss is reported as a toast.
      if (interactive && loadedPathsRef.current === paths) {
        setConflict(true);
        setIoError(
          "These labels changed elsewhere (the agent or another tab). Reload to load the latest — this discards your unsaved edits — or keep editing.",
        );
      } else {
        useStore
          .getState()
          .pushToast(
            `Edits to ${imgFileName} were not saved — its labels changed elsewhere (the agent or another tab) first.`,
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
        const hasContent = c.boxes.length + c.polygons.length > 0;
        // boxes -> partial; an empty save keeps a prior confirmed negative, else unannotated
        // (a negative needs an explicit Complete, not just an empty file).
        const newStatus = hasContent
          ? "partial"
          : current === "negative"
            ? "negative"
            : "unannotated";
        if (current !== newStatus) {
          setImageStatus(name, newStatus);
          // Best-effort status write; the labels are already saved.
          void classesApi.setImageStatus(projectRoot, name, newStatus).catch(() => {});
        }
      }
    }

    // Staleness guard: flushLeaving() fires this save without awaiting it, so by
    // the time the POST resolves the load effect may already have loaded the NEXT
    // image and repointed loadedPathsRef. Rewinding the ref here would make every
    // later save write the new image's shapes onto the old image's label file
    // (with echoed mtimes that match it, so the backend's 409 guard can't catch
    // it), and markClean() would silently drop edits already made on the new image.
    if (loadedPathsRef.current !== paths) return;

    loadedPathsRef.current = { ...paths, mtimes: result.base_mtimes };
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
      const labels = await api.annotate.load(paths.image, paths.det, paths.seg);
      loadLabels(labels);
      loadedPathsRef.current = { ...paths, mtimes: labels.base_mtimes };
      setIoError(null);
      setConflict(false);
      setSaveBlocked(false);
    } catch {
      setIoError("Reload failed — check the connection and try again.");
    }
  }

  // Flush telemetry + any unsaved edits for the image being left, using the paths
  // that canvas belongs to. Called before loading a different image and on unmount.
  function flushLeaving() {
    const leaving = useStore.getState().sessionTracking.currentImageName;
    if (leaving) emitImageSessionEvent(leaving);
    void save({ interactive: false });
  }

  // React to the agent writing labels (panel event). If it touched the file we're
  // viewing: reload on a clean canvas, or offer a Reload conflict prompt if dirty.
  // (Different file/layout → the StatusBar indicator already shows the activity.)
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
    const current = new Set([norm(paths.det), norm(paths.seg)].filter(Boolean));
    if (!written.some((w) => current.has(norm(w)))) return;
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
    const det = dataset.annotations_detect_dir
      ? `${dataset.annotations_detect_dir}/${stem}.json`
      : null;
    const seg = dataset.annotations_segment_dir
      ? `${dataset.annotations_segment_dir}/${stem}.json`
      : null;
    const key = `${imgPath}\0${det ?? ""}\0${seg ?? ""}`;

    // Already displaying this exact image + label target. Ignore — this is what
    // stops an unrelated store change (a WS state snapshot, a mode/class toggle,
    // any patchGui that swaps the dataset object) from re-reading disk and
    // discarding unsaved canvas edits.
    if (loadedKeyRef.current === key) return;

    // Switching images: flush the previous image's work first (to the paths it
    // belongs to), then load the new one.
    flushLeaving();

    let cancelled = false;
    void (async () => {
      try {
        const labels = await api.annotate.load(imgPath, det, seg);
        if (cancelled) return;
        loadLabels(labels);
        loadedKeyRef.current = key;
        loadedPathsRef.current = { image: imgPath, det, seg, mtimes: labels.base_mtimes };
        setSaveBlocked(false);
        setIoError(null);
        setConflict(false);
        startImageSessionTracking(currentImageName, labels.boxes.length + labels.polygons.length);
      } catch {
        if (cancelled) return;
        // Show a blank canvas but block saving so a transient load failure can't let an
        // empty canvas overwrite the labels still on disk. image_path stays empty so the
        // Complete checkbox won't derive a status from this blank canvas either.
        loadLabels({ image_path: "", img_width: 0, img_height: 0, boxes: [], polygons: [] });
        loadedKeyRef.current = key;
        loadedPathsRef.current = null;
        setSaveBlocked(true);
        setConflict(false);
        setIoError(
          "Could not load this image's labels — saving is disabled to avoid overwriting the labels on disk.",
        );
        startImageSessionTracking(currentImageName, 0);
      }
    })();
    return () => {
      cancelled = true;
    };
    // Keyed on image identity + label dirs only (see loadedKeyRef guard); save /
    // loadLabels / tracking actions are stable or ref-based.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imgPath, currentImageName, dataset.annotations_detect_dir, dataset.annotations_segment_dir]);

  function emitImageSessionEvent(imageName: string) {
    const state = useStore.getState();
    const tracking = state.sessionTracking;
    const projectRoot = state.gui.dataset.project_root;
    if (!projectRoot) return;
    if (tracking.currentImageName !== imageName || tracking.imageEnterTimeMs === null) return;

    const finalAnnotationCount = state.canvas.boxes.length + state.canvas.polygons.length;
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
      })
      .catch(() => {
        // Best-effort telemetry; annotation flow should never block on this.
      });
  }

  function commitPolygonAndTrack() {
    if (isLocked) return;
    if (commitCurrentPolygon()) incrementAnnotationsAdded(1);
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

  function selectClassByNumberKey(key: number) {
    // pick by ID if present in class registry
    const entries = useStore.getState().classes.list;
    const target = entries.find((c) => c.id === key);
    if (target) setActiveClass(key);
  }

  useKeyboardShortcuts([
    { keys: "ctrl+z", action: () => undo(), when: () => !isLocked },
    { keys: "ctrl+shift+z", action: () => redo(), when: () => !isLocked },
    { keys: "ctrl+y", action: () => redo(), when: () => !isLocked },
    { keys: "ctrl+s", action: () => void save(), when: () => !isLocked },
    { keys: "m", action: () => useStore.getState().setMode(mode === "box" ? "polygon" : "box") },
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
        }
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
      },
    },
    // Held-key auto-repeat (~30/s) would queue a full image render per tick — one flip per press.
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
    { keys: "0", action: () => selectClassByNumberKey(0) },
    { keys: "1", action: () => selectClassByNumberKey(1) },
    { keys: "2", action: () => selectClassByNumberKey(2) },
    { keys: "3", action: () => selectClassByNumberKey(3) },
    { keys: "4", action: () => selectClassByNumberKey(4) },
    { keys: "5", action: () => selectClassByNumberKey(5) },
    { keys: "6", action: () => selectClassByNumberKey(6) },
    { keys: "7", action: () => selectClassByNumberKey(7) },
    { keys: "8", action: () => selectClassByNumberKey(8) },
    { keys: "9", action: () => selectClassByNumberKey(9) },
  ]);

  // ── Snap helper (image-space) ───────────────────────────────────────

  function snapImagePoint(
    ix: number,
    iy: number,
    excludePolyVi?: [number, number],
  ): [number, number] {
    if (!annotateUi.snap) return [ix, iy];
    const sc = view.scale || 1;
    const thr = SNAP_RADIUS_CANVAS / sc; // image-space radius
    let best: [number, number] | null = null;
    let bestD = thr;
    canvas.polygons.forEach((poly, pi) => {
      poly.points.forEach(([px, py], vi) => {
        if (excludePolyVi && excludePolyVi[0] === pi && excludePolyVi[1] === vi) return;
        const d = Math.hypot(px - ix, py - iy);
        if (d < bestD) {
          bestD = d;
          best = [px, py];
        }
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
  // Throttle the "detect is derived" notice to once per image, not once per click.
  const derivedNoticeRef = useRef<string | null>(null);

  const onDown = (ix: number, iy: number, ev: Konva.KonvaEventObject<MouseEvent>) => {
    if (isLocked) return;
    if (ev.evt.button !== 0) return; // right-button drags must not fabricate boxes
    // A fresh press starts a new gesture: clear the drag flag first. A completed vertex drag
    // fires no trailing click, so without this the stale flag would swallow the next click
    // (e.g. an outside click meant to deselect), forcing a second click.
    didDragRef.current = false;
    if (mode === "box") {
      // Detect boxes are derived from polygons when any exist, so a drawn box would be
      // discarded on save — point the annotator at polygon mode instead of losing it.
      if (canvas.polygons.length > 0) {
        if (derivedNoticeRef.current !== currentImageName) {
          derivedNoticeRef.current = currentImageName;
          useStore
            .getState()
            .pushToast(
              "Detect boxes are derived from polygons on this image — switch to Polygon mode (M) to add an object.",
            );
        }
        return;
      }
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
      // Otherwise a press inside an existing (active-class) box selects it; empty space
      // deselects and starts a new box.
      for (let i = canvas.boxes.length - 1; i >= 0; i--) {
        const b = canvas.boxes[i];
        if (b.class_id === activeClass && ix >= b.x1 && ix <= b.x2 && iy >= b.y1 && iy <= b.y2) {
          setSelectedBoxIdx(i);
          return;
        }
      }
      setSelectedBoxIdx(null);
      const cx = Math.max(0, Math.min(canvas.imgWidth || ix, ix));
      const cy = Math.max(0, Math.min(canvas.imgHeight || iy, iy));
      setDrawing({ x1: cx, y1: cy, x2: cx, y2: cy, class_id: activeClass });
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
      // Try vertex grab
      for (let vi = 0; vi < poly.points.length; vi++) {
        const [px, py] = poly.points[vi];
        if (Math.hypot(px - ix, py - iy) < vertThr) {
          // Capture undo once at drag start; the drag itself uses dragVertex (no
          // per-mousemove undo push, which would otherwise flood the 30-entry stack).
          pushUndo();
          useStore.getState().setDraggingVertex([pi, vi]);
          didDragRef.current = true;
          return;
        }
      }
      // Try edge insert
      const edgeThr = EDGE_INSERT_THRESHOLD / sc;
      let bestEdge = -1;
      let bestDist = edgeThr;
      let bestProj: [number, number] | null = null;
      for (let ei = 0; ei < poly.points.length; ei++) {
        const [ax, ay] = poly.points[ei];
        const [bx, by] = poly.points[(ei + 1) % poly.points.length];
        const { dist, proj } = pointToSegmentDist(ix, iy, ax, ay, bx, by);
        if (dist < bestDist) {
          bestDist = dist;
          bestEdge = ei;
          bestProj = proj;
        }
      }
      if (bestEdge >= 0 && bestProj) {
        const newPts = poly.points.slice();
        newPts.splice(bestEdge + 1, 0, bestProj);
        updatePolygon(pi, { ...poly, points: newPts });
        useStore.getState().setDraggingVertex([pi, bestEdge + 1]);
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
  // frame firing means the callback runs on current — never stale — state.
  const pendingMoveRef = useRef<[number, number] | null>(null);
  const moveRafRef = useRef<number | null>(null);
  const processMoveRef = useRef<(ix: number, iy: number) => void>(() => {});
  processMoveRef.current = (ix: number, iy: number) => {
    setCursor([ix, iy]);
    if (isLocked) return;

    // Vertex drag
    const dragging = annotateUi.draggingVertex;
    if (dragging) {
      const [pi, vi] = dragging;
      const poly = canvas.polygons[pi];
      if (poly) {
        const [sx, sy] = snapImagePoint(ix, iy, [pi, vi]);
        const clamped: [number, number] = [
          Math.max(0, Math.min(canvas.imgWidth || sx, sx)),
          Math.max(0, Math.min(canvas.imgHeight || sy, sy)),
        ];
        dragVertex(pi, vi, clamped); // no per-move undo push (see onDown drag start)
      }
      return;
    }

    // Streaming (freehand): between the two clicks, drop a vertex each time the pointer has
    // moved far enough — no button held.
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
          dragBox(bDrag.idx, { ...b, x1, y1, x2, y2 }); // undo captured on down; spread keeps provenance
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
    if (boxDragRef.current) {
      boxDragRef.current = null;
      didDragRef.current = false;
      // The drag suppressed full pushes; the settled geometry ships now.
      canvasPusherRef.current.schedule(() => buildCanvasBodyRef.current(), true);
      return;
    }
    if (annotateUi.draggingVertex) {
      useStore.getState().setDraggingVertex(null);
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
        class_id: activeClass,
      };
      if (box.x2 - box.x1 > MIN_BOX_SIDE && box.y2 - box.y1 > MIN_BOX_SIDE) {
        addBox(box);
        incrementAnnotationsAdded(1);
      }
      setDrawing(null);
    }
  };

  const onClick = (ix: number, iy: number, ev: Konva.KonvaEventObject<MouseEvent>) => {
    if (isLocked) return;
    if (ev.evt.button !== 0) return;
    if (mode !== "polygon") return;
    if (annotateUi.draggingVertex) return;
    if (didDragRef.current) {
      didDragRef.current = false; // the trailing click of a vertex-drag release
      return;
    }

    // Stream (freehand): click to start, click again to stop + commit. Between the two clicks the
    // vertices follow the cursor (see processMove) — the button is never held down. Right-click
    // (onContextMenu) cancels a live stream.
    if (annotateUi.stream) {
      if (streamingRef.current) {
        streamingRef.current = false;
        if (canvas.currentPolygon.length >= 3) commitPolygonAndTrack();
        else setCurrentPolygon([]);
      } else {
        if (canvas.selectedPolygonIdx !== null) selectPolygon(null);
        pushUndo();
        const [sx, sy] = snapImagePoint(ix, iy);
        setCurrentPolygon([[sx, sy]]);
        streamingRef.current = true;
      }
      return;
    }

    // Placing vertices into a new polygon
    if (canvas.currentPolygon.length > 0) {
      const [sx, sy] = snapImagePoint(ix, iy);
      setCurrentPolygon([...canvas.currentPolygon, [sx, sy]]);
      return;
    }

    // Not currently drawing — clicking on a polygon selects it
    for (let pi = 0; pi < canvas.polygons.length; pi++) {
      if (pointInPolygon([ix, iy], canvas.polygons[pi].points)) {
        selectPolygon(pi);
        return;
      }
    }
    // Clicked empty space with nothing selected: start a new polygon
    if (canvas.selectedPolygonIdx === null) {
      const [sx, sy] = snapImagePoint(ix, iy);
      setCurrentPolygon([[sx, sy]]);
    } else {
      // Already had a selection and click didn't land on a polygon → deselect
      selectPolygon(null);
    }
  };

  const onDoubleClick = (_ix: number, _iy: number) => {
    if (isLocked) return;
    if (mode !== "polygon") return;
    if (canvas.currentPolygon.length >= 3) {
      commitPolygonAndTrack();
    }
  };

  const onContextMenu = (ix: number, iy: number, ev: Konva.KonvaEventObject<MouseEvent>) => {
    ev.evt.preventDefault();
    if (isLocked) return;
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
        for (let vi = 0; vi < poly.points.length; vi++) {
          const [px, py] = poly.points[vi];
          if (Math.hypot(px - ix, py - iy) < vertThr) {
            pushUndo();
            if (poly.points.length <= 3) {
              deletePolygon(pi);
            } else {
              const newPts = poly.points.slice();
              newPts.splice(vi, 1);
              updatePolygon(pi, { ...poly, points: newPts });
            }
            return;
          }
        }
        if (pointInPolygon([ix, iy], poly.points)) {
          deletePolygon(pi);
          return;
        }
      }
      selectPolygon(null);
      return;
    }
    // Box right-click delete — box mode only: in polygon mode a coincident (invisible)
    // detect box would swallow the click and silently fork the two GT layouts.
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
        if (pointInPolygon([ix, iy], canvas.polygons[pi].points)) {
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
  // Both radii resolve in screen px, then compensate by 1/s exactly once — mixing the
  // spaces here once made selected handles grow 1/s² and blanket the frame at fit zoom.
  const vertScreen = Math.max(3, Math.min(VERTEX_HANDLE_RADIUS * (1.6 - s * 0.2), 12));
  const selScreen = Math.max(vertScreen + 1, Math.min(VERTEX_HANDLE_RADIUS * (2.2 - s * 0.2), 16));
  const vertR = vertScreen * scaleLineW;
  const selVertR = selScreen * scaleLineW;
  const labelSize = Math.max(8, Math.min(Math.round(9 * (0.6 + s * 0.4)), 18)) * scaleLineW;

  if (!imgPath || !currentImageName) {
    return (
      <div className="flex-1 flex flex-col min-h-0">
        <AnnotateToolbar
          onSave={() => void save()}
          saveDisabled={saveDisabled}
          dirty={canvas.dirty}
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

  const imageUrl = imgPath ? api.images.url(imgPath, IMAGE_MAX_WIDTH) : null;

  const renderLabels = annotateUi.visible;
  const hoveredIdx = annotateUi.hoveredPolygonIdx;
  const draggingIdx = annotateUi.draggingVertex?.[0];

  // The detect layer is derived from polygons. In box mode, show those derived bounding
  // boxes (read-only) so the detect layer is inspectable in real time; with no polygons,
  // box mode edits the real boxes. Recomputed from the live polygon list each render.
  const boxesDerived = mode === "box" && canvas.polygons.length > 0;
  const boxesToRender = boxesDerived
    ? canvas.polygons.map((p): Box => {
        const [x1, y1, x2, y2] = polygonBbox(p.points);
        return { x1, y1, x2, y2, class_id: p.class_id };
      })
    : canvas.boxes;

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <AnnotateToolbar
        onSave={() => void save()}
        saveDisabled={saveDisabled}
        dirty={canvas.dirty}
      />
      <div className="relative flex-1 flex flex-col min-h-0">
        <CanvasStage
          imageUrl={imageUrl}
          imgWidth={canvas.imgWidth}
          imgHeight={canvas.imgHeight}
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
                  stroke={classColor(activeClass)}
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
                  stroke={classColor(activeClass)}
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
          {/* Committed shapes — memoized, cursor-independent (see AnnotationShapes) */}
          <AnnotationShapes
            boxes={boxesToRender}
            polygons={canvas.polygons}
            mode={mode}
            activeClass={activeClass}
            selectedPolygonIdx={canvas.selectedPolygonIdx}
            selectedBoxIdx={boxesDerived ? null : selectedBoxIdx}
            hoveredIdx={hoveredIdx}
            draggingIdx={draggingIdx}
            renderLabels={renderLabels}
            classes={classList}
            classColor={classColor}
            className={className}
            boxStroke={boxStroke}
            polyStroke={polyStroke}
            vertR={vertR}
            selVertR={selVertR}
            labelSize={labelSize}
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
          </div>
        )}

        <AnnotateLegend />
      </div>
    </div>
  );
}

/** Hover-triggered legend, anchored lower-left of the canvas. Lists the project's classes
 *  (outline colour = class) plus the selected-shape blue — the same grammar as Review. */
function AnnotateLegend() {
  const classes = useStore((s) => s.classes.list);
  return (
    <div className="group absolute bottom-3 left-3 z-20">
      <div className="pointer-events-none absolute bottom-full left-0 mb-2 w-max min-w-[8rem] translate-y-1 whitespace-nowrap rounded-md border border-tcip-border-hover bg-tcip-panel p-3 opacity-0 shadow-lg transition-all group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100">
        <h4 className="mb-2 text-[11px] font-semibold tracking-wide text-tcip-fg">
          Annotate Legend
        </h4>
        <ul className="space-y-1.5">
          {classes.map((c) => (
            <li key={c.id} className="flex items-center gap-2.5 text-[12px]">
              <span
                className="inline-block h-[13px] w-[18px] shrink-0 rounded-[2px] border-[2.5px]"
                style={{ borderColor: c.color }}
              />
              <span className="text-tcip-fg">{c.name}</span>
            </li>
          ))}
          <li className="flex items-center gap-2.5 text-[12px]">
            <span
              className="inline-block h-[13px] w-[18px] shrink-0 rounded-[2px] border-[2.5px]"
              style={{ borderColor: "#00BFFF" }}
            />
            <span className="text-tcip-fg">Selected</span>
          </li>
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

function BoxOverlay({
  box,
  stroke,
  width,
  labelSize,
  label,
  selected,
  handleR,
}: {
  box: Box;
  stroke: string;
  width: number;
  labelSize: number;
  label: string;
  selected?: boolean;
  handleR?: number;
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
      <HaloLabel x={box.x1} y={box.y1} text={label} fill={stroke} size={labelSize} />
    </>
  );
}

function PolygonOverlay({
  polygon,
  stroke,
  width,
  vertexRadius,
  showVertices,
  labelSize,
  label,
}: {
  polygon: PolygonShape;
  stroke: string;
  width: number;
  vertexRadius: number;
  showVertices: boolean;
  labelSize: number;
  label: string;
}) {
  if (polygon.points.length < 2) return null;
  const flat = polygon.points.flat();
  const [x0, y0] = polygon.points[0];
  return (
    <>
      <Line points={flat} closed stroke={stroke} strokeWidth={width} />
      {showVertices &&
        polygon.points.map(([x, y], i) => (
          <Circle
            key={`v-${i}`}
            x={x}
            y={y}
            radius={vertexRadius}
            fill={stroke}
            stroke="#ffffff"
            strokeWidth={width * 0.5}
          />
        ))}
      <HaloLabel x={x0} y={y0} text={label} fill={stroke} size={labelSize} />
    </>
  );
}

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
    for (const [x, y] of poly.points) {
      const d = Math.hypot(x - cursor[0], y - cursor[1]);
      if (d < bestD) {
        bestD = d;
        best = [x, y];
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
