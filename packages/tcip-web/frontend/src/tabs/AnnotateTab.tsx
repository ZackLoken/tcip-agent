import { useEffect, useRef, useState } from "react";
import { Circle, Line, Rect, Text } from "react-konva";
import Konva from "konva";

import { api } from "@/api/client";
import type { Mtimes } from "@/api/client";
import { classesApi } from "@/api/classes";
import { sessionsApi } from "@/api/sessions";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
import { useImageNav } from "@/hooks/useImageNav";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useStore } from "@/store";
import type { Box, DatasetSelection, PolygonShape, PredictionReference } from "@/store/types";

const SNAP_RADIUS_CANVAS = 15;
const VERTEX_HANDLE_RADIUS = 4;
const EDGE_INSERT_THRESHOLD = 6;
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

function pointInPolygon(pt: [number, number], poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

export function AnnotateTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const view = useStore((s) => s.gui.view);
  const mode = useStore((s) => s.gui.mode);
  const activeClass = useStore((s) => s.gui.active_class);
  const predRef = useStore((s) => s.gui.pred_reference);
  const classColor = useStore((s) => s.classColor);
  const className = useStore((s) => s.className);

  const canvas = useStore((s) => s.canvas);
  const loadLabels = useStore((s) => s.loadLabelsIntoCanvas);
  const addBox = useStore((s) => s.addBox);
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
  const isLocked = currentImageName ? imageStatus.byImage[currentImageName] === "complete" : false;

  // Image navigation (shared with TopBar + Review; honors the status filter).
  const nav = useImageNav();

  // ── Label load + save ───────────────────────────────────────────────

  // Save the current canvas to the paths it was actually loaded from. Reads the
  // live store + refs (not render closures), so it stays correct even when called
  // from an effect while the app is mid-transition to another image.
  async function save(opts?: { interactive?: boolean }) {
    // interactive=false is the auto-flush on navigate/unmount: a conflict there is
    // handled silently (don't clobber; the StatusBar agent indicator already told
    // the user), whereas an explicit Save surfaces the Reload prompt.
    const interactive = opts?.interactive ?? true;
    const paths = loadedPathsRef.current;
    if (!paths) return; // no confirmed load → refuse to overwrite on-disk labels
    const c = useStore.getState().canvas;
    if (!c.dirty) return;
    const projectRoot = useStore.getState().gui.dataset.project_root;

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
      });
    } catch {
      if (interactive)
        setIoError(
          "Could not save annotations — your edits are kept in the editor; press Save to retry.",
        );
      return;
    }

    if (result.status === "conflict") {
      // Someone else (agent or another tab) wrote this file since we loaded it.
      // Do NOT clobber their work; keep the canvas dirty.
      if (interactive) {
        setConflict(true);
        setIoError(
          "These labels changed elsewhere (the agent or another tab). Reload to load the latest — this discards your unsaved edits — or keep editing.",
        );
      }
      return;
    }

    loadedPathsRef.current = { ...paths, mtimes: result.base_mtimes };
    markClean();
    setIoError(null);
    setConflict(false);

    // Update per-image status unless it is pinned Complete. An empty save is a
    // confirmed negative (the 0-byte label file is preserved on disk), distinct
    // from an image that was never annotated.
    const name = paths.image.split(/[/\\]/).pop() ?? "";
    if (projectRoot && name) {
      const current = useStore.getState().imageStatus.byImage[name];
      if (current !== "complete") {
        const newStatus = c.boxes.length + c.polygons.length > 0 ? "partial" : "negative";
        if (current !== newStatus) {
          setImageStatus(name, newStatus);
          // Best-effort status write; the labels are already saved.
          void classesApi.setImageStatus(projectRoot, name, newStatus).catch(() => {});
        }
      }
    }
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
      ? `${dataset.annotations_detect_dir}/${stem}.txt`
      : null;
    const seg = dataset.annotations_segment_dir
      ? `${dataset.annotations_segment_dir}/${stem}.txt`
      : null;
    const key = `${imgPath} ${det ?? ""} ${seg ?? ""}`;

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
        // Show a blank canvas but BLOCK saving so a transient load failure can't
        // let an empty canvas overwrite the labels still on disk.
        loadLabels({ image_path: imgPath, img_width: 0, img_height: 0, boxes: [], polygons: [] });
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
      },
      when: () => !isLocked,
    },
    {
      keys: "escape",
      action: () => {
        setCurrentPolygon([]);
        setDrawing(null);
        selectPolygon(null);
      },
    },
    { keys: "arrowleft", action: () => stepImage(-1) },
    { keys: "arrowright", action: () => stepImage(1) },
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

  const onDown = (ix: number, iy: number) => {
    if (isLocked) return;
    if (mode === "box") {
      setDrawing({ x1: ix, y1: iy, x2: ix, y2: iy, class_id: activeClass });
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
        return;
      }
      // Click missed vertex + edge while polygon selected — check if still inside to keep selected
      if (!pointInPolygon([ix, iy], poly.points)) {
        selectPolygon(null);
      }
    }
  };

  const onMove = (ix: number, iy: number) => {
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

    // Box drag
    if (drawing) {
      setDrawing({ ...drawing, x2: ix, y2: iy });
      return;
    }

    // Polygon hover detection
    if (mode === "polygon" && canvas.currentPolygon.length === 0) {
      let hover: number | null = null;
      for (let pi = 0; pi < canvas.polygons.length; pi++) {
        if (pointInPolygon([ix, iy], canvas.polygons[pi].points)) {
          hover = pi;
          break;
        }
      }
      if (hover !== annotateUi.hoveredPolygonIdx) setHoveredPolygon(hover);
    }
  };

  const onUp = (ix: number, iy: number) => {
    if (isLocked) return;
    if (annotateUi.draggingVertex) {
      useStore.getState().setDraggingVertex(null);
      return;
    }
    if (mode === "box" && drawing) {
      const box: Box = {
        x1: Math.min(drawing.x1, ix),
        y1: Math.min(drawing.y1, iy),
        x2: Math.max(drawing.x1, ix),
        y2: Math.max(drawing.y1, iy),
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
    // Right-click cancels in-progress polygon first
    if (mode === "polygon" && canvas.currentPolygon.length > 0) {
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
    // Box right-click delete
    for (let i = 0; i < canvas.boxes.length; i++) {
      const b = canvas.boxes[i];
      if (ix >= b.x1 && ix <= b.x2 && iy >= b.y1 && iy <= b.y2) {
        deleteBox(i);
        return;
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
  const vertR = Math.max(3, Math.min(VERTEX_HANDLE_RADIUS * (1.6 - s * 0.2), 12)) * scaleLineW;
  const selVertR =
    Math.max(vertR, Math.min(VERTEX_HANDLE_RADIUS * (2.2 - s * 0.2), 16)) * scaleLineW;
  const labelSize = Math.max(8, Math.min(Math.round(9 * (0.6 + s * 0.4)), 18)) * scaleLineW;

  if (!imgPath || !currentImageName) {
    return (
      <div className="flex-1 flex flex-col">
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

  const imageUrl = imgPath ? api.images.url(imgPath) : null;

  const renderLabels = annotateUi.visible;
  const hoveredIdx = annotateUi.hoveredPolygonIdx;
  const draggingIdx = annotateUi.draggingVertex?.[0];

  return (
    <div className="flex-1 flex flex-col">
      <div className="relative flex-1">
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
        >
          {/* Boxes (only active class in box mode, like yolo-annotator) */}
          {renderLabels &&
            mode === "box" &&
            canvas.boxes.map((b, i) =>
              b.class_id === activeClass ? (
                <BoxOverlay
                  key={`box-${i}`}
                  box={b}
                  stroke={classColor(b.class_id)}
                  width={boxStroke}
                  labelSize={labelSize}
                  label={`${b.class_id}: ${className(b.class_id)}`}
                />
              ) : null,
            )}

          {/* Polygons */}
          {renderLabels &&
            canvas.polygons.map((p, i) => {
              const selected = canvas.selectedPolygonIdx === i;
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

          {/* In-progress polygon */}
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

          {/* Prediction reference */}
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

        {isLocked && (
          <div className="absolute top-3 right-3 rounded-md border border-tcip-border bg-tcip-panel/95 px-2 py-1 text-[11px] text-tcip-muted pointer-events-none">
            Locked (Complete). Uncheck Complete to edit.
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 px-3 py-1.5 border-t border-tcip-border bg-tcip-panel text-[11px]">
        <button
          className="tcip-btn text-[11px]"
          onClick={() => undo()}
          title="Ctrl+Z"
          disabled={isLocked}
        >
          ↶ Undo
        </button>
        <button
          className="tcip-btn text-[11px]"
          onClick={() => redo()}
          title="Ctrl+Shift+Z"
          disabled={isLocked}
        >
          ↷ Redo
        </button>
        {mode === "polygon" && (
          <button
            className="tcip-btn text-[11px]"
            onClick={() => commitPolygonAndTrack()}
            disabled={isLocked || canvas.currentPolygon.length < 3}
            title="Double-click or Enter"
          >
            ✓ Close polygon
          </button>
        )}
        <span className="flex-1" />
        <span className="text-tcip-muted">
          Boxes: {canvas.boxes.length} · Polys: {canvas.polygons.length}
        </span>
        <button
          className={canvas.dirty ? "tcip-btn-primary" : "tcip-btn"}
          onClick={() => void save()}
          disabled={!imgPath || isLocked || saveBlocked}
          title="Ctrl+S (auto-save on image change)"
        >
          {canvas.dirty ? "💾 Save" : "Saved"}
        </button>
      </div>
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
}: {
  box: Box;
  stroke: string;
  width: number;
  labelSize: number;
  label: string;
}) {
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
  const r = 12 / scale;
  return (
    <Circle
      x={best[0]}
      y={best[1]}
      radius={r}
      stroke="#FFE7B1"
      strokeWidth={2 / scale}
      dash={[3 / scale, 3 / scale]}
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
