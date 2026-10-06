import { useEffect, useMemo, useRef, useState } from "react";
import { Rect } from "react-konva";
import Konva from "konva";

import { api, type SaveLabelsBody } from "@/api/client";
import { subjectColor } from "@/api/subjects";
import { committedOf } from "@/api/http";
import { sessionsApi } from "@/api/sessions";
import type { ImageEventPayload } from "@/api/types.generated";
import { AnnotateLegend } from "@/components/annotate/AnnotateLegend";
import { AnnotationShapes } from "@/components/annotate/AnnotationShapes";
import { AttributePanel } from "@/components/annotate/AttributePanel";
import { FlagMarks } from "@/components/annotate/FlagMarks";
import { InProgressPolygon } from "@/components/annotate/InProgressPolygon";
import { ProposalShapes } from "@/components/annotate/ProposalShapes";
import { ReviewStrip } from "@/components/annotate/ReviewStrip";
import { SnapIndicator } from "@/components/annotate/SnapIndicator";
import { AnnotateToolbar } from "@/components/AnnotateToolbar";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
import { TabHeading } from "@/components/TabHeading";
import { useBandSelection } from "@/hooks/useBandSelection";
import { useImageBands } from "@/hooks/useImageBands";
import { useImageNav } from "@/hooks/useImageNav";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { usePrefetchAdjacentImages } from "@/hooks/usePrefetchAdjacentImages";
import { useRegionServes } from "@/hooks/useRegionServes";
import { useServingGrid } from "@/hooks/useServingGrid";
import { compositeParams } from "@/lib/bandSelection";
import { ANNOTATE_KEYS } from "@/lib/annotateKeys";
import type { LoadedImage } from "@/lib/imageLoader";
import { currentImage } from "@/lib/paths";
import {
  flagPlaces,
  flagRequest,
  imageFlags,
  keptItems,
  matchTypes,
  nearestNeighborOrder,
  reviewItems,
  sameItem,
  scopedOrder,
  shownProposals,
  stepTarget,
  type ItemFilters,
  type ReviewItem,
  type StepScope,
} from "@/lib/reviewItems";
import { NO_SUBJECT_DRAFT_COLOR, POINT_HIT_CANVAS, strokeWidths } from "@/lib/symbology";
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
  MIN_BOX_SIDE,
  pointInRings,
  pointToSegmentDist,
  withRing,
} from "@/lib/polygonGeometry";
import { applyEditDrag, hitTestEdit, type EditDrag } from "@/lib/editGeometry";
import { useSubjectColors } from "@/lib/subjectColors";
import { nextMode } from "@/lib/toolMode";
import { useStore } from "@/store";
import {
  isFinished,
  type Box,
  type PolygonShape,
  type ServedProposals,
  type SubjectState,
} from "@/store/types";

/** The gestures a save adjudicates beside the canvas content (one save door, its own fields). */
type Gestures = Pick<
  SaveLabelsBody,
  "bucket" | "accept" | "reject" | "complete" | "rect" | "proposals_hidden" | "flag" | "resolve"
>;

const SNAP_RADIUS_CANVAS = 15;
const EDGE_INSERT_THRESHOLD = 6;
const STREAM_MIN_DIST_CANVAS = 6; // screen px between vertices laid down in Stream (freehand) mode

const contributionsInFlight = new Set<ImageEventPayload>();

/** Post every held image visit of the open project not already in flight. Each is retired once
 *  the backend accepts it, or committed it and could not record the line (the gap toasted),
 *  and stays held for the next send otherwise. */
function sendHeldContributions() {
  const { heldContributions, openProject, retireContribution, pushToast } = useStore.getState();
  for (const contribution of heldContributions) {
    if (contribution.project_id !== openProject?.id || contributionsInFlight.has(contribution)) {
      continue;
    }
    contributionsInFlight.add(contribution);
    void sessionsApi
      .imageEvent(contribution)
      .then(
        () => retireContribution(contribution),
        (e: unknown) => {
          if (committedOf(e) === null) return;
          retireContribution(contribution);
          pushToast(e instanceof Error ? e.message : String(e));
        },
      )
      .finally(() => contributionsInFlight.delete(contribution));
  }
}

export function AnnotateTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const view = useStore((s) => s.gui.view);
  const setView = useStore((s) => s.setView);
  const setMode = useStore((s) => s.setMode);
  const mode = useStore((s) => s.gui.mode);
  const activeSubject = useStore((s) => s.gui.active_subject);
  // The subject registry (subject -> {description?, attributes?}); drives colors (name-derived,
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
  const startImageSessionTracking = useStore((s) => s.startImageSessionTracking);
  const incrementAnnotationsAdded = useStore((s) => s.incrementAnnotationsAdded);
  const closeSessionInterval = useStore((s) => s.closeSessionInterval);
  const heldContributions = useStore((s) => s.heldContributions);
  const openProjectId = useStore((s) => s.openProject?.id);
  useEffect(() => sendHeldContributions(), [heldContributions, openProjectId]);

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
  const boxDragRef = useRef<{ idx: number; drag: EditDrag } | null>(null);
  // Index of the point being dragged. A point has no vertices, so repositioning it is the whole
  // edit: one undo snapshot is taken when the drag starts (see onDown), like a box/vertex drag.
  const pointDragRef = useRef<number | null>(null);

  // I/O safety. The canvas belongs to exactly the image last loaded from disk:
  //  - loadedPathsRef: the image the current shapes came from. save() writes its labels, never
  //    a since-changed dataset's current image, which could write one image's shapes onto another's.
  //  - loadedKeyRef: gates reloads to a genuine image-identity change, so unrelated store updates
  //    (a WS snapshot, a mode/subject toggle) don't re-read disk and clobber unsaved edits.
  //  - saveBlocked: set when a load failed, so a blank canvas can't overwrite the labels on disk.
  const loadedKeyRef = useRef<string | null>(null);
  const loadedPathsRef = useRef<{
    image: string;
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
  const saveDisabled = !imgPath || saveBlocked;

  const bandsInfo = useImageBands(imgPath);
  const [bandSelection, setBandSelection] = useBandSelection(bandsInfo);

  // Every request for this view (the canvas' own and the prefetcher's warm-up) carries one set
  // of band params, so the two never warm and read different renders of the same image.
  const composite = compositeParams(bandsInfo, bandSelection);

  const nav = useImageNav();
  usePrefetchAdjacentImages(composite.bands, composite.stretch);

  const [baseFacts, setBaseFacts] = useState<LoadedImage | null>(null);
  const serving = useServingGrid(imgPath);
  const regions = useRegionServes({
    imagePath: imgPath,
    imgW: canvas.imgWidth,
    imgH: canvas.imgHeight,
    view,
    serving,
    baseFacts,
    composite,
  });

  function viewportRect(): [number, number, number, number] | null {
    const host = measureCanvasHost();
    if (!host) return null;
    const vp = computeViewport(view, host, canvas.imgWidth, canvas.imgHeight);
    return vp ? [vp.x, vp.y, vp.w, vp.h] : null;
  }
  // A region mark is offered only while the view holds less than the whole image.
  const viewRect = viewportRect();
  const viewIsRegion =
    !!viewRect &&
    (viewRect[0] > 0 ||
      viewRect[1] > 0 ||
      viewRect[0] + viewRect[2] < canvas.imgWidth ||
      viewRect[1] + viewRect[3] < canvas.imgHeight);

  function overview() {
    const host = measureCanvasHost();
    if (!host || canvas.imgWidth <= 0 || canvas.imgHeight <= 0) return;
    setView(fitView(host, canvas.imgWidth, canvas.imgHeight));
  }

  // ── Proposals: the selected bucket's, paired and decided server-side ────────
  const bucket = dataset.bucket;
  // Per image: hiding proposals while annotating is recorded on the mark the person then makes.
  const [hideProposals, setHideProposals] = useState(false);
  const [proposalTick, setProposalTick] = useState(0);
  const [served, setServed] = useState<{ key: string; served: ServedProposals } | null>(null);
  const proposalKey = `${imgPath ?? ""}\0${bucket ?? ""}`;
  useEffect(() => {
    if (!imgPath || !currentImageName || !bucket || hideProposals) return;
    let canceled = false;
    api.annotate.proposals(imgPath, bucket).then(
      (r) => {
        if (!canceled) setServed({ key: proposalKey, served: r });
      },
      (e: unknown) => {
        if (!canceled)
          useStore
            .getState()
            .pushToast(`Could not load proposals: ${e instanceof Error ? e.message : String(e)}`);
      },
    );
    return () => {
      canceled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [proposalKey, hideProposals, proposalTick]);
  const reviewing = !!bucket && !hideProposals && served?.key === proposalKey;
  const proposals = useMemo(() => (reviewing ? served.served.proposals : []), [reviewing, served]);
  const operatingPoint = reviewing ? served.served.operating_point : null;

  // ── The review path: items, filters, order and focus ───────────────────────

  // The confidence floor starts at the bucket's validated operating point and is the person's
  // to move, kept per bucket so another bucket starts at its own.
  const [confidenceEntry, setConfidenceEntry] = useState<{
    bucket: string;
    value: number | null;
  } | null>(null);
  const [matchFilter, setMatchFilter] = useState<ItemFilters["match"]>("all");
  const [scope, setScope] = useState<StepScope>("all");
  const filters: ItemFilters = {
    confidence:
      confidenceEntry?.bucket === bucket ? confidenceEntry.value : (operatingPoint?.conf ?? null),
    match: reviewing ? matchFilter : "all",
  };
  function setFilters(next: ItemFilters) {
    setMatchFilter(next.match);
    if (bucket) setConfidenceEntry({ bucket, value: next.confidence });
  }
  const matches = useMemo(
    () => matchTypes(canvas, proposals, reviewing),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [canvas.boxes, canvas.polygons, canvas.points, proposals, reviewing],
  );
  // The items are the selected tool's geometry only: a review steps boxes, polygons or points.
  const items = useMemo(
    () =>
      reviewItems({
        canvas,
        proposals,
        reviewing,
        mode,
        flags: canvas.flags,
        bucket: bucket ?? null,
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      canvas.boxes,
      canvas.polygons,
      canvas.points,
      canvas.flags,
      proposals,
      reviewing,
      mode,
      bucket,
    ],
  );
  const kept = useMemo(
    () => keptItems(items, filters),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [items, filters.confidence, filters.match],
  );
  const order = useMemo(() => scopedOrder(nearestNeighborOrder(kept), scope), [kept, scope]);
  const drawnProposals = useMemo(
    () => shownProposals(proposals, filters, mode),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [proposals, filters.confidence, filters.match, mode],
  );

  const focusedProposal = useStore((s) => s.annotateUi.focusedProposal);
  const [selectedProposal, setSelectedProposal] = useState<number | null>(null);
  // Focus belongs to the selected tool: switching tools drops it.
  useEffect(() => {
    setSelectedBoxIdx(null);
    setSelectedProposal(null);
    selectPolygon(null);
    selectPoint(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);
  // One focused item at a time: focusing a box or a proposal drops every other selection, and
  // a polygon or point selected on the canvas drops the box and the proposal.
  function selectBox(idx: number | null) {
    setSelectedBoxIdx(idx);
    if (idx !== null) {
      selectPolygon(null);
      selectPoint(null);
      setSelectedProposal(null);
    }
  }
  function selectProposal(index: number | null) {
    setSelectedProposal(index);
    if (index !== null) {
      setSelectedBoxIdx(null);
      selectPolygon(null);
      selectPoint(null);
    }
  }
  useEffect(() => {
    if (canvas.selectedPolygonIdx !== null || canvas.selectedPointIdx !== null) {
      setSelectedBoxIdx(null);
      setSelectedProposal(null);
    }
  }, [canvas.selectedPolygonIdx, canvas.selectedPointIdx]);
  useEffect(() => {
    selectProposal(focusedProposal);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusedProposal, currentImageName]);
  const focused: ReviewItem | null = useMemo(() => {
    const find = (kind: ReviewItem["kind"], ref: number | null) =>
      (ref === null ? undefined : items.find((i) => i.kind === kind && i.ref === ref)) ?? null;
    // A proposal pairing with an annotation is that annotation's item, never one of its own.
    const paired = proposals.find((p) => p.index === selectedProposal)?.paired ?? null;
    return (
      find("proposal", selectedProposal) ??
      items.find((i) => i.kind === "annotation" && paired !== null && i.index === paired) ??
      find(
        "annotation",
        { box: selectedBoxIdx, polygon: canvas.selectedPolygonIdx, point: canvas.selectedPointIdx }[
          mode
        ],
      )
    );
  }, [
    items,
    proposals,
    mode,
    selectedProposal,
    selectedBoxIdx,
    canvas.selectedPolygonIdx,
    canvas.selectedPointIdx,
  ]);

  /** Focus one item: select it, make its subject active and zoom the view to it, padded by its
   *  own extent (a point by a fortieth of the image's longer side, since it has none). */
  function focusItem(item: ReviewItem) {
    if (item.kind === "proposal") selectProposal(item.ref);
    else if (item.shape === "box") selectBox(item.ref);
    else if (item.shape === "polygon") selectPolygon(item.ref);
    else selectPoint(item.ref);
    setActiveSubject(item.subject);
    const host = measureCanvasHost();
    if (!host || canvas.imgWidth <= 0 || canvas.imgHeight <= 0) return;
    const [x0, y0, x1, y1] = item.bbox;
    const pad = Math.max(x1 - x0, y1 - y0, Math.max(canvas.imgWidth, canvas.imgHeight) / 40);
    setView(
      zoomToRect(
        { x0, y0, x1, y1 },
        { host, imgW: canvas.imgWidth, imgH: canvas.imgHeight, padX: pad, padY: pad },
      ),
    );
  }

  function stepItem(delta: number) {
    const next = stepTarget(order, focused, delta);
    if (next) focusItem(next);
  }

  function jumpToItem(oneBased: number) {
    const target = order[Math.max(1, Math.min(order.length, oneBased)) - 1];
    if (target) focusItem(target);
  }

  const position = order.findIndex((item) => sameItem(item, focused)) + 1;
  const unreviewedKept = scopedOrder(kept, "unreviewed");

  /** The undecided proposal an item stands for: itself, or the one pairing with it. */
  const proposalOf = (item: ReviewItem | null | undefined) =>
    item?.kind === "proposal" ? item.ref : item?.pairing;

  /** Accept or reject the focused item's proposal, or the first unreviewed item's when the
   *  focused item has none. */
  function decide(action: "accept" | "reject") {
    const target = proposalOf(focused) ?? proposalOf(unreviewedKept[0]);
    if (target === undefined || !bucket) return;
    void save({ gestures: { bucket, [action]: [target] } });
  }

  /** Edit the focused proposal: it is accepted, and the annotation the save appended (the
   *  document's last) is then focused in its place. */
  async function edit() {
    if (focused?.kind !== "proposal" || !bucket) return;
    await save({ gestures: { bucket, accept: [focused.ref] } });
    const saved = reviewItems({
      canvas: useStore.getState().canvas,
      proposals: [],
      reviewing: false,
      mode,
      flags: [],
      bucket: null,
    });
    const last = saved.reduce<ReviewItem | undefined>(
      (best, i) => ((i.index ?? -1) > (best?.index ?? -1) ? i : best),
      undefined,
    );
    if (last) focusItem(last);
  }

  // ── Flags: a comment on the focused item, or on the image when nothing is focused ──
  const [flagsOpen, setFlagsOpen] = useState(false);
  const openFlags = canvas.flags.filter((f) => f.resolved_by === null);
  const flagMarks = useMemo(
    () => flagPlaces(openFlags, drawnProposals, bucket ?? null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [canvas.flags, drawnProposals, bucket],
  );
  function raiseFlag(text: string) {
    void save({ gestures: { flag: [flagRequest(text, focused, bucket ?? null)] } });
  }
  function resolveFlag(id: string, reply: string) {
    void save({ gestures: { resolve: { [id]: reply } } });
  }

  async function markComplete(next: boolean, rect?: [number, number, number, number]) {
    const subject = dataset.subject;
    if (!subject) return;
    const stateOf = () => useStore.getState().canvas.completion[subject];
    const before = stateOf();
    await save({
      gestures: {
        complete: { [subject]: next },
        rect: rect ?? null,
        proposals_hidden: hideProposals,
      },
    });
    if (before !== "negative" && stateOf() === "negative") {
      useStore.getState().markNegativeConfirmed();
    }
  }

  // A box selection belongs to one image; leaving it drops the selection + any drag (and ends a
  // live freehand stream so it can't bleed vertices onto the next image).
  useEffect(() => {
    setHideProposals(false);
    setSelectedBoxIdx(null);
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

  // ── Live canvas push (agent visibility: capture_live_canvas) ──────────────
  // The ref always holds the freshest closure so the debounced pusher never reads stale state.
  const colorTick = useSubjectColors(); // bumps on a recolor, so swatches recompute below
  const subjectSwatches = useMemo(() => {
    void colorTick; // read only to force recompute; subjectColor() itself needs no argument for it
    return Object.keys(registry).map((name) => ({ name, color: subjectColor(name) }));
  }, [registry, colorTick]);
  const buildCanvasBodyRef = useRef<() => CanvasStateBody | null>(() => null);
  buildCanvasBodyRef.current = () => {
    const project = useStore.getState().openProject;
    if (!imgPath || !project) return null;
    // Never push mid-transition: attaching the previous image's still-live shapes to the new
    // image_path would show the agent a false canvas; wait until the loaded identity matches.
    if (loadedPathsRef.current?.image !== imgPath) return null;
    const host = measureCanvasHost();
    return {
      project_id: project.id,
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
        focused,
        matches,
        flagMarks,
        mode,
        activeSubject: activeSubject ?? "",
        visible: annotateUi.visible,
        colorFor: subjectColor,
        cutStart: cutStart
          ? { point: cutStart.point, color: subjectColor(cutStart.polygon.subject) }
          : null,
        cursor,
        proposals: drawnProposals,
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
    drawnProposals,
    matches,
    canvas.flags,
    focused,
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

  /** Save the current canvas, with any gestures it adjudicates, to the path it was loaded from,
   *  reading the live store and refs so a call mid-transition writes the right image. With
   *  `interactive` false (the auto-flush on navigate/unmount) a dropped save is a toast. */
  async function save(opts?: { interactive?: boolean; gestures?: Gestures }) {
    const interactive = opts?.interactive ?? true;
    const gestures = opts?.gestures;
    const paths = loadedPathsRef.current;
    if (!paths) return; // no confirmed load → refuse to overwrite the stored labels
    const c = useStore.getState().canvas;
    if (!c.dirty && !gestures) return;
    const imgFileName = paths.image.split(/[/\\]/).pop() ?? "image";

    let result;
    try {
      result = await api.annotate.save({
        image_path: paths.image,
        annotations: canvasToAnnotations({
          boxes: c.boxes,
          polygons: c.polygons,
          points: c.points,
          imageAnnotations: c.imageAnnotations,
        }),
        base_mtime: paths.mtime,
        user: useStore.getState().user,
        ...gestures,
      });
    } catch (e) {
      const detail = e instanceof Error ? e.message : String(e);
      // Identity check: a stale failure for a since-left image must not raise a
      // banner over the image now on screen.
      if (interactive && loadedPathsRef.current === paths) {
        setIoError(
          `Could not save annotations (${detail}). Your edits are kept in the editor; press Save to retry.`,
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

    // Staleness guard: flushLeaving() fires this save without awaiting it, so by
    // the time the POST resolves the load effect may already have loaded the next
    // image and repointed loadedPathsRef. Rewinding the ref here would make every
    // later save write the new image's shapes onto the old image's label document
    // (with an echoed mtime that matches it, so the backend's 409 guard can't catch
    // it), and markClean() would silently drop edits already made on the new image.
    if (loadedPathsRef.current !== paths) return;

    loadedPathsRef.current = { ...paths, mtime: result.base_mtime };
    setIoError(null);
    setConflict(false);
    if (!gestures) {
      markClean(result.completion, result.flags);
      return;
    }
    // An adjudication writes server-side content (an accepted proposal), so the document reloads.
    await reloadCurrent();
    if (gestures.bucket) setProposalTick((t) => t + 1);
  }

  // Re-fetch the current image's labels from disk, discarding local edits. Used to
  // resolve a conflict (409) or to pick up an agent write on a clean canvas.
  async function reloadCurrent() {
    const paths = loadedPathsRef.current;
    if (!paths) return;
    try {
      const labels = await api.annotate.load(paths.image);
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
    closeSessionInterval();
    void save({ interactive: false });
  }

  // React to a panel event writing labels: reload this image on a clean canvas, or offer a
  // Reload conflict prompt when dirty; a different file leaves the StatusBar indicator alone.
  useEffect(() => {
    if (
      !agentActivity ||
      agentActivity.panel !== "annotate" ||
      agentActivity.eventType !== "labels_written"
    )
      return;
    const paths = loadedPathsRef.current;
    if (!paths) return;
    const norm = (p: unknown) => (typeof p === "string" ? p.replace(/\\/g, "/") : "");
    if (norm(agentActivity.data.image_path) !== norm(paths.image)) return;
    if (useStore.getState().canvas.dirty) {
      setConflict(true);
      const client = agentActivity.client ?? "A process";
      setIoError(
        `${client} just updated this image's labels. Reload to load them (discards your unsaved edits), or keep editing.`,
      );
    } else {
      void reloadCurrent();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentActivity?.seq]);

  useEffect(() => {
    if (!imgPath || !currentImageName) return;
    const key = imgPath;

    // Already displaying this image: an unrelated store change (a WS snapshot, a mode or
    // subject toggle) must not re-read the labels and discard unsaved canvas edits.
    if (loadedKeyRef.current === key) return;

    // Switching images: flush the previous image's work first (to the path it
    // belongs to), then load the new one.
    flushLeaving();

    let canceled = false;
    void (async () => {
      try {
        const labels = await api.annotate.load(imgPath);
        if (canceled) return;
        loadLabels(labels);
        loadedKeyRef.current = key;
        loadedPathsRef.current = { image: imgPath, mtime: labels.base_mtime };
        setSaveBlocked(false);
        setIoError(null);
        setConflict(false);
        startImageSessionTracking(currentImageName);
      } catch {
        if (canceled) return;
        // A blank canvas with saving blocked: a transient load failure never overwrites the labels.
        loadLabels({
          image_path: "",
          img_width: 0,
          img_height: 0,
          boxes: [],
          polygons: [],
          points: [],
          imageAnnotations: [],
          completion: {},
          flags: [],
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
      canceled = true;
    };
    // Keyed on image identity only (see loadedKeyRef); the actions it calls are ref-based.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imgPath, currentImageName]);

  function commitPolygonAndTrack() {
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

  function selectSubjectByIndex(idx: number) {
    // Number keys pick the Nth declared subject (0-based).
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

  // Channeled "cut", so a repeat replaces the standing toast with a count.
  function requireCutSelection(): void {
    useStore
      .getState()
      .pushToast(
        "Select a polygon to cut, then click two points on either side of it.",
        "error",
        "cut",
      );
  }

  function multiRingCutRefusal(ringCount: number): string {
    return (
      `This shape covers ${ringCount} separate parts of one object; the cut applies to a ` +
      "single outline. Cut a part in polygon mode after the others are removed, or redraw it."
    );
  }

  const subjectState: SubjectState | null = canvas.loadedImagePath
    ? (canvas.completion[dataset.subject ?? ""] ?? "unannotated")
    : null;
  const subjectFinished = isFinished(subjectState);

  const K = ANNOTATE_KEYS;
  useKeyboardShortcuts([
    { keys: K.undo.keys, action: () => undo() },
    { keys: K.redo.keys, action: () => redo() },
    { keys: K.redoAlias.keys, action: () => redo() },
    { keys: K.save.keys, action: () => void save() },
    { keys: K.mode.keys, action: () => setMode(nextMode(mode)) },
    { keys: K.accept.keys, action: () => decide("accept"), when: () => unreviewedKept.length > 0 },
    { keys: K.reject.keys, action: () => decide("reject"), when: () => unreviewedKept.length > 0 },
    { keys: K.edit.keys, action: () => void edit(), when: () => focused?.kind === "proposal" },
    { keys: K.flag.keys, action: () => setFlagsOpen((open) => !open) },
    { keys: K.nextItem.keys, action: () => stepItem(1), when: () => order.length > 0 },
    { keys: K.previousItem.keys, action: () => stepItem(-1), when: () => order.length > 0 },
    {
      keys: K.complete.keys,
      action: () => markComplete(!subjectFinished),
      when: () => !!dataset.subject && subjectState !== null,
    },
    { keys: K.hideProposals.keys, action: () => setHideProposals((h) => !h), when: () => !!bucket },
    {
      keys: K.stream.keys,
      action: () => useStore.getState().setStream(!annotateUi.stream),
      when: () => mode === "polygon",
    },
    {
      keys: K.snap.keys,
      action: () => useStore.getState().setSnap(!annotateUi.snap),
      when: () => mode === "polygon",
    },
    {
      keys: K.cut.keys,
      action: () => useStore.getState().setCut(!annotateUi.cut),
      when: () => mode === "polygon",
    },
    {
      keys: K.delete.keys,
      action: () => {
        if (focused?.kind !== "annotation") return;
        if (focused.shape === "polygon") deletePolygon(focused.ref);
        else if (focused.shape === "point") deletePoint(focused.ref);
        else {
          deleteBox(focused.ref);
          selectBox(null);
        }
      },
    },
    {
      keys: K.cancel.keys,
      action: () => {
        setCurrentPolygon([]);
        setDrawing(null);
        selectPolygon(null);
        selectBox(null);
        selectPoint(null);
        selectProposal(null);
        setCutStart(null);
        useStore.getState().setCut(false);
      },
    },
    // Held-key auto-repeat (~30/s) would queue a full image render per tick: one flip per press.
    {
      keys: K.previous.keys,
      action: (e) => {
        if (!e.repeat) nav.stepImage(-1);
      },
    },
    {
      keys: K.next.keys,
      action: (e) => {
        if (!e.repeat) nav.stepImage(1);
      },
    },
    {
      keys: K.close.keys,
      action: () => commitPolygonAndTrack(),
      when: () => mode === "polygon" && canvas.currentPolygon.length >= 3,
    },
    ...Array.from(K.subject.keys, (key, i) => ({
      keys: key,
      action: () => selectSubjectByIndex(i),
    })),
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
          selectBox(i);
          return;
        }
      }
      selectBox(null);
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
            "The polygon changed since the first click; the cut was canceled. Select it and " +
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
    if (outsideImage(ix, iy)) return;
    if (mode !== "polygon") return;
    streamingRef.current = false; // a double-click ends laying even when too short to close
    if (canvas.currentPolygon.length >= 3) {
      commitPolygonAndTrack();
    }
  };

  const onContextMenu = (ix: number, iy: number, ev: Konva.KonvaEventObject<MouseEvent>) => {
    ev.evt.preventDefault();
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
  const widths = strokeWidths(s);
  const { scaleLineW, boxStroke, polyStroke, vertR } = widths;

  if (!imgPath || !currentImageName) {
    return (
      <div className="flex-1 flex flex-col min-h-0">
        <AnnotateToolbar
          onSave={() => void save()}
          saveDisabled={saveDisabled}
          dirty={canvas.dirty}
          subjectState={null}
          onComplete={markComplete}
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
      <TabHeading tab="annotate" />
      <AnnotateToolbar
        onSave={() => void save()}
        saveDisabled={saveDisabled}
        dirty={canvas.dirty}
        bandsInfo={bandsInfo}
        bandSelection={bandSelection}
        onBandSelectionChange={setBandSelection}
        subjectState={subjectState}
        onComplete={markComplete}
        onCompleteView={viewIsRegion && viewRect ? () => markComplete(true, viewRect) : undefined}
      />
      <ReviewStrip
        bucket={bucket ?? null}
        proposalsShown={!hideProposals}
        reviewing={reviewing}
        onProposalsShown={(shown) => setHideProposals(!shown)}
        operatingPoint={operatingPoint}
        filters={filters}
        onFilters={setFilters}
        scope={scope}
        onScope={setScope}
        counts={{
          items: kept.length,
          unreviewed: unreviewedKept.length,
          flags: openFlags.length,
        }}
        flags={focused ? focused.flags : imageFlags(canvas.flags)}
        flagsOpen={flagsOpen}
        onFlagsOpen={setFlagsOpen}
        onFlag={raiseFlag}
        onResolve={resolveFlag}
        focused={focused}
        position={position}
        total={order.length}
        onStep={stepItem}
        onJump={jumpToItem}
        onAccept={() => decide("accept")}
        onEdit={() => void edit()}
        onReject={() => decide("reject")}
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
                  stroke={activeSubject ? subjectColor(activeSubject) : NO_SUBJECT_DRAFT_COLOR}
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
          {renderLabels && (
            <ProposalShapes proposals={drawnProposals} focused={focused} widths={widths} />
          )}
          {/* Committed shapes: memoized, cursor-independent (see AnnotationShapes) */}
          <AnnotationShapes
            boxes={canvas.boxes}
            polygons={canvas.polygons}
            points={canvas.points}
            mode={mode}
            activeSubject={activeSubject}
            matches={matches}
            focused={focused}
            hoveredIdx={hoveredIdx}
            draggingIdx={draggingIdx}
            renderLabels={renderLabels}
            widths={widths}
          />
          {renderLabels && <FlagMarks places={flagMarks} size={widths.labelSize} />}
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

        <AttributePanel focused={focused} />

        <AnnotateLegend reviewing={reviewing} />
      </div>
    </div>
  );
}
