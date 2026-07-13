/**
 * Annotate-tab context toolbar: class/color, draw mode, snap/stream, image status, and
 * image navigation. This lives directly under the global TopBar (logo + tabs + status)
 * so tab-specific tools are decoupled — each tab owns its own toolbar rather than
 * cramming everything into the app-level bar.
 */

import { useMemo, useRef, useState } from "react";

import { classesApi, type ImageStatus } from "@/api/classes";
import { ColorPickerModal } from "@/components/ColorPickerModal";
import { useImageNav } from "@/hooks/useImageNav";
import { useStore } from "@/store";

const STATUS_FILTERS: { value: "all" | ImageStatus; label: string }[] = [
  { value: "all", label: "All" },
  { value: "complete", label: "Complete" },
  { value: "partial", label: "Partial" },
  { value: "negative", label: "Negative" },
  { value: "unannotated", label: "Unannotated" },
];

export function AnnotateToolbar() {
  const dataset = useStore((s) => s.gui.dataset);
  const mode = useStore((s) => s.gui.mode);
  const setMode = useStore((s) => s.setMode);
  const activeClass = useStore((s) => s.gui.active_class);
  const setActiveClass = useStore((s) => s.setActiveClass);
  const classes = useStore((s) => s.classes.list);
  const upsertClass = useStore((s) => s.upsertClass);
  const classColor = useStore((s) => s.classColor);
  const canvasBoxes = useStore((s) => s.canvas.boxes);
  const canvasPolygons = useStore((s) => s.canvas.polygons);
  const annotateUi = useStore((s) => s.annotateUi);
  const setVisible = useStore((s) => s.setVisible);
  const setSnap = useStore((s) => s.setSnap);
  const setStream = useStore((s) => s.setStream);
  const imageStatus = useStore((s) => s.imageStatus);
  const setStatusFilter = useStore((s) => s.setStatusFilter);
  const setImageStatus = useStore((s) => s.setImageStatus);

  const [pickerOpen, setPickerOpen] = useState(false);
  const [counterDraft, setCounterDraft] = useState<string | null>(null);
  const counterRef = useRef<HTMLInputElement | null>(null);

  // Counts per class, based on current canvas state.
  const classCounts = useMemo(() => {
    const counts = new Map<number, number>();
    if (mode === "box") {
      for (const b of canvasBoxes) counts.set(b.class_id, (counts.get(b.class_id) ?? 0) + 1);
    } else {
      for (const p of canvasPolygons) counts.set(p.class_id, (counts.get(p.class_id) ?? 0) + 1);
    }
    return counts;
  }, [mode, canvasBoxes, canvasPolygons]);

  const currentImage = dataset.image_list[dataset.current_image_index] ?? "—";
  const currentStatus: ImageStatus | undefined = currentImage
    ? imageStatus.byImage[currentImage]
    : undefined;
  // The canvas swaps asynchronously on flips — until it holds this image's labels,
  // Complete would derive its status from the previous image's shapes.
  const loadedImagePath = useStore((s) => s.canvas.loadedImagePath);
  const canvasReady =
    !!dataset.dataset_root &&
    !!dataset.date &&
    loadedImagePath === `${dataset.dataset_root}/images/${dataset.date}/${currentImage}`;
  // Shared navigation — same filtered traversal as the arrow keys and the Review tab.
  const nav = useImageNav();

  async function addNewClass() {
    const name = window.prompt("New class name:");
    if (!name) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    // Re-use existing class if name matches (case-insensitive)
    const existing = classes.find((c) => c.name.toLowerCase() === trimmed.toLowerCase());
    if (existing) {
      setActiveClass(existing.id);
      return;
    }
    const nextId = classes.length ? Math.max(...classes.map((c) => c.id)) + 1 : 0;
    try {
      const { color } = await classesApi.autoColor(nextId);
      const entry = { id: nextId, name: trimmed, color };
      upsertClass(entry);
      setActiveClass(nextId);
      if (dataset.project_root) {
        await classesApi.save(dataset.project_root, dataset.annotation_type, [...classes, entry]);
      }
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not add class: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function commitColor(newColor: string) {
    setPickerOpen(false);
    const entry = classes.find((c) => c.id === activeClass);
    if (!entry) return;
    const updated = { ...entry, color: newColor };
    upsertClass(updated);
    if (dataset.project_root) {
      const next = classes.map((c) => (c.id === activeClass ? updated : c));
      try {
        await classesApi.save(dataset.project_root, dataset.annotation_type, next);
      } catch (e) {
        useStore
          .getState()
          .pushToast(`Could not save class color: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
  }

  async function toggleComplete(next: boolean) {
    if (!currentImage || !dataset.project_root) return;
    const hasContent = canvasBoxes.length + canvasPolygons.length > 0;
    // Complete → content:complete, empty:negative (the intentional negative); uncheck → content:partial, empty:unannotated.
    const newStatus: ImageStatus = next
      ? hasContent
        ? "complete"
        : "negative"
      : hasContent
        ? "partial"
        : "unannotated";
    setImageStatus(currentImage, newStatus);
    try {
      await classesApi.setImageStatus(dataset.project_root, currentImage, newStatus);
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not update status: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // Class-dropdown options include the <New Class> sentinel
  const classOptions = useMemo(() => {
    const opts = classes.map((c) => ({
      value: String(c.id),
      label: `${c.id}: ${c.name} (${classCounts.get(c.id) ?? 0})`,
    }));
    opts.push({ value: "__new__", label: "<New Class>" });
    return opts;
  }, [classes, classCounts]);

  const activeEntry = classes.find((c) => c.id === activeClass);

  return (
    <div className="h-topbar flex items-center gap-2 px-3 border-b border-tcip-border bg-tcip-panel shrink-0">
      {/* Color swatch */}
      <button
        className="w-7 h-7 rounded border border-tcip-border mr-1 shrink-0"
        title={`Color: ${activeEntry?.color ?? "—"}  (click to edit)`}
        style={{ background: classColor(activeClass) }}
        onClick={() => setPickerOpen(true)}
        disabled={!activeEntry}
      />

      {/* Class dropdown */}
      <select
        className="tcip-select"
        value={String(activeClass)}
        onChange={(e) => {
          const v = e.target.value;
          if (v === "__new__") {
            void addNewClass();
          } else {
            setActiveClass(Number(v));
          }
        }}
      >
        {classOptions.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>

      {/* Visible checkbox */}
      <label className="flex items-center gap-1 text-[11px] ml-3">
        <input
          type="checkbox"
          checked={annotateUi.visible}
          onChange={(e) => setVisible(e.target.checked)}
        />
        Visible
      </label>

      {/* Mode toggle */}
      <div className="flex rounded overflow-hidden border border-tcip-border ml-3">
        <button
          onClick={() => setMode("box")}
          className={`px-2 h-7 text-[11px] transition-colors ${mode === "box" ? "bg-tcip-accent text-white" : "bg-tcip-panel text-tcip-fg hover:bg-tcip-hover"}`}
        >
          Box
        </button>
        <button
          onClick={() => setMode("polygon")}
          className={`px-2 h-7 text-[11px] transition-colors ${mode === "polygon" ? "bg-tcip-accent text-white" : "bg-tcip-panel text-tcip-fg hover:bg-tcip-hover"}`}
        >
          Polygon&nbsp;&nbsp;⬡
        </button>
      </div>

      {/* Stream / Snap (polygon only) */}
      <button
        className={`tcip-btn text-[11px] ml-1 ${annotateUi.stream ? "!bg-tcip-accent !text-white" : ""}`}
        onClick={() => setStream(!annotateUi.stream)}
        disabled={mode !== "polygon"}
        title="Stream vertices while dragging (v)"
      >
        Stream: {annotateUi.stream ? "On" : "Off"}
      </button>
      <button
        className={`tcip-btn text-[11px] ${annotateUi.snap ? "!bg-tcip-accent !text-white" : ""}`}
        onClick={() => setSnap(!annotateUi.snap)}
        disabled={mode !== "polygon"}
        title="Snap to nearest vertex (s)"
      >
        Snap: {annotateUi.snap ? "On" : "Off"}
      </button>

      {/* Complete + Status filter */}
      <label
        className="flex items-center gap-1 text-[11px] ml-3"
        title={canvasReady ? undefined : "Loading this image's labels…"}
      >
        <input
          type="checkbox"
          // Show a confirmed negative as checked too (completing an empty image sets "negative").
          checked={currentStatus === "complete" || currentStatus === "negative"}
          onChange={(e) => void toggleComplete(e.target.checked)}
          disabled={!currentImage || !canvasReady}
        />
        Complete
      </label>
      <select
        className="tcip-select text-[11px]"
        value={imageStatus.activeFilter}
        onChange={(e) => setStatusFilter(e.target.value as "all" | ImageStatus)}
        title="Status filter"
      >
        {STATUS_FILTERS.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>

      <div className="flex-1" />

      {/* Prev / counter / next — filtered position within the status filter */}
      <button
        className="tcip-btn text-[11px]"
        onClick={() => nav.stepImage(-1)}
        disabled={!nav.canPrev}
      >
        ◀&nbsp;&nbsp;Prev
      </button>
      <input
        ref={counterRef}
        className="tcip-input w-12 text-center font-mono text-[11px]"
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
      <span className="text-[11px] text-tcip-muted tabular-nums">/ {nav.total}</span>
      <button
        className="tcip-btn text-[11px]"
        onClick={() => nav.stepImage(1)}
        disabled={!nav.canNext}
      >
        Next&nbsp;&nbsp;▶
      </button>

      {/* Image name */}
      <div className="text-[11px] text-tcip-muted font-mono max-w-[200px] truncate ml-2">
        {currentImage}
      </div>

      {pickerOpen && activeEntry && (
        <ColorPickerModal
          title={`Color for ${activeEntry.id}: ${activeEntry.name}`}
          initialColor={activeEntry.color}
          onSubmit={commitColor}
          onCancel={() => setPickerOpen(false)}
        />
      )}
    </div>
  );
}
