/**
 * Annotate-tab context toolbar. Two rows matching the approved mockup:
 *   Row 1  — draw mode (Box/Polygon), the class picker pill, an Editor toggle, then the
 *            nav filter, image navigation, and the Complete checkbox.
 *   Editor — a second toolbar (collapsed by default, remembered) holding the tools you
 *            flip constantly (Snap / Stream / Show labels) plus Undo / Redo / Save.
 * Lives directly under the global TopBar; Undo/Redo/Save are wired up from AnnotateTab.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { classesApi, type ImageStatus } from "@/api/classes";
import { ColorPickerModal } from "@/components/ColorPickerModal";
import { useImageNav } from "@/hooks/useImageNav";
import { useStore } from "@/store";

const STATUS_FILTERS: { value: "all" | ImageStatus; label: string }[] = [
  { value: "all", label: "All" },
  { value: "partial", label: "Partial" },
  { value: "complete", label: "Complete" },
  { value: "negative", label: "Negative" },
  { value: "unannotated", label: "Unannotated" },
];

/** A pressed-state tool button with a status dot, matching the mockup's Editor tools. */
function Etool({
  label,
  pressed,
  onClick,
  disabled,
  title,
}: {
  label: string;
  pressed: boolean;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`flex h-7 items-center gap-2 rounded border px-3 text-[12px] transition-colors disabled:opacity-40 ${
        pressed
          ? "border-tcip-accent/55 bg-tcip-accent/20 text-tcip-fg"
          : "border-tcip-border bg-tcip-bg text-tcip-muted hover:border-tcip-border-hover hover:text-tcip-fg"
      }`}
    >
      <span
        className={`h-2 w-2 rounded-full ${pressed ? "bg-tcip-accent" : "bg-tcip-muted"}`}
        aria-hidden
      />
      {label}
    </button>
  );
}

export function AnnotateToolbar({
  onSave,
  saveDisabled,
  dirty,
}: {
  onSave: () => void;
  saveDisabled: boolean;
  dirty: boolean;
}) {
  const dataset = useStore((s) => s.gui.dataset);
  const mode = useStore((s) => s.gui.mode);
  const setMode = useStore((s) => s.setMode);
  const activeClass = useStore((s) => s.gui.active_class);
  const setActiveClass = useStore((s) => s.setActiveClass);
  const classes = useStore((s) => s.classes.list);
  const upsertClass = useStore((s) => s.upsertClass);
  const canvasBoxes = useStore((s) => s.canvas.boxes);
  const canvasPolygons = useStore((s) => s.canvas.polygons);
  const annotateUi = useStore((s) => s.annotateUi);
  const setVisible = useStore((s) => s.setVisible);
  const setSnap = useStore((s) => s.setSnap);
  const setStream = useStore((s) => s.setStream);
  const imageStatus = useStore((s) => s.imageStatus);
  const setStatusFilter = useStore((s) => s.setStatusFilter);
  const setImageStatus = useStore((s) => s.setImageStatus);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);

  // Editor shelf: collapsed by default, remembered across sessions.
  const [editorOpen, setEditorOpen] = useState<boolean>(() => {
    try {
      return localStorage.getItem("tcip.annotate.editorOpen") === "1";
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem("tcip.annotate.editorOpen", editorOpen ? "1" : "0");
    } catch {
      /* storage disabled — the shelf just won't persist */
    }
  }, [editorOpen]);

  const [classMenuOpen, setClassMenuOpen] = useState(false);
  const [pickerClassId, setPickerClassId] = useState<number | null>(null);
  const [counterDraft, setCounterDraft] = useState<string | null>(null);
  const counterRef = useRef<HTMLInputElement | null>(null);

  // Counts per class from the current canvas state (box mode counts boxes, else polygons).
  const classCounts = useMemo(() => {
    const counts = new Map<number, number>();
    const src = mode === "box" ? canvasBoxes : canvasPolygons;
    for (const s of src) counts.set(s.class_id, (counts.get(s.class_id) ?? 0) + 1);
    return counts;
  }, [mode, canvasBoxes, canvasPolygons]);

  const currentImage = dataset.image_list[dataset.current_image_index] ?? "—";
  const currentStatus: ImageStatus | undefined = currentImage
    ? imageStatus.byImage[currentImage]
    : undefined;
  const loadedImagePath = useStore((s) => s.canvas.loadedImagePath);
  const canvasReady =
    !!dataset.dataset_root &&
    !!dataset.date &&
    loadedImagePath === `${dataset.dataset_root}/images/${dataset.date}/${currentImage}`;
  const nav = useImageNav();

  const activeEntry = classes.find((c) => c.id === activeClass);
  const activeCount = classCounts.get(activeClass) ?? 0;

  async function addNewClass() {
    const name = window.prompt("New class name:");
    if (!name) return;
    const trimmed = name.trim();
    if (!trimmed) return;
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
        await classesApi.save(
          dataset.project_root, dataset.annotation_type, [...classes, entry],
          dataset.dataset_root, dataset.annotations_detect_dir, dataset.annotations_segment_dir);
      }
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not add class: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function commitColor(newColor: string) {
    const id = pickerClassId;
    setPickerClassId(null);
    const entry = classes.find((c) => c.id === id);
    if (!entry) return;
    const updated = { ...entry, color: newColor };
    upsertClass(updated);
    if (dataset.project_root) {
      const next = classes.map((c) => (c.id === entry.id ? updated : c));
      try {
        await classesApi.save(
          dataset.project_root, dataset.annotation_type, next,
          dataset.dataset_root, dataset.annotations_detect_dir, dataset.annotations_segment_dir);
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
    const newStatus: ImageStatus = next
      ? hasContent
        ? "complete"
        : "negative"
      : hasContent
        ? "partial"
        : "unannotated";
    setImageStatus(currentImage, newStatus);
    try {
      await classesApi.setImageStatus(
        dataset.project_root,
        currentImage,
        newStatus,
        dataset.annotation_type,
        dataset.date,
      );
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not update status: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  const pickerEntry = classes.find((c) => c.id === pickerClassId);

  return (
    <div className="shrink-0 border-b border-tcip-border bg-tcip-panel">
      {/* Row 1 — mode + class + Editor toggle, then navigation */}
      <div className="h-topbar flex items-center gap-3 px-3">
        {/* Draw mode */}
        <div
          className="inline-flex gap-0.5 rounded border border-tcip-border bg-tcip-bg p-0.5"
          role="group"
          aria-label="Draw mode"
        >
          <button
            aria-pressed={mode === "box"}
            onClick={() => setMode("box")}
            className={`flex h-6 items-center gap-1.5 rounded-[4px] px-2.5 text-[12px] font-semibold transition-colors ${
              mode === "box" ? "bg-tcip-accent text-white" : "text-tcip-muted hover:text-tcip-fg"
            }`}
          >
            <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
              <rect
                x="2.5"
                y="3.5"
                width="11"
                height="9"
                rx="1"
                stroke="currentColor"
                strokeWidth="1.6"
              />
            </svg>
            Box
          </button>
          <button
            aria-pressed={mode === "polygon"}
            onClick={() => setMode("polygon")}
            className={`flex h-6 items-center gap-1.5 rounded-[4px] px-2.5 text-[12px] font-semibold transition-colors ${
              mode === "polygon"
                ? "bg-tcip-accent text-white"
                : "text-tcip-muted hover:text-tcip-fg"
            }`}
          >
            <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
              <path
                d="M8 2l5 3.5-2 6H5l-2-6z"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinejoin="round"
              />
            </svg>
            Polygon
          </button>
        </div>

        {/* Class picker pill */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setClassMenuOpen((o) => !o)}
            aria-expanded={classMenuOpen}
            disabled={!activeEntry}
            className="flex h-[30px] items-center gap-2 rounded border border-tcip-border bg-tcip-bg px-2.5 text-[12px] text-tcip-fg hover:border-tcip-border-hover disabled:opacity-50"
          >
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={{ background: activeEntry?.color ?? "#666" }}
              aria-hidden
            />
            <span className="font-semibold">{activeEntry?.name ?? "—"}</span>
            <span className="font-mono text-tcip-muted">({activeCount})</span>
            <svg viewBox="0 0 10 10" width="9" height="9" fill="none" aria-hidden="true">
              <path
                d="M2 4l3 3 3-3"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-tcip-muted"
              />
            </svg>
          </button>
          {classMenuOpen && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setClassMenuOpen(false)} />
              <div className="absolute left-0 top-full z-20 mt-1 w-56 rounded-md border border-tcip-border bg-tcip-panel py-1 text-[12px] shadow-lg">
                {classes.map((c) => (
                  <div key={c.id} className="flex items-center gap-2 px-2 hover:bg-tcip-hover">
                    <button
                      type="button"
                      title="Edit colour"
                      aria-label={`Edit colour for ${c.name}`}
                      onClick={() => {
                        setClassMenuOpen(false);
                        setPickerClassId(c.id);
                      }}
                      className="h-3.5 w-3.5 shrink-0 rounded-sm border border-tcip-border"
                      style={{ background: c.color }}
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setActiveClass(c.id);
                        setClassMenuOpen(false);
                      }}
                      className="flex flex-1 items-center py-1 text-left text-tcip-fg"
                    >
                      <span className={c.id === activeClass ? "font-semibold" : ""}>{c.name}</span>
                      <span className="ml-auto font-mono text-tcip-muted">
                        {classCounts.get(c.id) ?? 0}
                      </span>
                    </button>
                  </div>
                ))}
                <div className="mt-1 border-t border-tcip-border pt-1">
                  <button
                    type="button"
                    onClick={() => {
                      setClassMenuOpen(false);
                      void addNewClass();
                    }}
                    className="w-full px-2 py-1 text-left text-tcip-accent hover:bg-tcip-hover"
                  >
                    + New class
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Editor toggle — drops the second toolbar */}
        <button
          type="button"
          onClick={() => setEditorOpen((o) => !o)}
          aria-expanded={editorOpen}
          className={`flex h-[30px] items-center gap-2 rounded border px-3 text-[12px] font-semibold transition-colors ${
            editorOpen
              ? "border-tcip-border bg-tcip-hover text-tcip-fg"
              : "border-tcip-border bg-tcip-bg text-tcip-muted hover:border-tcip-border-hover hover:text-tcip-fg"
          }`}
        >
          <svg
            viewBox="0 0 16 16"
            width="11"
            height="11"
            fill="none"
            aria-hidden="true"
            className={`transition-transform ${editorOpen ? "rotate-90" : ""}`}
          >
            <path
              d="M6 4l4 4-4 4"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Editor
        </button>

        <div className="flex-1" />

        {/* Nav filter */}
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-bold uppercase tracking-wide text-tcip-muted">
            Filter
          </span>
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
        </div>

        {/* Image navigation */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold uppercase tracking-wide text-tcip-muted">
            Image
          </span>
          <span
            className="max-w-[150px] truncate font-mono text-[11px] text-tcip-fg"
            title={currentImage}
          >
            {currentImage}
          </span>
          <button
            className="tcip-btn text-[11px]"
            onClick={() => nav.stepImage(-1)}
            disabled={!nav.canPrev}
            aria-label="Previous image"
          >
            ◀
          </button>
          <input
            ref={counterRef}
            className="tcip-input w-10 text-center font-mono text-[11px]"
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
          <span className="font-mono text-[11px] tabular-nums text-tcip-muted">/ {nav.total}</span>
          <button
            className="tcip-btn text-[11px]"
            onClick={() => nav.stepImage(1)}
            disabled={!nav.canNext}
            aria-label="Next image"
          >
            ▶
          </button>
        </div>

        {/* Complete */}
        <label
          className="flex items-center gap-1.5 text-[12px]"
          title={canvasReady ? undefined : "Loading this image's labels…"}
        >
          <input
            type="checkbox"
            checked={currentStatus === "complete" || currentStatus === "negative"}
            onChange={(e) => void toggleComplete(e.target.checked)}
            disabled={!currentImage || !canvasReady}
          />
          Complete
        </label>
      </div>

      {/* Editor second toolbar — the tools, plus Undo / Redo / Save */}
      {editorOpen && (
        <div className="flex items-center gap-3 border-t border-tcip-border px-3 py-2">
          <div className="flex items-center gap-2.5">
            <Etool
              label="Snap"
              pressed={annotateUi.snap}
              onClick={() => setSnap(!annotateUi.snap)}
              disabled={mode !== "polygon"}
              title="Snap to nearest vertex (s)"
            />
            <Etool
              label="Stream"
              pressed={annotateUi.stream}
              onClick={() => setStream(!annotateUi.stream)}
              disabled={mode !== "polygon"}
              title="Freehand: click to start laying vertices, click to pause, double-click to close (v)"
            />
            <Etool
              label="Show labels"
              pressed={annotateUi.visible}
              onClick={() => setVisible(!annotateUi.visible)}
              title="Show or hide annotation overlays"
            />
          </div>
          <div className="flex-1" />
          <div className="flex items-center gap-2">
            <button className="tcip-btn text-[12px]" onClick={() => undo()} title="Undo (Ctrl+Z)">
              ↶&nbsp;&nbsp;Undo
            </button>
            <button
              className="tcip-btn text-[12px]"
              onClick={() => redo()}
              title="Redo (Ctrl+Shift+Z)"
            >
              ↷&nbsp;&nbsp;Redo
            </button>
            <button
              className={dirty ? "tcip-btn-primary text-[12px]" : "tcip-btn text-[12px]"}
              onClick={onSave}
              disabled={saveDisabled}
              title="Save (Ctrl+S) — also auto-saves on image change"
            >
              {dirty ? "Save" : "Saved"}
            </button>
          </div>
        </div>
      )}

      {pickerEntry && (
        <ColorPickerModal
          title={`Color for ${pickerEntry.id}: ${pickerEntry.name}`}
          initialColor={pickerEntry.color}
          onSubmit={commitColor}
          onCancel={() => setPickerClassId(null)}
        />
      )}
    </div>
  );
}
