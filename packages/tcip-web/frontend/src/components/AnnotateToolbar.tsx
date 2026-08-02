/**
 * Annotate-tab context toolbar. Two rows matching the approved mockup:
 *   Row 1: draw mode (Point/Box/Polygon), the subject picker pill, an Editor toggle, then the
 *          nav filter, image navigation, and the Complete checkbox.
 *   Editor: a second toolbar (collapsed by default, remembered) holding the tools you
 *           flip constantly (Snap / Stream / Show labels) plus Undo / Redo / Save.
 * Lives directly under the global TopBar; Undo/Redo/Save are wired up from AnnotateTab.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import type { ImageBandsResponse } from "@/api/client";
import { classesApi, subjectColor, type ImageStatus } from "@/api/classes";
import { BandPicker } from "@/components/BandPicker";
import { useImageNav } from "@/hooks/useImageNav";
import type { BandSelection } from "@/lib/bandSelection";
import { useStore } from "@/store";

// Progression order (start state first, terminal states last), matches Review's parallel
// status filter, which already reads Unreviewed before Reviewed.
const STATUS_FILTERS: { value: "all" | ImageStatus; label: string }[] = [
  { value: "all", label: "All" },
  { value: "unannotated", label: "Unannotated" },
  { value: "partial", label: "Partial" },
  { value: "complete", label: "Complete" },
  { value: "negative", label: "Negative" },
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
  isLocked,
  bandsInfo,
  bandSelection,
  onBandSelectionChange,
}: {
  onSave: () => void;
  saveDisabled: boolean;
  dirty: boolean;
  // True when the current image's status is Complete/Negative: edits and saves are blocked.
  isLocked?: boolean;
  // Band-composite picker (multispectral only): omitted/null for a standard RGB dataset.
  bandsInfo?: ImageBandsResponse | null;
  bandSelection?: BandSelection | null;
  onBandSelectionChange?: (next: BandSelection) => void;
}) {
  const dataset = useStore((s) => s.gui.dataset);
  const mode = useStore((s) => s.gui.mode);
  const setMode = useStore((s) => s.setMode);
  const activeSubject = useStore((s) => s.gui.active_subject);
  const setActiveSubject = useStore((s) => s.setActiveSubject);
  const registry = useStore((s) => s.registry.subjects);
  const setRegistry = useStore((s) => s.setRegistry);
  const canvasBoxes = useStore((s) => s.canvas.boxes);
  const canvasPolygons = useStore((s) => s.canvas.polygons);
  const canvasPoints = useStore((s) => s.canvas.points);
  const canvasImageAnnotations = useStore((s) => s.canvas.imageAnnotations);
  const annotateUi = useStore((s) => s.annotateUi);
  const setVisible = useStore((s) => s.setVisible);
  const setSnap = useStore((s) => s.setSnap);
  const setStream = useStore((s) => s.setStream);
  const imageStatus = useStore((s) => s.imageStatus);
  const setStatusFilter = useStore((s) => s.setStatusFilter);
  const setImageStatus = useStore((s) => s.setImageStatus);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);

  const subjectNames = useMemo(() => Object.keys(registry), [registry]);

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
      /* storage disabled, the shelf just won't persist */
    }
  }, [editorOpen]);

  const [subjectMenuOpen, setSubjectMenuOpen] = useState(false);
  const [counterDraft, setCounterDraft] = useState<string | null>(null);
  const counterRef = useRef<HTMLInputElement | null>(null);

  // Counts per subject across everything on the current canvas (boxes + polygons + points + ratings).
  const subjectCounts = useMemo(() => {
    const counts = new Map<string, number>();
    const bump = (subj: string) => counts.set(subj, (counts.get(subj) ?? 0) + 1);
    for (const b of canvasBoxes) bump(b.subject);
    for (const p of canvasPolygons) bump(p.subject);
    for (const p of canvasPoints) bump(p.subject);
    for (const a of canvasImageAnnotations) bump(a.subject);
    return counts;
  }, [canvasBoxes, canvasPolygons, canvasPoints, canvasImageAnnotations]);

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

  const activeCount = activeSubject ? (subjectCounts.get(activeSubject) ?? 0) : 0;

  async function addNewSubject() {
    const name = window.prompt("New subject name:");
    if (!name) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    if (subjectNames.includes(trimmed)) {
      setActiveSubject(trimmed);
      return;
    }
    const next = { ...registry, [trimmed]: {} };
    setRegistry(next);
    setActiveSubject(trimmed);
    if (dataset.project_root) {
      try {
        await classesApi.save(
          dataset.project_root,
          next,
          dataset.dataset_root,
          dataset.annotations_dir,
        );
      } catch (e) {
        useStore
          .getState()
          .pushToast(`Could not add subject: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
  }

  async function toggleComplete(next: boolean) {
    if (!currentImage || !dataset.project_root) return;
    const hasContent =
      canvasBoxes.length +
        canvasPolygons.length +
        canvasPoints.length +
        canvasImageAnnotations.length >
      0;
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
        dataset.subject,
        dataset.date,
        dataset.dataset_root,
        dataset.annotations_dir,
      );
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not update status: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <div className="shrink-0 border-b border-tcip-border bg-tcip-panel">
      {/* Row 1: mode + subject + Editor toggle, then navigation */}
      <div className="h-topbar flex items-center gap-3 px-3">
        {/* Draw mode */}
        <div
          className="inline-flex gap-0.5 rounded border border-tcip-border bg-tcip-bg p-0.5"
          role="group"
          aria-label="Draw mode"
        >
          <button
            aria-pressed={mode === "point"}
            onClick={() => setMode("point")}
            title="Point: click to place one location (a prompt or landmark), drag it to move, right-click to remove"
            className={`flex h-6 items-center gap-1.5 rounded-[4px] px-2.5 text-[12px] font-semibold transition-colors ${
              mode === "point" ? "bg-tcip-accent text-white" : "text-tcip-muted hover:text-tcip-fg"
            }`}
          >
            {/* The canvas mark in miniature: ticks converging on a core, so the tool and the shape
                it authors read as the same thing. */}
            <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
              <path
                d="M8 1.6v2.7M8 11.7v2.7M1.6 8h2.7M11.7 8h2.7"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
              <circle cx="8" cy="8" r="2" fill="currentColor" />
            </svg>
            Point
          </button>
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

        {/* Subject picker pill */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setSubjectMenuOpen((o) => !o)}
            aria-expanded={subjectMenuOpen}
            className="flex h-[30px] items-center gap-2 rounded border border-tcip-border bg-tcip-bg px-2.5 text-[12px] text-tcip-fg hover:border-tcip-border-hover"
          >
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={{ background: activeSubject ? subjectColor(activeSubject) : "#666" }}
              aria-hidden
            />
            <span className="font-semibold">{activeSubject ?? "select subject"}</span>
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
          {subjectMenuOpen && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setSubjectMenuOpen(false)} />
              <div className="absolute left-0 top-full z-20 mt-1 w-56 rounded-md border border-tcip-border bg-tcip-panel py-1 text-[12px] shadow-lg">
                {subjectNames.map((name) => (
                  <div key={name} className="flex items-center gap-2 px-2 hover:bg-tcip-hover">
                    <span
                      className="h-3.5 w-3.5 shrink-0 rounded-sm border border-tcip-border"
                      style={{ background: subjectColor(name) }}
                      aria-hidden
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setActiveSubject(name);
                        setSubjectMenuOpen(false);
                      }}
                      className="flex flex-1 items-center py-1 text-left text-tcip-fg"
                    >
                      <span className={name === activeSubject ? "font-semibold" : ""}>{name}</span>
                      <span className="ml-auto font-mono text-tcip-muted">
                        {subjectCounts.get(name) ?? 0}
                      </span>
                    </button>
                  </div>
                ))}
                <div className="mt-1 border-t border-tcip-border pt-1">
                  <button
                    type="button"
                    onClick={() => {
                      setSubjectMenuOpen(false);
                      void addNewSubject();
                    }}
                    className="w-full px-2 py-1 text-left text-tcip-accent hover:bg-tcip-hover"
                  >
                    + New subject
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Editor toggle: drops the second toolbar */}
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

      {/* Editor second toolbar: the tools, plus Undo / Redo / Save */}
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
          {bandsInfo && bandsInfo.band_count > 3 && bandSelection && onBandSelectionChange && (
            <>
              <span aria-hidden className="h-5 w-px bg-tcip-border" />
              <BandPicker
                bandCount={bandsInfo.band_count}
                bands={bandsInfo.bands}
                selection={bandSelection}
                onChange={onBandSelectionChange}
              />
            </>
          )}
          <div className="flex-1" />
          <div className="flex items-center gap-2">
            {isLocked && (
              <span className="text-[12px] text-tcip-muted">Complete; uncheck to edit</span>
            )}
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
              title="Save (Ctrl+S), also auto-saves on image change"
            >
              {dirty ? "Save" : "Saved"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
