import { useMemo, useRef, useState } from "react";

import { classesApi, type ImageStatus } from "@/api/classes";
import { ColorPickerModal } from "@/components/ColorPickerModal";
import { useStore } from "@/store";
import type { TabName } from "@/store/types";

const TABS: { id: TabName; label: string }[] = [
  { id: "annotate", label: "Annotate" },
  { id: "review", label: "Review" },
  { id: "training", label: "Training" },
  { id: "tuning", label: "Tuning" },
  { id: "inference", label: "Inference" },
  { id: "results", label: "Results" },
];

const STATUS_FILTERS: { value: "all" | ImageStatus; label: string }[] = [
  { value: "all", label: "All" },
  { value: "complete", label: "Complete" },
  { value: "partial", label: "Partial" },
  { value: "unannotated", label: "Unannotated" },
];

export function TopBar() {
  const activeTab = useStore((s) => s.gui.active_tab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const dataset = useStore((s) => s.gui.dataset);
  const patchGui = useStore((s) => s.patchGui);
  const mode = useStore((s) => s.gui.mode);
  const setMode = useStore((s) => s.setMode);
  const activeClass = useStore((s) => s.gui.active_class);
  const setActiveClass = useStore((s) => s.setActiveClass);
  const classes = useStore((s) => s.classes.list);
  const upsertClass = useStore((s) => s.upsertClass);
  const classColor = useStore((s) => s.classColor);
  const wsStatus = useStore((s) => s.wsStatus);
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

  // Counts per class, based on current canvas state (Annotate tab).
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
  const totalVisible = dataset.image_list.length;
  const currentStatus: ImageStatus | undefined = currentImage
    ? imageStatus.byImage[currentImage]
    : undefined;

  function jumpToIndex(oneBased: number) {
    if (!dataset.image_list.length) return;
    const idx = Math.max(1, Math.min(dataset.image_list.length, oneBased)) - 1;
    patchGui({ dataset: { ...dataset, current_image_index: idx } });
  }

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
    const { color } = await classesApi.autoColor(nextId);
    const entry = { id: nextId, name: trimmed, color };
    upsertClass(entry);
    setActiveClass(nextId);
    if (dataset.project_root) {
      await classesApi.save(dataset.project_root, [...classes, entry]);
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
      await classesApi.save(dataset.project_root, next);
    }
  }

  async function toggleComplete(next: boolean) {
    if (!currentImage || !dataset.project_root) return;
    const newStatus: ImageStatus = next
      ? "complete"
      : canvasBoxes.length + canvasPolygons.length > 0
      ? "partial"
      : "unannotated";
    setImageStatus(currentImage, newStatus);
    await classesApi.setImageStatus(dataset.project_root, currentImage, newStatus);
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
      <div className="font-semibold tracking-wide mr-2 text-tcip-fg">TCIP</div>

      {/* Tabs */}
      <div className="flex items-center gap-1 mr-4">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`px-3 h-7 rounded text-[12px] ${
              activeTab === t.id
                ? "bg-tcip-accent text-white"
                : "bg-transparent text-tcip-fg hover:bg-tcip-border"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Annotate-only center controls */}
      {activeTab === "annotate" && (
        <>
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
              className={`px-2 h-7 text-[11px] ${mode === "box" ? "bg-tcip-accent text-white" : "bg-tcip-panel text-tcip-fg"}`}
            >
              Box
            </button>
            <button
              onClick={() => setMode("polygon")}
              className={`px-2 h-7 text-[11px] ${mode === "polygon" ? "bg-tcip-accent text-white" : "bg-tcip-panel text-tcip-fg"}`}
            >
              Polygon ⬡
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
          <label className="flex items-center gap-1 text-[11px] ml-3">
            <input
              type="checkbox"
              checked={currentStatus === "complete"}
              onChange={(e) => void toggleComplete(e.target.checked)}
              disabled={!currentImage}
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
        </>
      )}

      <div className="flex-1" />

      {/* Dataset breadcrumb */}
      <div className="text-[11px] text-tcip-muted truncate max-w-md">
        {dataset.dataset_root && dataset.date ? (
          <>
            {dataset.dataset_root.split(/[/\\]/).slice(-2).join("/")} · {dataset.date}
          </>
        ) : (
          "no dataset selected"
        )}
      </div>

      {/* Prev / counter / next */}
      <button
        className="tcip-btn text-[11px]"
        onClick={() => jumpToIndex(dataset.current_image_index)}
        disabled={!dataset.image_list.length || dataset.current_image_index <= 0}
      >
        ◀ Prev
      </button>

      <input
        ref={counterRef}
        className="tcip-input w-12 text-center font-mono text-[11px]"
        value={counterDraft ?? String(dataset.current_image_index + 1)}
        onChange={(e) => setCounterDraft(e.target.value.replace(/[^0-9]/g, ""))}
        onFocus={() => setCounterDraft(String(dataset.current_image_index + 1))}
        onBlur={() => setCounterDraft(null)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            const num = parseInt(counterDraft ?? "", 10);
            if (!Number.isNaN(num)) jumpToIndex(num);
            setCounterDraft(null);
            counterRef.current?.blur();
          } else if (e.key === "Escape") {
            setCounterDraft(null);
            counterRef.current?.blur();
          }
        }}
      />
      <span className="text-[11px] text-tcip-muted tabular-nums">/ {totalVisible}</span>

      <button
        className="tcip-btn text-[11px]"
        onClick={() => jumpToIndex(dataset.current_image_index + 2)}
        disabled={!dataset.image_list.length || dataset.current_image_index >= totalVisible - 1}
      >
        Next ▶
      </button>

      {/* Image name */}
      <div className="text-[11px] text-tcip-muted font-mono max-w-[200px] truncate ml-2">
        {currentImage}
      </div>

      {/* WS pill */}
      <div className="flex items-center gap-1 text-[11px] ml-2">
        <span
          className={`w-2 h-2 rounded-full ${
            wsStatus === "connected"
              ? "bg-tcip-tp"
              : wsStatus === "connecting"
              ? "bg-tcip-fn"
              : "bg-tcip-fp"
          }`}
        />
        <span className="text-tcip-muted">{wsStatus}</span>
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
