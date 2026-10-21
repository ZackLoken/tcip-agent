/**
 * Annotate-tab toolbar, one row: the tool mode (Point, Box, Polygon), the engine prompt, the
 * subject picker, the Edit menu (Snap, Stream, Cut, Show labels, Undo, Redo), Save, the band
 * picker for a multispectral image, then the image stepper and the Complete control.
 */

import { useMemo, useState, type ReactNode } from "react";

import type { ImageBandsResponse } from "@/api/client";
import { subjectColor } from "@/api/subjects";
import type { PhrasePrompt } from "@/api/types.generated";
import { EnginePrompt } from "@/components/annotate/EnginePrompt";
import { BandPicker } from "@/components/BandPicker";
import { Stepper } from "@/components/Stepper";
import { useImageNav } from "@/hooks/useImageNav";
import { showsBandPicker, type BandSelection } from "@/lib/bandSelection";
import { saveRegistry } from "@/lib/registrySave";
import { useSubjectColors } from "@/lib/subjectColors";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";
import { isFinished, type SubjectState } from "@/store/types";

/** A button that drops a menu beneath itself; the menu closes on any click outside it. */
function MenuButton({
  label,
  pressed,
  children,
}: {
  label: string;
  pressed?: boolean;
  children: (close: () => void) => ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className={`flex h-[30px] items-center gap-2 rounded border px-2.5 font-semibold transition-colors ${
          open || pressed
            ? "border-tcip-border bg-tcip-hover text-tcip-fg"
            : "border-tcip-border bg-tcip-bg text-tcip-muted hover:border-tcip-border-hover hover:text-tcip-fg"
        }`}
      >
        {label}
      </button>
      {open && (
        <>
          <div role="presentation" className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div
            role="menu"
            className="absolute left-0 top-full z-20 mt-1 min-w-44 rounded-md border border-tcip-border bg-tcip-panel py-1 shadow-lg"
          >
            {children(() => setOpen(false))}
          </div>
        </>
      )}
    </div>
  );
}

/** One menu row: a tool that stays pressed, or an action that closes the menu. */
function MenuItem({
  label,
  onClick,
  pressed,
  disabled,
  title,
}: {
  label: string;
  onClick: () => void;
  pressed?: boolean;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      role={pressed === undefined ? "menuitem" : "menuitemcheckbox"}
      aria-checked={pressed}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-tcip-fg hover:bg-tcip-hover disabled:opacity-40 disabled:hover:bg-transparent"
    >
      {pressed !== undefined && (
        <span
          className={`h-2 w-2 rounded-full ${pressed ? "bg-tcip-accent" : "bg-tcip-border"}`}
          aria-hidden
        />
      )}
      {label}
    </button>
  );
}

const MODE_ICONS = {
  point: (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
      <path
        d="M8 1.6v2.7M8 11.7v2.7M1.6 8h2.7M11.7 8h2.7"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="8" cy="8" r="2" fill="currentColor" />
    </svg>
  ),
  box: (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
      <rect x="2.5" y="3.5" width="11" height="9" rx="1" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  ),
  polygon: (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
      <path
        d="M8 2l5 3.5-2 6H5l-2-6z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  ),
} as const;

const MODE_TITLES = {
  point: "Point: click to place, drag to move, right-click to remove",
  box: "Box: drag to draw",
  polygon: "Polygon: click each vertex, double-click to close",
} as const;

export function AnnotateToolbar({
  onSave,
  onCancelSave,
  saveDisabled,
  dirty,
  bandsInfo,
  bandSelection,
  onBandSelectionChange,
  subjectState,
  onComplete,
  onCompleteView,
  proposing,
  onPhrase,
  onPropose,
}: {
  onSave: () => void;
  /** Abandon the save in flight, which the Save button offers while one is. */
  onCancelSave: () => void;
  saveDisabled: boolean;
  dirty: boolean;
  // Band-composite picker (multispectral only): omitted/null for a standard RGB dataset.
  bandsInfo?: ImageBandsResponse | null;
  bandSelection?: BandSelection | null;
  onBandSelectionChange?: (next: BandSelection) => void;
  // The dataset subject's state on this image as the backend serves it; null until loaded.
  subjectState: SubjectState | null;
  // Mark the dataset subject complete over the whole image (true) or withdraw its marks (false).
  onComplete: (next: boolean) => void;
  // Mark the dataset subject complete over the region in view; omitted when the view holds it all.
  onCompleteView?: () => void;
  // Whether an engine prompt is in flight, the send of a phrase, and the send of the held prompts.
  proposing: boolean;
  onPhrase: (prompt: PhrasePrompt) => void;
  onPropose: () => void;
}) {
  const dataset = useStore((s) => s.gui.dataset);
  const projectRoot = useStore(selectProjectRoot);
  const mode = useStore((s) => s.gui.mode);
  const setMode = useStore((s) => s.setMode);
  const activeSubject = useStore((s) => s.gui.active_subject);
  const setActiveSubject = useStore((s) => s.setActiveSubject);
  const registry = useStore((s) => s.registry.subjects);
  const discovered = useStore((s) => s.registry.discovered);
  const canvasBoxes = useStore((s) => s.canvas.boxes);
  const canvasPolygons = useStore((s) => s.canvas.polygons);
  const canvasPoints = useStore((s) => s.canvas.points);
  const canvasImageAnnotations = useStore((s) => s.canvas.imageAnnotations);
  const saving = useStore((s) => s.canvas.saving !== null);
  const annotateUi = useStore((s) => s.annotateUi);
  const setVisible = useStore((s) => s.setVisible);
  const setSnap = useStore((s) => s.setSnap);
  const setStream = useStore((s) => s.setStream);
  const setCut = useStore((s) => s.setCut);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);

  const subjectNames = useMemo(() => Object.keys(registry), [registry]);
  useSubjectColors(); // re-render on a recolor: the swatches below call subjectColor() fresh

  const [subjectMenuOpen, setSubjectMenuOpen] = useState(false);

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

  const currentImage = dataset.image_list[dataset.current_image_index] ?? null;
  const nav = useImageNav();
  const finished = isFinished(subjectState);
  const polygonTools = mode === "polygon";

  const activeCount = activeSubject ? (subjectCounts.get(activeSubject) ?? 0) : 0;

  async function addNewSubject() {
    const found = discovered.filter((n) => !subjectNames.includes(n));
    const name = window.prompt(
      found.length
        ? `New subject (in this dataset's labels: ${found.join(", ")}):`
        : "New subject:",
    );
    if (!name) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    if (subjectNames.includes(trimmed)) {
      setActiveSubject(trimmed);
      return;
    }
    const previousSubject = activeSubject;
    setActiveSubject(trimmed);
    const landed = await saveRegistry({ ...registry, [trimmed]: {} }, "Could not add the subject");
    if (!landed && projectRoot && dataset.dataset_root) setActiveSubject(previousSubject);
  }

  return (
    <div className="shrink-0 border-b border-tcip-border bg-tcip-panel">
      <div className="h-topbar flex items-center gap-3 px-3">
        <div
          className="inline-flex gap-0.5 rounded border border-tcip-border bg-tcip-bg p-0.5"
          role="group"
          aria-label="Tool"
        >
          {(["point", "box", "polygon"] as const).map((tool) => (
            <button
              key={tool}
              aria-pressed={mode === tool}
              onClick={() => setMode(tool)}
              title={MODE_TITLES[tool]}
              className={`flex h-7 items-center gap-1.5 rounded-[4px] px-2.5 font-semibold transition-colors ${
                mode === tool ? "bg-tcip-accent text-white" : "text-tcip-muted hover:text-tcip-fg"
              }`}
            >
              {MODE_ICONS[tool]}
              {tool.charAt(0).toUpperCase() + tool.slice(1)}
            </button>
          ))}
        </div>

        <EnginePrompt proposing={proposing} onPhrase={onPhrase} onPropose={onPropose} />

        {/* Subject picker */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setSubjectMenuOpen((o) => !o)}
            aria-expanded={subjectMenuOpen}
            className="flex h-[30px] items-center gap-2 rounded border border-tcip-border bg-tcip-bg px-2.5 text-tcip-fg hover:border-tcip-border-hover"
          >
            <span
              className="h-2.5 w-2.5 rounded-xs"
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
              <div
                role="presentation"
                className="fixed inset-0 z-10"
                onClick={() => setSubjectMenuOpen(false)}
              />
              <div className="absolute left-0 top-full z-20 mt-1 w-56 rounded-md border border-tcip-border bg-tcip-panel py-1 shadow-lg">
                {subjectNames.map((name) => (
                  <div key={name} className="flex items-center gap-2 px-2 hover:bg-tcip-hover">
                    <span
                      className="h-3.5 w-3.5 shrink-0 rounded-xs border border-tcip-border"
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
                    New subject
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        <MenuButton label="Edit" pressed={annotateUi.snap || annotateUi.stream || annotateUi.cut}>
          {(close) => (
            <>
              <MenuItem
                label="Snap"
                pressed={annotateUi.snap}
                onClick={() => setSnap(!annotateUi.snap)}
                disabled={!polygonTools}
                title="Snap to the nearest vertex (s)"
              />
              <MenuItem
                label="Stream"
                pressed={annotateUi.stream}
                onClick={() => setStream(!annotateUi.stream)}
                disabled={!polygonTools}
                title="Freehand vertices (v)"
              />
              <MenuItem
                label="Cut"
                pressed={annotateUi.cut}
                onClick={() => setCut(!annotateUi.cut)}
                disabled={!polygonTools}
                title={
                  polygonTools
                    ? "Click two points on either side of the selected polygon to split it (x)"
                    : "Cut: in polygon mode only"
                }
              />
              <div className="my-1 border-t border-tcip-border" />
              <MenuItem
                label="Show labels"
                pressed={annotateUi.visible}
                onClick={() => setVisible(!annotateUi.visible)}
              />
              <div className="my-1 border-t border-tcip-border" />
              <MenuItem
                label="Undo"
                onClick={() => {
                  undo();
                  close();
                }}
                title="Ctrl+Z"
              />
              <MenuItem
                label="Redo"
                onClick={() => {
                  redo();
                  close();
                }}
                title="Ctrl+Shift+Z"
              />
            </>
          )}
        </MenuButton>

        <button
          className={dirty ? "tcip-btn-primary h-[30px]" : "tcip-btn h-[30px]"}
          onClick={saving ? onCancelSave : onSave}
          disabled={saveDisabled && !saving}
          title={saving ? "Click to cancel this save" : "Save (Ctrl+S)"}
        >
          {saving ? "Saving" : dirty ? "Save" : "Saved"}
        </button>

        {bandsInfo &&
          bandSelection &&
          onBandSelectionChange &&
          showsBandPicker(bandsInfo, bandSelection) && (
            <BandPicker
              bandCount={bandsInfo.band_count}
              bands={bandsInfo.bands}
              selection={bandSelection}
              onChange={onBandSelectionChange}
              sampled={bandsInfo.sampled}
              pixelFraction={bandsInfo.pixel_fraction}
              overviewSize={bandsInfo.overview_size}
            />
          )}

        <div className="flex-1" />

        <Stepper
          label="Image"
          noun="image"
          name={currentImage}
          position={nav.position}
          total={nav.total}
          canPrev={nav.canPrev}
          canNext={nav.canNext}
          onStep={nav.stepImage}
          onJump={nav.jumpToPosition}
        />

        <label
          className="flex items-center gap-1.5"
          title={
            !dataset.subject
              ? "Select a subject first"
              : subjectState !== null
                ? `Every ${dataset.subject} on this image is annotated (c)`
                : "Loading this image's labels"
          }
        >
          <input
            type="checkbox"
            checked={finished}
            onChange={(e) => onComplete(e.target.checked)}
            disabled={!currentImage || subjectState === null || !dataset.subject}
          />
          {subjectState === "negative" ? "Complete (none)" : "Complete"}
        </label>
        {onCompleteView && (
          <MenuButton label="▾">
            {(close) => (
              <MenuItem
                label="Complete the region in view"
                onClick={() => {
                  onCompleteView();
                  close();
                }}
                disabled={!dataset.subject || subjectState === null}
              />
            )}
          </MenuButton>
        )}
      </div>
    </div>
  );
}
