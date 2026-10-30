/**
 * Annotate-tab toolbar, one row in three groups: the tool mode (Point, Box, Polygon) with the
 * engine prompt; the subject picker, the Edit menu (Snap, Stream, Cut, Undo, Redo), the View menu
 * (Labels, Proposals, Predictions), Save and the band picker for a multispectral image; then the
 * image stepper with its filter by annotation status, and the Complete control.
 */

import { useMemo, useState, type ReactNode } from "react";

import { subjectColor } from "@/api/subjects";
import type { ImageBands } from "@/api/types.generated";
import { AttributePanel } from "@/components/annotate/AttributePanel";
import { EnginePrompt } from "@/components/annotate/EnginePrompt";
import { BandPicker } from "@/components/BandPicker";
import { Stepper } from "@/components/Stepper";
import { useImageNav } from "@/hooks/useImageNav";
import { ANNOTATE_KEYS } from "@/lib/annotateKeys";
import { showsBandPicker, type BandSelection } from "@/lib/bandSelection";
import {
  filteredIndices,
  IMAGE_FILTER_WORDS,
  IMAGE_FILTERS,
  type ImageFilter,
} from "@/lib/imageFilter";
import { currentImage } from "@/lib/paths";
import { saveRegistry } from "@/lib/registrySave";
import type { ReviewItem } from "@/lib/reviewItems";
import { useSubjectColors } from "@/lib/subjectColors";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";
import { isFinished, type AnnotateView, type SubjectState } from "@/store/types";

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

/** A thin vertical rule between the toolbar's groups. */
function Divider() {
  return <div className="h-5 w-px shrink-0 bg-tcip-border" aria-hidden />;
}

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
  proposing,
  onPhrase,
  onPropose,
  view,
  onView,
  predictionsAvailable,
  focused,
}: {
  onSave: () => void;
  /** Abandon the save in flight, which the Save button offers while one is. */
  onCancelSave: () => void;
  saveDisabled: boolean;
  dirty: boolean;
  // Band-composite picker (multispectral only): omitted/null for a standard RGB dataset.
  bandsInfo?: ImageBands | null;
  bandSelection?: BandSelection | null;
  onBandSelectionChange?: (next: BandSelection) => void;
  // The dataset subject's state on this image as the backend serves it; null until loaded.
  subjectState: SubjectState | null;
  // Mark the dataset subject complete over the whole image (true) or withdraw its marks (false).
  onComplete: (next: boolean) => void;
  // Whether an engine call is in flight, the find by phrase, and the send of the held prompts.
  proposing: boolean;
  onPhrase: (phrase: string) => void;
  onPropose: () => void;
  // Which overlays the canvas shows beside the labels.
  view: AnnotateView;
  onView: (next: AnnotateView) => void;
  // Whether a predictions bucket is selected, so the Predictions entry has something to show.
  predictionsAvailable: boolean;
  // The focused item, whose attributes the subject menu edits when it is an annotation.
  focused: ReviewItem | null;
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
  const setImageFilter = useStore((s) => s.setImageFilter);
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

  const imageName = currentImage(dataset)?.name ?? null;
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
    const description = window.prompt(`What is one ${trimmed}? (its description)`);
    if (description === null) return;
    const previousSubject = activeSubject;
    setActiveSubject(trimmed);
    const landed = await saveRegistry(
      { ...registry, [trimmed]: { description } },
      "Could not add the subject",
    );
    if (!landed && projectRoot && dataset) setActiveSubject(previousSubject);
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

        <Divider />

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
              <div className="absolute left-0 top-full z-20 mt-1 max-h-[75vh] w-72 overflow-auto rounded-md border border-tcip-border bg-tcip-panel py-1 shadow-lg">
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
                <AttributePanel focused={focused} />
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

        <MenuButton
          label="View"
          pressed={
            !annotateUi.visible || !view.proposals || (predictionsAvailable && !view.predictions)
          }
        >
          {() => (
            <>
              <MenuItem
                label="Labels"
                pressed={annotateUi.visible}
                onClick={() => setVisible(!annotateUi.visible)}
              />
              <MenuItem
                label="Proposals"
                pressed={view.proposals}
                onClick={() => onView({ ...view, proposals: !view.proposals })}
                title={`The proposals staged on this image (${ANNOTATE_KEYS.hideProposals.label})`}
              />
              <MenuItem
                label="Predictions"
                pressed={view.predictions}
                onClick={() => onView({ ...view, predictions: !view.predictions })}
                disabled={!predictionsAvailable}
                title={
                  predictionsAvailable
                    ? `The selected model's predictions (${ANNOTATE_KEYS.hideProposals.label})`
                    : "Choose predictions on the project card"
                }
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
              info={bandsInfo}
              selection={bandSelection}
              onChange={onBandSelectionChange}
            />
          )}

        <div className="flex-1" />

        <Divider />

        <Stepper
          label="Image"
          noun="image"
          name={imageName}
          position={nav.position}
          total={nav.total}
          canPrev={nav.canPrev}
          canNext={nav.canNext}
          onStep={nav.stepImage}
          onJump={nav.jumpToPosition}
        />

        {dataset && (
          <select
            aria-label="Images"
            className="tcip-select h-[30px]"
            value={annotateUi.imageFilter}
            onChange={(e) => setImageFilter(e.target.value as ImageFilter)}
            title={`Which images the stepper steps through, by what the labels say of ${dataset.subject ?? "the subject"}`}
          >
            {IMAGE_FILTERS.map((f) => (
              <option key={f} value={f}>
                {`${IMAGE_FILTER_WORDS[f]} (${filteredIndices(dataset.images, annotateUi.imageStates, f).length})`}
              </option>
            ))}
          </select>
        )}

        <label
          className="flex items-center gap-1.5"
          title={
            !dataset?.subject
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
            disabled={!imageName || subjectState === null || !dataset?.subject}
          />
          {subjectState === "negative" ? "Complete (none)" : "Complete"}
        </label>
      </div>
    </div>
  );
}
