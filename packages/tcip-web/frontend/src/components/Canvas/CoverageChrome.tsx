/**
 * Coverage grid chrome for the open raster: the overlay toggle, a key naming what each overlay
 * mark means, the grid's own derivation line, the attestation control for the cell under the
 * viewport center (or the cell a Map click just opened, while it stays in view), and the errors
 * and previous-lattice facts a breeder needs before trusting or acting on any of it. Shown for
 * every raster the grid route serves a lattice for, single-cell rasters included, so a one-cell
 * image's own attestation is reachable from the GUI; only the Map tool itself stays withdrawn on
 * a single-cell raster (AnnotateToolbar). Floats bottom-right, completing the canvas' floating-
 * chrome grammar (legend bottom-left, attributes top-right, Overview top-left), and collapses
 * like the app's other disclosure panels, remembered per session the way the overlay toggle
 * already is.
 *
 * The status announcement, the read-error notice and the previous-lattice notices render as their
 * own blocks above the collapsible panel, never inside its body: a remembered collapse must never
 * hide a refusal or an announcement from a breeder who reopens the image later in the session.
 *
 * The attestation control's label always states the write it performs and the subject it writes
 * for, read from the raw stored set: "Attest <cell> complete for <subject>" when the cell has
 * never been attested, "Unattest <cell> for <subject>" when it is attested and fresh, "Re-attest
 * <cell> for <subject> (changed since attested)" when it is stale. It is offered only while the
 * overlay is on and can draw, since attesting a cell the breeder cannot see is acting on an unseen
 * state; on a raster with no drawable overlay (a single cell, or no grid at all) the control is
 * offered directly, since there is nothing to hide it behind. When the current grid differs from
 * an existing record's grid, the first press only arms a confirmation (the write would discard
 * every attestation made on the previous lattice) with an explicit Cancel beside it; a second
 * press on the same control performs it. The armed state, its cancellation and every error are
 * announced (role="status") for a screen-reader user who is not looking at the panel.
 */

import { useEffect, useId, useState } from "react";

import { CollapsibleSection } from "@/components/CollapsibleSection";
import type { OtherLatticeAttestation } from "@/hooks/useRegionCompleteness";
import { useDisclosure } from "@/hooks/useDisclosure";
import {
  breederReadErrorReason,
  meetsBar,
  type CellAttestedView,
  type WorkingScale,
} from "@/lib/coverage";
import type { ReplaceRequired } from "@/lib/coverageTracker";

/** The sr-only line naming an attestation's own scale provenance: the view scale it was pressed
 *  at, and whether this image's own coverage record showed the cell already seen at that write
 *  (through `meetsBar`, the one comparison), so a reader can tell an attestation made from a
 *  whole-frame view from one made after the sweep reached the record. */
function attestedViewLine(cell: string, entry: CellAttestedView | undefined): string | null {
  if (!entry) return null;
  const n = entry.view_scale === null ? "an unstated" : `${(entry.view_scale * 100).toFixed(1)}%`;
  const workingScale = entry.working_scale_at_write;
  const atScale = entry.seen_on_record.at_scale;
  if (atScale === null || !entry.seen_on_record.grid_matched) {
    return `${cell} attested at ${n} zoom, not seen on record`;
  }
  const s = `${(atScale * 100).toFixed(1)}%`;
  if (workingScale === null) {
    return `${cell} attested at ${n} zoom, seen on record at ${s}, no working scale recorded at attestation`;
  }
  const met = meetsBar(atScale, workingScale);
  const against = ` against a working scale of ${(workingScale.zoom * 100).toFixed(1)}%`;
  return `${cell} attested at ${n} zoom, seen on record at ${s}${against}${met ? "" : " (below it)"}`;
}

/** Keyboard-operable disclosure for the overlay's marks, the pattern CollapsibleSection already
 *  uses elsewhere: a real button toggles a named region, no hover required. Opens upward from the
 *  panel's own header, since the panel floats near the bottom of the canvas and a downward-opening
 *  panel runs past the window edge; closes whenever the surrounding chrome collapses, since a
 *  reader who collapsed the chrome did not ask to keep this dropdown up. */
function CoverageKey(props: { panelOpen: boolean }) {
  const [open, setOpen] = useState(false);
  const regionId = useId();

  useEffect(() => {
    if (!props.panelOpen) setOpen(false);
  }, [props.panelOpen]);

  return (
    <div className="relative">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={regionId}
        onClick={() => setOpen((o) => !o)}
        className="rounded border border-tcip-border bg-tcip-bg px-1.5 py-0.5 text-[10px] font-semibold text-tcip-muted hover:text-tcip-fg"
      >
        Key
      </button>
      {open && (
        <div
          id={regionId}
          role="region"
          aria-label="Coverage overlay key"
          className="absolute bottom-full right-0 z-30 mb-1 w-56 rounded-md border border-tcip-border-hover bg-tcip-panel p-2.5 text-[10px] text-tcip-fg shadow-lg"
        >
          <ul className="space-y-1">
            <li>
              swept: every part of the cell has been on screen at the working scale, any session
            </li>
            <li>
              short dashes, no fill: swept this session, not yet saved -- the same cells the "not
              yet saved" state line names
            </li>
            <li>count in parentheses: saved annotations of the active subject</li>
            <li>solid border: attested complete, active subject</li>
            <li>dotted border: attested complete, another subject</li>
            <li>struck through: attested but changed since attested</li>
          </ul>
        </div>
      )}
    </div>
  );
}

/** The replace-hold notice's own sentence: at `cellsSeen` 0 the record served cells at native
 *  resolution but swept none, so "0 cells seen" would read as a measurement rather than the
 *  absence it is; the zero case names the record itself instead. The server's 409 stays the
 *  authority on whether a hold applies at all, this only picks the wording once it does. */
function replaceNoticeText(replaceRequired: ReplaceRequired): string {
  const { cellsSeen, cols, rows } = replaceRequired;
  if (cellsSeen === 0) {
    return (
      `a previous lattice's record (${cols}x${rows}) with no cells seen; progress on this ` +
      "lattice is not saved until you replace it"
    );
  }
  return (
    `${cellsSeen} cell${cellsSeen === 1 ? "" : "s"} seen on a previous lattice (${cols}x${rows}); ` +
    "progress on this lattice is not saved until you replace it"
  );
}

/** The panel's own visible working-scale line: states the set zoom and who set it, read
 *  verbatim off `WorkingScale.source` rather than reassembled here. */
function workingScaleLineText(scale: WorkingScale, subject: string): string {
  return `Working scale for ${subject}: ${(scale.zoom * 100).toFixed(1)}% (${scale.source})`;
}

function attestLabel(subject: string, cell: string, complete: boolean, stale: boolean): string {
  if (!complete) return `Attest ${cell} complete for ${subject}`;
  if (stale) return `Re-attest ${cell} for ${subject} (changed since attested)`;
  return `Unattest ${cell} for ${subject}`;
}

/** The lattice states that hold at least one cell, one line each, for a screen-reader user who
 *  cannot see the overlay's canvas pixels ("swept: B1, B2", "saved for fruit: A1 (2)",
 *  "attested: B2", "attested for leaf: B1", "changed since attested: B2"). Cell names sorted for
 *  a stable reading order, other-subject lines sorted by subject name. */
function stateLines(props: {
  subject: string | null;
  swept: ReadonlySet<string>;
  pending: ReadonlySet<string>;
  activeComplete: ReadonlySet<string>;
  activeStale: ReadonlySet<string>;
  otherComplete: Readonly<Record<string, readonly string[]>>;
  annotationCounts: Record<string, number>;
}): string[] {
  const lines: string[] = [];
  const swept = Array.from(props.swept).sort();
  if (swept.length) lines.push(`swept: ${swept.join(", ")}`);
  // Only a swept cell whose facts are still pending gets the overlay's short-dash mark
  // (CoverageOverlay's own `swept && pending`); this line names the same set.
  const pending = Array.from(props.swept)
    .filter((name) => props.pending.has(name))
    .sort();
  if (pending.length) lines.push(`not yet saved: ${pending.join(", ")}`);
  if (props.subject) {
    const saved = Object.entries(props.annotationCounts)
      .filter(([, n]) => n > 0)
      .sort(([a], [b]) => a.localeCompare(b));
    if (saved.length) {
      lines.push(
        `saved for ${props.subject}: ${saved.map(([name, n]) => `${name} (${n})`).join(", ")}`,
      );
    }
  }
  const attested = Array.from(props.activeComplete).sort();
  if (attested.length) lines.push(`attested: ${attested.join(", ")}`);
  for (const subj of Object.keys(props.otherComplete).sort()) {
    lines.push(`attested for ${subj}: ${props.otherComplete[subj].join(", ")}`);
  }
  const stale = Array.from(props.activeStale).sort();
  if (stale.length) lines.push(`changed since attested: ${stale.join(", ")}`);
  return lines;
}

/** The numeric grid-zoom control: screen pixels per native pixel, the same number the status
 *  bar shows as a percentage. Accepts a positive number only; the caller's own route refuses a
 *  non-positive one by name, so this control refuses locally rather than sending a request known
 *  to fail. */
function GridZoomControl(props: { subject: string; onSet: (zoom: number) => void }) {
  const [text, setText] = useState("");
  const parsed = Number(text);
  const valid = text.trim() !== "" && Number.isFinite(parsed) && parsed > 0;

  return (
    <div className="flex items-center gap-1.5">
      <label className="text-tcip-muted" htmlFor="tcip-grid-zoom-input">
        Grid zoom for {props.subject}
      </label>
      <input
        id="tcip-grid-zoom-input"
        type="number"
        step="0.1"
        min="0"
        value={text}
        onChange={(e) => setText(e.target.value)}
        className="w-16 rounded border border-tcip-border bg-tcip-bg px-1.5 py-0.5 text-tcip-fg"
      />
      <button
        type="button"
        disabled={!valid}
        onClick={() => {
          if (valid) props.onSet(parsed);
        }}
        className="rounded border border-tcip-border bg-tcip-bg px-2 py-0.5 font-semibold text-tcip-fg hover:border-tcip-border-hover disabled:opacity-50"
      >
        Set
      </button>
    </div>
  );
}

export function CoverageChrome(props: {
  subject: string | null;
  derivation: string;
  /** Why no coverage lattice could be derived (no subject, no set zoom, or the canvas host not
   *  yet measured), or null once one is loaded (see `settled` for "not yet answered"). */
  reason: string | null;
  /** Whether the grid fetch has answered at all: tells "not yet answered" from "answered, no
   *  lattice" so the panel never renders `reason` as a fact before the read is done. */
  settled: boolean;
  /** Set only when the current grid came from an already-worked image's own recorded lattice and
   *  the subject's current zoom would derive a different tile size. */
  freshDerivationDiffers: boolean | null;
  onRederiveLattice: () => void;
  onSetGridZoom: (zoom: number) => void;
  gridFetchError: string | null;
  readError: string | null;
  countsError: string | null;
  /** Whether this raster's grid can actually draw an overlay (more than one cell, grid loaded):
   *  the overlay toggle is offered only then, and the attest control skips the overlay-on gate
   *  when it is false, since there is no overlay for it to hide the cell behind. */
  canOverlay: boolean;
  overlayOn: boolean;
  onToggleOverlay: () => void;
  currentCellName: string | null;
  currentCellComplete: boolean;
  currentCellStale: boolean;
  otherLattice: OtherLatticeAttestation | null;
  /** The tracker's own replace hold: a view-coverage sweep record for this image/subject on a
   *  grid other than the current one, or null. While it stands no further sweep is posted until
   *  `onArmReplace` confirms discarding it. */
  replaceRequired: ReplaceRequired | null;
  onArmReplace: () => void;
  swept: ReadonlySet<string>;
  /** Cells seen locally, not yet acknowledged by the server (see coverageTracker.ts). */
  pending: ReadonlySet<string>;
  /** Cells recorded seen but not meeting the current bar (or every recorded cell while there is
   *  no bar): the chrome's own "coarser" remainder line. */
  coarserCount: number;
  /** The active subject's working scale (the set grid zoom), or null (see `workingScaleReason`). */
  workingScale: WorkingScale | null;
  workingScaleReason: string | null;
  /** The image's own whole-frame fit scale (unclamped), so the panel can state when the working
   *  scale sits coarser than any real view could ever be. */
  fitScale: number | null;
  activeComplete: ReadonlySet<string>;
  activeStale: ReadonlySet<string>;
  /** The active subject's scale provenance per attested cell, on the current grid. */
  activeCellsAttestedView: Readonly<Record<string, CellAttestedView>>;
  /** Another subject's complete cells on the current grid, keyed by that subject's name. */
  otherComplete: Readonly<Record<string, readonly string[]>>;
  annotationCounts: Record<string, number>;
  onAttest: (complete: boolean) => void;
}) {
  const [confirmPending, setConfirmPending] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const [replaceArmed, setReplaceArmed] = useState(false);
  const { open: panelOpen, toggle: togglePanel } = useDisclosure(
    "tcip.annotate.coverageChromeOpen",
    true,
  );

  const cell = props.currentCellName;
  const destructive = !!props.otherLattice;

  useEffect(() => {
    setConfirmPending(false);
  }, [props.currentCellName, props.otherLattice]);

  // Tied to the hold alone, never to the current cell: a breeder deciding whether to replace
  // must not lose that decision by panning to another cell first.
  useEffect(() => {
    setReplaceArmed(false);
  }, [props.replaceRequired]);

  function press() {
    if (destructive && !confirmPending) {
      setConfirmPending(true);
      const n = props.otherLattice!.count;
      setAnnouncement(
        `Confirmation armed: press Confirm to discard ${n} previous attestation${n === 1 ? "" : "s"}, or Cancel.`,
      );
      return;
    }
    setConfirmPending(false);
    setAnnouncement("");
    props.onAttest(!props.currentCellComplete || props.currentCellStale);
  }

  function cancel() {
    setConfirmPending(false);
    setAnnouncement("Attestation cancelled.");
  }

  function pressReplace() {
    if (!replaceArmed) {
      setReplaceArmed(true);
      const n = props.replaceRequired!.cellsSeen;
      setAnnouncement(
        `Replace armed: press Confirm to discard ${n} cell${n === 1 ? "" : "s"} seen on the previous lattice, or Cancel.`,
      );
      return;
    }
    setReplaceArmed(false);
    setAnnouncement("");
    props.onArmReplace();
  }

  function cancelReplace() {
    setReplaceArmed(false);
    setAnnouncement("Replace cancelled.");
  }

  let label = "";
  if (cell && props.subject) {
    if (destructive && confirmPending) {
      const n = props.otherLattice!.count;
      label = `Confirm: attest ${cell} for ${props.subject} (discards ${n} previous attestation${n === 1 ? "" : "s"})`;
    } else {
      label = attestLabel(props.subject, cell, props.currentCellComplete, props.currentCellStale);
    }
  }

  const lines = stateLines({
    subject: props.subject,
    swept: props.swept,
    pending: props.pending,
    activeComplete: props.activeComplete,
    activeStale: props.activeStale,
    otherComplete: props.otherComplete,
    annotationCounts: props.annotationCounts,
  });

  const attestedViewLineText = cell
    ? attestedViewLine(cell, props.activeCellsAttestedView[cell])
    : null;

  const readErrorReason = props.readError ? breederReadErrorReason(props.readError) : null;
  const noticeClass =
    "rounded-md border bg-tcip-panel/95 px-3 py-1.5 text-[11px] shadow-lg backdrop-blur";

  const bar = props.workingScale;
  const barPct = bar ? bar.zoom * 100 : null;
  const belowFitScale = bar !== null && props.fitScale !== null && bar.zoom < props.fitScale;
  const workingScaleLine =
    bar && props.subject ? workingScaleLineText(bar, props.subject) : props.workingScaleReason;
  const coarserLine =
    props.coarserCount === 0
      ? null
      : bar
        ? `${props.coarserCount} cell${props.coarserCount === 1 ? "" : "s"} on record ${
            props.coarserCount === 1 ? "was" : "were"
          } seen at a coarser scale than ${props.subject}'s working scale`
        : `${props.coarserCount} cell${props.coarserCount === 1 ? "" : "s"} on record ${
            props.coarserCount === 1 ? "was" : "were"
          } fully on screen; no working scale to judge ${props.coarserCount === 1 ? "it" : "them"} against`;

  return (
    <div className="absolute bottom-3 right-3 z-20 flex w-64 flex-col items-stretch gap-1.5">
      <p role="status" className="sr-only">
        {announcement}
      </p>
      {readErrorReason && (
        <p role="status" className={`${noticeClass} border-tcip-fp/40 text-tcip-fp`}>
          this image&apos;s labels could not be read, so nothing can be attested until they are
          fixed: {readErrorReason}
        </p>
      )}
      {props.otherLattice && (
        <p className={`${noticeClass} border-tcip-border text-tcip-warn`}>
          {props.otherLattice.count} cell{props.otherLattice.count === 1 ? "" : "s"} attested on a
          previous lattice ({props.otherLattice.cols}x{props.otherLattice.rows}).
        </p>
      )}
      {props.replaceRequired && (
        <div className={`${noticeClass} flex flex-col gap-1.5 border-tcip-border text-tcip-warn`}>
          <p>{replaceNoticeText(props.replaceRequired)}.</p>
          <div className="flex gap-1.5">
            <button
              type="button"
              onClick={pressReplace}
              className={`flex-1 rounded border px-2 py-1 text-left text-[11px] font-semibold transition-colors ${
                replaceArmed
                  ? "border-tcip-fp/60 bg-tcip-fp/15 text-tcip-fp"
                  : "border-tcip-border bg-tcip-bg text-tcip-fg hover:border-tcip-border-hover"
              }`}
            >
              {replaceArmed
                ? `Confirm: discard ${props.replaceRequired.cellsSeen} cell${
                    props.replaceRequired.cellsSeen === 1 ? "" : "s"
                  } seen on the previous lattice`
                : "Replace"}
            </button>
            {replaceArmed && (
              <button
                type="button"
                onClick={cancelReplace}
                className="rounded border border-tcip-border bg-tcip-bg px-2 py-1 text-[11px] font-semibold text-tcip-muted hover:text-tcip-fg"
              >
                Cancel
              </button>
            )}
          </div>
        </div>
      )}
      {belowFitScale && (
        <p className={`${noticeClass} border-tcip-border text-tcip-muted`}>
          {props.subject}&apos;s working scale ({barPct!.toFixed(1)}%) is coarser than the
          whole-image view, so every view meets it.
        </p>
      )}
      {props.settled && props.reason && (
        <p className={`${noticeClass} border-tcip-border text-tcip-muted`}>{props.reason}</p>
      )}
      {props.freshDerivationDiffers && (
        <div
          className={`${noticeClass} flex items-center justify-between gap-1.5 border-tcip-border text-tcip-warn`}
        >
          <span>the grid zoom has changed since this image's lattice was recorded.</span>
          <button
            type="button"
            onClick={props.onRederiveLattice}
            className="rounded border border-tcip-border bg-tcip-bg px-2 py-1 text-[11px] font-semibold text-tcip-fg hover:border-tcip-border-hover"
          >
            Re-derive lattice
          </button>
        </div>
      )}
      <CollapsibleSection
        className="w-64 rounded-md border border-tcip-border bg-tcip-panel/95 px-3 py-2 text-[11px] text-tcip-fg shadow-lg backdrop-blur"
        title={props.subject ? `Coverage for ${props.subject}` : "Coverage grid"}
        right={
          <div className="flex items-center gap-1.5">
            <CoverageKey panelOpen={panelOpen} />
            {props.canOverlay && (
              <button
                type="button"
                aria-pressed={props.overlayOn}
                onClick={props.onToggleOverlay}
                className={`rounded border px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                  props.overlayOn
                    ? "border-tcip-accent/55 bg-tcip-accent/20 text-tcip-fg"
                    : "border-tcip-border bg-tcip-bg text-tcip-muted hover:text-tcip-fg"
                }`}
              >
                {props.overlayOn ? "Overlay on" : "Overlay off"}
              </button>
            )}
          </div>
        }
        open={panelOpen}
        onToggle={togglePanel}
      >
        <div className="flex flex-col gap-2">
          {props.subject && <GridZoomControl subject={props.subject} onSet={props.onSetGridZoom} />}

          {props.gridFetchError ? (
            <p className="text-tcip-fp">coverage grid unavailable: {props.gridFetchError}</p>
          ) : props.derivation ? (
            <p className="text-tcip-muted">{props.derivation}. A cell is not a training tile.</p>
          ) : null}

          {props.countsError && !readErrorReason && (
            <p className="text-tcip-warn">
              saved-annotation counts unavailable: {props.countsError}
            </p>
          )}

          {props.subject && workingScaleLine && (
            <p className="text-tcip-muted">{workingScaleLine}</p>
          )}
          {coarserLine && <p className="text-tcip-muted">{coarserLine}</p>}

          {lines.length > 0 && (
            <ul className="sr-only">
              {lines.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          )}
          {attestedViewLineText && <p className="sr-only">{attestedViewLineText}</p>}

          {cell &&
            !readErrorReason &&
            (!props.canOverlay || props.overlayOn ? (
              props.subject && (
                <div className="flex gap-1.5">
                  <button
                    type="button"
                    onClick={press}
                    className={`flex-1 rounded border px-2 py-1 text-left text-[11px] font-semibold transition-colors ${
                      destructive && confirmPending
                        ? "border-tcip-fp/60 bg-tcip-fp/15 text-tcip-fp"
                        : "border-tcip-border bg-tcip-bg text-tcip-fg hover:border-tcip-border-hover"
                    }`}
                  >
                    {label}
                  </button>
                  {destructive && confirmPending && (
                    <button
                      type="button"
                      onClick={cancel}
                      className="rounded border border-tcip-border bg-tcip-bg px-2 py-1 text-[11px] font-semibold text-tcip-muted hover:text-tcip-fg"
                    >
                      Cancel
                    </button>
                  )}
                </div>
              )
            ) : (
              <p className="text-tcip-muted">Turn the overlay on to attest {cell}.</p>
            ))}
        </div>
      </CollapsibleSection>
    </div>
  );
}
