/**
 * Coverage grid chrome for a multi-cell raster: the overlay toggle, a key naming what each
 * overlay mark means, the grid's own derivation line, the attestation control for the cell under
 * the viewport center, and the errors and previous-lattice fact a breeder needs before trusting
 * or acting on any of it. Floats bottom-right, completing the canvas' floating-chrome grammar
 * (legend bottom-left, attributes top-right, Overview top-left).
 *
 * The attestation control's label always states the write it performs, read from the raw stored
 * set: "Attest" when the cell has never been attested, "Unattest" when it is attested and fresh,
 * "Re-attest ... (changed since attested)" when it is stale. When the current grid differs from
 * an existing record's grid, the first press only arms a confirmation (the write would discard
 * every attestation made on the previous lattice); a second press performs it.
 */

import { useEffect, useState } from "react";

import type { OtherLatticeAttestation } from "@/hooks/useRegionCompleteness";

/** Hover-triggered key for the overlay's marks, the AnnotateLegend pattern: nothing on the
 *  overlay itself names a mark, so this is the one place a breeder can look it up. */
function CoverageKey() {
  return (
    <div className="group relative">
      <button
        type="button"
        className="rounded border border-tcip-border bg-tcip-bg px-1.5 py-0.5 text-[10px] font-semibold text-tcip-muted hover:text-tcip-fg"
      >
        Key
      </button>
      <div className="pointer-events-none absolute right-0 top-full z-30 mt-1 w-52 translate-y-1 rounded-md border border-tcip-border-hover bg-tcip-panel p-2.5 text-[10px] text-tcip-fg opacity-0 shadow-lg transition-all group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100">
        <ul className="space-y-1">
          <li>
            Green tint, dashed border: swept (viewed at working scale, this or an earlier session)
          </li>
          <li>(n) beside a cell name: saved annotations of the active subject</li>
          <li>Bright border: attested complete, active subject</li>
          <li>Dim border: attested complete, another subject</li>
          <li>Struck through: attested but changed since attested</li>
        </ul>
      </div>
    </div>
  );
}

export function CoverageChrome(props: {
  derivation: string;
  gridFetchError: string | null;
  readError: string | null;
  countsError: string | null;
  overlayOn: boolean;
  onToggleOverlay: () => void;
  currentCellName: string | null;
  currentCellComplete: boolean;
  currentCellStale: boolean;
  otherLattice: OtherLatticeAttestation | null;
  onAttest: (complete: boolean) => void;
}) {
  const [confirmPending, setConfirmPending] = useState(false);

  useEffect(() => {
    setConfirmPending(false);
  }, [props.currentCellName, props.otherLattice]);

  const cell = props.currentCellName;
  const destructive = !!props.otherLattice;

  function press() {
    if (destructive && !confirmPending) {
      setConfirmPending(true);
      return;
    }
    setConfirmPending(false);
    props.onAttest(!props.currentCellComplete || props.currentCellStale);
  }

  let label: string;
  if (destructive && confirmPending) {
    label = `Confirm: attest ${cell} (discards ${props.otherLattice!.count} previous attestation${props.otherLattice!.count === 1 ? "" : "s"})`;
  } else if (!props.currentCellComplete) {
    label = `Attest ${cell} complete`;
  } else if (props.currentCellStale) {
    label = `Re-attest ${cell} (changed since attested)`;
  } else {
    label = `Unattest ${cell}`;
  }

  return (
    <div className="absolute bottom-3 right-3 z-20 flex w-64 flex-col gap-2 rounded-md border border-tcip-border bg-tcip-panel/95 px-3 py-2 text-[11px] text-tcip-fg shadow-lg backdrop-blur">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-wide text-tcip-muted">
          Coverage grid
        </span>
        <div className="flex items-center gap-1.5">
          <CoverageKey />
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
        </div>
      </div>

      {props.gridFetchError ? (
        <p className="text-tcip-fp">coverage grid unavailable: {props.gridFetchError}</p>
      ) : props.derivation ? (
        <p className="text-tcip-muted">{props.derivation}. A cell is not a training tile.</p>
      ) : null}

      {props.readError && (
        <p className="text-tcip-fp">completeness unavailable: {props.readError}</p>
      )}
      {props.countsError && !props.readError && (
        <p className="text-tcip-warn">saved-annotation counts unavailable: {props.countsError}</p>
      )}

      {props.otherLattice && (
        <p className="text-tcip-warn">
          {props.otherLattice.count} cell{props.otherLattice.count === 1 ? "" : "s"} attested on a
          previous lattice ({props.otherLattice.cols}x{props.otherLattice.rows}).
        </p>
      )}

      {cell && !props.readError && (
        <button
          type="button"
          onClick={press}
          className={`rounded border px-2 py-1 text-left text-[11px] font-semibold transition-colors ${
            destructive && confirmPending
              ? "border-tcip-fp/60 bg-tcip-fp/15 text-tcip-fp"
              : "border-tcip-border bg-tcip-bg text-tcip-fg hover:border-tcip-border-hover"
          }`}
        >
          {label}
        </button>
      )}
    </div>
  );
}
