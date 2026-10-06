import { useState } from "react";

import { subjectColor } from "@/api/subjects";
import { ColorPickerModal } from "@/components/ColorPickerModal";
import {
  resetSubjectColorOverride,
  setSubjectColorOverride,
  useSubjectColors,
} from "@/lib/subjectColors";
import { FOCUS_HALO, legendRows, type LegendRow } from "@/lib/symbology";
import { useStore } from "@/store";

/** One row's swatch, drawn with the row's own color and line style, haloed for the focus row. */
function Swatch({ row }: { row: LegendRow }) {
  return (
    <span
      className={`inline-block h-[13px] w-[18px] shrink-0 rounded-[2px] border-[2.5px] ${
        row.style === "dotted" ? "border-dotted" : ""
      }`}
      style={{
        borderColor: row.color,
        boxShadow: row.halo
          ? `0 0 0 ${FOCUS_HALO.widthFactor}px color-mix(in srgb, ${FOCUS_HALO.color} ${FOCUS_HALO.opacity * 100}%, transparent)`
          : undefined,
      }}
    />
  );
}

/** Legend, anchored lower-left of the canvas: reveals on hover, on keyboard focus within it, or
 *  by toggling the Legend button (click, Enter, Space). Its rows come from the symbology's own
 *  constants: the review statuses while proposals are shown, else the dataset's subjects (a
 *  subject row opens this browser's color picker), then the line styles and the focus halo. */
export function AnnotateLegend({ reviewing }: { reviewing: boolean }) {
  const registry = useStore((s) => s.registry.subjects);
  useSubjectColors(); // re-render on a recolor so the swatches below never show a stale color
  const [pickerSubject, setPickerSubject] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const panelId = "annotate-legend-panel";
  const rows = legendRows(Object.keys(registry), reviewing);
  return (
    <div className="group absolute bottom-3 left-3 z-20">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 rounded-full border border-tcip-border bg-tcip-panel/90 px-2.5 py-1 text-[11px] text-tcip-muted backdrop-blur hover:border-tcip-border-hover hover:text-tcip-fg"
      >
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.4" />
          <path
            d="M8 7.2v3.4M8 5.2v.05"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
        Legend
      </button>
      <div
        id={panelId}
        className={
          "absolute bottom-full left-0 mb-2 w-max min-w-[8rem] translate-y-1 whitespace-nowrap rounded-md border border-tcip-border-hover bg-tcip-panel p-3 shadow-lg transition-all group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:translate-y-0 group-focus-within:opacity-100 " +
          (open ? "pointer-events-auto translate-y-0 opacity-100" : "pointer-events-none opacity-0")
        }
      >
        <h4 className="mb-2 text-[11px] font-semibold tracking-wide text-tcip-fg">
          Annotate Legend
        </h4>
        <ul className="space-y-1.5">
          {rows.map((row) =>
            row.subject ? (
              <li key={row.text}>
                <button
                  type="button"
                  onClick={() => setPickerSubject(row.subject!)}
                  title={`Change ${row.subject}'s color (this browser only)`}
                  className="flex w-full items-center gap-2.5 rounded text-[12px] hover:bg-tcip-hover"
                >
                  <Swatch row={row} />
                  <span className="text-tcip-fg">{row.text}</span>
                </button>
              </li>
            ) : (
              <li key={row.text} className="flex items-center gap-2.5 text-[12px]">
                <Swatch row={row} />
                <span className={row.color === "currentColor" ? "text-tcip-muted" : "text-tcip-fg"}>
                  {row.text}
                </span>
              </li>
            ),
          )}
        </ul>
      </div>
      {pickerSubject && (
        <ColorPickerModal
          title={`${pickerSubject}'s color (this browser only; derives from the name elsewhere)`}
          initialColor={subjectColor(pickerSubject)}
          onSubmit={(hex) => {
            setSubjectColorOverride(pickerSubject, hex);
            setPickerSubject(null);
          }}
          onReset={() => {
            resetSubjectColorOverride(pickerSubject);
            setPickerSubject(null);
          }}
          onCancel={() => setPickerSubject(null)}
        />
      )}
    </div>
  );
}
