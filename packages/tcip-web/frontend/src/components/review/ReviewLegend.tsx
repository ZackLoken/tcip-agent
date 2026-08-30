import { useEffect, useRef, useState } from "react";

import { LegendRow } from "@/components/review/LegendRow";
import type { ReviewColors } from "@/lib/reviewColors";

/** Legend anchored lower-left of the canvas (same pattern as Annotate). Opens on hover for a
 *  quick view and pins open on click so a swatch can be recoloured without the popover slipping
 *  away; clicking outside unpins. Solid = outcome, dashed blue = the detection under review. */
export function ReviewLegend({
  colors,
  items,
  onEdit,
}: {
  colors: ReviewColors;
  /** The rows to list, in order: ReviewTab's own COLOR_LABELS (plain-language label + outcome
   *  key + dashed flag per row), passed in rather than read from a module constant here since
   *  that constant is tab-specific. */
  items: { key: keyof ReviewColors; label: string; dashed?: boolean }[];
  onEdit: (key: keyof ReviewColors) => void;
}) {
  const [pinned, setPinned] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!pinned) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setPinned(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [pinned]);
  const shown = pinned
    ? "pointer-events-auto translate-y-0 opacity-100"
    : "pointer-events-none translate-y-1 opacity-0 group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100";
  return (
    <div ref={rootRef} className="group absolute bottom-3 left-3 z-20">
      <div
        className={`absolute bottom-full left-0 mb-2 w-max min-w-[10rem] whitespace-nowrap rounded-md border border-tcip-border-hover bg-tcip-panel p-3 shadow-lg transition-all ${shown}`}
      >
        <h4 className="mb-2 text-[11px] font-semibold tracking-wide text-tcip-fg">Review Legend</h4>
        <ul className="space-y-1.5">
          {items.map((c) => (
            <LegendRow
              key={c.key}
              color={colors[c.key]}
              dashed={c.dashed}
              label={c.label}
              onEdit={() => onEdit(c.key)}
            />
          ))}
        </ul>
        <p className="mt-2 border-t border-tcip-border pt-1.5 text-[10px] text-tcip-muted">
          Click a swatch to recolour
        </p>
      </div>
      <button
        type="button"
        onClick={() => setPinned((p) => !p)}
        aria-pressed={pinned}
        className={`flex items-center gap-1.5 rounded-full border bg-tcip-panel/90 px-2.5 py-1 text-[11px] backdrop-blur hover:border-tcip-border-hover hover:text-tcip-fg ${pinned ? "border-tcip-border-hover text-tcip-fg" : "border-tcip-border text-tcip-muted"}`}
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
    </div>
  );
}
