import { subjectColor } from "@/api/classes";
import { useStore } from "@/store";

/** Hover-triggered legend, anchored lower-left of the canvas. Lists the dataset's subjects
 *  (outline colour = subject, GUI-local) plus the selected-shape blue, the same grammar as
 *  Review. In box mode, an extra row explains the dashed boxes: a polygon's own read-only bounds,
 *  not a second editable annotation. */
export function AnnotateLegend() {
  const registry = useStore((s) => s.registry.subjects);
  const mode = useStore((s) => s.gui.mode);
  const names = Object.keys(registry);
  return (
    <div className="group absolute bottom-3 left-3 z-20">
      <div className="pointer-events-none absolute bottom-full left-0 mb-2 w-max min-w-[8rem] translate-y-1 whitespace-nowrap rounded-md border border-tcip-border-hover bg-tcip-panel p-3 opacity-0 shadow-lg transition-all group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100">
        <h4 className="mb-2 text-[11px] font-semibold tracking-wide text-tcip-fg">
          Annotate Legend
        </h4>
        <ul className="space-y-1.5">
          {names.map((name) => (
            <li key={name} className="flex items-center gap-2.5 text-[12px]">
              <span
                className="inline-block h-[13px] w-[18px] shrink-0 rounded-[2px] border-[2.5px]"
                style={{ borderColor: subjectColor(name) }}
              />
              <span className="text-tcip-fg">{name}</span>
            </li>
          ))}
          <li className="flex items-center gap-2.5 text-[12px]">
            <span
              className="inline-block h-[13px] w-[18px] shrink-0 rounded-[2px] border-[2.5px]"
              style={{ borderColor: "#00BFFF" }}
            />
            <span className="text-tcip-fg">Selected</span>
          </li>
          {mode === "box" && (
            <li className="flex items-center gap-2.5 text-[12px]">
              <span
                className="inline-block h-[13px] w-[18px] shrink-0 rounded-[2px] border-[2.5px] border-dashed"
                style={{ borderColor: "currentColor" }}
              />
              <span className="text-tcip-muted">Dashed = polygon&apos;s box (read-only)</span>
            </li>
          )}
        </ul>
      </div>
      <button
        type="button"
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
    </div>
  );
}
