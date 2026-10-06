import { useRef, useState } from "react";

import { UNSET_GLYPH } from "@/lib/glyphs";

/** A stepper over a list: its label, the current entry's name, a previous button, a position
 *  entry that jumps on Enter, the total and a next button. The image stepper and the item
 *  stepper are both this component, so they read and work the same way. */
export function Stepper({
  label,
  noun,
  name,
  position,
  total,
  canPrev,
  canNext,
  onStep,
  onJump,
}: {
  /** The visible label beside the stepper. */
  label: string;
  /** The singular noun the controls are named by ("Previous image", "Item position"). */
  noun: string;
  /** The current entry's name, shown truncated; the unset glyph when there is none. */
  name: string | null;
  /** 1-based; 0 when nothing is current. */
  position: number;
  total: number;
  canPrev: boolean;
  canNext: boolean;
  onStep: (delta: number) => void;
  onJump: (oneBased: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const Noun = noun.charAt(0).toUpperCase() + noun.slice(1);
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] font-bold uppercase tracking-wide text-tcip-muted">{label}</span>
      <span
        className="max-w-[150px] truncate font-mono text-[11px] text-tcip-fg"
        title={name ?? UNSET_GLYPH}
      >
        {name ?? UNSET_GLYPH}
      </span>
      <button
        className="tcip-btn text-[11px]"
        onClick={() => onStep(-1)}
        disabled={!canPrev}
        aria-label={`Previous ${noun}`}
      >
        ◀
      </button>
      <input
        ref={inputRef}
        aria-label={`${Noun} position`}
        title={`${Noun} position: type a number and press Enter to jump`}
        className="tcip-input w-10 text-center font-mono text-[11px]"
        value={draft ?? (position > 0 ? String(position) : "")}
        onChange={(e) => setDraft(e.target.value.replace(/[^0-9]/g, ""))}
        onFocus={() => setDraft(String(position || 1))}
        onBlur={() => setDraft(null)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            const num = parseInt(draft ?? "", 10);
            if (!Number.isNaN(num)) onJump(num);
            setDraft(null);
            inputRef.current?.blur();
          } else if (e.key === "Escape") {
            setDraft(null);
            inputRef.current?.blur();
          }
        }}
      />
      <span className="font-mono text-[11px] tabular-nums text-tcip-muted">/ {total}</span>
      <button
        className="tcip-btn text-[11px]"
        onClick={() => onStep(1)}
        disabled={!canNext}
        aria-label={`Next ${noun}`}
      >
        ▶
      </button>
    </div>
  );
}
