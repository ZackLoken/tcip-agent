import { useEffect, useRef, useState } from "react";

/** An attribute's values as chips, each with a remove control when `onRemove` is given. */
export function ValueChips({
  values,
  onRemove,
}: {
  values: string[];
  onRemove?: (value: string) => void;
}) {
  return (
    <span className="flex flex-wrap gap-1">
      {values.map((v) => (
        <span key={v} className="tcip-badge border border-tcip-border text-tcip-fg">
          {v}
          {onRemove && (
            <button
              type="button"
              className="ml-1 text-tcip-muted hover:text-tcip-fg"
              aria-label={`Remove ${v}`}
              onClick={() => onRemove(v)}
            >
              ✕
            </button>
          )}
        </span>
      ))}
    </span>
  );
}

/** The `+ value` control of one declared attribute: a box that declares each value entered
 *  (Enter or Add) and stays open for the next until Escape or a second `+ value`. */
export function AddValue({ attr, onAdd }: { attr: string; onAdd: (value: string) => void }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);
  const commit = () => {
    const value = draft.trim();
    if (!value) return;
    onAdd(value);
    setDraft("");
  };
  return (
    <>
      <button
        type="button"
        title={`Declare a new ${attr} value`}
        aria-label={`Add ${/^[aeiou]/i.test(attr) ? "an" : "a"} ${attr} value`}
        className="shrink-0 text-tcip-muted hover:text-tcip-fg"
        onClick={() => {
          setOpen((o) => !o);
          setDraft("");
        }}
      >
        + value
      </button>
      {open && (
        <span className="mt-1 flex w-full items-center gap-1.5">
          <input
            ref={inputRef}
            className="tcip-input flex-1 text-[13px]"
            placeholder="Value"
            aria-label={`new ${attr} value`}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit();
              if (e.key === "Escape") setOpen(false);
            }}
          />
          <button
            type="button"
            className="tcip-btn text-[13px]"
            disabled={!draft.trim()}
            onClick={commit}
          >
            Add
          </button>
        </span>
      )}
    </>
  );
}
