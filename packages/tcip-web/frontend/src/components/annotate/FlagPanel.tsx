import { useState } from "react";

import type { Flag } from "@/store/types";

/** The flags on one target (the focused item, or the image as a whole): each open flag with who
 *  raised it and a reply that resolves it, and a comment box that raises a new one. */
export function FlagPanel({
  target,
  flags,
  onRaise,
  onResolve,
  onClose,
}: {
  /** What a new flag lands on, as the stepper names it ("pod box", "this image"). */
  target: string;
  /** The open flags on that target. */
  flags: Flag[];
  onRaise: (text: string) => void;
  onResolve: (id: string, reply: string) => void;
  onClose: () => void;
}) {
  const [text, setText] = useState("");
  const [replies, setReplies] = useState<Record<string, string>>({});
  const raise = () => {
    if (!text.trim()) return;
    onRaise(text.trim());
    setText("");
  };
  return (
    <div
      role="dialog"
      aria-label={`Flags on ${target}`}
      className="absolute right-0 top-full z-30 mt-1 w-80 rounded-md border border-tcip-border-hover bg-tcip-panel p-3 text-[12px] shadow-lg"
      onKeyDown={(e) => {
        if (e.key === "Escape") onClose();
      }}
    >
      <div className="mb-2 flex items-center justify-between">
        <h4 className="font-semibold text-tcip-fg">Flags on {target}</h4>
        <button
          type="button"
          aria-label="Close flags"
          className="text-tcip-muted hover:text-tcip-fg"
          onClick={onClose}
        >
          ✕
        </button>
      </div>
      {flags.length === 0 && <p className="mb-2 text-tcip-muted">No open flag here.</p>}
      <ul className="mb-2 space-y-2">
        {flags.map((flag) => (
          <li key={flag.id} className="rounded border border-tcip-border p-2">
            <p className="text-tcip-fg">{flag.text}</p>
            <p className="mb-1.5 text-[11px] text-tcip-muted">
              {flag.by}, {flag.at.slice(0, 10)}
            </p>
            <div className="flex gap-1.5">
              <input
                aria-label={`Reply to "${flag.text}"`}
                className="tcip-input h-7 flex-1 text-[11px]"
                placeholder="reply (optional)"
                value={replies[flag.id] ?? ""}
                onChange={(e) => setReplies({ ...replies, [flag.id]: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === "Enter") onResolve(flag.id, (replies[flag.id] ?? "").trim());
                }}
              />
              <button
                type="button"
                className="tcip-btn h-7 text-[11px]"
                onClick={() => onResolve(flag.id, (replies[flag.id] ?? "").trim())}
              >
                Resolve
              </button>
            </div>
          </li>
        ))}
      </ul>
      <div className="flex gap-1.5">
        <input
          autoFocus
          aria-label="New flag comment"
          className="tcip-input h-7 flex-1 text-[11px]"
          placeholder="what needs a second look?"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") raise();
          }}
        />
        <button
          type="button"
          className="tcip-btn-primary h-7 text-[11px]"
          disabled={!text.trim()}
          onClick={raise}
        >
          Flag
        </button>
      </div>
    </div>
  );
}
