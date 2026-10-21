import { useState } from "react";

import type { PhrasePrompt } from "@/api/types.generated";
import { useStore } from "@/store";

/** The engine prompt control: one toggle holding each placed point and drawn box as a prompt
 *  for the segmenter, a Propose button sending them together, and a phrase for the detector
 *  sent through `onPhrase`. While `proposing`, every send is refused. */
export function EnginePrompt({
  proposing,
  onPhrase,
  onPropose,
}: {
  proposing: boolean;
  onPhrase: (prompt: PhrasePrompt) => void;
  onPropose: () => void;
}) {
  const armed = useStore((s) => s.annotateUi.prompt);
  const pending = useStore((s) => s.annotateUi.pendingPrompts);
  const setPrompt = useStore((s) => s.setPrompt);
  const [phrase, setPhrase] = useState("");
  const held = pending.points.length + pending.boxes.length;
  const canFind = !proposing && phrase.trim() !== "";

  return (
    <div
      role="group"
      aria-label="Prompt"
      aria-busy={proposing}
      className={`inline-flex items-center gap-1.5 rounded border p-0.5 ${
        armed ? "border-tcip-accent bg-tcip-accent/15" : "border-tcip-border bg-tcip-bg"
      }`}
    >
      <button
        type="button"
        aria-pressed={armed}
        onClick={() => setPrompt(!armed)}
        title="Click objects or draw boxes around them, then propose"
        className={`flex h-7 items-center gap-1.5 rounded-[4px] px-2.5 font-semibold transition-colors ${
          armed ? "bg-tcip-accent text-white" : "text-tcip-muted hover:text-tcip-fg"
        }`}
      >
        <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
          <rect
            x="2.5"
            y="2.5"
            width="11"
            height="11"
            rx="3"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeDasharray="2.4 2.4"
            strokeLinecap="round"
          />
        </svg>
        {proposing ? "Proposing" : "Prompt"}
      </button>
      {armed && (
        <>
          <form
            className="flex items-center gap-1.5"
            onSubmit={(e) => {
              e.preventDefault();
              if (canFind) onPhrase({ phrase });
            }}
          >
            <input
              type="text"
              aria-label="Phrase"
              placeholder="Find by phrase"
              className="tcip-input h-7 w-44"
              value={phrase}
              onChange={(e) => setPhrase(e.target.value)}
            />
            <button type="submit" className="tcip-btn h-7" disabled={!canFind}>
              Find
            </button>
          </form>
          <button
            type="button"
            className="tcip-btn-primary h-7"
            disabled={proposing || held === 0}
            onClick={onPropose}
          >
            {held > 0 ? `Propose ${held}` : "Propose"}
          </button>
        </>
      )}
    </div>
  );
}
