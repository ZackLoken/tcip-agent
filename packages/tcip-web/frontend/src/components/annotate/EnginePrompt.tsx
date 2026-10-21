import { useId, useState } from "react";

import { CROSS_TILE_MERGE_DESCRIPTIONS, type DetectorParams } from "@/api/types.generated";
import { DisclosureChevron } from "@/components/CollapsibleSection";
import { useDisclosure } from "@/hooks/useDisclosure";
import { useStore } from "@/store";

// transformers' post_process_grounded_object_detection defaults: threshold and text_threshold.
const THRESHOLD_PREFILL = 0.25;
// SAHI get_sliced_prediction's defaults: postprocess type and match threshold.
const MERGE_PREFILL = { postprocess: "greedynmm", cross_tile_nms: 0.5 };

/** A number input for one detector setting, its label beside it. */
function Setting({
  label,
  value,
  onChange,
  title,
}: {
  label: string;
  value: number;
  onChange: (next: number) => void;
  title: string;
}) {
  return (
    <label className="flex items-center gap-1 text-[13px] text-tcip-muted" title={title}>
      {label}
      <input
        type="number"
        aria-label={label}
        className="tcip-input h-6 w-14 text-[13px]"
        min={0}
        max={1}
        step={0.05}
        value={Number.isNaN(value) ? "" : value}
        onChange={(e) => onChange(e.target.value === "" ? NaN : Number(e.target.value))}
      />
    </label>
  );
}

/** The engine prompt control: one toggle arming the next placed point or drawn box as a prompt
 *  for the segmenter, and, while armed, a phrase with the detector's settings sent through
 *  `onPrompt`. While `proposing`, the toggle says so and the send is refused. */
export function EnginePrompt({
  proposing,
  onPrompt,
}: {
  proposing: boolean;
  onPrompt: (prompt: DetectorParams) => void;
}) {
  const armed = useStore((s) => s.annotateUi.prompt);
  const setPrompt = useStore((s) => s.setPrompt);
  const [phrase, setPhrase] = useState("");
  const [boxThreshold, setBoxThreshold] = useState(THRESHOLD_PREFILL);
  const [textThreshold, setTextThreshold] = useState(THRESHOLD_PREFILL);
  const [merge, setMerge] = useState(MERGE_PREFILL);
  const tiling = useDisclosure();
  const tilingId = useId();

  const stated = [boxThreshold, textThreshold, merge.cross_tile_nms];
  const canFind = !proposing && phrase.trim() !== "" && stated.every(Number.isFinite);

  return (
    <div
      role="group"
      aria-label="Engine prompt"
      aria-busy={proposing}
      className={`relative inline-flex items-center gap-1.5 rounded border p-0.5 ${
        armed
          ? "border-dotted border-tcip-accent bg-tcip-accent/15"
          : "border-tcip-border bg-tcip-bg"
      }`}
    >
      <button
        type="button"
        aria-pressed={armed}
        onClick={() => setPrompt(!armed)}
        title="Prompt: while on, click an object or draw a box around it to get its outline, or describe what to look for and find every one; each comes back as a proposal to accept, correct or reject"
        className={`flex h-6 items-center gap-1.5 rounded-[4px] px-2.5 text-[13px] font-semibold transition-colors ${
          armed ? "bg-tcip-accent text-white" : "text-tcip-muted hover:text-tcip-fg"
        }`}
      >
        {/* A dotted outline, the stroke the canvas draws a proposal in. */}
        <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
          <rect
            x="2.5"
            y="2.5"
            width="11"
            height="11"
            rx="3"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeDasharray="1.6 2.2"
            strokeLinecap="round"
          />
        </svg>
        {proposing ? "Proposing" : "Prompt"}
      </button>
      {armed && (
        <form
          className="flex items-center gap-1.5 pr-0.5"
          onSubmit={(e) => {
            e.preventDefault();
            if (!canFind) return;
            onPrompt({
              phrase,
              box_threshold: boxThreshold,
              text_threshold: textThreshold,
              ...merge,
            });
          }}
        >
          <input
            type="text"
            aria-label="Phrase"
            placeholder="Describe what to find"
            className="tcip-input h-6 w-44 text-[13px]"
            value={phrase}
            onChange={(e) => setPhrase(e.target.value)}
          />
          <Setting
            label="Box"
            value={boxThreshold}
            onChange={setBoxThreshold}
            title="How sure the detector must be that a box holds what you described: raise it for fewer, surer boxes, lower it for more, less sure ones"
          />
          <Setting
            label="Text"
            value={textThreshold}
            onChange={setTextThreshold}
            title="How well the words must fit a box for them to label it: raise it to keep a label only when the words fit the box that well, lower it to let looser words label it"
          />
          <button
            type="button"
            onClick={tiling.toggle}
            aria-expanded={tiling.open}
            aria-controls={tilingId}
            className="flex h-6 items-center gap-1 px-1 text-[13px] text-tcip-muted hover:text-tcip-fg"
          >
            <DisclosureChevron open={tiling.open} />
            Tiling
          </button>
          <button type="submit" className="tcip-btn h-6 text-[13px]" disabled={!canFind}>
            Find
          </button>
          {tiling.open && (
            <div
              id={tilingId}
              title="The image is searched in tiles sized from what is found on it, and an object two tiles both see is joined into one box"
              className="absolute left-0 top-full z-20 mt-1 flex items-center gap-2 rounded-md border border-tcip-border bg-tcip-panel px-2 py-1.5 shadow-lg"
            >
              <label className="flex items-center gap-1 text-[13px] text-tcip-muted">
                Merge
                <select
                  aria-label="Merge"
                  className="tcip-select h-6 text-[13px]"
                  value={merge.postprocess}
                  onChange={(e) => setMerge({ ...merge, postprocess: e.target.value })}
                >
                  {Object.entries(CROSS_TILE_MERGE_DESCRIPTIONS).map(([name, description]) => (
                    <option key={name} value={name}>
                      {description}
                    </option>
                  ))}
                </select>
              </label>
              <Setting
                label="Match"
                value={merge.cross_tile_nms}
                onChange={(cross_tile_nms) => setMerge({ ...merge, cross_tile_nms })}
                title="How alike two tiles' boxes must be to count as one object: raise it to join only boxes that nearly coincide, lower it to join boxes that only partly overlap"
              />
            </div>
          )}
        </form>
      )}
    </div>
  );
}
