import { useState } from "react";

import type { AttributeDef } from "@/api/classes";
import { UNSET_GLYPH } from "@/lib/glyphs";

/** One `<select>` per declared attribute of the subject; empty resets the value. Each row also
 *  offers `+ value` when `onAddValue` is given, so a missing value name can be declared right
 *  where a breeder first needed it. */
export function AttributeEditors({
  subject,
  attributes,
  registry,
  onChange,
  onAddValue,
}: {
  subject: string;
  attributes: Record<string, string>;
  registry: Record<string, { attributes?: Record<string, AttributeDef> }>;
  onChange: (attr: string, value: string) => void;
  /** Declares a new value for one of the subject's already-declared attributes. Omitted where the
   *  caller has nowhere to save a grown registry (e.g. a read-only context). */
  onAddValue?: (attr: string, value: string) => void;
}) {
  const defs = registry[subject]?.attributes ?? {};
  const entries = Object.entries(defs);
  const [addingValueFor, setAddingValueFor] = useState<string | null>(null);
  const [valueDraft, setValueDraft] = useState("");
  if (entries.length === 0) {
    return <p className="text-tcip-muted">No attributes declared for {subject}.</p>;
  }
  return (
    <>
      {entries.map(([name, def]) => (
        <div key={name} className="mb-1">
          <div className="flex items-center gap-1.5">
            <label className="flex flex-1 items-center gap-1.5">
              <span className="w-20 shrink-0 truncate text-tcip-muted" title={name}>
                {name}
              </span>
              <select
                className="tcip-select flex-1 text-[11px]"
                value={attributes[name] ?? ""}
                onChange={(e) => onChange(name, e.target.value)}
              >
                <option value="" aria-label={`no ${name} value`}>
                  {UNSET_GLYPH}
                </option>
                {def.values.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            {onAddValue && (
              <button
                type="button"
                title={`Declare a new ${name} value`}
                aria-label={`Add a ${name} value`}
                className="shrink-0 text-tcip-muted hover:text-tcip-fg"
                onClick={() => {
                  setAddingValueFor(addingValueFor === name ? null : name);
                  setValueDraft("");
                }}
              >
                + value
              </button>
            )}
          </div>
          {addingValueFor === name && (
            <div className="mt-1 flex items-center gap-1.5 pl-[5.5rem]">
              <input
                autoFocus
                className="tcip-input flex-1 text-[11px]"
                placeholder="new value name"
                value={valueDraft}
                onChange={(e) => setValueDraft(e.target.value)}
              />
              <button
                type="button"
                className="tcip-btn text-[11px]"
                disabled={!valueDraft.trim()}
                onClick={() => {
                  onAddValue?.(name, valueDraft.trim());
                  setValueDraft("");
                  setAddingValueFor(null);
                }}
              >
                Add
              </button>
            </div>
          )}
        </div>
      ))}
    </>
  );
}
