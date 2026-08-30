import type { AttributeDef } from "@/api/classes";

// The reset option's glyph, by code point rather than the literal character (an em dash).

/** One `<select>` per declared attribute of the subject; empty resets the value. */
export function AttributeEditors({
  subject,
  attributes,
  registry,
  onChange,
}: {
  subject: string;
  attributes: Record<string, string>;
  registry: Record<string, { attributes?: Record<string, AttributeDef> }>;
  onChange: (attr: string, value: string) => void;
}) {
  const defs = registry[subject]?.attributes ?? {};
  const entries = Object.entries(defs);
  if (entries.length === 0) {
    return <p className="text-tcip-muted">No attributes declared for {subject}.</p>;
  }
  return (
    <>
      {entries.map(([name, def]) => (
        <label key={name} className="mb-1 flex items-center gap-1.5">
          <span className="w-20 shrink-0 truncate text-tcip-muted" title={name}>
            {name}
          </span>
          <select
            className="tcip-select flex-1 text-[11px]"
            value={attributes[name] ?? ""}
            onChange={(e) => onChange(name, e.target.value)}
          >
            <option value="">-</option>
            {def.values.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
      ))}
    </>
  );
}
