import type { ImageBandInfo } from "@/api/client";
import type { BandSelection, Stretch } from "@/lib/bandSelection";

/**
 * An R/G/B band-composite picker plus a stretch select — collapses to a single dropdown (every
 * channel set together) when the source is single-band. The mount-level gate (band_count > 3,
 * never shown for a standard RGB dataset) lives with each caller; this component only renders
 * whatever bandCount/bands it's given.
 */
export function BandPicker({
  bandCount,
  bands,
  selection,
  onChange,
}: {
  bandCount: number;
  bands: ImageBandInfo[];
  selection: BandSelection;
  onChange: (next: BandSelection) => void;
}) {
  const single = bandCount === 1;
  const channels = single ? (["r"] as const) : (["r", "g", "b"] as const);

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-bold uppercase tracking-wide text-tcip-muted">
        {single ? "Band" : "Bands"}
      </span>
      {channels.map((channel) => (
        <select
          key={channel}
          aria-label={single ? "Band" : `${channel.toUpperCase()} band`}
          className="tcip-select text-[11px]"
          value={selection[channel]}
          onChange={(e) => {
            const value = e.target.value;
            onChange(
              single
                ? { ...selection, r: value, g: value, b: value }
                : { ...selection, [channel]: value },
            );
          }}
        >
          {bands.map((b) => (
            <option key={b.name} value={b.name}>
              {b.wavelength_nm != null ? `${b.name} (${b.wavelength_nm}nm)` : b.name}
            </option>
          ))}
        </select>
      ))}
      <select
        aria-label="Stretch"
        className="tcip-select text-[11px]"
        value={selection.stretch}
        onChange={(e) => onChange({ ...selection, stretch: e.target.value as Stretch })}
      >
        <option value="minmax">Min-Max</option>
        <option value="percent_clip">Percent Clip</option>
      </select>
    </div>
  );
}
