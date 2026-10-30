import {
  COMPOSITE_STRETCHES,
  type CompositeStretch,
  type ImageBand,
  type ImageBands,
} from "@/api/types.generated";
import type { BandSelection } from "@/lib/bandSelection";

/** How the picker names each stretch the server composites through. */
const STRETCH_LABELS: Record<CompositeStretch, string> = {
  minmax: "Min-Max",
  percent_clip: "Percent Clip",
};

/** How a band reads in the picker: its wavelength where the sensor reported one, and otherwise
 *  what the file says it holds, so a transparency band is not offered as if it were a color a
 *  viewer might want to look at. */
function bandLabel(band: ImageBand): string {
  if (band.wavelength_nm != null) return `${band.name} (${band.wavelength_nm}nm)`;
  if (band.interpretation === "alpha") return `${band.name} (alpha)`;
  return band.name;
}

/** A share of pixels as a percentage a reader can act on: whole numbers where there are some,
 *  one significant digit where the sample is a sliver. */
function samplePercent(fraction: number): string {
  const pct = fraction * 100;
  if (pct >= 10) return `${Math.round(pct)}`;
  if (pct >= 1) return pct.toFixed(1);
  return pct.toPrecision(1);
}

/**
 * An R/G/B band-composite picker plus a stretch select, collapses to a single dropdown (every
 * channel set together) when the source is single-band. The mount-level gate (the server's
 * `plain`, never shown for a frame displayed as its own pixels) lives with each caller; this
 * component only renders whatever bands `info` reports.
 *
 * When the reported band ranges were read from part of the raster rather than all of it, the
 * picker says so: the stretch a viewer picks is built on those numbers, and a value in a window
 * the sample missed is not in them.
 */
export function BandPicker({
  info: { band_count: bandCount, bands, sampled, pixel_fraction: pixelFraction, overview_size },
  selection,
  onChange,
}: {
  info: ImageBands;
  selection: BandSelection;
  onChange: (next: BandSelection) => void;
}) {
  const single = bandCount === 1;
  const channels = single ? (["r"] as const) : (["r", "g", "b"] as const);

  return (
    <div className="flex items-center gap-1.5">
      <span className="tcip-label">{single ? "Band" : "Bands"}</span>
      {channels.map((channel) => (
        <select
          key={channel}
          aria-label={single ? "Band" : `${channel.toUpperCase()} band`}
          className="tcip-select text-[13px]"
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
              {bandLabel(b)}
            </option>
          ))}
        </select>
      ))}
      <select
        aria-label="Stretch"
        className="tcip-select text-[13px]"
        value={selection.stretch}
        onChange={(e) =>
          onChange({ ...selection, stretch: COMPOSITE_STRETCHES[e.target.selectedIndex] })
        }
      >
        {COMPOSITE_STRETCHES.map((stretch) => (
          <option key={stretch} value={stretch}>
            {STRETCH_LABELS[stretch]}
          </option>
        ))}
      </select>
      {sampled && pixelFraction !== null && (
        <span
          className="text-[12px] text-tcip-muted"
          title="These band ranges were read from a seeded sample of this raster's pixels, not from every pixel, so a more extreme value in a window the sample missed is not in them."
        >
          {`stats from a ${samplePercent(pixelFraction)}% pixel sample`}
        </span>
      )}
      {overview_size !== null && (
        <span
          className="text-[12px] text-tcip-muted"
          title="These band ranges were read from a reduced-resolution overview of this raster, not from its native pixels; averaged overview values sit inside the native range, so the true extremes can lie outside these numbers."
        >
          {`stats from a ${overview_size[0]}x${overview_size[1]} overview`}
        </span>
      )}
    </div>
  );
}
