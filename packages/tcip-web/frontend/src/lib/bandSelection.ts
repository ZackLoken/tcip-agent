import type { ImageBandInfo } from "@/api/client";

export type Stretch = "minmax" | "percent_clip";

export interface BandSelection {
  r: string;
  g: string;
  b: string;
  stretch: Stretch;
}

/** First three declared bands as the initial R/G/B assignment (falling back to the first band
 *  when fewer than three are reported, though the picker is never mounted below band_count > 3). */
export function defaultBandSelection(bands: ImageBandInfo[]): BandSelection {
  const names = bands.map((b) => b.name);
  return {
    r: names[0] ?? "",
    g: names[1] ?? names[0] ?? "",
    b: names[2] ?? names[0] ?? "",
    stretch: "minmax",
  };
}
