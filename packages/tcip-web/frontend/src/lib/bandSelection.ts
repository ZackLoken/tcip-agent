import type { ImageBandInfo, ImageBandsResponse } from "@/api/client";
import type { CoverageViewing } from "@/api/types.generated";

/** The stretch modes a band picker offers, the server's own vocabulary minus the ``none`` mode a
 *  composite selection never asks for. */
export type Stretch = Exclude<NonNullable<CoverageViewing["stretch"]>, "none">;

export interface BandSelection {
  r: string;
  g: string;
  b: string;
  stretch: Stretch;
}

/** The band interpretations an ordinary colour frame carries, in order. */
const RGBA_INTERPRETATIONS = "red,green,blue,alpha";

/** Whether a source's four bands are an ordinary 8-bit RGBA frame rather than four captured
 *  spectral bands.
 *
 *  Such a frame has no band choice to make: its fourth band is alpha, and the three that remain
 *  are the colours themselves, so it displays as its own pixels instead of through a stretch a
 *  viewer picks bands for. Decided on what the server read from the file (`interpretation`), never
 *  on the band count, which an equally four-band multispectral capture shares. */
export function isPlainColourFrame(bandsInfo: ImageBandsResponse): boolean {
  return (
    bandsInfo.bands.length === 4 &&
    bandsInfo.bands.every((b) => b.dtype === "uint8") &&
    bandsInfo.bands.map((b) => b.interpretation).join(",") === RGBA_INTERPRETATIONS
  );
}

/** The `bands`/`stretch` an image request carries for the current selection, or neither when the
 *  source has no band picker (a plain RGB dataset serves its own pixels).
 *
 *  One expression of that, shared by everything that requests an image for the same view: the
 *  canvas and the prefetcher would otherwise warm and read two different renders of one image. */
export function compositeParams(
  bandsInfo: ImageBandsResponse | null,
  selection: BandSelection | null,
): { bands?: string; stretch?: Stretch } {
  if (!bandsInfo || bandsInfo.band_count <= 3 || !selection) return {};
  return { bands: `${selection.r},${selection.g},${selection.b}`, stretch: selection.stretch };
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
