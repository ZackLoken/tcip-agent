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

/** The band-set signature a selection is held per: the image's band names, in declared order,
 *  joined. Two images share a selection only when they share this exact signature. */
export function bandSetSignature(bands: ImageBandInfo[]): string {
  return bands.map((b) => b.name).join(",");
}

/** Whether every one of `selection`'s three band names is actually among `bandsInfo`'s own bands.
 *  A selection kept for one band-set signature must never be applied to different metadata, the
 *  transient shape a fast image switch can produce while a fetch is still in flight. */
function selectionMatchesBands(bandsInfo: ImageBandsResponse, selection: BandSelection): boolean {
  const names = new Set(bandsInfo.bands.map((b) => b.name));
  return names.has(selection.r) && names.has(selection.g) && names.has(selection.b);
}

/** The `bands`/`stretch` an image request carries for the current selection, or neither when the
 *  source has no band picker (a plain RGB dataset serves its own pixels), the frame is an ordinary
 *  colour photo, or the selection names bands this metadata doesn't actually carry.
 *
 *  One expression of that, shared by everything that requests an image for the same view: the
 *  canvas and the prefetcher would otherwise warm and read two different renders of one image. */
export function compositeParams(
  bandsInfo: ImageBandsResponse | null,
  selection: BandSelection | null,
): { bands?: string; stretch?: Stretch } {
  if (!bandsInfo || bandsInfo.band_count <= 3 || !selection) return {};
  if (isPlainColourFrame(bandsInfo) || !selectionMatchesBands(bandsInfo, selection)) return {};
  return { bands: `${selection.r},${selection.g},${selection.b}`, stretch: selection.stretch };
}

/** Whether a band picker may show for this frame and selection: multispectral, not a plain colour
 *  photo, and a selection whose names this metadata actually declares. The one gate every render of
 *  the picker (Annotate's toolbar, Review's own) shares, so neither shows one for the other's stale
 *  metadata mid-switch. */
export function showsBandPicker(
  bandsInfo: ImageBandsResponse | null | undefined,
  selection: BandSelection | null | undefined,
): boolean {
  if (!bandsInfo || bandsInfo.band_count <= 3 || !selection) return false;
  return !isPlainColourFrame(bandsInfo) && selectionMatchesBands(bandsInfo, selection);
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
