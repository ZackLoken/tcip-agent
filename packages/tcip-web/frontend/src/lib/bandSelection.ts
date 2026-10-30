import type { CompositeStretch, ImageBand, ImageBands } from "@/api/types.generated";

export interface BandSelection {
  r: string;
  g: string;
  b: string;
  stretch: CompositeStretch;
}

/** The band-set signature a selection is held per: the image's band names, in declared order,
 *  joined. Two images share a selection only when they share this exact signature. */
export function bandSetSignature(bands: ImageBand[]): string {
  return bands.map((b) => b.name).join(",");
}

/** Whether every one of `selection`'s three band names is actually among `bandsInfo`'s own bands.
 *  A selection kept for one band-set signature must never be applied to different metadata, the
 *  transient shape a fast image switch can produce while a fetch is still in flight. */
function selectionMatchesBands(bandsInfo: ImageBands, selection: BandSelection): boolean {
  const names = new Set(bandsInfo.bands.map((b) => b.name));
  return names.has(selection.r) && names.has(selection.g) && names.has(selection.b);
}

/** The three band names an image request composites, in display order. */
export type CompositeBands = readonly [string, string, string];

/** The `bands`/`stretch` an image request carries for the current selection, or neither when the
 *  frame displays as its own pixels (the server's `plain`) or the selection names bands this
 *  metadata doesn't actually carry.
 *
 *  One expression of that, shared by everything that requests an image for the same view: the
 *  canvas and the prefetcher would otherwise warm and read two different renders of one image. */
export function compositeParams(
  bandsInfo: ImageBands | null,
  selection: BandSelection | null,
): { bands?: CompositeBands; stretch?: CompositeStretch } {
  if (!bandsInfo || !selection || bandsInfo.plain) return {};
  if (!selectionMatchesBands(bandsInfo, selection)) return {};
  return { bands: [selection.r, selection.g, selection.b], stretch: selection.stretch };
}

/** Whether a band picker may show for this frame and selection: exactly when the frame's image
 *  request carries a composite (`compositeParams`), so the picker never shows for stale metadata
 *  mid-switch. */
export function showsBandPicker(
  bandsInfo: ImageBands | null,
  selection: BandSelection | null,
): boolean {
  return compositeParams(bandsInfo, selection).bands !== undefined;
}

/** The server's default bands (`default_bands`) as the initial R/G/B assignment, at its default
 *  stretch (`default_stretch`). */
export function defaultBandSelection(bandsInfo: ImageBands): BandSelection {
  const [r, g, b] = bandsInfo.default_bands;
  return { r, g, b, stretch: bandsInfo.default_stretch };
}
