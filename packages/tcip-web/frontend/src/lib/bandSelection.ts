import type { CompositeStretch, ImageBand, ImageBands } from "@/api/types.generated";

export interface BandSelection {
  r: string;
  g: string;
  b: string;
  stretch: CompositeStretch;
}

/** The band choice an image offers: over a raster read with bands, its metadata, the selection
 *  held for its band set and the setter that records another; none for an image whose metadata
 *  is not in or that displays as its own pixels (the server's plain read). */
export type BandChoice =
  | {
      kind: "banded";
      info: ImageBands;
      selection: BandSelection;
      choose: (next: BandSelection) => void;
    }
  | { kind: "none" };

/** The band choice of an image offering none. */
export const NO_BAND_CHOICE: BandChoice = { kind: "none" };

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

/** The `bands`/`stretch` an image request carries for `choice`'s selection, or neither when the
 *  image offers no band choice or the selection names bands its metadata doesn't carry. */
export function compositeParams(choice: BandChoice): {
  bands?: CompositeBands;
  stretch?: CompositeStretch;
} {
  if (choice.kind === "none" || !selectionMatchesBands(choice.info, choice.selection)) return {};
  const { r, g, b, stretch } = choice.selection;
  return { bands: [r, g, b], stretch };
}

/** Whether a band picker may show for `choice`: exactly when the image request carries a
 *  composite (`compositeParams`), so the picker never shows for stale metadata mid-switch. */
export function showsBandPicker(choice: BandChoice): boolean {
  return compositeParams(choice).bands !== undefined;
}

/** The server's default bands (`default_bands`) as the initial R/G/B assignment, at its default
 *  stretch (`default_stretch`). */
export function defaultBandSelection(bandsInfo: ImageBands): BandSelection {
  const [r, g, b] = bandsInfo.default_bands;
  return { r, g, b, stretch: bandsInfo.default_stretch };
}
