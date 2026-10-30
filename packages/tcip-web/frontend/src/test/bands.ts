import type { ImageBands } from "@/api/types.generated";
import produced from "@/test/bandAnswers.json";

/** What `tools/generate_bands_fixture.py` received from the bands route: a band group of four
 *  16-bit bands stating their wavelengths, a band group of three 8-bit bands stating none, a
 *  four-band 8-bit TIFF and a four-band 16-bit one naming each band's color interpretation, a
 *  plain RGB photo, a four-band raster past the stats sample's budget, its bounds read off part of
 *  its pixels, and a band group of one 16-bit band. */
// A JSON import reads each tuple as an array, which no tuple type admits without `unknown`.
export const BAND_ANSWERS = produced as unknown as Record<
  "four_spectral" | "three_unlabeled" | "rgba" | "deep_rgba" | "plain" | "sampled" | "single",
  ImageBands
>;
