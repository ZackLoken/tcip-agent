import { describe, expect, it } from "vitest";

import {
  bandSetSignature,
  compositeParams,
  defaultBandSelection,
  showsBandPicker,
} from "@/lib/bandSelection";
import { BAND_ANSWERS } from "@/test/bands";

const { four_spectral: FOUR, rgba: RGBA } = BAND_ANSWERS;
const FOUR_SPECTRAL = FOUR.bands;
const NAMES = FOUR_SPECTRAL.map((band) => band.name);

/** A selection of `FOUR`'s first three bands in reverse order, at its served default stretch. */
const SELECTION = { r: NAMES[2], g: NAMES[1], b: NAMES[0], stretch: FOUR.default_stretch };

/** `SELECTION` with its blue channel naming a band `FOUR` does not declare. */
const STALE = { ...SELECTION, b: `${NAMES[0]}-elsewhere` };

/** A selection of `RGBA`'s default bands, the ones a composite of it would read. */
const RGBA_SELECTION = defaultBandSelection(RGBA);

describe("defaultBandSelection", () => {
  it("assigns the server's default bands to R, G, B at the server's default stretch", () => {
    const [r, g, b] = FOUR.default_bands;
    expect(defaultBandSelection(FOUR)).toEqual({ r, g, b, stretch: FOUR.default_stretch });
  });
});

describe("bandSetSignature", () => {
  it("joins the declared band names in order, so two images with the same bands share one", () => {
    expect(bandSetSignature(FOUR_SPECTRAL)).toBe(NAMES.join(","));
  });

  it("differs when the band set differs, even by one band", () => {
    expect(bandSetSignature(FOUR_SPECTRAL)).not.toBe(bandSetSignature(FOUR_SPECTRAL.slice(0, 3)));
  });
});

describe("compositeParams", () => {
  it("carries bands/stretch for a multispectral frame whose bands the selection actually names", () => {
    expect(compositeParams(FOUR, SELECTION)).toEqual({
      bands: [SELECTION.r, SELECTION.g, SELECTION.b],
      stretch: SELECTION.stretch,
    });
  });

  it("carries nothing for a frame the server displays plain, even with a selection in hand", () => {
    expect(RGBA.plain).toBe(true);
    expect(compositeParams(RGBA, RGBA_SELECTION)).toEqual({});
  });

  it("carries nothing when the selection names a band this metadata doesn't declare", () => {
    expect(compositeParams(FOUR, STALE)).toEqual({});
  });

  it("carries nothing with no bandsInfo or no selection", () => {
    expect(compositeParams(null, SELECTION)).toEqual({});
    expect(compositeParams(FOUR, null)).toEqual({});
  });
});

describe("showsBandPicker", () => {
  it("shows for a multispectral frame with a selection matching its bands", () => {
    expect(showsBandPicker(FOUR, SELECTION)).toBe(true);
  });

  it("hides for a frame the server displays plain", () => {
    expect(showsBandPicker(RGBA, RGBA_SELECTION)).toBe(false);
  });

  it("hides when the selection names a band the current metadata doesn't declare", () => {
    expect(showsBandPicker(FOUR, STALE)).toBe(false);
  });

  it("hides while bandsInfo or the selection is not yet known", () => {
    expect(showsBandPicker(null, SELECTION)).toBe(false);
    expect(showsBandPicker(FOUR, null)).toBe(false);
  });
});
