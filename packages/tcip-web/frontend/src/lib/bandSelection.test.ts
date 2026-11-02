import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { type ImageBands, PLAIN } from "@/api/types.generated";
import { useBandSelection } from "@/hooks/useBandSelection";
import {
  bandSetSignature,
  compositeParams,
  defaultBandSelection,
  showsBandPicker,
  type BandChoice,
  type BandSelection,
} from "@/lib/bandSelection";
import { BAND_ANSWERS } from "@/test/bands";

const { four_spectral: FOUR, rgba: RGBA } = BAND_ANSWERS;
const FOUR_SPECTRAL = FOUR.bands;
const NAMES = FOUR_SPECTRAL.map((band) => band.name);

/** A selection of `FOUR`'s first three bands in reverse order, at its served default stretch. */
const SELECTION = { r: NAMES[2], g: NAMES[1], b: NAMES[0], stretch: FOUR.default_stretch };

/** `SELECTION` with its blue channel naming a band `FOUR` does not declare. */
const STALE = { ...SELECTION, b: `${NAMES[0]}-elsewhere` };

/** The band choice `useBandSelection` answers for `bandsInfo`. */
function choiceOf(bandsInfo: ImageBands | null): BandChoice {
  return renderHook(() => useBandSelection(bandsInfo)).result.current;
}

/** The band choice answered for `bandsInfo`, holding `selection` as the person chose it. */
function chosen(bandsInfo: ImageBands, selection: BandSelection): BandChoice {
  const choice = choiceOf(bandsInfo);
  if (choice.kind !== "banded") throw new Error("the image offers no band choice");
  return { ...choice, selection };
}

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
    expect(compositeParams(chosen(FOUR, SELECTION))).toEqual({
      bands: [SELECTION.r, SELECTION.g, SELECTION.b],
      stretch: SELECTION.stretch,
    });
  });

  it("carries nothing for a frame the server displays plain", () => {
    expect(RGBA.read.kind).toBe(PLAIN);
    expect(compositeParams(choiceOf(RGBA))).toEqual({});
  });

  it("carries nothing when the selection names a band this metadata doesn't declare", () => {
    expect(compositeParams(chosen(FOUR, STALE))).toEqual({});
  });

  it("carries nothing with no bandsInfo", () => {
    expect(compositeParams(choiceOf(null))).toEqual({});
  });
});

describe("showsBandPicker", () => {
  it("shows for a multispectral frame with a selection matching its bands", () => {
    expect(showsBandPicker(chosen(FOUR, SELECTION))).toBe(true);
  });

  it("hides for a frame the server displays plain", () => {
    expect(showsBandPicker(choiceOf(RGBA))).toBe(false);
  });

  it("hides when the selection names a band the current metadata doesn't declare", () => {
    expect(showsBandPicker(chosen(FOUR, STALE))).toBe(false);
  });

  it("hides while bandsInfo is not yet known", () => {
    expect(showsBandPicker(choiceOf(null))).toBe(false);
  });
});
