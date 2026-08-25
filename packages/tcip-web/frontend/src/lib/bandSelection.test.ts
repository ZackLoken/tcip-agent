import { describe, expect, it } from "vitest";

import type { ImageBandInfo, ImageBandsResponse } from "@/api/client";
import {
  bandSetSignature,
  compositeParams,
  defaultBandSelection,
  isPlainColourFrame,
  showsBandPicker,
} from "@/lib/bandSelection";

function bandsResponse(bands: ImageBandInfo[]): ImageBandsResponse {
  return { band_count: bands.length, bands };
}

const RGBA: ImageBandInfo[] = ["red", "green", "blue", "alpha"].map((interpretation) => ({
  name: interpretation,
  wavelength_nm: null,
  dtype: "uint8",
  min: 0,
  max: 255,
  interpretation,
}));

const FOUR_SPECTRAL: ImageBandInfo[] = [
  { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
  { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
  { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
  { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
];

describe("isPlainColourFrame", () => {
  it("recognises an 8-bit RGBA frame, which has no band choice to make", () => {
    expect(isPlainColourFrame(bandsResponse(RGBA))).toBe(true);
  });

  it("leaves a four-band spectral capture to the picker", () => {
    expect(isPlainColourFrame(bandsResponse(FOUR_SPECTRAL))).toBe(false);
  });

  it("decides on what the file says, not on the band count", () => {
    const unlabelled = RGBA.map(({ interpretation: _drop, ...band }) => band);
    expect(isPlainColourFrame(bandsResponse(unlabelled))).toBe(false);
  });

  it("does not take a four-band 16-bit capture for a colour frame", () => {
    const deep = RGBA.map((band) => ({ ...band, dtype: "uint16", max: 65535 }));
    expect(isPlainColourFrame(bandsResponse(deep))).toBe(false);
  });
});

describe("defaultBandSelection", () => {
  it("assigns the first three declared bands to R, G, B and defaults to Min-Max", () => {
    const bands = [
      { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
      { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
      { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
      { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
    ];
    expect(defaultBandSelection(bands)).toEqual({
      r: "Blue",
      g: "Green",
      b: "Red",
      stretch: "minmax",
    });
  });

  it("falls back to the only band for every channel when fewer than three are reported", () => {
    const bands = [
      { name: "Panchromatic", wavelength_nm: null, dtype: "uint16", min: 0, max: 4095 },
    ];
    expect(defaultBandSelection(bands)).toEqual({
      r: "Panchromatic",
      g: "Panchromatic",
      b: "Panchromatic",
      stretch: "minmax",
    });
  });
});

describe("bandSetSignature", () => {
  it("joins the declared band names in order, so two images with the same bands share one", () => {
    expect(bandSetSignature(FOUR_SPECTRAL)).toBe("Blue,Green,Red,NIR");
  });

  it("differs when the band set differs, even by one band", () => {
    expect(bandSetSignature(FOUR_SPECTRAL)).not.toBe(bandSetSignature(FOUR_SPECTRAL.slice(0, 3)));
  });
});

describe("compositeParams", () => {
  const selection = { r: "Red", g: "Green", b: "Blue", stretch: "minmax" as const };

  it("carries bands/stretch for a multispectral frame whose bands the selection actually names", () => {
    expect(compositeParams(bandsResponse(FOUR_SPECTRAL), selection)).toEqual({
      bands: "Red,Green,Blue",
      stretch: "minmax",
    });
  });

  it("carries nothing for a plain colour frame, even with a selection in hand", () => {
    const rgbaSelection = { r: "red", g: "green", b: "blue", stretch: "minmax" as const };
    expect(compositeParams(bandsResponse(RGBA), rgbaSelection)).toEqual({});
  });

  it("carries nothing when the selection names a band this metadata doesn't declare", () => {
    const stale = { r: "Red", g: "Green", b: "SWIR", stretch: "minmax" as const };
    expect(compositeParams(bandsResponse(FOUR_SPECTRAL), stale)).toEqual({});
  });

  it("carries nothing with no bandsInfo or no selection", () => {
    expect(compositeParams(null, selection)).toEqual({});
    expect(compositeParams(bandsResponse(FOUR_SPECTRAL), null)).toEqual({});
  });
});

describe("showsBandPicker", () => {
  const selection = { r: "Red", g: "Green", b: "Blue", stretch: "minmax" as const };

  it("shows for a multispectral frame with a selection matching its bands", () => {
    expect(showsBandPicker(bandsResponse(FOUR_SPECTRAL), selection)).toBe(true);
  });

  it("hides for a plain colour frame", () => {
    const rgbaSelection = { r: "red", g: "green", b: "blue", stretch: "minmax" as const };
    expect(showsBandPicker(bandsResponse(RGBA), rgbaSelection)).toBe(false);
  });

  it("hides when the selection names a band the current metadata doesn't declare", () => {
    const stale = { r: "Red", g: "Green", b: "SWIR", stretch: "minmax" as const };
    expect(showsBandPicker(bandsResponse(FOUR_SPECTRAL), stale)).toBe(false);
  });

  it("hides while bandsInfo or the selection is not yet known", () => {
    expect(showsBandPicker(null, selection)).toBe(false);
    expect(showsBandPicker(bandsResponse(FOUR_SPECTRAL), null)).toBe(false);
  });
});
