import { describe, expect, it } from "vitest";

import { defaultBandSelection } from "@/lib/bandSelection";

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
