import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { BandPicker } from "@/components/BandPicker";
import { defaultBandSelection, type BandSelection } from "@/lib/bandSelection";

afterEach(cleanup);

const FOUR_BANDS = [
  { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
  { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
  { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
  { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
];

function renderPicker(bandCount: number, selection: BandSelection, onChange = vi.fn()) {
  render(
    <BandPicker
      bandCount={bandCount}
      bands={bandCount === 1 ? [FOUR_BANDS[0]] : FOUR_BANDS}
      selection={selection}
      onChange={onChange}
    />,
  );
  return onChange;
}

describe("BandPicker", () => {
  it("offers one R, one G and one B select for a multi-band source", () => {
    const selection = defaultBandSelection(FOUR_BANDS);
    renderPicker(4, selection);
    expect(screen.getByLabelText("R band")).toBeInTheDocument();
    expect(screen.getByLabelText("G band")).toBeInTheDocument();
    expect(screen.getByLabelText("B band")).toBeInTheDocument();
    expect(screen.getByLabelText("Stretch")).toBeInTheDocument();
  });

  it("collapses to a single Band dropdown when band_count is 1", () => {
    const selection: BandSelection = { r: "Blue", g: "Blue", b: "Blue", stretch: "minmax" };
    renderPicker(1, selection);
    expect(screen.getByLabelText("Band")).toBeInTheDocument();
    expect(screen.queryByLabelText("R band")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("G band")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("B band")).not.toBeInTheDocument();
  });

  it("the single dropdown sets r, g and b together", () => {
    const selection: BandSelection = { r: "Blue", g: "Blue", b: "Blue", stretch: "minmax" };
    const onChange = renderPicker(1, selection);
    fireEvent.change(screen.getByLabelText("Band"), { target: { value: "Blue" } });
    expect(onChange).toHaveBeenCalledWith({ r: "Blue", g: "Blue", b: "Blue", stretch: "minmax" });
  });

  it("changing one channel of a multi-band triad leaves the others untouched", () => {
    const selection = defaultBandSelection(FOUR_BANDS); // Blue, Green, Red
    const onChange = renderPicker(4, selection);
    fireEvent.change(screen.getByLabelText("R band"), { target: { value: "NIR" } });
    expect(onChange).toHaveBeenCalledWith({
      r: "NIR",
      g: "Green",
      b: "Red",
      stretch: "minmax",
    });
  });

  it("offers exactly Min-Max and Percent Clip as stretch options", () => {
    renderPicker(4, defaultBandSelection(FOUR_BANDS));
    const options = Array.from(screen.getByLabelText("Stretch").querySelectorAll("option")).map(
      (o) => o.textContent,
    );
    expect(options).toEqual(["Min-Max", "Percent Clip"]);
  });
});
