import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ImageBands } from "@/api/types.generated";
import { BandPicker } from "@/components/BandPicker";
import { defaultBandSelection, type BandSelection } from "@/lib/bandSelection";
import { BAND_ANSWERS } from "@/test/bands";

afterEach(cleanup);

const FOUR = BAND_ANSWERS.four_spectral;
const FOUR_BANDS = FOUR.bands;
const SINGLE = BAND_ANSWERS.single;

/** The picker over the produced answer `info` holding `selection`; returns its change handler. */
function renderPicker(info: ImageBands, selection: BandSelection, onChange = vi.fn()) {
  render(<BandPicker info={info} selection={selection} onChange={onChange} />);
  return onChange;
}

/** The selection of a one-band answer: its one band on every channel. */
function singleSelection(): BandSelection {
  const [{ name }] = SINGLE.bands;
  return { r: name, g: name, b: name, stretch: SINGLE.default_stretch };
}

describe("BandPicker", () => {
  it("offers one R, one G and one B select for a multi-band source", () => {
    const selection = defaultBandSelection(FOUR);
    renderPicker(FOUR, selection);
    expect(screen.getByLabelText("R band")).toBeInTheDocument();
    expect(screen.getByLabelText("G band")).toBeInTheDocument();
    expect(screen.getByLabelText("B band")).toBeInTheDocument();
    expect(screen.getByLabelText("Stretch")).toBeInTheDocument();
  });

  it("collapses to a single Band dropdown for a one-band source", () => {
    expect(SINGLE.band_count).toBe(1);
    renderPicker(SINGLE, singleSelection());
    expect(screen.getByLabelText("Band")).toBeInTheDocument();
    expect(screen.queryByLabelText("R band")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("G band")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("B band")).not.toBeInTheDocument();
  });

  it("the single dropdown sets r, g and b together", () => {
    const selection = singleSelection();
    const onChange = renderPicker(SINGLE, selection);
    fireEvent.change(screen.getByLabelText("Band"), { target: { value: selection.r } });
    expect(onChange).toHaveBeenCalledWith(selection);
  });

  it("changing one channel of a multi-band triad leaves the others untouched", () => {
    const selection = defaultBandSelection(FOUR);
    const fourth = FOUR_BANDS[3].name;
    const onChange = renderPicker(FOUR, selection);
    fireEvent.change(screen.getByLabelText("R band"), { target: { value: fourth } });
    expect(onChange).toHaveBeenCalledWith({ ...selection, r: fourth });
  });

  it("offers exactly Min-Max and Percent Clip as stretch options", () => {
    renderPicker(FOUR, defaultBandSelection(FOUR));
    const options = Array.from(screen.getByLabelText("Stretch").querySelectorAll("option")).map(
      (o) => o.textContent,
    );
    expect(options).toEqual(["Min-Max", "Percent Clip"]);
  });

  it("says the band ranges came from a sample, and from how much of the raster", () => {
    const sampled = BAND_ANSWERS.sampled;
    renderPicker(sampled, defaultBandSelection(sampled));
    const percent = Math.round(Number(sampled.pixel_fraction) * 100);
    expect(screen.getByText(`stats from a ${percent}% pixel sample`)).toBeInTheDocument();
  });

  it("names a transparency band as one, so it is not offered as a color to look at", () => {
    const rgba = BAND_ANSWERS.deep_rgba;
    const alpha = rgba.bands[3];
    expect(alpha.interpretation).toBe("alpha");
    renderPicker(rgba, defaultBandSelection(rgba));
    const options = Array.from(screen.getByLabelText("R band").querySelectorAll("option")).map(
      (o) => o.textContent,
    );
    expect(options[3]).toBe(`${alpha.name} (alpha)`);
  });

  it("says nothing when the reported ranges are the raster's own exact bounds", () => {
    expect(FOUR.sampled).toBe(false);
    renderPicker(FOUR, defaultBandSelection(FOUR));
    expect(screen.queryByText(/pixel sample/)).not.toBeInTheDocument();
  });
});
