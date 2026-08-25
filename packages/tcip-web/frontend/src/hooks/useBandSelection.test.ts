import { beforeEach, describe, expect, it } from "vitest";
import { act, renderHook } from "@testing-library/react";

import type { ImageBandInfo, ImageBandsResponse } from "@/api/client";
import { useBandSelection } from "@/hooks/useBandSelection";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

function bandsResponse(bands: ImageBandInfo[]): ImageBandsResponse {
  return { band_count: bands.length, bands };
}

const SET_A: ImageBandInfo[] = [
  { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
  { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
  { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
  { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
];

const SET_B: ImageBandInfo[] = [
  { name: "B1", wavelength_nm: 440, dtype: "uint16", min: 0, max: 65535 },
  { name: "B2", wavelength_nm: 490, dtype: "uint16", min: 0, max: 65535 },
  { name: "B3", wavelength_nm: 530, dtype: "uint16", min: 0, max: 65535 },
];

const RGBA: ImageBandInfo[] = ["red", "green", "blue", "alpha"].map((interpretation) => ({
  name: interpretation,
  wavelength_nm: null,
  dtype: "uint8",
  min: 0,
  max: 255,
  interpretation,
}));

describe("useBandSelection", () => {
  it("returns null with a no-op setter while bandsInfo itself is null", () => {
    const { result } = renderHook(() => useBandSelection(null));
    const [selection, setSelection] = result.current;
    expect(selection).toBeNull();

    act(() => setSelection({ r: "Red", g: "Green", b: "Blue", stretch: "minmax" }));
    expect(useStore.getState().bandSelection.byBandSet).toEqual({});
  });

  it("returns null for a plain colour frame, and its setter writes nothing", () => {
    const { result } = renderHook(() => useBandSelection(bandsResponse(RGBA)));
    const [selection, setSelection] = result.current;
    expect(selection).toBeNull();

    act(() => setSelection({ r: "red", g: "green", b: "blue", stretch: "minmax" }));
    expect(useStore.getState().bandSelection.byBandSet).toEqual({});
  });

  it("a selection changed in one instance is the selection a second instance reads over the same band set", () => {
    const tabA = renderHook(() => useBandSelection(bandsResponse(SET_A)));
    const tabB = renderHook(() => useBandSelection(bandsResponse(SET_A)));

    act(() => tabA.result.current[1]({ r: "NIR", g: "Red", b: "Green", stretch: "percent_clip" }));
    tabB.rerender();

    expect(tabB.result.current[0]).toEqual({
      r: "NIR",
      g: "Red",
      b: "Green",
      stretch: "percent_clip",
    });
  });

  it("a composite chosen over one band set survives a detour through another and returns", () => {
    const hook = renderHook(({ bandsInfo }) => useBandSelection(bandsInfo), {
      initialProps: { bandsInfo: bandsResponse(SET_A) },
    });

    act(() => hook.result.current[1]({ r: "NIR", g: "Red", b: "Green", stretch: "minmax" }));
    expect(hook.result.current[0]).toEqual({ r: "NIR", g: "Red", b: "Green", stretch: "minmax" });

    // Detour through a differently-banded image: neither read nor rewritten by this switch.
    hook.rerender({ bandsInfo: bandsResponse(SET_B) });
    expect(hook.result.current[0]).not.toEqual({
      r: "NIR",
      g: "Red",
      b: "Green",
      stretch: "minmax",
    });

    // Back to the original band set: the earlier choice is exactly what comes back.
    hook.rerender({ bandsInfo: bandsResponse(SET_A) });
    expect(hook.result.current[0]).toEqual({ r: "NIR", g: "Red", b: "Green", stretch: "minmax" });
  });

  it("keeps the same setter and default-selection identity across a render with no signature change", () => {
    const hook = renderHook(({ bandsInfo }) => useBandSelection(bandsInfo), {
      initialProps: { bandsInfo: bandsResponse(SET_A) },
    });
    const [firstSelection, firstSetter] = hook.result.current;

    // A fresh bandsInfo object, but the same band-set signature: a dependency array keyed off
    // either the selection or the setter must not see a new identity here.
    hook.rerender({ bandsInfo: bandsResponse(SET_A) });
    const [secondSelection, secondSetter] = hook.result.current;

    expect(secondSetter).toBe(firstSetter);
    expect(secondSelection).toBe(firstSelection);
  });
});
