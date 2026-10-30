import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { useBandSelection } from "@/hooks/useBandSelection";
import { defaultBandSelection } from "@/lib/bandSelection";
import { useStore } from "@/store";
import { BAND_ANSWERS } from "@/test/bands";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

const { four_spectral: SET_A, three_unlabeled: SET_B, rgba: RGBA } = BAND_ANSWERS;
const [FIRST, SECOND, THIRD, FOURTH] = SET_A.bands.map((band) => band.name);

/** A selection over `SET_A` other than its default, its fourth band first. */
const CHOSEN = { r: FOURTH, g: THIRD, b: SECOND, stretch: SET_A.default_stretch };

/** A selection of `bandsInfo`'s default bands. */
const ownBands = defaultBandSelection;

describe("useBandSelection", () => {
  it("returns null with a no-op setter while bandsInfo itself is null", () => {
    const { result } = renderHook(() => useBandSelection(null));
    const [selection, setSelection] = result.current;
    expect(selection).toBeNull();

    act(() => setSelection(ownBands(SET_A)));
    expect(useStore.getState().bandSelection.byBandSet).toEqual({});
  });

  it("returns null for a frame the server displays plain, and its setter writes nothing", () => {
    const { result } = renderHook(() => useBandSelection(RGBA));
    const [selection, setSelection] = result.current;
    expect(selection).toBeNull();

    act(() => setSelection(ownBands(RGBA)));
    expect(useStore.getState().bandSelection.byBandSet).toEqual({});
  });

  it("a selection changed in one instance is the selection a second instance reads over the same band set", () => {
    const tabA = renderHook(() => useBandSelection(SET_A));
    const tabB = renderHook(() => useBandSelection(SET_A));
    const clipped = { ...CHOSEN, stretch: "percent_clip" as const };

    act(() => tabA.result.current[1](clipped));
    tabB.rerender();

    expect(tabB.result.current[0]).toEqual(clipped);
  });

  it("a composite chosen over one band set survives a detour through another and returns", () => {
    const hook = renderHook(({ bandsInfo }) => useBandSelection(bandsInfo), {
      initialProps: { bandsInfo: SET_A },
    });
    expect(hook.result.current[0]).not.toEqual(CHOSEN);
    expect(FIRST).not.toBe(CHOSEN.r);

    act(() => hook.result.current[1](CHOSEN));
    expect(hook.result.current[0]).toEqual(CHOSEN);

    // Detour through a differently-banded image: neither read nor rewritten by this switch.
    hook.rerender({ bandsInfo: SET_B });
    expect(hook.result.current[0]).not.toEqual(CHOSEN);

    // Back to the original band set: the earlier choice is exactly what comes back.
    hook.rerender({ bandsInfo: SET_A });
    expect(hook.result.current[0]).toEqual(CHOSEN);
  });

  it("keeps the same setter and default-selection identity across a render with no signature change", () => {
    const hook = renderHook(({ bandsInfo }) => useBandSelection(bandsInfo), {
      initialProps: { bandsInfo: SET_A },
    });
    const [firstSelection, firstSetter] = hook.result.current;

    // A fresh bandsInfo object, but the same band-set signature: a dependency array keyed off
    // either the selection or the setter must not see a new identity here.
    hook.rerender({ bandsInfo: { ...SET_A } });
    const [secondSelection, secondSetter] = hook.result.current;

    expect(secondSetter).toBe(firstSetter);
    expect(secondSelection).toBe(firstSelection);
  });
});
