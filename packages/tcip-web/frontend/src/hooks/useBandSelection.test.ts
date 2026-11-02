import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { useBandSelection } from "@/hooks/useBandSelection";
import { defaultBandSelection, NO_BAND_CHOICE, type BandChoice } from "@/lib/bandSelection";
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

/** The banded choice `choice` is; a choice offering none throws. */
function banded(choice: BandChoice): Extract<BandChoice, { kind: "banded" }> {
  if (choice.kind !== "banded") throw new Error("the image offers no band choice");
  return choice;
}

describe("useBandSelection", () => {
  it("offers no band choice while bandsInfo itself is null", () => {
    const { result } = renderHook(() => useBandSelection(null));
    expect(result.current).toEqual(NO_BAND_CHOICE);
  });

  it("offers no band choice for a frame the server displays plain, and records nothing", () => {
    const { result } = renderHook(() => useBandSelection(RGBA));
    expect(result.current).toEqual(NO_BAND_CHOICE);
    expect(useStore.getState().bandSelection.byBandSet).toEqual({});
  });

  it("a selection changed in one instance is the selection a second instance reads over the same band set", () => {
    const tabA = renderHook(() => useBandSelection(SET_A));
    const tabB = renderHook(() => useBandSelection(SET_A));
    const clipped = { ...CHOSEN, stretch: "percent_clip" as const };

    act(() => banded(tabA.result.current).choose(clipped));
    tabB.rerender();

    expect(banded(tabB.result.current).selection).toEqual(clipped);
  });

  it("a composite chosen over one band set survives a detour through another and returns", () => {
    const hook = renderHook(({ bandsInfo }) => useBandSelection(bandsInfo), {
      initialProps: { bandsInfo: SET_A },
    });
    const selection = () => banded(hook.result.current).selection;
    expect(selection()).toEqual(ownBands(SET_A));
    expect(FIRST).not.toBe(CHOSEN.r);

    act(() => banded(hook.result.current).choose(CHOSEN));
    expect(selection()).toEqual(CHOSEN);

    // Detour through a differently-banded image: neither read nor rewritten by this switch.
    hook.rerender({ bandsInfo: SET_B });
    expect(selection()).toEqual(ownBands(SET_B));

    // Back to the original band set: the earlier choice is exactly what comes back.
    hook.rerender({ bandsInfo: SET_A });
    expect(selection()).toEqual(CHOSEN);
  });

  it("keeps the same setter and default-selection identity across a render with no signature change", () => {
    const hook = renderHook(({ bandsInfo }) => useBandSelection(bandsInfo), {
      initialProps: { bandsInfo: SET_A },
    });
    const first = banded(hook.result.current);

    // A fresh bandsInfo object, but the same band-set signature: a dependency array keyed off
    // either the selection or the setter must not see a new identity here.
    hook.rerender({ bandsInfo: { ...SET_A } });
    const second = banded(hook.result.current);

    expect(second.choose).toBe(first.choose);
    expect(second.selection).toBe(first.selection);
  });
});
