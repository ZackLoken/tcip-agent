import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, loadedLabels } from "@/api/client";
import { usePrefetchAdjacentImages } from "@/hooks/usePrefetchAdjacentImages";
import { currentImage } from "@/lib/paths";
import { useStore } from "@/store";
import { selectCanvasMatchesDataset } from "@/store/slices/canvas";
import { LABEL_ANSWERS } from "@/test/labels";
import { GROUPED_SELECTION, selected, SEPARATED_SELECTION } from "@/test/store";

const SELECTION = selected({ count: 2 }, SEPARATED_SELECTION);
const [FIRST, SECOND] = SELECTION.images;

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState((s) => ({ gui: { ...s.gui, dataset: SELECTION } }));
});

afterEach(() => {
  useStore.setState(initialStoreState, true);
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("the selection's image places", () => {
  it("names the current image at the place the selection carries", () => {
    expect(currentImage(SELECTION)).toEqual(FIRST);
  });

  it("matches the canvas to the dataset by the places the selection carries", () => {
    const loaded = (image_path: string) => {
      useStore
        .getState()
        .loadLabelsIntoCanvas(loadedLabels({ ...LABEL_ANSWERS.loaded, image_path }));
      return selectCanvasMatchesDataset(useStore.getState());
    };

    expect(loaded(SECOND.path)).toBe(true);
    expect(loaded(GROUPED_SELECTION.images[0].path)).toBe(false);
  });

  it("warms the next image at the place the selection carries", () => {
    vi.useFakeTimers();
    const url = vi.spyOn(api.images, "url");

    renderHook(() => usePrefetchAdjacentImages(1000));
    act(() => {
      vi.advanceTimersByTime(600);
    });

    expect(url.mock.calls.map(([path]) => path)).toEqual([SECOND.path]);
  });
});
