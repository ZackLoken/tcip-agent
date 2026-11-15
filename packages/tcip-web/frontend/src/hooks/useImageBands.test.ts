import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { ImageBands } from "@/api/types.generated";
import { useImageBands } from "@/hooks/useImageBands";
import { BAND_ANSWERS } from "@/test/bands";

describe("useImageBands", () => {
  it("returns null and fetches nothing for a null path", () => {
    const spy = vi.spyOn(api.images, "bands");
    const { result } = renderHook(() => useImageBands(null));
    expect(result.current).toBeNull();
    expect(spy).not.toHaveBeenCalled();
  });

  it("resolves the reported band contract for a given path", async () => {
    vi.spyOn(api.images, "bands").mockResolvedValue(BAND_ANSWERS.four_spectral);
    const { result } = renderHook(() =>
      useImageBands("C:/data/images/2026-01-01/DJI_0001.bandgroup"),
    );
    await waitFor(() => expect(result.current?.band_count).toBe(4));
  });

  it("answers null for a new path until that path's own answer arrives", async () => {
    const first = "C:/data/images/2026-01-01/DJI_0001.bandgroup";
    const second = "C:/data/images/2026-01-01/img2.jpg";
    const secondAnswer = Promise.withResolvers<ImageBands>();
    vi.spyOn(api.images, "bands").mockImplementation((path) =>
      path === first ? Promise.resolve(BAND_ANSWERS.four_spectral) : secondAnswer.promise,
    );
    const { result, rerender } = renderHook((path) => useImageBands(path), {
      initialProps: first,
    });
    await waitFor(() => expect(result.current).toBe(BAND_ANSWERS.four_spectral));

    rerender(second);
    expect(result.current).toBeNull();

    await act(() => Promise.resolve(secondAnswer.resolve(BAND_ANSWERS.plain)));
    expect(result.current).toBe(BAND_ANSWERS.plain);
  });

  it("resolves null (rather than throwing) when the fetch fails", async () => {
    vi.spyOn(api.images, "bands").mockRejectedValue(new Error("404 not found"));
    const { result } = renderHook(() => useImageBands("C:/data/images/2026-01-01/img1.jpg"));
    await waitFor(() => expect(result.current).toBeNull());
  });
});
