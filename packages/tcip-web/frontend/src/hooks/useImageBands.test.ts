import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { useImageBands } from "@/hooks/useImageBands";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useImageBands", () => {
  it("returns null and fetches nothing for a null path", () => {
    const spy = vi.spyOn(api.images, "bands");
    const { result } = renderHook(() => useImageBands(null));
    expect(result.current).toBeNull();
    expect(spy).not.toHaveBeenCalled();
  });

  it("resolves the reported band contract for a given path", async () => {
    vi.spyOn(api.images, "bands").mockResolvedValue({
      band_count: 4,
      bands: [
        { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
        { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
        { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
        { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
      ],
    });
    const { result } = renderHook(() =>
      useImageBands("C:/data/images/2026-01-01/DJI_0001.bandgroup"),
    );
    await waitFor(() => expect(result.current?.band_count).toBe(4));
  });

  it("resolves null (rather than throwing) when the fetch fails", async () => {
    vi.spyOn(api.images, "bands").mockRejectedValue(new Error("404 not found"));
    const { result } = renderHook(() => useImageBands("C:/data/images/2026-01-01/img1.jpg"));
    await waitFor(() => expect(result.current).toBeNull());
  });
});
