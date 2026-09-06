import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { classesApi } from "@/api/classes";
import { useImageStatusHydrate } from "@/hooks/useImageStatusHydrate";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

afterEach(() => {
  vi.restoreAllMocks();
});

const PARAMS = {
  projectRoot: "C:/proj",
  subject: "subject_a",
  datasetRoot: "C:/data",
  datasetDate: "2026-01-01",
  annotationsDir: "C:/data/annotations/2026-01-01",
  imageList: ["img1.jpg"],
};

describe("useImageStatusHydrate", () => {
  it("flags a stored complete whose derived token is negative as stale, never writes it", async () => {
    vi.spyOn(classesApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      stale_definition: [],
    });
    vi.spyOn(classesApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "negative" },
      unreadable: [],
    });
    const bulk = vi
      .spyOn(classesApi, "setImageStatusBulk")
      .mockResolvedValue({ status: "ok", n: 0, digest_unstamped: [] });

    renderHook(() => useImageStatusHydrate(PARAMS));

    await waitFor(() => expect(useStore.getState().imageStatus.staleMarks).toEqual(["img1.jpg"]));
    expect(bulk).not.toHaveBeenCalled();
    expect(useStore.getState().imageStatus.byImage["img1.jpg"]).toBe("complete");
  });

  it("heals an unconfirmed name from unannotated to partial and writes it", async () => {
    vi.spyOn(classesApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "unannotated" },
      stale_definition: [],
    });
    vi.spyOn(classesApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "partial" },
      unreadable: [],
    });
    const bulk = vi
      .spyOn(classesApi, "setImageStatusBulk")
      .mockResolvedValue({ status: "ok", n: 1, digest_unstamped: [] });

    renderHook(() => useImageStatusHydrate(PARAMS));

    await waitFor(() =>
      expect(useStore.getState().imageStatus.byImage["img1.jpg"]).toBe("partial"),
    );
    expect(useStore.getState().imageStatus.staleMarks).toEqual([]);
    expect(bulk).toHaveBeenCalledWith(
      "C:/proj",
      { "img1.jpg": "partial" },
      "subject_a",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
      undefined,
    );
  });

  it("flags a digest-stale name with no content disagreement", async () => {
    vi.spyOn(classesApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      stale_definition: ["img1.jpg"],
    });
    vi.spyOn(classesApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      unreadable: [],
    });
    vi.spyOn(classesApi, "setImageStatusBulk").mockResolvedValue({
      status: "ok",
      n: 0,
      digest_unstamped: [],
    });

    renderHook(() => useImageStatusHydrate(PARAMS));

    await waitFor(() => expect(useStore.getState().imageStatus.staleMarks).toEqual(["img1.jpg"]));
  });

  it("unions a digest-stale name with a separately content-stale name", async () => {
    const params = { ...PARAMS, imageList: ["img1.jpg", "img2.jpg"] };
    vi.spyOn(classesApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete", "img2.jpg": "complete" },
      stale_definition: ["img2.jpg"],
    });
    vi.spyOn(classesApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "negative", "img2.jpg": "complete" },
      unreadable: [],
    });
    vi.spyOn(classesApi, "setImageStatusBulk").mockResolvedValue({
      status: "ok",
      n: 0,
      digest_unstamped: [],
    });

    renderHook(() => useImageStatusHydrate(params));

    await waitFor(() =>
      expect(useStore.getState().imageStatus.staleMarks).toEqual(["img1.jpg", "img2.jpg"]),
    );
  });

  it("leaves a digest-stale name out of staleMarks when it is not in the loaded image list", async () => {
    vi.spyOn(classesApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      stale_definition: ["img1.jpg", "img_outside.jpg"],
    });
    vi.spyOn(classesApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      unreadable: [],
    });
    vi.spyOn(classesApi, "setImageStatusBulk").mockResolvedValue({
      status: "ok",
      n: 0,
      digest_unstamped: [],
    });

    renderHook(() => useImageStatusHydrate(PARAMS));

    await waitFor(() => expect(useStore.getState().imageStatus.staleMarks).toEqual(["img1.jpg"]));
  });

  it("does nothing with no subject selected: nothing to scope image status to yet", () => {
    const load = vi.spyOn(classesApi, "loadImageStatus");
    renderHook(() => useImageStatusHydrate({ ...PARAMS, subject: null }));
    expect(load).not.toHaveBeenCalled();
  });

  it("clears a prior dataset's stale marks even when this one has no subject selected yet", () => {
    useStore.setState((s) => ({ imageStatus: { ...s.imageStatus, staleMarks: ["img1.jpg"] } }));
    renderHook(() => useImageStatusHydrate({ ...PARAMS, subject: null }));
    expect(useStore.getState().imageStatus.staleMarks).toEqual([]);
  });
});
