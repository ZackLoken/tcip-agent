import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { subjectsApi } from "@/api/subjects";
import { StructuredRefusalError } from "@/api/http";
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
    vi.spyOn(subjectsApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      stale_definition: [],
    });
    vi.spyOn(subjectsApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "negative" },
      unreadable: [],
    });
    const bulk = vi
      .spyOn(subjectsApi, "setImageStatusBulk")
      .mockResolvedValue({ status: "ok", n: 0, digest_unstamped: [] });

    renderHook(() => useImageStatusHydrate(PARAMS));

    await waitFor(() => expect(useStore.getState().imageStatus.staleMarks).toEqual(["img1.jpg"]));
    expect(bulk).not.toHaveBeenCalled();
    expect(useStore.getState().imageStatus.byImage["img1.jpg"]).toBe("complete");
  });

  it("heals an unconfirmed name from unannotated to partial and writes it", async () => {
    vi.spyOn(subjectsApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "unannotated" },
      stale_definition: [],
    });
    vi.spyOn(subjectsApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "partial" },
      unreadable: [],
    });
    const bulk = vi
      .spyOn(subjectsApi, "setImageStatusBulk")
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
    vi.spyOn(subjectsApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      stale_definition: ["img1.jpg"],
    });
    vi.spyOn(subjectsApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      unreadable: [],
    });
    vi.spyOn(subjectsApi, "setImageStatusBulk").mockResolvedValue({
      status: "ok",
      n: 0,
      digest_unstamped: [],
    });

    renderHook(() => useImageStatusHydrate(PARAMS));

    await waitFor(() => expect(useStore.getState().imageStatus.staleMarks).toEqual(["img1.jpg"]));
  });

  it("unions a digest-stale name with a separately content-stale name", async () => {
    const params = { ...PARAMS, imageList: ["img1.jpg", "img2.jpg"] };
    vi.spyOn(subjectsApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete", "img2.jpg": "complete" },
      stale_definition: ["img2.jpg"],
    });
    vi.spyOn(subjectsApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "negative", "img2.jpg": "complete" },
      unreadable: [],
    });
    vi.spyOn(subjectsApi, "setImageStatusBulk").mockResolvedValue({
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
    vi.spyOn(subjectsApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      stale_definition: ["img1.jpg", "img_outside.jpg"],
    });
    vi.spyOn(subjectsApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "complete" },
      unreadable: [],
    });
    vi.spyOn(subjectsApi, "setImageStatusBulk").mockResolvedValue({
      status: "ok",
      n: 0,
      digest_unstamped: [],
    });

    renderHook(() => useImageStatusHydrate(PARAMS));

    await waitFor(() => expect(useStore.getState().imageStatus.staleMarks).toEqual(["img1.jpg"]));
  });

  it("does nothing with no subject selected: nothing to scope image status to yet", () => {
    const load = vi.spyOn(subjectsApi, "loadImageStatus");
    renderHook(() => useImageStatusHydrate({ ...PARAMS, subject: null }));
    expect(load).not.toHaveBeenCalled();
  });

  it("clears a prior dataset's stale marks even when this one has no subject selected yet", () => {
    useStore.setState((s) => ({ imageStatus: { ...s.imageStatus, staleMarks: ["img1.jpg"] } }));
    renderHook(() => useImageStatusHydrate({ ...PARAMS, subject: null }));
    expect(useStore.getState().imageStatus.staleMarks).toEqual([]);
  });

  it("continues with its own writes and toasts the message when the bulk write's audit line is lost", async () => {
    vi.spyOn(subjectsApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "unannotated" },
      stale_definition: [],
    });
    vi.spyOn(subjectsApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "partial" },
      unreadable: [],
    });
    const message = "gui_set_image_status_bulk completed and its audit entry could not be written";
    vi.spyOn(subjectsApi, "setImageStatusBulk").mockRejectedValue(
      new StructuredRefusalError(
        { error: "audit_entry_not_written", message, committed: { status: "ok", n: 1 } },
        409,
        message,
      ),
    );

    renderHook(() => useImageStatusHydrate(PARAMS));

    await waitFor(() =>
      expect(useStore.getState().imageStatus.byImage["img1.jpg"]).toBe("partial"),
    );
    expect(useStore.getState().toasts.map((t) => t.message)).toContainEqual(message);
  });

  it("skips its own writes on any other bulk-write failure", async () => {
    vi.spyOn(subjectsApi, "loadImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "unannotated" },
      stale_definition: [],
    });
    vi.spyOn(subjectsApi, "deriveImageStatus").mockResolvedValue({
      statuses: { "img1.jpg": "partial" },
      unreadable: [],
    });
    vi.spyOn(subjectsApi, "setImageStatusBulk").mockRejectedValue(new Error("network down"));

    renderHook(() => useImageStatusHydrate(PARAMS));

    await waitFor(() =>
      expect(useStore.getState().toasts.at(-1)?.message).toBe(
        "Could not load the image status for this project.",
      ),
    );
    expect(useStore.getState().imageStatus.byImage["img1.jpg"]).toBeUndefined();
  });
});
