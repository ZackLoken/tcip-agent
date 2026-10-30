import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import * as client from "@/api/client";
import { OVERVIEWS_REQUIRED } from "@/api/types.generated";
import { STALL_MS, useOverviewBuild, writtenLabel } from "@/hooks/useOverviewBuild";
import { IMAGE_ANSWERS } from "@/test/images";

const { build: BUILD, running: RUNNING, completed: COMPLETED, failed: FAILED } = IMAGE_ANSWERS;
const PATH = BUILD.path;
const URL = `/api/images?path=${PATH}`;

describe("useOverviewBuild", () => {
  it("starts nothing while the image is loading normally (no refusal condition read)", () => {
    const build = vi.spyOn(client.api.images, "buildOverviews");
    const { result } = renderHook(() => useOverviewBuild(URL, PATH, null));
    expect(result.current.building).toBe(false);
    expect(build).not.toHaveBeenCalled();
  });

  it("builds when the loader's read headers named the missing overviews, and shows the wait", async () => {
    vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(BUILD);
    vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(RUNNING);

    const { result } = renderHook(() => useOverviewBuild(URL, PATH, OVERVIEWS_REQUIRED));
    await waitFor(() => expect(result.current.building).toBe(true));
    expect(client.api.images.buildOverviews).toHaveBeenCalledWith({ path: PATH });
    await waitFor(() => expect(result.current.bytesWritten).toBe(RUNNING.bytes_written));
  });

  it("reads a sidecar's bytes in decimal megabytes", () => {
    expect(writtenLabel(4_200_000)).toBe("4.2 MB written");
  });

  it("clears the wait and asks for the image again once the build completes", async () => {
    vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(BUILD);
    vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(COMPLETED);

    const { result } = renderHook(() => useOverviewBuild(URL, PATH, OVERVIEWS_REQUIRED));
    await waitFor(() => expect(result.current.reloadToken).toBe(1));
    expect(result.current).toEqual({
      building: false,
      bytesWritten: COMPLETED.bytes_written,
      error: null,
      reloadToken: 1,
    });
  });

  it("reports a failed build instead of leaving the viewer waiting", async () => {
    vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(BUILD);
    vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(FAILED);

    const { result } = renderHook(() => useOverviewBuild(URL, PATH, OVERVIEWS_REQUIRED));
    await waitFor(() => expect(result.current.error).toBe(FAILED.error));
    expect(result.current.building).toBe(false);
    expect(result.current.reloadToken).toBe(0);
  });

  it("stops waiting on a build whose reported sidecar size never moves", async () => {
    vi.useFakeTimers();
    try {
      vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(BUILD);
      vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(RUNNING);

      const { result } = renderHook(() => useOverviewBuild(URL, PATH, OVERVIEWS_REQUIRED));
      await act(() => vi.advanceTimersByTimeAsync(STALL_MS * 2));
      expect(result.current.building).toBe(false);
      expect(result.current.error).toContain("stopped reporting progress");
      expect(result.current.error).toContain(`held at ${writtenLabel(RUNNING.bytes_written)}`);
      expect(result.current.reloadToken).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("starts one build per image, so a build that fails is never retried in a loop", async () => {
    const build = vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(BUILD);
    vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(FAILED);

    const { result, rerender } = renderHook(() => useOverviewBuild(URL, PATH, OVERVIEWS_REQUIRED));
    await waitFor(() => expect(result.current.error).toBe(FAILED.error));
    rerender();
    rerender();
    expect(build).toHaveBeenCalledTimes(1);
  });
});
