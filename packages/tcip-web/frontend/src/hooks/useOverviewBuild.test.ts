import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import * as client from "@/api/client";
import { STALL_MS, useOverviewBuild } from "@/hooks/useOverviewBuild";

const URL = "/api/images?path=C:/data/mosaic.tif";
const PATH = "C:/data/mosaic.tif";

afterEach(() => {
  vi.restoreAllMocks();
});

function job(status: client.OverviewJob["status"], progress = 0, error: string | null = null) {
  return { job_id: "ovr-1", path: PATH, status, progress, error };
}

describe("useOverviewBuild", () => {
  it("starts nothing while the image is loading normally", () => {
    const refusal = vi.spyOn(client.api.images, "refusal");
    const { result } = renderHook(() => useOverviewBuild(URL, PATH, false));
    expect(result.current.building).toBe(false);
    expect(refusal).not.toHaveBeenCalled();
  });

  it("starts nothing when the load failed for some other reason", async () => {
    vi.spyOn(client.api.images, "refusal").mockResolvedValue(null);
    const build = vi.spyOn(client.api.images, "buildOverviews");
    const { result } = renderHook(() => useOverviewBuild(URL, PATH, true));
    await waitFor(() => expect(client.api.images.refusal).toHaveBeenCalled());
    expect(build).not.toHaveBeenCalled();
    expect(result.current.building).toBe(false);
  });

  it("builds when the server named the missing overviews, and shows the wait", async () => {
    vi.spyOn(client.api.images, "refusal").mockResolvedValue("overviews_required");
    vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(job("running"));
    vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(job("running", 0.4));

    const { result } = renderHook(() => useOverviewBuild(URL, PATH, true));
    await waitFor(() => expect(result.current.building).toBe(true));
    expect(client.api.images.buildOverviews).toHaveBeenCalledWith(PATH);
    await waitFor(() => expect(result.current.progress).toBe(0.4));
  });

  it("clears the wait and asks for the image again once the build completes", async () => {
    vi.spyOn(client.api.images, "refusal").mockResolvedValue("overviews_required");
    vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(job("running"));
    vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(job("completed", 1));

    const { result } = renderHook(() => useOverviewBuild(URL, PATH, true));
    await waitFor(() => expect(result.current.reloadToken).toBe(1));
    expect(result.current.building).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("reports a failed build instead of leaving the viewer waiting", async () => {
    vi.spyOn(client.api.images, "refusal").mockResolvedValue("overviews_required");
    vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(job("running"));
    vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(
      job("failed", 0.2, "no room on the imagery volume"),
    );

    const { result } = renderHook(() => useOverviewBuild(URL, PATH, true));
    await waitFor(() => expect(result.current.error).toBe("no room on the imagery volume"));
    expect(result.current.building).toBe(false);
    expect(result.current.reloadToken).toBe(0);
  });

  it("stops waiting on a build whose reported progress never moves", async () => {
    vi.useFakeTimers();
    try {
      vi.spyOn(client.api.images, "refusal").mockResolvedValue("overviews_required");
      vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(job("running"));
      vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(job("running", 0.25));

      const { result } = renderHook(() => useOverviewBuild(URL, PATH, true));
      await vi.advanceTimersByTimeAsync(STALL_MS * 2);
      expect(result.current.building).toBe(false);
      expect(result.current.error).toContain("stopped reporting progress");
      expect(result.current.reloadToken).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("starts one build per image, so a build that fails is never retried in a loop", async () => {
    vi.spyOn(client.api.images, "refusal").mockResolvedValue("overviews_required");
    const build = vi.spyOn(client.api.images, "buildOverviews").mockResolvedValue(job("running"));
    vi.spyOn(client.api.images, "overviewJob").mockResolvedValue(job("failed", 0, "denied"));

    const { result, rerender } = renderHook(() => useOverviewBuild(URL, PATH, true));
    await waitFor(() => expect(result.current.error).toBe("denied"));
    rerender();
    rerender();
    expect(build).toHaveBeenCalledTimes(1);
  });
});
