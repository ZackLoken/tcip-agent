import { afterEach, describe, expect, it, vi } from "vitest";

import { trainingApi } from "@/api/training";

describe("trainingApi.cancel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs to the run's cancel endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ run_id: "r1", status: "running", cancel_requested: true }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    const res = await trainingApi.cancel("r1");
    expect(res.cancel_requested).toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/training/runs/r1/cancel",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
