import { describe, expect, it, vi } from "vitest";

import { trainingApi } from "@/api/training";
import { cancelAnswer } from "@/test/training";

describe("trainingApi.cancel", () => {
  it("POSTs to the run's cancel endpoint, naming the person", async () => {
    const answer = cancelAnswer({ experiment_id: "r1" });
    const fetchMock = vi.fn().mockResolvedValue(Response.json(answer));
    vi.stubGlobal("fetch", fetchMock);

    const res = await trainingApi.cancel("r1", "jordan");
    expect(res).toEqual(answer);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/training/runs/r1/cancel",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ user: "jordan" }) }),
    );
  });
});
