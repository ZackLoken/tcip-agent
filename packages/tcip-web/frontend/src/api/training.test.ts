import { describe, expect, it } from "vitest";

import { trainingApi } from "@/api/training";
import { fetchAnswers } from "@/test/refusals";
import { CANCEL } from "@/test/training";

describe("trainingApi.cancel", () => {
  it("POSTs to the run's cancel endpoint, naming the person", async () => {
    const answer = CANCEL;
    const fetchMock = fetchAnswers(200, answer);

    const res = await trainingApi.cancel("r1", { user: "jordan" });
    expect(res).toEqual(answer);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/training/runs/r1/cancel",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ user: "jordan" }) }),
    );
  });
});
