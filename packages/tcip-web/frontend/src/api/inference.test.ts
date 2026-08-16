import { afterEach, describe, expect, it, vi } from "vitest";

import { StructuredRefusalError } from "@/api/http";
import { inferenceApi, operationalizationRefusalOf, resultsApi } from "@/api/inference";

function stubFetch(status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    } as Response),
  );
}

describe("results api error handling (asJson)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the parsed body on a 2xx", async () => {
    stubFetch(200, {
      models: [{ name: "m", checkpoint_path: "/w.pt", tags: ["best"] }],
    });
    const res = await resultsApi.registeredModels("/proj");
    expect(res.models).toHaveLength(1);
    expect(res.models[0].checkpoint_path).toBe("/w.pt");
  });

  it("throws on a non-2xx instead of silently returning the error body", async () => {
    // Regression: a 404 error body ({detail}) used to be returned as-is, so callers
    // read `.models`/`.rows` off undefined and crashed on the next render.
    stubFetch(404, { detail: "no plant mapping" });
    await expect(resultsApi.registeredModels("/proj")).rejects.toThrow("no plant mapping");
  });
});

describe("exportCsv refusal decoding", () => {
  const REQUEST = {
    project_root: "C:/proj",
    mapping_path: "C:/proj/.tcip/state/plant_mapping.json",
    predictions_by_date: {},
    trait: "catkin_50per_date",
  };

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("carries a structured refusal through the blob path parsed, not stringified", async () => {
    const detail = {
      kind: "operationalization",
      state: 1,
      trait: "catkin_50per_date",
      delivery_kind: "state_crossing_dates",
      message: "no operationalization is recorded for a state_crossing_dates delivery",
    };
    stubFetch(400, { detail });

    const thrown = await resultsApi.exportCsv(REQUEST, "curves", "x.csv").catch((e: unknown) => e);
    expect(thrown).toBeInstanceOf(StructuredRefusalError);
    expect((thrown as Error).message).not.toContain("[object Object]");
    expect(operationalizationRefusalOf(thrown)?.delivery_kind).toBe("state_crossing_dates");
  });

  it("keeps the status-only message when the refusal body carries no detail", async () => {
    stubFetch(500, {});
    await expect(resultsApi.exportCsv(REQUEST, "curves", "x.csv")).rejects.toThrow(
      "export_csv failed: 500",
    );
  });
});

describe("inferenceApi.cancel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs to the job's cancel endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ job_id: "j1", status: "running", cancel_requested: true }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    const res = await inferenceApi.cancel("j1");
    expect(res.cancel_requested).toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/inference/jobs/j1/cancel",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
