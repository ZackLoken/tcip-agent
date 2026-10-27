import { describe, expect, it, vi } from "vitest";

import { StructuredRefusalError } from "@/api/http";
import { inferenceApi, operationalizationRefusalOf, resultsApi } from "@/api/inference";
import { ACKNOWLEDGED_BY_HEADER } from "@/api/types.generated";
import { COUNT_EXPORT_HEADERS } from "@/test/deliveryRecords";

function stubFetch(status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      json: () => Promise.resolve(body),
    }),
  );
}

describe("results api error handling (asJson)", () => {
  it("returns the parsed body on a 2xx", async () => {
    stubFetch(200, {
      models: [{ name: "m", checkpoint_path: "/w.pt", tags: ["best"] }],
    });
    const res = await resultsApi.registeredModels();
    expect(res.models).toHaveLength(1);
    expect(res.models[0].checkpoint_path).toBe("/w.pt");
  });

  it("throws on a non-2xx instead of silently returning the error body", async () => {
    // A 404 error body ({detail}) must not be returned as-is: a caller reading `.models`/
    // `.rows` off it would get undefined and crash on the next render.
    stubFetch(404, { detail: "no plant mapping" });
    await expect(resultsApi.registeredModels()).rejects.toThrow("no plant mapping");
  });
});

describe("downloadCsv refusal decoding", () => {
  const REQUEST = {
    mapping_name: "valley",
    dataset_root: "C:/data",
    buckets: [],
    trait: "stage_50per_date",
    plants: ["PLOT-01"],
    payload: "curves" as const,
    filename: "x.csv",
    user: "jordan",
  };

  it("carries a structured refusal through the blob path parsed, not stringified", async () => {
    const detail = {
      kind: "operationalization",
      message:
        "its confirmed revision states no operationalization for a state_crossing_dates delivery",
    };
    stubFetch(400, { detail });

    const thrown = await resultsApi.downloadCsv(REQUEST).catch((e: unknown) => e);
    expect(thrown).toBeInstanceOf(StructuredRefusalError);
    expect((thrown as Error).message).not.toContain("[object Object]");
    expect(operationalizationRefusalOf(thrown)?.message).toBe(detail.message);
  });

  it("keeps the status-only message when the refusal body carries no detail", async () => {
    stubFetch(500, {});
    await expect(resultsApi.downloadCsv(REQUEST)).rejects.toThrow("500");
  });
});

describe("downloadCountCsv headers", () => {
  const REQUEST = {
    delivery: {
      kind: "per_image_count" as const,
      dataset_root: "C:/data",
      bucket: "chain/2026-02-11",
      trait: "stem",
    },
    filename: "counts.csv",
    user: "breeder",
  };

  function answering(headers: Record<string, string>) {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("image,count\n", { headers })));
  }

  it("decodes the three headers the count-export door answered", async () => {
    answering(COUNT_EXPORT_HEADERS);
    const { headers } = await resultsApi.downloadCountCsv(REQUEST);
    expect(headers).toEqual({
      savedTo: "/scratch/workspace/counts/results_export/counts.csv",
      validated: true,
      acknowledgedBy: "",
    });
  });

  it("refuses an answer the door sent without its acknowledger header", async () => {
    const { [ACKNOWLEDGED_BY_HEADER]: _removed, ...withoutAcknowledger } = COUNT_EXPORT_HEADERS;
    answering(withoutAcknowledger);
    await expect(resultsApi.downloadCountCsv(REQUEST)).rejects.toThrow(ACKNOWLEDGED_BY_HEADER);
  });
});

describe("inferenceApi.cancel", () => {
  it("POSTs to the job's cancel endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ job_id: "j1", status: "running" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const res = await inferenceApi.cancel("j1", "jordan");
    expect(res.job_id).toBe("j1");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/inference/jobs/j1/cancel",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ user: "jordan" }) }),
    );
  });
});
