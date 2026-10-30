import { describe, expect, it } from "vitest";

import { refusalOf, StructuredRefusalError } from "@/api/http";
import { inferenceApi, resultsApi } from "@/api/inference";
import {
  EXPORT_HEADER,
  type LaunchInferencePayload,
  NOTHING_STATED,
  OPERATIONALIZATION,
  type OperationalizationRefusal,
} from "@/api/types.generated";
import {
  ACKNOWLEDGED_COUNT_EXPORT,
  COUNT_EXPORT,
  eventOf,
  REGISTERED_MODELS,
  RESULTS_ANSWERS,
} from "@/test/deliveryRecords";
import { INFERENCE_ANSWERS } from "@/test/inference";
import { fetchAnswers, fetchFile } from "@/test/refusals";

const LAUNCH: LaunchInferencePayload = {
  checkpoint_path: "C:/data/model.pt",
  dataset_root: "C:/data",
  date: "2026-01-01",
  bucket_name: "baseline",
  stated: NOTHING_STATED,
  assessment_id: null,
  user: "jordan",
};

describe("inference and results api answers", () => {
  it("returns the parsed body on a 2xx", async () => {
    fetchAnswers(200, REGISTERED_MODELS);
    expect(await resultsApi.registeredModels()).toEqual(REGISTERED_MODELS);
  });

  it("throws on a non-2xx instead of silently returning the error body", async () => {
    // A 404 error body ({detail}) must not be returned as-is: a caller reading a field off it
    // would get undefined and crash on the next render.
    const { status, body } = INFERENCE_ANSWERS.missing_checkpoint;
    fetchAnswers(status, body);
    await expect(inferenceApi.launch(LAUNCH)).rejects.toThrow(body.detail);
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
    acknowledgment: null,
  };

  it("carries a structured refusal through the blob path parsed, not stringified", async () => {
    const { status, body } = RESULTS_ANSWERS.operationalization_refusal;
    fetchAnswers(status, body);

    const raised = await resultsApi.downloadCsv(REQUEST).catch((e: unknown) => e);
    expect(raised).toBeInstanceOf(StructuredRefusalError);
    expect((raised as Error).message).not.toContain("[object Object]");
    expect(refusalOf<OperationalizationRefusal>(raised, OPERATIONALIZATION)).toEqual(body.detail);
  });

  it("keeps the status-only message when the refusal body carries no detail", async () => {
    // A transport-level failure: a 500 whose body no route wrote.
    fetchAnswers(500, {});
    await expect(resultsApi.downloadCsv(REQUEST)).rejects.toThrow("500");
  });
});

describe("downloadCountCsv's reported export", () => {
  const REQUEST = {
    delivery: {
      kind: "per_image_count" as const,
      dataset_root: "C:/data",
      bucket: "chain/2026-02-11",
      trait: "stem",
    },
    filename: "counts.csv",
    user: "breeder",
    acknowledgment: null,
  };

  /** Make the door's call answer the produced `exported` file with `headers`. */
  function answering(exported: typeof COUNT_EXPORT, headers = exported.headers) {
    fetchFile(exported.file, headers);
  }

  it("reads the export the door reported as what its event records", async () => {
    answering(COUNT_EXPORT);
    const event = eventOf(COUNT_EXPORT);
    const { blob, exported } = await resultsApi.downloadCountCsv(REQUEST);
    expect(exported).toMatchObject({
      saved_path: event.output_path,
      validated: event.validated,
      acknowledged_by: null,
    });
    expect(await blob.text()).toBe(COUNT_EXPORT.file);
  });

  it("reads the acknowledging person of an unvalidated export", async () => {
    answering(ACKNOWLEDGED_COUNT_EXPORT);
    const event = eventOf(ACKNOWLEDGED_COUNT_EXPORT);
    const { exported } = await resultsApi.downloadCountCsv(REQUEST);
    expect(exported).toMatchObject({
      validated: event.validated,
      acknowledged_by: event.acknowledgment?.acknowledged_by,
    });
  });

  it("refuses an answer the door sent without its export header", async () => {
    const { [EXPORT_HEADER]: _removed, ...withoutExport } = COUNT_EXPORT.headers;
    answering(COUNT_EXPORT, withoutExport);
    await expect(resultsApi.downloadCountCsv(REQUEST)).rejects.toThrow(EXPORT_HEADER);
  });
});

describe("inferenceApi.cancel", () => {
  it("POSTs to the job's cancel endpoint", async () => {
    fetchAnswers(200, INFERENCE_ANSWERS.cancel);

    expect(await inferenceApi.cancel("j1", { user: "jordan" })).toEqual(INFERENCE_ANSWERS.cancel);
    expect(fetch).toHaveBeenCalledWith(
      "/api/inference/jobs/j1/cancel",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ user: "jordan" }) }),
    );
  });
});
