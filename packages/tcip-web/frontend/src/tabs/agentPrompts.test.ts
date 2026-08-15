import { describe, expect, it } from "vitest";

import { defaultSweepRequest, defaultTrainingRequest } from "@/tabs/agentPrompts";

describe("defaultTrainingRequest", () => {
  it("names the dataset and the subject when both are selected", () => {
    const text = defaultTrainingRequest("/data/valley", "subject_a");
    expect(text).toContain("for the dataset at /data/valley, subject subject_a");
  });

  it("drops each part of the scope that is not selected", () => {
    expect(defaultTrainingRequest("/data/valley", null)).toContain(
      "training run for the dataset at /data/valley.",
    );
    expect(defaultTrainingRequest(null, "subject_a")).toContain(
      "training run for subject subject_a.",
    );
    expect(defaultTrainingRequest(null, null)).toContain("training run.");
  });

  it("asks the agent to decide on tiling from the data rather than naming a threshold", () => {
    const text = defaultTrainingRequest(null, null);
    expect(text).toContain("tiles or on whole frames");
    expect(text).not.toMatch(/\d+\s*(px|pixels)/);
  });

  it("raises the sampler choice as a layout condition, not a size or a named sampler", () => {
    const text = defaultTrainingRequest(null, null);
    expect(text).toContain("sampler choice");
    expect(text).toContain("strip-layout");
    expect(text).not.toMatch(/tile_locality/);
    expect(text).not.toMatch(/large (raster|mosaic|image)/i);
  });
});

describe("defaultSweepRequest", () => {
  it("omits the dataset clause when nothing is selected", () => {
    expect(defaultSweepRequest(null)).toContain("Run a hyperparameter sweep.");
    expect(defaultSweepRequest("/data/valley")).toContain("sweep for the dataset at /data/valley.");
  });
});
