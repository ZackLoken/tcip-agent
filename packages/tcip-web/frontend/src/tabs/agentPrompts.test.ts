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

describe("both composer defaults", () => {
  // snake_case, a dotted module path (segments at least two characters, so "e.g." never
  // trips it), a tools/ path, a .py suffix, or a backticked token.
  const IDENTIFIER_SHAPED =
    /[a-zA-Z]+_[a-zA-Z_]+|[a-zA-Z][a-zA-Z0-9_]+(?:\.[a-zA-Z][a-zA-Z0-9_]+)+|tools\/|\.py\b|`[^`]+`/;

  it("names no tool, script or function the agent should call", () => {
    // "leaf", not "subject_a": an underscored subject name would trip the identifier check
    // on the breeder's own data, not on wording this function chose.
    expect(defaultTrainingRequest(null, null)).not.toMatch(IDENTIFIER_SHAPED);
    expect(defaultTrainingRequest("/data/valley", "leaf")).not.toMatch(IDENTIFIER_SHAPED);
    expect(defaultSweepRequest(null)).not.toMatch(IDENTIFIER_SHAPED);
    expect(defaultSweepRequest("/data/valley")).not.toMatch(IDENTIFIER_SHAPED);
  });

  it("catches every shape the guard names, and nothing narrower", () => {
    expect("tuning.launch").toMatch(IDENTIFIER_SHAPED);
    expect("`preflight`").toMatch(IDENTIFIER_SHAPED);
    expect("tools/list_tools.py").toMatch(IDENTIFIER_SHAPED);
    // An abbreviation, not a module path: each segment must be at least two characters.
    expect("e.g.").not.toMatch(IDENTIFIER_SHAPED);
  });
});
