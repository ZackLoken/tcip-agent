import { describe, expect, it } from "vitest";

import { buildOptunaSpace, DEFAULT_HPO_PARAMS, parseNumList, type HpoParam } from "@/tabs/hpoSpace";

describe("parseNumList", () => {
  it("parses a comma list and drops blanks/non-numbers", () => {
    expect(parseNumList("2, 4, 8")).toEqual([2, 4, 8]);
    expect(parseNumList("2, , x, 4")).toEqual([2, 4]);
    expect(parseNumList("")).toEqual([]);
  });
});

describe("buildOptunaSpace", () => {
  it("emits Optuna-typed specs for the default params", () => {
    const space = buildOptunaSpace(DEFAULT_HPO_PARAMS);
    expect(space.lr).toEqual({ type: "loguniform", low: 1e-5, high: 1e-2 });
    expect(space.weight_decay).toEqual({ type: "loguniform", low: 1e-5, high: 1e-2 });
    expect(space.batch_size).toEqual({ type: "categorical", choices: [2, 4] });
    // Architecture axes (backbone / head / min_size) are model-specific and no longer swept.
    expect(space.backbone).toBeUndefined();
    expect(space.head).toBeUndefined();
    expect(space.min_size).toBeUndefined();
  });

  it("excludes disabled params and empty categoricals", () => {
    const params: HpoParam[] = [
      { key: "lr", label: "lr", kind: "loguniform", enabled: false, low: 1, high: 2 },
      { key: "head", label: "head", kind: "choices", enabled: true, options: ["a"], selected: [] },
      { key: "bs", label: "bs", kind: "numlist", enabled: true, values: [] },
    ];
    expect(buildOptunaSpace(params)).toEqual({});
  });
});
