import { describe, expect, it } from "vitest";

import {
  buildSearchSpace,
  DEFAULT_HPO_PARAMS,
  parseNumList,
  SCHEDULERS,
  SEARCH_ALGORITHMS,
  type HpoParam,
} from "@/tabs/hpoSpace";

describe("parseNumList", () => {
  it("parses a comma list and drops blanks/non-numbers", () => {
    expect(parseNumList("2, 4, 8")).toEqual([2, 4, 8]);
    expect(parseNumList("2, , x, 4")).toEqual([2, 4]);
    expect(parseNumList("")).toEqual([]);
  });
});

describe("buildSearchSpace", () => {
  it("emits typed specs for the default params", () => {
    const space = buildSearchSpace(DEFAULT_HPO_PARAMS);
    expect(space.lr).toEqual({ type: "loguniform", low: 1e-5, high: 1e-2 });
    expect(space.weight_decay).toEqual({ type: "loguniform", low: 1e-5, high: 1e-2 });
    expect(space.batch_size).toEqual({ type: "categorical", choices: [2, 4] });
    // Architecture axes (backbone / head / min_size) are model-specific and not swept.
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
    expect(buildSearchSpace(params)).toEqual({});
  });
});

describe("agent strategy menus", () => {
  it("offers native + backend search algorithms and the native schedulers", () => {
    expect(SEARCH_ALGORITHMS).toContain("random");
    expect(SEARCH_ALGORITHMS).toContain("grid");
    expect(SEARCH_ALGORITHMS).toContain("optuna");
    expect(SEARCH_ALGORITHMS).toContain("bohb");
    expect(SCHEDULERS).toContain("asha");
    expect(SCHEDULERS).toContain("pbt");
    expect(SCHEDULERS).toContain("none");
  });
});
