import { describe, expect, it } from "vitest";

import { sweepDrawsOf } from "@/api/tuning";

describe("sweepDrawsOf", () => {
  it("narrows a full result into groups, the best block, and the draw count", () => {
    const result = {
      split_draws: 2,
      best_value_spread: { mean: 1.5, std: 0.1, min: 1.4, max: 1.6, seeds_complete: [1, 2] },
      split_sensitivity: [
        {
          point: { "training_source.lr": 0.01 },
          block: {
            seeds: [1, 2],
            values: [1.4, 1.6],
            mean: 1.5,
            std: 0.1,
            min: 1.4,
            max: 1.6,
            n: 2,
            n_complete: 2,
            seeds_complete: [1, 2],
          },
          eligible: true,
        },
      ],
    };

    const draws = sweepDrawsOf(result);

    expect(draws).not.toBeNull();
    expect(draws?.splitDraws).toBe(2);
    expect(draws?.best).toEqual({
      mean: 1.5,
      std: 0.1,
      min: 1.4,
      max: 1.6,
      n: null,
      n_complete: null,
      seeds: [],
      seeds_complete: [1, 2],
    });
    expect(draws?.groups).toEqual([
      {
        point: { "training_source.lr": 0.01 },
        block: {
          mean: 1.5,
          std: 0.1,
          min: 1.4,
          max: 1.6,
          n: 2,
          n_complete: 2,
          seeds: [1, 2],
          seeds_complete: [1, 2],
        },
        eligible: true,
      },
    ]);
  });

  it("is null for a result carrying no split_sensitivity array", () => {
    expect(sweepDrawsOf({ best_params: {} })).toBeNull();
    expect(sweepDrawsOf({ split_sensitivity: "not an array" })).toBeNull();
    expect(sweepDrawsOf(null)).toBeNull();
    expect(sweepDrawsOf("not an object")).toBeNull();
  });

  it("narrows a malformed block to null cells rather than dropping the group", () => {
    const result = {
      split_draws: 1,
      best_value_spread: null,
      best_value_state:
        "no eligible point: every drawn point had an errored or never-answered draw",
      split_sensitivity: [
        { point: null, block: { mean: "not a number", seeds: "not an array" }, eligible: "yes" },
        { point: { k: 1 } },
      ],
    };

    const draws = sweepDrawsOf(result);

    expect(draws).not.toBeNull();
    expect(draws?.best).toBeNull();
    expect(draws?.bestState).toBe(
      "no eligible point: every drawn point had an errored or never-answered draw",
    );
    expect(draws?.groups).toEqual([
      {
        point: null,
        block: {
          mean: null,
          std: null,
          min: null,
          max: null,
          n: null,
          n_complete: null,
          seeds: [],
          seeds_complete: [],
        },
        eligible: false,
      },
      {
        point: { k: 1 },
        block: {
          mean: null,
          std: null,
          min: null,
          max: null,
          n: null,
          n_complete: null,
          seeds: [],
          seeds_complete: [],
        },
        eligible: false,
      },
    ]);
  });
});
