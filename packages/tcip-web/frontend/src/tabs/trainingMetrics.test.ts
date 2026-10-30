import { describe, expect, it, vi } from "vitest";

import { METRICS_ROW_STAMPS, NOT_FINITE_SUFFIX } from "@/api/types.generated";
import { mergeMetric, numericMetricKeys } from "@/tabs/trainingMetrics";
import { METRIC_FRAMES } from "@/test/training";

// The backend's declared spelling stands in for a value no literal here shares, so a helper that
// spells the suffix itself instead of reading the generated module fails these tests.
vi.mock("@/api/types.generated", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/types.generated")>()),
  NOT_FINITE_SUFFIX: "__not_finite",
}));

describe("mergeMetric (training stream de-dup)", () => {
  const [first, second] = METRIC_FRAMES.map((frame) => frame.row);

  it("appends rows with distinct epochs", () => {
    expect(first.epoch).not.toBe(second.epoch);
    expect(mergeMetric(mergeMetric([], first), second)).toEqual([first, second]);
  });

  it("upserts a replayed epoch instead of duplicating it, the later frame whole", () => {
    const rows = mergeMetric([first, second], first);
    expect(rows).toEqual([first, second]);
    const regrown = { ...second, selection: 0.25 };
    expect(mergeMetric(rows, regrown)).toEqual([first, regrown]);
  });
});

describe("numericMetricKeys (the logged metric tables' filter)", () => {
  it("drops a produced row's stamps and keeps its numeric metrics", () => {
    const [{ row }] = METRIC_FRAMES;
    const keys = numericMetricKeys(row);
    expect(keys.length).toBeGreaterThan(0);
    for (const stamp of METRICS_ROW_STAMPS) expect(keys).not.toContain(stamp);
    for (const key of keys) expect(typeof row[key]).toBe("number");
  });

  it("drops a key whose stamped value is not a number, such as a selection label", () => {
    expect(numericMetricKeys({ val_map50: 0.7, selection_label: "held-out" })).toEqual([
      "val_map50",
    ]);
  });

  it("drops a metric's companion under the backend's declared non-finite suffix", () => {
    expect(
      numericMetricKeys({ train_loss: 0.5, [`train_loss${NOT_FINITE_SUFFIX}`]: 1, train_state: 2 }),
    ).toEqual(["train_loss", "train_state"]);
  });

  it("answers no keys for a record that is absent", () => {
    expect(numericMetricKeys(undefined)).toEqual([]);
    expect(numericMetricKeys(null)).toEqual([]);
  });
});
