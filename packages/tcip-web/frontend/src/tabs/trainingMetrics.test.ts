import { describe, expect, it, vi } from "vitest";

import { NOT_FINITE_SUFFIX, VAL_METRIC_PREFIX } from "@/api/types.generated";
import { bareMetricName, mergeMetric, numericMetricKeys } from "@/tabs/trainingMetrics";

// The backend's declared spellings stand in for values no literal here shares, so a helper that
// spells a suffix or prefix itself instead of reading the generated module fails these tests.
vi.mock("@/api/types.generated", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/types.generated")>()),
  NOT_FINITE_SUFFIX: "__not_finite",
  VAL_METRIC_PREFIX: "heldout__",
}));

describe("mergeMetric (training stream de-dup)", () => {
  it("appends rows with distinct epochs", () => {
    let rows = mergeMetric([], { epoch: 0, train_loss: 1 });
    rows = mergeMetric(rows, { epoch: 1, train_loss: 0.5 });
    expect(rows).toHaveLength(2);
  });

  it("upserts a replayed epoch instead of duplicating it", () => {
    let rows: Parameters<typeof mergeMetric>[0] = [
      { epoch: 0, train_loss: 1 },
      { epoch: 1, train_loss: 0.5 },
    ];
    rows = mergeMetric(rows, { epoch: 0, train_loss: 1 });
    expect(rows).toHaveLength(2);
    rows = mergeMetric(rows, { epoch: 1, train_loss: 0.42 });
    expect(rows).toHaveLength(2);
    expect(rows[1].train_loss).toBe(0.42);
  });

  it("keeps a loss and the selection logged after it at one epoch, the second frame whole", () => {
    let rows = mergeMetric([], { epoch: 2, train_loss: 0.5 });
    rows = mergeMetric(rows, { epoch: 2, train_loss: 0.5, selection: 0.25 });
    expect(rows).toEqual([{ epoch: 2, train_loss: 0.5, selection: 0.25 }]);
  });
});

describe("numericMetricKeys (the rank chooser and the logged tables' shared filter)", () => {
  it("drops the epoch and timestamp alongside a numeric metric", () => {
    expect(numericMetricKeys({ epoch: 3, timestamp: 123, train_loss: 0.5 })).toEqual([
      "train_loss",
    ]);
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

describe("bareMetricName", () => {
  it("strips the backend's declared validation prefix and nothing else", () => {
    expect(bareMetricName(`${VAL_METRIC_PREFIX}map50`)).toBe("map50");
    expect(bareMetricName("val_map50")).toBe("val_map50");
    expect(bareMetricName("train_loss")).toBe("train_loss");
  });
});
