import { describe, expect, it } from "vitest";

import { mergeMetric, metricKey, numericMetricKeys, runOrderLine } from "@/tabs/trainingMetrics";

describe("mergeMetric (training stream de-dup)", () => {
  it("appends rows with distinct epochs", () => {
    let rows = mergeMetric([], { epoch: 0, train_loss: 1 });
    rows = mergeMetric(rows, { epoch: 1, train_loss: 0.5 });
    expect(rows).toHaveLength(2);
  });

  it("upserts a replayed epoch instead of duplicating it", () => {
    // Regression: a WS reconnect replays row 0 from the start; the old plain-append
    // (and the removed seed-GET) double-plotted these. Upsert keeps one point per epoch.
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

  it("keys by step when epoch is absent, and appends keyless rows", () => {
    expect(metricKey({ step: 3 })).toBe(3);
    expect(metricKey({ train_loss: 1 })).toBeUndefined();
    let rows = mergeMetric([], { train_loss: 1 });
    rows = mergeMetric(rows, { train_loss: 1 });
    expect(rows).toHaveLength(2); // no epoch/step -> can't dedupe
  });
});

describe("numericMetricKeys (the rank chooser and the logged table's shared filter)", () => {
  it("drops epoch, step and timestamp alongside a numeric metric", () => {
    expect(numericMetricKeys({ epoch: 3, step: 9, timestamp: 123, train_loss: 0.5 })).toEqual([
      "train_loss",
    ]);
  });

  it("drops a key whose stamped value is not a number, such as a selection label", () => {
    expect(numericMetricKeys({ val_map50: 0.7, selection_label: "held-out" })).toEqual([
      "val_map50",
    ]);
  });

  it("drops a metric's own _state companion key", () => {
    expect(numericMetricKeys({ train_loss: 0.5, train_loss_state: "nan" })).toEqual(["train_loss"]);
  });

  it("answers no keys for a record that is absent", () => {
    expect(numericMetricKeys(undefined)).toEqual([]);
    expect(numericMetricKeys(null)).toEqual([]);
  });
});

describe("runOrderLine (the Training/Tuning tabs' shared order sentence)", () => {
  it("names the app, not a browser window, for both a run and a sweep noun", () => {
    expect(runOrderLine("run", "experiment id")).toBe(
      "Runs this app's own launches first, in launch order; every other recorded run follows, sorted by experiment id.",
    );
    expect(runOrderLine("sweep", "sweep id")).toBe(
      "Sweeps this app's own launches first, in launch order; every other recorded sweep follows, sorted by sweep id.",
    );
  });
});
