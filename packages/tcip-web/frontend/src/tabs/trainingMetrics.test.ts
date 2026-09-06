import { describe, expect, it } from "vitest";

import {
  defaultChartSeries,
  mergeMetric,
  metricKey,
  numericMetricKeys,
  runOrderLine,
} from "@/tabs/trainingMetrics";

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

describe("defaultChartSeries (the live metrics chart's default series rule)", () => {
  it("plots both loss keys plus the selection line, unmerged, when selection is not a loss", () => {
    const rows = [
      { epoch: 0, train_loss: 0.9, val_loss: 0.8, selection: 0.5, selection_metric: "map50" },
      { epoch: 1, train_loss: 0.7, val_loss: 0.6, selection: 0.6, selection_metric: "map50" },
    ];
    const series = defaultChartSeries(["train_loss", "val_loss", "selection"], rows);
    expect(series.allKeys).toBe(false);
    expect(series.keys).toEqual(["train_loss", "val_loss", "selection"]);
    expect(series.labels.selection).toBe("selection (map50)");
    expect(series.labels.train_loss).toBeUndefined();
  });

  it("merges selection into val_loss when a validated run selects on loss", () => {
    const rows = [
      { epoch: 0, train_loss: 0.9, val_loss: 0.4, selection: 0.4, selection_metric: "loss" },
    ];
    const series = defaultChartSeries(["train_loss", "val_loss", "selection"], rows);
    expect(series.keys).toEqual(["train_loss", "val_loss"]);
    expect(series.labels.val_loss).toBe("val_loss (selection)");
    expect(series.labels.train_loss).toBeUndefined();
  });

  it("merges selection into train_loss when there is no validation loader", () => {
    const rows = [{ epoch: 0, train_loss: 0.27, selection: 0.27, selection_metric: "loss" }];
    const series = defaultChartSeries(["train_loss", "selection"], rows);
    expect(series.keys).toEqual(["train_loss"]);
    expect(series.labels.train_loss).toBe("train_loss (selection)");
  });

  it("falls back to every numeric key only when the log has neither a loss key nor selection", () => {
    const rows = [{ epoch: 0, val_map50: 0.7, lr: 0.001 }];
    const series = defaultChartSeries(["val_map50", "lr"], rows);
    expect(series.allKeys).toBe(true);
    expect(series.keys).toEqual(["val_map50", "lr"]);
  });

  it("leaves a bespoke non-loss key out of the default series and merges nothing on it", () => {
    const rows = [
      { epoch: 0, train_loss: 0.5, loss_total: 1.2, selection: 0.5, selection_metric: "loss" },
    ];
    const series = defaultChartSeries(["train_loss", "loss_total", "selection"], rows);
    // loss_total does not end in "loss", so it never joins the default series or the merge.
    expect(series.keys).toEqual(["train_loss"]);
    expect(series.labels.train_loss).toBe("train_loss (selection)");
  });

  it("plots selection alone, unmerged, when no row names a selection_metric", () => {
    const rows = [{ epoch: 0, train_loss: 0.5, selection: 0.5 }];
    const series = defaultChartSeries(["train_loss", "selection"], rows);
    expect(series.keys).toEqual(["train_loss", "selection"]);
    expect(series.labels.selection).toBe("selection");
  });
});

describe("runOrderLine (the Training/Tuning tabs' shared order sentence)", () => {
  it("names the running process, not a browser window, for both a run and a sweep noun", () => {
    expect(runOrderLine("run", "experiment id")).toBe(
      "Runs this running process itself launched come first, in launch order; every other recorded run follows, sorted by experiment id.",
    );
    expect(runOrderLine("sweep", "sweep id")).toBe(
      "Sweeps this running process itself launched come first, in launch order; every other recorded sweep follows, sorted by sweep id.",
    );
  });
});
