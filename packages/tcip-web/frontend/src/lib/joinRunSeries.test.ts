import { describe, expect, it } from "vitest";

import type { MetricRow } from "@/api/training";
import { joinRunSeries, metricKeysAcross } from "@/lib/joinRunSeries";

describe("joinRunSeries", () => {
  it("joins runs sharing the same epochs into one point per epoch", () => {
    const { points } = joinRunSeries(
      [
        {
          runId: "a",
          rows: [
            { epoch: 0, loss: 1 },
            { epoch: 1, loss: 0.5 },
          ],
        },
        {
          runId: "b",
          rows: [
            { epoch: 0, loss: 2 },
            { epoch: 1, loss: 1.5 },
          ],
        },
      ],
      "loss",
    );
    expect(points).toEqual([
      { x: 0, a: 1, b: 2 },
      { x: 1, a: 0.5, b: 1.5 },
    ]);
  });

  it("keeps each run's own value at the epochs the other run lacks, rather than filling zero", () => {
    // Different lengths: run "a" logged three epochs, run "b" only two.
    const { points } = joinRunSeries(
      [
        {
          runId: "a",
          rows: [
            { epoch: 0, loss: 1 },
            { epoch: 1, loss: 0.5 },
            { epoch: 2, loss: 0.2 },
          ],
        },
        { runId: "b", rows: [{ epoch: 0, loss: 2 }] },
      ],
      "loss",
    );
    expect(points).toEqual([
      { x: 0, a: 1, b: 2 },
      { x: 1, a: 0.5 },
      { x: 2, a: 0.2 },
    ]);
    // A run's own line ends where its finite values end: no "b" key past epoch 0.
    expect(points[1].b).toBeUndefined();
    expect(points[2].b).toBeUndefined();
  });

  it("keys by step when a run's rows carry no epoch, matching the tab's one row identity", () => {
    const { points } = joinRunSeries([{ runId: "a", rows: [{ step: 5, loss: 0.9 }] }], "loss");
    expect(points).toEqual([{ x: 5, a: 0.9 }]);
  });

  it("drops a keyless row (no epoch or step) from the overlay rather than guessing a position", () => {
    const { points } = joinRunSeries(
      [{ runId: "a", rows: [{ loss: 0.9 }, { epoch: 0, loss: 0.5 }] }],
      "loss",
    );
    expect(points).toEqual([{ x: 0, a: 0.5 }]);
  });

  it("skips a non-finite value at a shared epoch, never plotting it as zero", () => {
    const { points } = joinRunSeries(
      [
        { runId: "a", rows: [{ epoch: 0, loss: Number.NaN }] },
        { runId: "b", rows: [{ epoch: 0, loss: 0.3 }] },
      ],
      "loss",
    );
    expect(points).toEqual([{ x: 0, b: 0.3 }]);
  });

  it("counts a dropped row per run, independent of the metric being charted", () => {
    const { droppedByRun } = joinRunSeries(
      [
        { runId: "a", rows: [{ loss: 0.9 }, { epoch: 0, loss: 0.5 }] },
        { runId: "b", rows: [{ epoch: 0, loss: 0.3 }] },
      ],
      "loss",
    );
    expect(droppedByRun).toEqual({ a: 1, b: 0 });
  });

  it("counts a null-epoch, keyless row as dropped, the same identity rule metricKey applies", () => {
    // epoch: null is not typeof "number", so metricKey treats it the same as a missing epoch;
    // a caller re-implementing the rule as `=== undefined` would miss this row.
    const { points, droppedByRun } = joinRunSeries(
      [{ runId: "a", rows: [{ epoch: null, loss: 0.9 } as unknown as MetricRow] }],
      "loss",
    );
    expect(points).toEqual([]);
    expect(droppedByRun).toEqual({ a: 1 });
  });
});

describe("metricKeysAcross", () => {
  it("unions the numeric keys across every run, excluding epoch/step", () => {
    const keys = metricKeysAcross([
      { runId: "a", rows: [{ epoch: 0, loss: 1, val_map50: 0.4 }] },
      { runId: "b", rows: [{ step: 1, val_loss: 0.2 }] },
    ]);
    expect(keys).toEqual(["loss", "val_loss", "val_map50"]);
  });
});
