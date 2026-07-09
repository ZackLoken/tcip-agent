import { describe, expect, it } from "vitest";

import { mergeMetric, metricKey } from "@/tabs/trainingMetrics";

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
