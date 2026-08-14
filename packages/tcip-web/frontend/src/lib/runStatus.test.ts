import { describe, expect, it } from "vitest";

import { TERMINAL_STATUSES } from "@/lib/runStatus";

// The four tokens below are the backend's own terminal set (jobstore.TERMINAL_STATUSES),
// written out here rather than imported, so the two copies are compared instead of one.
describe("run polling stop condition", () => {
  it("stops on every status the backend treats as final", () => {
    expect(Array.from(TERMINAL_STATUSES).sort()).toEqual([
      "cancelled",
      "completed",
      "failed",
      "interrupted",
    ]);
  });

  it("keeps polling a run that can still change state", () => {
    expect(TERMINAL_STATUSES.has("pending")).toBe(false);
    expect(TERMINAL_STATUSES.has("running")).toBe(false);
    expect(TERMINAL_STATUSES.has("queued")).toBe(false);
  });
});
