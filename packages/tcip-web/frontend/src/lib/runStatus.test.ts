import { describe, expect, it } from "vitest";

import type { JobStatus } from "@/api/types.generated";
import { TERMINAL_STATUSES } from "@/lib/runStatus";

// pending/running are typed against the generated vocabulary; queued is a tuning-sweep status
// outside it, kept as a plain literal probe since nothing generated pins it.
describe("run polling stop condition", () => {
  it("keeps polling a run that can still change state", () => {
    const nonTerminal: JobStatus[] = ["pending", "running"];
    for (const status of nonTerminal) expect(TERMINAL_STATUSES.has(status)).toBe(false);
    expect(TERMINAL_STATUSES.has("queued")).toBe(false);
  });
});
