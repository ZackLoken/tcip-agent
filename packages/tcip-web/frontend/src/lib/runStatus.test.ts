import { describe, expect, it } from "vitest";

import type { JobState_Completed_ } from "@/api/types.generated";
import { TERMINAL_STATES } from "@/lib/runStatus";

describe("run polling stop condition", () => {
  it("keeps polling a run that can still change state", () => {
    const nonTerminal: JobState_Completed_["status"][] = ["pending", "running"];
    for (const status of nonTerminal) expect(TERMINAL_STATES.has(status)).toBe(false);
  });
});
