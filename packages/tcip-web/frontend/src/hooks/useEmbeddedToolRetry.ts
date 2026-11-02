/**
 * Retries an embedded-tool launch attempt on a timer until it settles.
 */

import { useEffect, useRef, useState } from "react";

import type { NoLogsRefusal, TensorboardLaunch } from "@/api/types.generated";

/** The retry cadence, in milliseconds. */
export const EMBEDDED_TOOL_RETRY_MS = 3000;

/** What a board's launch has come to: not answered yet, its launch's answer, the refusal that
 * its run wrote no logs, or a request that never reached an answer, with its error's message. */
export type EmbeddedToolOutcome =
  { kind: "pending" } | TensorboardLaunch | NoLogsRefusal | { kind: "unreached"; message: string };

export const PENDING: EmbeddedToolOutcome = { kind: "pending" };

export interface EmbeddedToolStepResult {
  outcome: EmbeddedToolOutcome;
  /** True once a further attempt would change nothing: a board is serving, or the run/sweep
   * behind it reached a state a retry can't recover from. False keeps the timer running. */
  done: boolean;
}

export function useEmbeddedToolRetry(
  key: string | null,
  active: boolean,
  attempt: number,
  step: () => Promise<EmbeddedToolStepResult>,
  retryMs: number = EMBEDDED_TOOL_RETRY_MS,
): EmbeddedToolOutcome {
  const [outcome, setOutcome] = useState<EmbeddedToolOutcome>(PENDING);
  const stepRef = useRef(step);
  stepRef.current = step;

  useEffect(() => {
    setOutcome(PENDING);
    if (!active) return;
    let canceled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      const result = await stepRef.current();
      if (canceled) return;
      setOutcome(result.outcome);
      if (!result.done) timer = setTimeout(() => void tick(), retryMs);
    };
    void tick();
    return () => {
      canceled = true;
      if (timer) clearTimeout(timer);
    };
    // A key change (a direct switch from one run or sweep to another) resets and restarts too.
  }, [key, active, attempt, retryMs]);

  return outcome;
}
