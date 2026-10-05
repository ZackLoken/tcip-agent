/**
 * Retries an embedded-tool launch attempt on a timer until it settles.
 */

import { useEffect, useRef, useState } from "react";

/** The retry cadence, in milliseconds. */
export const EMBEDDED_TOOL_RETRY_MS = 3000;

export interface EmbeddedToolOutcome {
  url: string | null;
  error: string | null;
}

export interface EmbeddedToolStepResult extends EmbeddedToolOutcome {
  /** True once a further attempt would change nothing: a url landed, or the run/sweep behind
   * it reached a state a retry can't recover from. False keeps the timer running. */
  done: boolean;
}

export function useEmbeddedToolRetry(
  key: string | null,
  active: boolean,
  attempt: number,
  step: () => Promise<EmbeddedToolStepResult>,
  retryMs: number = EMBEDDED_TOOL_RETRY_MS,
): EmbeddedToolOutcome {
  const [outcome, setOutcome] = useState<EmbeddedToolOutcome>({ url: null, error: null });
  const stepRef = useRef(step);
  stepRef.current = step;

  useEffect(() => {
    setOutcome({ url: null, error: null });
    if (!active) return;
    let canceled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      const result = await stepRef.current();
      if (canceled) return;
      setOutcome({ url: result.url, error: result.error });
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
