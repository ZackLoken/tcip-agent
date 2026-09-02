/**
 * Retries an embedded-tool launch attempt on a timer until it settles, the one polling shape
 * the Training tab's run TensorBoard panel and the Tuning tab's sweep TensorBoard panel both
 * need instead of each keeping its own copy of the same loop.
 */

import { useEffect, useRef, useState } from "react";

/** The retry cadence both panels share; a run's own TensorBoard launch is idempotent, so
 * there is nothing costly about trying again on this cadence while nothing has served yet. */
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
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      const result = await stepRef.current();
      if (cancelled) return;
      setOutcome({ url: result.url, error: result.error });
      if (!result.done) timer = setTimeout(() => void tick(), retryMs);
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [active, attempt, retryMs]);

  return outcome;
}
