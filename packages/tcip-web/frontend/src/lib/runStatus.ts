import { TERMINAL_STATES as GENERATED_TERMINAL_STATES } from "@/api/types.generated";

/** States a training run or a sweep never leaves, so a poll keyed on one can stop. */
export const TERMINAL_STATES: ReadonlySet<string> = new Set(GENERATED_TERMINAL_STATES);
