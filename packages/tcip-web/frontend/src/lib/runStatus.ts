import { TERMINAL_STATUSES as GENERATED_TERMINAL_STATUSES } from "@/api/types.generated";

/** Statuses a training run or a sweep never leaves, so a poll keyed on one can stop. */
export const TERMINAL_STATUSES: ReadonlySet<string> = new Set(GENERATED_TERMINAL_STATUSES);
