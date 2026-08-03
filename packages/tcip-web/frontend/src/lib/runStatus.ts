/**
 * Statuses a training run or a sweep never leaves, so a poll keyed on one can stop.
 * Mirrors the backend's `jobstore.TERMINAL_STATUSES`.
 */
export const TERMINAL_STATUSES: ReadonlySet<string> = new Set([
  "completed",
  "failed",
  "cancelled",
  "interrupted",
]);
