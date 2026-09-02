/** Test-only reset for the shared coverage-outbox singleton: drop every queued payload and any
 *  pending retry timer, so one test's leftover state never leaks into the next. Never a method
 *  on the shipped `coverageOutbox` itself; this is the one importer of the module's own
 *  test-only reset function. */
import { coverageOutbox, resetCoverageOutboxForTests } from "@/lib/coverageTracker";

export function resetCoverageOutbox(): void {
  resetCoverageOutboxForTests(coverageOutbox);
}
