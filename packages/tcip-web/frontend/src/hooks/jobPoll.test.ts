import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { PriorityQueueJob } from "@/api/types.generated";
import { JOB_POLL_MS, pollJob } from "@/hooks/jobPoll";
import { QUEUE_ANSWERS } from "@/test/queue";

const { running: RUNNING, completed: COMPLETED, failed: FAILED } = QUEUE_ANSWERS;

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

/** Poll with `read`; return each read handed on and each error. */
function polled(read: (jobId: string) => Promise<PriorityQueueJob>) {
  const reads: PriorityQueueJob[] = [];
  const errors: unknown[] = [];
  const cancel = pollJob(
    RUNNING.job_id,
    read,
    (job) => reads.push(job),
    (e) => errors.push(e),
  );
  return { reads, errors, cancel };
}

describe("pollJob", () => {
  it("hands each read on and stops at the first read of a job no longer live, however it ended", async () => {
    for (const end of [COMPLETED, FAILED]) {
      const read = vi.fn().mockResolvedValueOnce(RUNNING).mockResolvedValue(end);
      const { reads } = polled(read);

      await vi.advanceTimersByTimeAsync(JOB_POLL_MS * 5);

      expect(reads).toEqual([RUNNING, end]);
      expect(read).toHaveBeenCalledTimes(2);
      expect(read).toHaveBeenCalledWith(RUNNING.job_id);
    }
  });

  it("stops on a read that fails, handing the error on", async () => {
    const refused = new Error("the job route refused");
    const read = vi.fn().mockRejectedValue(refused);
    const { reads, errors } = polled(read);

    await vi.advanceTimersByTimeAsync(JOB_POLL_MS * 5);

    expect(reads).toEqual([]);
    expect(errors).toEqual([refused]);
    expect(read).toHaveBeenCalledTimes(1);
  });

  it("reads nothing more once canceled", async () => {
    const read = vi.fn().mockResolvedValue(RUNNING);
    const { reads, cancel } = polled(read);

    await vi.advanceTimersByTimeAsync(JOB_POLL_MS);
    cancel();
    await vi.advanceTimersByTimeAsync(JOB_POLL_MS * 5);

    expect(reads).toEqual([RUNNING]);
    expect(read).toHaveBeenCalledTimes(1);
  });
});
