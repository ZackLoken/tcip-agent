/** The held review queue job, read back while it runs. */

import { useEffect } from "react";

import { api } from "@/api/client";
import { messageOf } from "@/api/http";
import { pollJob } from "@/hooks/jobPoll";
import { useStore } from "@/store";

/**
 * Poll the held review queue job (`annotateUi.reviewQueue`) while it runs, holding each read.
 *
 * A failed job's error is shown as the backend wrote it. A read that fails is shown and drops
 * the held job.
 */
export function useReviewQueuePoll(): void {
  const running = useStore((s) =>
    s.annotateUi.reviewQueue?.live ? s.annotateUi.reviewQueue.job_id : null,
  );

  useEffect(() => {
    if (running === null) return;
    const { setReviewQueue, pushToast } = useStore.getState();
    return pollJob(
      running,
      api.annotate.queueJob,
      (job) => {
        setReviewQueue(job);
        if (job.error !== null) pushToast(job.error);
      },
      (e) => {
        setReviewQueue(null);
        pushToast(messageOf(e));
      },
    );
  }, [running]);
}
