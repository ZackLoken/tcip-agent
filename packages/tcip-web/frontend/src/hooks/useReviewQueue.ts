/** The held review queue job, read back while it runs. */

import { useEffect } from "react";

import { api } from "@/api/client";
import { messageOf } from "@/api/http";
import { pollJob, unended } from "@/hooks/jobPoll";
import { useStore } from "@/store";

/**
 * Poll the held review queue job (`annotateUi.reviewQueue`) while it runs, holding each read.
 *
 * A failed job's error, as the backend wrote it, and a read that fails are the job's one entry,
 * standing on a key made from the job's id; a read that fails also drops the held job.
 */
export function useReviewQueuePoll(): void {
  const running = useStore((s) => {
    const held = s.annotateUi.reviewQueue;
    return held && unended(held.state) ? held.job_id : null;
  });

  useEffect(() => {
    const { setReviewQueue, raiseNotification, annotateUi } = useStore.getState();
    const held = annotateUi.reviewQueue;
    if (running === null || held === null) return;
    const failed = (sentence: string) =>
      raiseNotification({
        kind: "job",
        level: "error",
        text: { sentence },
        context: { target: `review queue of ${held.capture}` },
        key: `review-queue:${running}`,
        retell: false,
      });
    return pollJob(
      running,
      api.annotate.queueJob,
      (job) => {
        setReviewQueue(job);
        if (job.state.status === "failed") failed(job.state.error);
      },
      (e) => {
        setReviewQueue(null);
        failed(messageOf(e));
      },
    );
  }, [running]);
}
