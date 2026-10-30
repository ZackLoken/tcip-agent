/** Reading one background job back by its id until it ends. */

/** How often a job is read back. Long enough that a job running for minutes is not polled
 *  thousands of times, short enough that its progress moves and its end shows within a second. */
export const JOB_POLL_MS = 700;

/**
 * Read the job `jobId` names through `read` every {@link JOB_POLL_MS}, handing each read to
 * `onRead`; returns the cancel that stops it.
 *
 * The poll stops once a read says the job is no longer `live` (that read handed on first), when a
 * read fails (the error handed to `onError`), and when the cancel is called, which a caller holding
 * the poll in an effect calls on unmount or from inside `onRead`.
 */
export function pollJob<J extends { live: boolean }>(
  jobId: string,
  read: (jobId: string) => Promise<J>,
  onRead: (job: J) => void,
  onError: (error: unknown) => void,
): () => void {
  let canceled = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const next = () => {
    timer = setTimeout(() => {
      read(jobId).then(
        (job) => {
          if (canceled) return;
          onRead(job);
          if (job.live && !canceled) next();
        },
        (error: unknown) => {
          if (!canceled) onError(error);
        },
      );
    }, JOB_POLL_MS);
  };
  next();
  return () => {
    canceled = true;
    if (timer !== null) clearTimeout(timer);
  };
}
