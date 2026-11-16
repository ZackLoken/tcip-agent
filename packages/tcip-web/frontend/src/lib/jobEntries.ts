/** Each background job's one entry in the notifications area, told from the listings' answers. */

import { ended } from "@/hooks/jobPoll";
import { TERMINAL_STATES } from "@/lib/runStatus";
import { tell } from "@/lib/tell";
import { jobKey } from "@/lib/trainingKeys";
import { useStore } from "@/store";
import type { JobListings, JobsTold, ListingName } from "@/store/slices/jobListings";
import { LEVELS, type MessageLevel, selectStanding } from "@/store/slices/notifications";

/** One job as an answer lists it: its id, how the person reads its name, its state, and the
 *  sentence telling it. */
interface ListedJob {
  id: string;
  name: string;
  state: string;
  sentence: string;
}

/** One raise of a job's entry, and whether its key is released after it. */
interface JobRaise {
  id: string;
  level: MessageLevel;
  sentence: string;
  retell: boolean;
  release: boolean;
}

/** Whether `state` is one a run or sweep reads while its sign of life is stale and no end is
 *  written: terminal to the training tab, never an end. */
const stalled = (state: string) => TERMINAL_STATES.has(state) && !ended(state);

/** The job `name` in `state`, with its `progress` while it runs and its `error` when it has one. */
function listedJob(
  id: string,
  name: string,
  state: string,
  progress: string | null,
  error: string | null,
): ListedJob {
  const running = progress !== null && !ended(state) && !stalled(state);
  const said = `${name}: ${state}${running ? `, ${progress}` : ""}`;
  return { id, name, state, sentence: error ? `${said}. ${error}` : said };
}

/** The jobs each listing's answer lists: a training answer's runs and sweeps, a sweep's trials
 *  not among them, and an inference answer's jobs. */
const LISTED: { [N in ListingName]: (answer: JobListings[N]["answer"]) => ListedJob[] } = {
  training: ({ runs, sweeps }) => [
    ...runs.map((r) =>
      listedJob(
        r.experiment_id,
        `Training run ${r.experiment_id}`,
        r.state,
        r.current_epoch === null ? null : `epoch ${r.current_epoch}`,
        r.status_error,
      ),
    ),
    ...sweeps.map((s) =>
      listedJob(s.sweep_id, `Sweep ${s.sweep_id}`, s.state, null, s.status_error),
    ),
  ],
  inference: (jobs) =>
    Object.values(jobs).map((j) =>
      listedJob(
        j.job_id,
        `Inference job ${j.job_id}`,
        j.state.status,
        j.total > 0 ? `${j.done} of ${j.total} images` : null,
        "error" in j.state ? j.state.error : null,
      ),
    ),
};

/** How loud an entry telling a job in `state` is. */
function levelOf(state: string): MessageLevel {
  if (state === "failed") return "error";
  return stalled(state) ? "warning" : "info";
}

/**
 * Tell the jobs `answer`, the listing `listing` answered for the open project `project`, lists,
 * against what the last answer for that project and listing told.
 *
 * The first answer tells each unended job; a later one tells a job not seen unended before
 * that is unended now, or ended and not listed before. A job told unended is watched: a
 * sentence changed since the last told updates its standing entry in place, and none is raised
 * once its entry is dismissed. Its end, its state becoming or leaving stalled, and its absence
 * from an answer are told again, announced as new; an end or an absence also ends the watch and
 * releases the entry's key, so the same end answered again tells nothing.
 */
export function tellJobs<N extends ListingName>(
  listing: N,
  project: string,
  answer: JobListings[N]["answer"],
): void {
  const { jobsTold, keepJobsTold, releaseNotification } = useStore.getState();
  const before: JobsTold | undefined = jobsTold[project]?.[listing];
  const key = (id: string) => jobKey(listing, project, id);
  const listed = LISTED[listing](answer);
  const watched: JobsTold["watched"] = {};
  const raises: JobRaise[] = [];
  for (const job of listed) {
    const { id, name, state, sentence } = job;
    const seen = before?.watched[id];
    const over = ended(state);
    const raise = { id, level: levelOf(state), sentence, release: over };
    if (seen === undefined) {
      if (!over || (before !== undefined && !before.listed.includes(id)))
        raises.push({ ...raise, retell: true });
    } else if (over || stalled(state) !== stalled(seen.state)) {
      raises.push({ ...raise, retell: true });
    } else if (sentence !== seen.sentence && selectStanding(key(id))(useStore.getState())) {
      raises.push({ ...raise, retell: false });
    }
    if (!over) watched[id] = { name, state, sentence };
  }
  const listedIds = listed.map((j) => j.id);
  for (const [id, { name }] of Object.entries(before?.watched ?? {})) {
    if (listedIds.includes(id)) continue;
    const sentence = `${name}: no longer listed; its end was not seen here`;
    raises.push({ id, level: "warning", sentence, retell: true, release: true });
  }
  keepJobsTold(project, listing, { listed: listedIds, watched });
  // The live region announces the last raise alone, so an assertive one goes after every polite.
  const assertive = (r: JobRaise) => Number(LEVELS[r.level].live === "assertive");
  for (const r of raises.sort((a, b) => assertive(a) - assertive(b))) {
    tell("job", r.level, r.sentence, r.id, key(r.id), r.retell);
    if (r.release) releaseNotification(key(r.id));
  }
}
