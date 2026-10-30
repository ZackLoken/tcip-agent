import type { trainingApi } from "@/api/training";
import type {
  ComparedRun,
  Comparison,
  NoLogsRefusal,
  Ranking,
  RunRow,
  SplitChoices,
  SweepGroup,
  TrainingDetail,
  TrainingListing,
  TrainingRowFrame,
  TrainingStatusFrame,
} from "@/api/types.generated";
import { declaredClient } from "@/store/slices/agentActivity";
import type { Refusal } from "@/test/refusals";
import produced from "@/test/trainingAnswers.json";

type Answer<K extends keyof typeof trainingApi> = Awaited<ReturnType<(typeof trainingApi)[K]>>;

/** The refusals the generator received, each with its own detail: a relaunch of a run whose
 *  builder no import resolves, a ranking whose every carrier is unverified, a ranking by a metric
 *  no checkpoint carries (three tool dicts the route lifts), a TensorBoard over a run with no
 *  logs, and one over a failed run with no logs. */
interface Refusals {
  refused_relaunch: Refusal<{ message: string; issues: string[] }>;
  unverified_refusal: Refusal<{ message: string; all_unverified: boolean }>;
  absent_metric_refusal: Refusal<{ message: string }>;
  no_logs_refusal: Refusal<NoLogsRefusal>;
  crashed_no_logs_refusal: Refusal<NoLogsRefusal>;
}

/** What `tools/generate_training_fixture.py` received from the training routes over the runs, the
 *  two sweeps and the trials its scenes laid down through the platform's own writers. */
const PRODUCED = produced as unknown as {
  listing: TrainingListing;
  listing_before_failure: TrainingListing;
  details: TrainingDetail[];
  compare: Comparison;
  compare_same: Comparison;
  compare_one: Comparison;
  compare_unread: Comparison;
  compare_unset: Comparison;
  compare_rich: Comparison;
  compare_unstamped: Comparison;
  compare_unranked: Comparison;
  compare_bound: Comparison;
  compare_verified: Comparison;
  compare_best: Ranking;
  rich_best: Ranking;
  unranked_best: Ranking;
  verified_best: Ranking;
  split_choices: SplitChoices;
  moved_split_choices: SplitChoices;
  cancel: Answer<"cancel">;
  run_relaunch: Answer<"relaunch">;
  sweep_relaunch: Answer<"relaunch">;
  frames: [...TrainingRowFrame[], TrainingStatusFrame];
  unknown_frame: TrainingStatusFrame;
} & Refusals;

/** The one item of `found`, which holds `what`; any other count throws. */
function only<T>(found: T[], what: string): T {
  if (found.length !== 1) throw new Error(`the produced answers hold ${found.length} ${what}`);
  return found[0];
}

const { runs: LISTED_RUNS, sweeps: LISTED_SWEEPS } = PRODUCED.listing;
const STATUS = PRODUCED.frames.at(-1) as TrainingStatusFrame;
const STREAMED = STATUS.experiment_id;

/** The produced listing, taken once every scene ran. */
export const TRAINING_LISTING = PRODUCED.listing;

/** The produced listing taken while the run `RUNS.streamed` names still ran, before it failed. */
export const LISTING_BEFORE_FAILURE = PRODUCED.listing_before_failure;

/** The listing's runs by what their scene left: `unlaunched` running with no launch event,
 *  `relaunched` running from a person's relaunch with no agent declared, `agent` running from a
 *  declared agent's launch, `completed` ended with a best selection value, `empty` completed
 *  having logged nothing, `streamed` failed after logging an epoch and a batch row, and
 *  `unlogged` failed with no logs. */
export const RUNS = {
  unlaunched: only(
    LISTED_RUNS.filter((r) => r.state === "running" && r.launch === null),
    "running runs no launch event names",
  ),
  relaunched: only(
    LISTED_RUNS.filter((r) => r.relaunched_from !== null),
    "relaunched runs",
  ),
  agent: only(
    LISTED_RUNS.filter((r) => Boolean(r.launch?.agent_client_name)),
    "runs an agent declared",
  ),
  completed: only(
    LISTED_RUNS.filter((r) => r.state === "completed" && r.best_metric !== null),
    "completed runs with a best value",
  ),
  empty: only(
    LISTED_RUNS.filter((r) => r.state === "completed" && r.current_epoch === null),
    "completed runs that logged nothing",
  ),
  streamed: only(
    LISTED_RUNS.filter((r) => r.experiment_id === STREAMED),
    "runs the stream watched",
  ),
  unlogged: only(
    LISTED_RUNS.filter((r) => r.state === "failed" && r.experiment_id !== STREAMED),
    "failed runs with no logs",
  ),
} satisfies Record<string, RunRow>;

/** The client the agent run's launch declared, as the GUI names a declared client. */
export const AGENT_CLIENT = (() => {
  const client = declaredClient(RUNS.agent.launch);
  if (client === null) throw new Error("the produced agent run declares no client");
  return client;
})();

/** The listing's sweeps: `searched`, whose one completed trial its outcome names beside one still
 *  running, and `relaunched`, started again and holding no trial yet. */
export const SWEEPS = {
  searched: only(
    LISTED_SWEEPS.filter((s) => s.trials.length > 0),
    "sweeps holding trials",
  ),
  relaunched: only(
    LISTED_SWEEPS.filter((s) => s.trials.length === 0),
    "sweeps holding no trial",
  ),
} satisfies Record<string, SweepGroup>;

/** The searched sweep's trials: the one it completed and the one still running. */
export const TRIALS = {
  completed: only(
    SWEEPS.searched.trials.filter((t) => t.state === "completed"),
    "completed trials",
  ),
  running: only(
    SWEEPS.searched.trials.filter((t) => t.state === "running"),
    "running trials",
  ),
};

/** The produced detail of the run, trial or sweep `id` names; an id the generator read no detail
 *  for throws. */
export function detailOf(id: string): TrainingDetail {
  return only(
    PRODUCED.details.filter((d) => d.run?.experiment_id === id || d.sweep?.sweep_id === id),
    `details of ${id}`,
  );
}

/** The comparisons the generator received: of a run that registered nothing and a completed one
 *  over another dataset (`compare`), of two runs over one dataset neither registered (`same`), of
 *  one run (`one`), of a run and an id naming none (`unread`), of the first run and a run over images
 *  under no dataset root, whose launch recorded no fingerprint (`unset`), of the first run and a
 *  completed one
 *  whose checkpoint carries a declared metric, an undeclared one and a string (`rich`), of a
 *  completed run whose checkpoint carries no metric while its log does and the first run
 *  (`unstamped`), of a completed run carrying only an undeclared metric and the completed one
 *  (`unranked`), of a relaunch bound to a selection and the run it relaunched (`bound`), and of a
 *  run the default trainer completed and the unstamped run (`verified`). */
export const COMPARISONS = {
  compare: PRODUCED.compare,
  same: PRODUCED.compare_same,
  one: PRODUCED.compare_one,
  unread: PRODUCED.compare_unread,
  unset: PRODUCED.compare_unset,
  rich: PRODUCED.compare_rich,
  unstamped: PRODUCED.compare_unstamped,
  unranked: PRODUCED.compare_unranked,
  bound: PRODUCED.compare_bound,
  verified: PRODUCED.compare_verified,
};

/** `comparison`'s columns, each a run the route read (the entry carrying `run`); an unread column
 *  throws. */
export function readColumns(comparison: Comparison): ComparedRun[] {
  return comparison.experiments.map((entry) => {
    if (!("run" in entry)) throw new Error(`the comparison could not read ${entry.experiment_id}`);
    return entry;
  });
}

/** The rankings by a declared metric the generator received: of the `compare`, `rich` and
 *  `unranked` comparisons' runs including unverified checkpoints, and of the `verified`
 *  comparison's runs leaving them out (`verified`). */
export const RANKINGS = {
  compare: PRODUCED.compare_best,
  rich: PRODUCED.rich_best,
  unranked: PRODUCED.unranked_best,
  verified: PRODUCED.verified_best,
};

/** The produced data choices of the run `RUNS.unlaunched` names: its recorded choice, the
 *  dataset's drawn partition offered, and a partition whose record will not read offered disabled
 *  with its read error. */
export const SPLIT_CHOICES = PRODUCED.split_choices;

/** The produced data choices of a run whose images were removed once it was opened: its recorded
 *  choice refused for the missing directory, and no other partition offered. */
export const MOVED_SPLIT_CHOICES = PRODUCED.moved_split_choices;

/** The produced answer to a cancellation of the run `RUNS.unlaunched` names. */
export const CANCEL = PRODUCED.cancel;

/** The produced answer to a relaunch of the first run. */
export const RUN_RELAUNCH = PRODUCED.run_relaunch;

/** The produced answer to a relaunch of the sweep. */
export const SWEEP_RELAUNCH = PRODUCED.sweep_relaunch;

/** The training routes' produced refusals, each read whole or through `thrown`. */
export const TRAINING_REFUSALS: Refusals = PRODUCED;

/** The frames the metrics stream sent, in order, while the run `RUNS.streamed` names logged a batch
 *  of its first epoch, that epoch's row, an epoch row whose loss is not finite, a third epoch row
 *  and a batch of a fourth epoch it never finished, before it failed. */
export const STREAM = PRODUCED.frames.slice(0, -1) as TrainingRowFrame[];

/** The stream's epoch frames, in order. */
export const METRIC_FRAMES = STREAM.filter((frame) => frame.type === "metric");

/** The stream's batch frames, in order: one of the first epoch, one of the unfinished fourth. */
export const BATCH_FRAMES = STREAM.filter((frame) => frame.type === "batch");

/** The stream's terminal frame, sent once the run failed. */
export const STATUS_FRAME = STATUS;

/** The produced frame the metrics stream sends for an id naming no run. */
export const UNKNOWN_RUN_FRAME = PRODUCED.unknown_frame;
