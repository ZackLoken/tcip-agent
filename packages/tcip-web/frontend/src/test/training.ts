import { decodeRefusal } from "@/api/http";
import type {
  AsRecordedChoice,
  CompareBestResult,
  CompareExperiment,
  CompareRegistryEntry,
  CompareResult,
  SelectionChoice,
  SplitChoices,
  trainingApi,
} from "@/api/training";
import type {
  RunRow,
  SweepGroup,
  TrainingDetail,
  TrainingListing,
  TrainingRowFrame,
  TrainingStatusFrame,
} from "@/api/types.generated";
import produced from "@/test/trainingAnswers.json";

type Answer<K extends keyof typeof trainingApi> = Awaited<ReturnType<(typeof trainingApi)[K]>>;

/** A refusal the generator received: the status the route answered and the body it sent, whose
 *  detail is `D`. */
interface Refusal<D> {
  status: number;
  body: { detail: D };
}

/** The refusals the generator received, each with its own detail: a relaunch of a run whose
 *  builder no import resolves, a ranking whose every carrier is unverified, a ranking by a metric
 *  no checkpoint carries, a TensorBoard over a run with no logs, and one over a failed run with no
 *  logs. */
interface Refusals {
  refused_relaunch: Refusal<{ message: string; issues: string[] }>;
  unverified_refusal: Refusal<{ message: string; all_unverified: boolean }>;
  absent_metric_refusal: Refusal<{ message: string }>;
  no_logs_refusal: Refusal<{ message: string; no_logs: boolean }>;
  crashed_no_logs_refusal: Refusal<{ message: string; no_logs: boolean }>;
}

/** What `tools/generate_training_fixture.py` received from the training routes over two runs, a
 *  sweep holding one trial, a run whose builder no import resolves and a run completed with no
 *  optimizer step, each laid down by the platform's own writers. */
const PRODUCED = produced as {
  listing: TrainingListing;
  detail: TrainingDetail;
  compare: CompareResult & {
    experiments: (CompareExperiment & { registry: CompareRegistryEntry[] })[];
  };
  compare_best: CompareBestResult;
  metric_directions: Answer<"metricDirections">;
  split_choices: SplitChoices;
  cancel: Answer<"cancel">;
  run_relaunch: Answer<"relaunch">;
  sweep_relaunch: Answer<"relaunch">;
  metric_frame: TrainingRowFrame;
  batch_frame: TrainingRowFrame;
  status_frame: TrainingStatusFrame;
  unknown_frame: TrainingStatusFrame;
} & Refusals;

const [RUN] = PRODUCED.listing.runs;
const [SWEEP] = PRODUCED.listing.sweeps;
const [RUN_COLUMN, COMPLETED_COLUMN] = PRODUCED.compare.experiments;
const [REGISTERED] = COMPLETED_COLUMN.registry;

/** The produced first run's row with `change` laid over it. */
export function runRow(change: Partial<RunRow>): RunRow {
  return { ...RUN, ...change };
}

/** The produced sweep's trial row with `change` laid over it. */
export function trialRow(change: Partial<RunRow>): RunRow {
  return { ...SWEEP.trials[0], ...change };
}

/** The produced sweep with `change` laid over it. */
export function sweepGroup(change: Partial<SweepGroup>): SweepGroup {
  return { ...SWEEP, ...change };
}

/** The metric the produced sweep's recorded base config ranks its trials by. */
export const SWEEP_SELECTION_METRIC = (
  SWEEP.input.base_config as { evaluation: { selection_metric: string } }
).evaluation.selection_metric;

/** The produced listing with `change` laid over it. */
export function trainingListing(change: Partial<TrainingListing>): TrainingListing {
  return { ...PRODUCED.listing, ...change };
}

/** The produced first run's detail with `change` laid over it. */
export function trainingDetail(change: Partial<TrainingDetail>): TrainingDetail {
  return { ...PRODUCED.detail, ...change };
}

/** The produced comparison with `change` laid over it, counting the columns it then holds. */
export function compareResult(change: Partial<Omit<CompareResult, "count">>): CompareResult {
  const compared = { ...PRODUCED.compare, ...change };
  return { ...compared, count: compared.experiments.length };
}

/** The produced first run's comparison column, which registered nothing, with `change` laid
 *  over it. */
export function compareEntry(change: Partial<CompareExperiment>): CompareExperiment {
  return { ...RUN_COLUMN, ...change };
}

/** The produced completed run's registered checkpoint, as its comparison column lists it, with
 *  `change` laid over it. */
export function registryEntry(change: Partial<CompareRegistryEntry>): CompareRegistryEntry {
  return { ...REGISTERED, ...change };
}

/** The produced ranking of the comparison with `change` laid over it. */
export function compareBestAnswer(change: Partial<CompareBestResult>): CompareBestResult {
  return { ...PRODUCED.compare_best, ...change };
}

/** The produced metric directions with each of `change`'s laid over them. */
export function metricDirections(change: Record<string, boolean>): Answer<"metricDirections"> {
  return { higher_is_better: { ...PRODUCED.metric_directions.higher_is_better, ...change } };
}

/** The produced first run's data choices with `change` laid over them. */
export function splitChoices(change: Partial<SplitChoices>): SplitChoices {
  return { ...PRODUCED.split_choices, ...change };
}

/** The produced first run's as-recorded data choice with `change` laid over it. */
export function asRecordedChoice(change: Partial<AsRecordedChoice>): AsRecordedChoice {
  return { ...PRODUCED.split_choices.as_recorded, ...change };
}

/** The produced first run's one recorded partition choice with `change` laid over it. */
export function selectionChoice(change: Partial<SelectionChoice>): SelectionChoice {
  return { ...PRODUCED.split_choices.selections[0], ...change };
}

/** The produced first run's cancellation answer with `change` laid over it. */
export function cancelAnswer(change: Partial<Answer<"cancel">>): Answer<"cancel"> {
  return { ...PRODUCED.cancel, ...change };
}

/** The produced answer to a relaunch of the first run. */
export const RUN_RELAUNCH = PRODUCED.run_relaunch;

/** The produced answer to a relaunch of the sweep. */
export const SWEEP_RELAUNCH = PRODUCED.sweep_relaunch;

/** The detail the produced refusal `name` carries. */
export function refusalDetail<N extends keyof Refusals>(name: N): Refusals[N]["body"]["detail"] {
  return PRODUCED[name].body.detail;
}

/** The error the client decodes from the produced refusal `name`. */
export function producedRefusal(name: keyof Refusals): Promise<Error> {
  const { body, status } = PRODUCED[name];
  return decodeRefusal(new Response(JSON.stringify(body), { status }));
}

/** The produced epoch frame of the metrics stream, carrying `row`. */
export function metricFrame(row: TrainingRowFrame["row"]): TrainingRowFrame {
  return { ...PRODUCED.metric_frame, row };
}

/** The produced batch frame of the metrics stream, carrying `row`. */
export function batchFrame(row: TrainingRowFrame["row"]): TrainingRowFrame {
  return { ...PRODUCED.batch_frame, row };
}

/** The produced terminal frame of the metrics stream, its run's row with `change` laid over it. */
export function statusFrame(change: Partial<RunRow>): TrainingStatusFrame {
  const { status } = PRODUCED.status_frame;
  return { ...PRODUCED.status_frame, status: status && { ...status, ...change } };
}

/** The produced frame the metrics stream sends for an id naming no run. */
export const UNKNOWN_RUN_FRAME = PRODUCED.unknown_frame;
