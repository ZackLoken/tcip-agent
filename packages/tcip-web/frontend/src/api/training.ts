/** Training-tab specific REST + WebSocket helpers. */

import { getJson, postJson, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type {
  TrainingDetail,
  TrainingListing,
  TrainingMetricFrame,
  TrainingStatusFrame,
} from "@/api/types.generated";
import { createReconnectingSocket, jsonFrameHandlers } from "@/lib/reconnectingSocket";

export interface TensorboardLaunch {
  url?: string;
  port?: number;
  pid?: number;
  logdir?: string;
  /** Present instead of a url when the process died during startup; `output` is what it wrote. */
  error?: string;
  output?: string;
}

export interface MetricRow {
  // The producer (tcip_store.values) writes null for a non-finite metric value.
  [metric: string]: number | string | null | undefined;
}

export interface AsRecordedChoice {
  case: "bound" | "drawn";
  line: string;
  compatible: boolean;
  reason: string | null;
}

export interface SelectionChoice {
  selection_dir: string;
  enabled: boolean;
  reason: string | null;
  seed: number | null;
  group_by: string | null;
  train: number;
  val: number;
  calibration: number;
  /** The recorded data.split keys choosing this partition drops (seed, group_by, ...). */
  replaced_split_keys: string[];
}

export interface SplitChoices {
  as_recorded: AsRecordedChoice;
  selections: SelectionChoice[];
}

/** One entry a comparison's own experiment registered, reduced to what leaves the backend. */
export interface CompareRegistryEntry {
  name: string;
  metrics: Record<string, number | string | null> | null;
  metrics_source: string | null;
  registered_at: string | null;
}

/** The run's own partition, from its resolved record (never the launch config's own pre-launch
 * intent), reduced to the four states a comparison names. */
export interface CompareSplit {
  case: "bound" | "drawn" | "spatial" | "none";
  selection_dir?: string;
  seed?: number | null;
}

/** One marked experiment's own column in the comparison, every value labeled by which record
 * it came from. `error` alone (no other field) marks an id compare_experiments could not even
 * read; every other field is absent only on that entry. */
export interface CompareExperiment {
  experiment_id: string;
  error?: string;
  state?: string | null;
  n_epochs?: number;
  last_logged_metrics?: MetricRow;
  rows_after_end?: number | null;
  /** The status record's own failure reason; null for a run that never failed. */
  status_error?: string | null;
  /** The config's builder; null when the config names none (never a fabricated "unknown"). */
  model?: string | null;
  task?: string | null;
  subject?: string | null;
  dataset_id?: string | null;
  dataset_fingerprint?: string | null;
  split?: CompareSplit;
  /** The run's own completed checkpoint as a one-entry list; empty for a run that did not
   * complete. */
  registry?: CompareRegistryEntry[];
}

export interface CompareResult {
  experiments: CompareExperiment[];
  count: number;
  /** null when any compared id is an error entry, or a fingerprint is missing/unrecorded/mixed. */
  same_dataset_fingerprint: boolean | null;
}

/** One entry the rank excluded for being unverified (metrics_source is not "trainer"). */
export interface CompareBestExcluded {
  name: string;
  metrics_source: string | null;
}

/** The marked comparison's own best-model answer, projected: no checkpoint path, config or
 * file size ever leaves the backend. */
export interface CompareBestResult {
  name: string;
  experiment_id: string | null;
  metrics: Record<string, number | string | null>;
  metrics_source: string | null;
  /** The metric the answer was ranked by. */
  ranking_basis: string;
  higher_is_better: boolean;
  direction_source: string;
  excluded_unverified: CompareBestExcluded[];
}

export const trainingApi = {
  listSplitChoices: (experiment_id: string) =>
    getJson<SplitChoices>(ROUTES.getTrainingConfigsByExperimentIdSplits(experiment_id)),

  /** Start a new run or sweep from the recorded one ``relaunched_from`` names; the answer names
   * a run by ``experiment_id`` and a sweep by ``sweep_id``. */
  relaunch: (relaunched_from: string, user: string, selection_dir?: string | null) =>
    postJson<{ experiment_id?: string; sweep_id?: string; [k: string]: unknown }>(
      ROUTES.postTrainingRuns,
      selection_dir ? { relaunched_from, selection_dir, user } : { relaunched_from, user },
    ),

  listRuns: () => getJson<TrainingListing>(ROUTES.getTrainingRuns),

  getRun: (experiment_id: string) =>
    getJson<TrainingDetail>(ROUTES.getTrainingRunsByExperimentId(experiment_id)),

  launchTensorboard: (experiment_id: string) =>
    postJson<TensorboardLaunch>(
      ROUTES.postTrainingRunsByExperimentIdTensorboard(experiment_id),
      {},
    ),

  cancel: (experiment_id: string, user: string) =>
    postJson<{ experiment_id: string; state: string; cancel_requested: boolean }>(
      ROUTES.postTrainingRunsByExperimentIdCancel(experiment_id),
      { user },
    ),

  compare: (experiment_ids: string[]) =>
    postJson<CompareResult>(ROUTES.postTrainingCompare, { experiment_ids }),

  compareBest: (params: {
    experiment_ids: string[];
    metric: string;
    higher_is_better?: boolean | null;
    include_unverified?: boolean;
  }) => postJson<CompareBestResult>(ROUTES.postTrainingCompareBest, params),

  /** evaluation.py's own declared-direction table, unaudited: a plain read the comparison's
   * metric chooser groups its stamped keys by, never a call through the rank tool. */
  metricDirections: () =>
    getJson<{ higher_is_better: Record<string, boolean> }>(ROUTES.getTrainingMetricDirections),
};

export type TrainingStreamMsg = TrainingMetricFrame | TrainingStatusFrame;

/**
 * Open a live metrics stream for a training run or trial of the open project, auto-reconnecting
 * with capped backoff.
 * The server replays all rows from the start on each (re)connect, so the consumer must
 * dedupe by epoch. A ``status`` frame carrying a report is terminal; one carrying only
 * ``error`` names an id no record claims (selected at its launch moment, before the record
 * exists, or simply unknown), and the socket keeps reconnecting under backoff until a report
 * arrives. That error-only frame never resets the backoff, so the reconnect delay grows to the
 * cap while the run stays unknown.
 */
export function openTrainingStream(
  experiment_id: string,
  onMessage: (msg: TrainingStreamMsg) => void,
): () => void {
  const url = wsUrl(ROUTES.socketTrainingRunsByExperimentIdStream(experiment_id));
  const socket = createReconnectingSocket({
    url,
    ...jsonFrameHandlers<TrainingStreamMsg>(
      onMessage,
      (frame) => frame.type === "status" && frame.status != null,
      (frame) => frame.type === "metric" || (frame.type === "status" && frame.status != null),
    ),
  });
  socket.start();
  return () => socket.stop();
}
