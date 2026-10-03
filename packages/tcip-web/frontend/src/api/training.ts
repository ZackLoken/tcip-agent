/** Training-tab specific REST + WebSocket helpers. */

import { getJson, postJson, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type { TrainingMetricFrame, TrainingStatusFrame } from "@/api/types.generated";
import { createReconnectingSocket, jsonFrameHandlers } from "@/lib/reconnectingSocket";
import type { DeclaredIdentity } from "@/store/slices/agentActivity";

export interface TrainingRunSummary {
  /** A training run's own id: an experiment record's, always (no record, no run). */
  experiment_id: string;
  status: string;
  current_epoch?: number;
  best_metric?: number;
  /** The bare metric name (val_-unprefixed) ``best_metric`` was selected on; null when the
   * run's config cannot resolve one. */
  best_metric_name?: string | null;
  output_dir?: string;
  /** The agent identity the run's launch event carries: empty for a launch no agent declared
   * itself to, null when no launch event names the run. */
  launch?: DeclaredIdentity | null;
  /** The run directory's last sign of life (ISO-8601): no process id is recorded anywhere, so
   * this is the one signal a stale ``running`` row (its process gone, read as live for the rest
   * of the heartbeat window) can show. */
  heartbeat?: string | null;
}

export interface TrainingRunDetail {
  experiment_id?: string;
  status?: string;
  epoch?: number | null;
  best_metric?: number | null;
  output_dir?: string | null;
  error?: string;
  /** Set only while a TensorBoard this backend started is still serving the run. */
  tensorboard_url?: string | null;
}

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
  epoch?: number;
  step?: number;
  // The producer (tcip_store.values) writes null for a non-finite metric value.
  [metric: string]: number | string | null | undefined;
}

export interface LaunchableConfig {
  experiment_id: string;
  builder: string | null;
  task: string | null;
  images_dir: string | null;
  subject: string | null;
  created: string | null;
  state: string;
  parent_experiment: string | null;
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
  n_rows?: number;
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
  higher_is_better: boolean;
  direction_source: string;
  excluded_unverified: CompareBestExcluded[];
}

export const trainingApi = {
  listConfigs: () => getJson<{ configs: LaunchableConfig[] }>(ROUTES.getTrainingConfigs),

  listSplitChoices: (experiment_id: string) =>
    getJson<SplitChoices>(ROUTES.getTrainingConfigsByExperimentIdSplits(experiment_id)),

  relaunch: (experiment_id: string, user: string, selection_dir?: string | null) =>
    postJson<{ experiment_id?: string; [k: string]: unknown }>(
      ROUTES.postTrainingRuns,
      selection_dir ? { experiment_id, selection_dir, user } : { experiment_id, user },
    ),

  listRuns: () => getJson<{ runs: TrainingRunSummary[] }>(ROUTES.getTrainingRuns),

  getRun: (experiment_id: string) =>
    getJson<TrainingRunDetail>(ROUTES.getTrainingRunsByExperimentId(experiment_id)),

  launchTensorboard: (experiment_id: string) =>
    postJson<TensorboardLaunch>(
      ROUTES.postTrainingRunsByExperimentIdTensorboard(experiment_id),
      {},
    ),

  cancel: (experiment_id: string, user: string) =>
    postJson<{ experiment_id: string; status: string; cancel_requested: boolean }>(
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
 * Open a live metrics stream for a training run of the open project, auto-reconnecting with
 * capped backoff.
 * The server replays all rows from the start on each (re)connect, so the consumer must
 * dedupe by epoch/step. A ``status`` frame carrying a report is terminal; one carrying only
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
