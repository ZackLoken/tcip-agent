/** Training-tab specific REST + WebSocket helpers. */

import { getJson, postJson, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type { TrainingMetricFrame, TrainingStatusFrame } from "@/api/types.generated";
import { createReconnectingSocket, jsonFrameHandlers } from "@/lib/reconnectingSocket";

export interface TrainingRunSummary {
  run_id: string;
  status: string;
  current_epoch?: number;
  best_metric?: number;
  /** The bare metric name (val_-unprefixed) ``best_metric`` was selected on; null when the
   * run's config cannot resolve one. */
  best_metric_name?: string | null;
  output_dir?: string;
  config_summary?: Record<string, unknown>;
  /** Set on every run reconstructed from a disk record and absent from this process's own run
   * registry, whatever its status; a process-locality fact only, never a statement about who
   * launched it (that is ``launched_by``, below). */
  external?: boolean;
  /** Who launched this run, from the record itself: ``{"launcher": "gui" | "agent" | "process"
   * | <other>}``, the identity fields alongside ``"agent"`` when an MCP handshake declared them,
   * or absent when the launch's tracking never reached the stamp, or the stamp failed. */
  launched_by?: Record<string, unknown> | null;
  /** Set once _ensure_experiment resolves this run's tracked experiment; null until then. */
  experiment_id?: string | null;
  /** Set when experiment tracking itself raised; null when it succeeded or never ran. */
  experiment_error?: string | null;
}

export interface TrainingRunDetail {
  run_id?: string;
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

export interface SplitManifestChoice {
  manifest_dir: string;
  enabled: boolean;
  reason: string | null;
  seed: number | null;
  group_by: string | null;
  train: number;
  val: number;
  calibration: number;
  other_dates: number;
  /** The recorded data.split keys choosing this partition drops (seed, group_by, ...). */
  replaced_split_keys: string[];
}

export interface SplitChoices {
  as_recorded: AsRecordedChoice;
  manifests: SplitManifestChoice[];
}

/** One entry a comparison's own experiment registered, reduced to what leaves the backend. */
export interface CompareRegistryEntry {
  name: string;
  metrics: Record<string, number | string | null> | null;
  metrics_source: string | null;
  registered_at: string | null;
}

/** The run's own partition, from its persisted split record (never the launch config's own
 * pre-launch intent), reduced to the four states a comparison names. */
export interface CompareSplit {
  case: "bound" | "drawn" | "none" | "error";
  manifest_dir?: string;
  seed?: number | null;
  error?: string;
}

/** One refused post-terminal mutation the platform audit log recorded against an experiment. */
export interface CompareRefusedMutation {
  timestamp: string | null;
  arguments: Record<string, unknown>;
}

/** One marked experiment's own column in the comparison, every value labelled by which record
 * it came from. `error` alone (no other field) marks an id compare_experiments could not even
 * read; every other field is absent only on that entry. */
export interface CompareExperiment {
  experiment_id: string;
  error?: string;
  recorded_state?: string | null;
  state?: string | null;
  log_locked?: boolean;
  n_epochs?: number;
  n_rows?: number;
  last_logged_metrics?: MetricRow;
  rows_after_end?: number | null;
  refused_mutations?: CompareRefusedMutation[];
  /** The status record's own failure reason; null for a run that never failed. */
  status_error?: string | null;
  /** The config's builder; null when the config names none (never a fabricated "unknown"). */
  model?: string | null;
  task?: string | null;
  subject?: string | null;
  dataset_id?: string | null;
  dataset_fingerprint?: string | null;
  fingerprint_formula_unrecorded?: boolean;
  split?: CompareSplit;
  /** This experiment's own registered entries; absent, with registry_error naming why, when
   * the project's registry index can't be read or matched at all. */
  registry?: CompareRegistryEntry[];
  registry_error?: string;
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

  relaunch: (experiment_id: string, split_manifest_dir?: string | null) =>
    postJson<{ run_id?: string; experiment_id?: string; [k: string]: unknown }>(
      ROUTES.postTrainingRuns,
      split_manifest_dir ? { experiment_id, split_manifest_dir } : { experiment_id },
    ),

  listRuns: () => getJson<{ runs: TrainingRunSummary[] }>(ROUTES.getTrainingRuns),

  getRun: (run_id: string) => getJson<TrainingRunDetail>(ROUTES.getTrainingRunsByRunId(run_id)),

  launchTensorboard: (run_id: string) =>
    postJson<TensorboardLaunch>(ROUTES.postTrainingRunsByRunIdTensorboard(run_id), {}),

  cancel: (run_id: string) =>
    postJson<{ run_id: string; status: string; cancel_requested: boolean }>(
      ROUTES.postTrainingRunsByRunIdCancel(run_id),
      {},
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
 * Open a live metrics stream for a training run, auto-reconnecting with capped backoff.
 * The server replays all rows from the start on each (re)connect, so the consumer must
 * dedupe by epoch/step. The ``status`` frame is always terminal, whether it carries a
 * known run's report or an unknown run's ``error``; once seen we stop reconnecting.
 */
export function openTrainingStream(
  project_root: string,
  run_id: string,
  onMessage: (msg: TrainingStreamMsg) => void,
): () => void {
  const url = wsUrl(
    `${ROUTES.socketTrainingRunsByRunIdStream(run_id)}?project_root=${encodeURIComponent(project_root)}`,
  );
  const socket = createReconnectingSocket({
    url,
    ...jsonFrameHandlers<TrainingStreamMsg>(onMessage, (frame) => frame.type === "status"),
  });
  socket.start();
  return () => socket.stop();
}
