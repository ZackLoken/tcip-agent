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
  output_dir?: string;
  config_summary?: Record<string, unknown>;
  external?: boolean; // reconstructed from experiment records: running in another process
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
  [metric: string]: number | string | undefined;
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
}

export interface SplitChoices {
  as_recorded: AsRecordedChoice;
  manifests: SplitManifestChoice[];
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
