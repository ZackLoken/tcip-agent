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

export const trainingApi = {
  validate: (config: unknown) =>
    postJson<{ valid: boolean; issues: string[] }>(ROUTES.postTrainingValidate, { config }),

  launch: (config: unknown, output_dir: string) =>
    postJson<{ run_id?: string; [k: string]: unknown }>(ROUTES.postTrainingLaunch, {
      config,
      output_dir,
    }),

  listRuns: () => getJson<{ runs: TrainingRunSummary[] }>(ROUTES.getTrainingRuns),

  getRun: (run_id: string) => getJson<TrainingRunDetail>(ROUTES.getTrainingRunsByRunId(run_id)),

  launchTensorboard: (run_id: string) =>
    postJson<TensorboardLaunch>(ROUTES.postTrainingRunsByRunIdTensorboard(run_id), {}),

  getMetrics: (project_root: string, run_id: string) =>
    getJson<{ metrics: MetricRow[]; exists: boolean }>(
      `${ROUTES.getTrainingRunsByRunIdMetrics(run_id)}?project_root=${encodeURIComponent(project_root)}`,
    ),

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
