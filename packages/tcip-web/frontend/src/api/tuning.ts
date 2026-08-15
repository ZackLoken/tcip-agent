/** Tuning (HPO) API helpers for the Tuning tab. */

import { getJson, postJson } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type { TensorboardLaunch } from "@/api/training";

export interface Sweep {
  sweep_id: string;
  status: string;
  error: string | null;
  has_result: boolean;
  /** True for a sweep recovered from its on-disk manifest rather than launched here. */
  external?: boolean;
}

export interface SweepDetail {
  sweep_id: string;
  status: string;
  error?: string | null;
  result: unknown;
}

export interface SweepTrial {
  trial_id: string;
  has_metrics: boolean;
  params: Record<string, unknown>;
  unconsumed_params: string[];
}

export const tuningApi = {
  listSweeps: () => getJson<{ sweeps: Sweep[] }>(ROUTES.getTuningSweeps),

  getSweep: (sweep_id: string) => getJson<SweepDetail>(ROUTES.getTuningSweepsBySweepId(sweep_id)),

  listTrials: (sweep_id: string) =>
    getJson<{ sweep_id: string; trials: SweepTrial[] }>(
      ROUTES.getTuningSweepsBySweepIdTrials(sweep_id),
    ),

  getTrialMetrics: (sweep_id: string, trial_id: string) =>
    getJson<{ metrics: Record<string, unknown>[]; exists: boolean }>(
      ROUTES.getTuningSweepsBySweepIdTrialsByTrialIdMetrics(sweep_id, trial_id),
    ),

  launch: (body: {
    base_config: unknown;
    param_space: unknown;
    n_trials: number;
    output_dir: string;
    search_alg: string;
    scheduler: string;
  }) => postJson<{ sweep_id?: string; [k: string]: unknown }>(ROUTES.postTuningLaunch, body),

  /** Ray runs one cluster per process, so its dashboard is not scoped to a sweep. */
  getRayDashboard: () => getJson<{ url: string | null }>(ROUTES.getTuningRayDashboard),

  launchSweepTensorboard: (sweep_id: string) =>
    postJson<TensorboardLaunch>(ROUTES.postTuningSweepsBySweepIdTensorboard(sweep_id), {}),

  launchTrialTensorboard: (sweep_id: string, trial_id: string) =>
    postJson<TensorboardLaunch>(
      ROUTES.postTuningSweepsBySweepIdTrialsByTrialIdTensorboard(sweep_id, trial_id),
      {},
    ),

  stopTrialTensorboard: (sweep_id: string, trial_id: string) =>
    postJson<{ status: string; pid?: number }>(
      ROUTES.postTuningSweepsBySweepIdTrialsByTrialIdTensorboardStop(sweep_id, trial_id),
      {},
    ),
};
