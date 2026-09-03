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
  n_trials?: number | null;
  search_alg?: string | null;
  scheduler?: string | null;
  param_space_keys?: string[];
  /** Whether the manifest carries a base_config: the relaunchable marker. */
  relaunchable?: boolean;
  /** Why this sweep cannot be relaunched, in the platform's own words; null when it can be. */
  reason?: string | null;
  cancel_requested?: boolean;
  /** The sweep this one was relaunched from, or null when it was not a relaunch. */
  relaunched_from?: string | null;
  /** Draws per sampled point (run_hyperparameter_search's own data.split.seed grid axis); null/1 for no draws. */
  split_draws?: number | null;
  /** Whether the recorded base_config redraws train/val inside a bound split manifest's own
   * members, rather than sweeping seeds over a drawn split. */
  redraws_within_manifest?: boolean;
  /** Whether run_hyperparameter_search has written this sweep's first manifest yet. False in the pre-manifest
   * window a relaunch opens (the route registers the job before it answers), so a caller keys
   * its not-yet-recorded state on this rather than on a 404 that window never produces. */
  has_manifest: boolean;
}

export interface SweepDetail {
  sweep_id: string;
  status: string;
  error?: string | null;
  result: unknown;
  /** Whether run_hyperparameter_search has written this sweep's first manifest yet; see Sweep.has_manifest. */
  has_manifest: boolean;
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

  relaunch: (study_name: string) =>
    postJson<{ sweep_id?: string; [k: string]: unknown }>(ROUTES.postTuningSweeps, { study_name }),

  cancel: (sweep_id: string) =>
    postJson<{ study_name: string; status: string; cancel_requested: boolean }>(
      ROUTES.postTuningSweepsBySweepIdCancel(sweep_id),
      {},
    ),

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
