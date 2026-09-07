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
  /** Whether the manifest carries a base_config and names no caller-supplied data.split.seed
   * axis at one draw: the relaunchable marker. */
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

/** One split_draws point's own summary over its draws (training_tools.group_split_draws's own
 *  block); a malformed count narrows to null rather than a counted zero. The block's ninth key,
 *  values, is not mirrored here: the table renders no raw list. */
export interface SplitDrawBlock {
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  n: number | null;
  n_complete: number | null;
  seeds: (number | null)[];
  seeds_complete: (number | null)[];
}

/** One split_sensitivity entry: a point's own params (null for Ray's own never-answered row,
 *  which names no point to group under), its draws block, and whether it is eligible for best. */
export interface SplitSensitivityGroup {
  point: Record<string, unknown> | null;
  block: SplitDrawBlock;
  eligible: boolean;
}

/** The split_draws spread a completed sweep's result carries when split_draws was above 1. */
export interface SweepDraws {
  groups: SplitSensitivityGroup[];
  best: SplitDrawBlock | null;
  bestState: string | null;
  splitDraws: number;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function numberOrNullArray(value: unknown): (number | null)[] {
  return Array.isArray(value) ? value.map(numberOrNull) : [];
}

function splitDrawBlockOf(value: unknown): SplitDrawBlock {
  const v = typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
  return {
    mean: numberOrNull(v.mean),
    std: numberOrNull(v.std),
    min: numberOrNull(v.min),
    max: numberOrNull(v.max),
    n: numberOrNull(v.n),
    n_complete: numberOrNull(v.n_complete),
    seeds: numberOrNullArray(v.seeds),
    seeds_complete: numberOrNullArray(v.seeds_complete),
  };
}

function splitSensitivityGroupOf(value: unknown): SplitSensitivityGroup {
  const v = typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
  const point =
    typeof v.point === "object" && v.point !== null ? (v.point as Record<string, unknown>) : null;
  return { point, block: splitDrawBlockOf(v.block), eligible: v.eligible === true };
}

/**
 * A completed sweep's split_draws spread, narrowed field by field with null fallbacks (the
 * form of api/inference.ts's own refusal narrowers, extended to narrow the arrays), or null
 * when result carries no split_sensitivity array (split_draws was 1, or the result predates it).
 */
export function sweepDrawsOf(result: unknown): SweepDraws | null {
  if (typeof result !== "object" || result === null) return null;
  const r = result as Record<string, unknown>;
  if (!Array.isArray(r.split_sensitivity)) return null;
  return {
    groups: r.split_sensitivity.map(splitSensitivityGroupOf),
    best: r.best_value_spread == null ? null : splitDrawBlockOf(r.best_value_spread),
    bestState: typeof r.best_value_state === "string" ? r.best_value_state : null,
    splitDraws: numberOrNull(r.split_draws) ?? 1,
  };
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
