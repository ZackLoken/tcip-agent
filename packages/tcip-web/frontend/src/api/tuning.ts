/** Tuning (HPO) API helpers for the Tuning tab. */

import { getJson, postJson } from "@/api/http";

export interface Sweep {
  sweep_id: string;
  status: string;
  error: string | null;
  has_result: boolean;
}

export interface SweepDetail {
  sweep_id: string;
  status: string;
  result: unknown;
}

export const tuningApi = {
  listSweeps: () => getJson<{ sweeps: Sweep[] }>("/api/tuning/sweeps"),

  getSweep: (sweep_id: string) => getJson<SweepDetail>(`/api/tuning/sweeps/${sweep_id}`),

  launch: (body: {
    base_config: unknown;
    param_space: unknown;
    n_trials: number;
    output_dir: string;
    search_alg: string;
    scheduler: string;
  }) => postJson<{ sweep_id?: string; [k: string]: unknown }>("/api/tuning/launch", body),
};
