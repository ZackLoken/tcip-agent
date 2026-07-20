/**
 * HPO search-space model for the Tuning tab's structured form. Covers exactly the params
 * the backend's `_apply_hpo_params` actually sweeps (lr / weight_decay / batch_size) and
 * builds the typed param-space the `run_hpo` tool feeds to Ray Tune — so users configure a
 * sweep with a form instead of hand-writing JSON. Architecture axes are model-specific
 * (`builder_kwargs`, unknown to the GUI) and not swept generically.
 */

export type HpoParam =
  | { key: string; label: string; kind: "loguniform"; enabled: boolean; low: number; high: number }
  | {
      key: string;
      label: string;
      kind: "choices";
      enabled: boolean;
      options: string[];
      selected: string[];
    }
  | { key: string; label: string; kind: "numlist"; enabled: boolean; values: number[] };

export const DEFAULT_HPO_PARAMS: HpoParam[] = [
  { key: "lr", label: "Learning rate", kind: "loguniform", enabled: true, low: 1e-5, high: 1e-2 },
  {
    key: "weight_decay",
    label: "Weight decay",
    kind: "loguniform",
    enabled: true,
    low: 1e-5,
    high: 1e-2,
  },
  { key: "batch_size", label: "Batch size", kind: "numlist", enabled: true, values: [2, 4] },
];

/**
 * Search algorithms Ray Tune can drive. `random`/`grid` are native; the rest need their
 * backend installed on the server (an unavailable pick surfaces as a clear sweep error).
 * The agent normally chooses per task/data — this menu is here for the human's direct use.
 */
export const SEARCH_ALGORITHMS = [
  "random",
  "grid",
  "optuna",
  "bayesopt",
  "hyperopt",
  "nevergrad",
  "ax",
  "hebo",
  "zoopt",
  "bohb",
] as const;

/** Trial schedulers (all native). `bohb` pairs with the bohb searcher; `none` runs every
 *  trial to completion. */
export const SCHEDULERS = ["asha", "hyperband", "bohb", "pbt", "median", "none"] as const;

/** Parse a comma-separated list of numbers, dropping blanks / non-numbers.
 *  (Blanks must be dropped BEFORE Number() — `Number("")` is 0, not NaN.) */
export function parseNumList(s: string): number[] {
  return s
    .split(",")
    .map((x) => x.trim())
    .filter((x) => x !== "")
    .map((x) => Number(x))
    .filter((n) => Number.isFinite(n));
}

/**
 * Build the typed search space from the form params. Only enabled params are included; an
 * enabled categorical with no selected values/choices is skipped (an empty `choices` would
 * make the backend raise).
 */
export function buildSearchSpace(params: HpoParam[]): Record<string, unknown> {
  const space: Record<string, unknown> = {};
  for (const p of params) {
    if (!p.enabled) continue;
    if (p.kind === "loguniform") {
      // Skip a param whose bounds were cleared to NaN — an unbounded log-uniform makes
      // the sweep launch and then crash at runtime.
      if (Number.isFinite(p.low) && Number.isFinite(p.high)) {
        space[p.key] = { type: "loguniform", low: p.low, high: p.high };
      }
    } else if (p.kind === "choices") {
      if (p.selected.length > 0) space[p.key] = { type: "categorical", choices: p.selected };
    } else if (p.values.length > 0) {
      space[p.key] = { type: "categorical", choices: p.values };
    }
  }
  return space;
}
