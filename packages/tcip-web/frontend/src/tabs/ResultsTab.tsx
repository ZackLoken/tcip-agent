import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api } from "@/api/client";
import {
  resultsApi,
  type PhenologyRequest,
  type OnsetRow,
  type PerPlantRow,
  type PlantMappingSummary,
} from "@/api/inference";
import { useStore } from "@/store";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";

interface DateRow {
  date: string;
  [plantId: string]: number | string | null;
}

/**
 * Parse an ISO date (`YYYY-MM-DD`) into a sortable integer. Mirrors the backend `_date_key`
 * in results.py so the chart's date order matches the server-computed onset table.
 */
function dateKey(date: string): number {
  const parts = date.split("-");
  if (parts.length !== 3) return 0;
  // Match the backend _date_key (Python int()): reject junk-suffixed parts like "15b"
  // so the chart's date order agrees with the server-computed onset ordering.
  if (!parts.every((p) => /^\d+$/.test(p))) return 0;
  const [y, m, d] = parts.map((x) => parseInt(x, 10));
  return y * 10000 + m * 100 + d;
}

export function ResultsTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const projectRoot = dataset.project_root;
  const datasetRoot = dataset.dataset_root;

  const [mappingPath, setMappingPath] = useState(
    projectRoot ? `${projectRoot}/.tcip/state/plant_mapping.json` : "",
  );
  // True unless a computed run reported that its predictions carried no positive-state class.
  const [positiveClassUnassessed, setPositiveClassUnassessed] = useState(false);

  // Dataset tree (dates + which models actually have predictions per date) drives the structured
  // per-date picker below, never a hand-edited JSON blob; models_with_predictions is the same
  // primitive the backend already computes this from, via api.dataset.tree.
  const [dates, setDates] = useState<string[]>([]);
  const [modelsByDate, setModelsByDate] = useState<Record<string, string[]>>({});
  const [datesError, setDatesError] = useState<string | null>(null);
  // The model picked per date; "" means "skip this date" (dropped before compute()).
  const [dateModel, setDateModel] = useState<Record<string, string>>({});

  // Plant-mapping build inputs.
  const [plantCsvText, setPlantCsvText] = useState("");
  const [nnTolerance, setNnTolerance] = useState(10);
  const [buildSummary, setBuildSummary] = useState<PlantMappingSummary | null>(null);
  const [buildMsg, setBuildMsg] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);

  const [curves, setCurves] = useState<PerPlantRow[]>([]);
  const [onset, setOnset] = useState<OnsetRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  // The exact request the displayed numbers came from: the CSV door recomputes from these inputs
  // rather than being handed the rows, so export and screen share one producer.
  const [lastRequest, setLastRequest] = useState<PhenologyRequest | null>(null);
  // Reconciled evidence for what is currently displayed. `provisional` is true whenever a dimension
  // lacked on-disk backing, so the tables can say so instead of rendering a phenology date as "valid".
  const [provisional, setProvisional] = useState(false);
  const [validity, setValidity] = useState<Record<string, string>>({});
  const [unvalidatedRefusal, setUnvalidatedRefusal] = useState<string | null>(null);

  // The trait a delivery is computed for, resolved from this project's own registered traits
  // (never assumed): auto-selected when there is exactly one, left blank (with an explicit
  // error, not a silent guess) when there are zero, offered as a choice when there are several.
  const [availableTraits, setAvailableTraits] = useState<string[]>([]);
  const [trait, setTrait] = useState("");
  const [traitError, setTraitError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectRoot) return;
    setTrait("");
    setTraitError(null);
    void resultsApi
      .traits(projectRoot)
      .then((res) => {
        setAvailableTraits(res.traits);
        if (res.traits.length === 0) {
          setTraitError("No trait is registered for this project yet.");
        } else if (res.traits.length === 1) {
          setTrait(res.traits[0]);
        }
      })
      .catch((e) => {
        setAvailableTraits([]);
        setTraitError(
          `Could not load this project's registered traits: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [projectRoot]);

  const refreshDatasetTree = useCallback(() => {
    if (!datasetRoot) return;
    void api.dataset
      .tree(datasetRoot)
      .then((t) => {
        setDates(t.dates_with_images);
        setModelsByDate(t.models_by_date);
        // Default each date to its first model with predictions; a date with none stays "" (skip).
        setDateModel(
          Object.fromEntries(t.dates_with_images.map((d) => [d, t.models_by_date[d]?.[0] ?? ""])),
        );
        setDatesError(null);
      })
      .catch((e) => {
        setDatesError(
          `Could not load this dataset's dates: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [datasetRoot]);

  useEffect(() => {
    refreshDatasetTree();
  }, [refreshDatasetTree]);

  // Same prediction-dir convention prefillPreds always used, kept as one place, now fed by the
  // structured picker's selections instead of a hand-typed date -> path JSON object.
  function predDirFor(date: string, model: string): string {
    return model ? `${datasetRoot}/predictions/${model}/${date}/detect` : "";
  }

  async function buildMapping() {
    if (!datasetRoot) return;
    const paths = plantCsvText
      .split(/[\n,]/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (paths.length === 0) {
      setBuildMsg("Add at least one plant CSV path.");
      return;
    }
    setBuilding(true);
    setBuildMsg(null);
    setBuildSummary(null);
    try {
      const res = await resultsApi.buildPlantMapping({
        images_root: `${datasetRoot}/images`,
        plant_csv_paths: paths,
        nn_tolerance_m: nnTolerance,
        persist_path: mappingPath || undefined,
      });
      setBuildSummary(res.summary);
      setBuildMsg(`Mapping built + saved to ${mappingPath}`);
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Build mapping failed: ${e instanceof Error ? e.message : String(e)}`);
      setBuildMsg(null);
    } finally {
      setBuilding(false);
    }
  }

  async function compute(acknowledgeUnvalidated = false) {
    if (!projectRoot) return;
    if (!trait) {
      setError(traitError ?? "Pick a trait before computing.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const predsMap: Record<string, string> = {};
      for (const d of dates) {
        const dir = predDirFor(d, dateModel[d] ?? "");
        if (dir) predsMap[d] = dir;
      }
      const request = {
        project_root: projectRoot,
        mapping_path: mappingPath,
        predictions_by_date: predsMap,
        trait,
        acknowledge_unvalidated: acknowledgeUnvalidated,
      };
      setLastRequest(request);
      const curveRes = await resultsApi.perPlantCurves(request);
      // The numbers and the evidence that qualifies them arrive together, so the tables below can
      // never render an unvalidated phenology measurement as though it were a delivery.
      setProvisional(curveRes.provisional);
      setValidity(curveRes.validated);
      setUnvalidatedRefusal(null);
      const unclassified = curveRes.positive_class_assessed === false;
      setPositiveClassUnassessed(unclassified);
      setCurves(curveRes.rows ?? []);
      if (unclassified) {
        // No positive-state class → the fraction is not a phenology measurement. Don't derive
        // milestones from it at all, so there is nothing to export (belt-and-braces with the
        // disabled export buttons + the compute_phenology MCP tool's hard refusal).
        setOnset([]);
      } else {
        // Same inputs, not the curve rows: the server recomputes rather than trusting a table the
        // client hands back, so a milestone date and the curve it was read off cannot disagree.
        const onsetRes = await resultsApi.onsetDates(request);
        setOnset(onsetRes.rows ?? []);
      }
    } catch (e) {
      // The server refuses unvalidated evidence by default. Surface why, plus the one-click way to
      // see the numbers anyway (clearly marked provisional), so an uncalibrated operating point is
      // a signposted next step rather than a dead end.
      const detail = e instanceof Error ? e.message : String(e);
      if (!acknowledgeUnvalidated && /unvalidated|not validated/i.test(detail)) {
        setUnvalidatedRefusal(detail);
        setCurves([]);
        setOnset([]);
      } else {
        setError(detail);
      }
    } finally {
      setLoading(false);
    }
  }

  async function downloadCsv(payload: "curves" | "milestones", filename: string) {
    if (!lastRequest) return;
    try {
      const blob = await resultsApi.exportCsv(lastRequest, payload, filename);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      useStore
        .getState()
        .pushToast(`CSV export failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // Measurement-integrity guard: never export a phenology CSV built on predictions that carry no
  // positive-state class, or on provisional evidence. Mirrors compute_phenology, which
  // hard-refuses both, so the GUI and the agent surface behave identically (see CLAUDE.md
  // invariant). The server refuses either case regardless; these keep the button from promising
  // what it can't do.
  const exportBlocked = positiveClassUnassessed || provisional;
  const downloadOnsetCsv = () => {
    if (exportBlocked) return;
    void downloadCsv("milestones", `${trait}_phenology.csv`);
  };
  const downloadCurvesCsv = () => {
    if (exportBlocked) return;
    // A curve export is the same delivered phenology measurement as the milestone one, just
    // un-summarised: same producer, same gate.
    void downloadCsv("curves", `${trait}_curves.csv`);
  };

  const chartData: DateRow[] = useMemo(() => {
    const byDate: Record<string, DateRow> = {};
    for (const r of curves) {
      byDate[r.date] ??= { date: r.date };
      byDate[r.date][r.plant_id] = r.ratio;
    }
    return Object.values(byDate).sort((a, b) => dateKey(a.date) - dateKey(b.date));
  }, [curves]);

  const plantKeys = useMemo(() => {
    const set = new Set<string>();
    curves.forEach((r) => set.add(r.plant_id));
    return Array.from(set);
  }, [curves]);

  // Which delivery dimensions the reconciled evidence actually failed on, reused to make the
  // agent hand-off below specific to what's missing rather than a generic "go calibrate" ask.
  const unvalidatedDims = useMemo(
    () =>
      Object.entries(validity)
        .filter(([, state]) => state === "false")
        .map(([dim]) => dim),
    [validity],
  );

  // What a breeder can't act on themselves: the backend refuses to deliver phenology until a
  // calibrated export_predictions + calibrate_classifier_operating_point stand behind it (see
  // results.py's _refusal). Hand that off to the agent instead of leaving the tool names on
  // screen with no next step.
  function calibrationRequest(detail: string | null): string {
    const dims = unvalidatedDims.length > 0 ? unvalidatedDims.join(", ") : "the operating point";
    const subject = trait ? `the "${trait}" trait` : "this trait";
    return (
      `Phenology delivery for ${subject} is blocked: ${dims} not validated on disk. ` +
      "Please produce the predictions via a calibrated export_predictions and calibrate the " +
      "classifier via calibrate_classifier_operating_point so this validates, then let me know " +
      "when it's ready so I can recompute here." +
      (detail ? ` Details from the app: ${detail}` : "")
    );
  }

  // Milestone columns are read generically off whatever the (threaded) trait's spec returned,
  // never hardcoded to one trait's own column names, so a different trait's rows render instead
  // of showing empty.
  const milestoneColumns = useMemo(() => {
    const known = new Set([
      "plant_id",
      "accession",
      "n_datapoints",
      "n_dates_unclassified",
      "n_dates_missing_images",
      "n_observed_dates",
    ]);
    const cols = new Set<string>();
    onset.forEach((r) => {
      Object.keys(r).forEach((k) => {
        // `_date` only: each milestone's `*_date_bound` is rendered beside its own date below
        // rather than as a column of its own.
        if (!known.has(k) && k.endsWith("_date")) cols.add(k);
      });
    });
    return Array.from(cols).sort();
  }, [onset]);

  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      {traitError && <div className="tcip-panel p-3 text-[11px] text-tcip-fp">{traitError}</div>}
      {availableTraits.length > 1 && (
        <div className="tcip-panel p-3 flex items-center gap-2">
          <label className="tcip-label">Trait</label>
          <select
            className="tcip-input w-auto"
            value={trait}
            onChange={(e) => setTrait(e.target.value)}
          >
            <option value="" disabled>
              Choose a trait…
            </option>
            {availableTraits.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
      )}
      {/* Plant mapping: build (from geolocated images + plant CSVs) or point at an existing file */}
      <div className="tcip-panel p-4">
        <div className="tcip-heading mb-3">Plant mapping</div>
        <div className="grid grid-cols-[1fr_1fr] gap-3">
          <div className="flex flex-col gap-1">
            <label className="tcip-label">
              Mapping file (built here, or an existing one to load)
            </label>
            <input
              className="tcip-input"
              value={mappingPath}
              onChange={(e) => setMappingPath(e.target.value)}
              placeholder="…/.tcip/state/plant_mapping.json"
            />
            <label className="tcip-label mt-1">Plant CSV path(s), one per line</label>
            <textarea
              className="tcip-input h-16 font-mono text-[11px] leading-4"
              value={plantCsvText}
              onChange={(e) => setPlantCsvText(e.target.value)}
              placeholder="…/plants_block_A.csv"
              spellCheck={false}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="tcip-label">NN tolerance (m)</label>
            <input
              className="tcip-input"
              type="number"
              step="1"
              min="0"
              value={nnTolerance}
              onChange={(e) => setNnTolerance(parseFloat(e.target.value) || 0)}
            />
            <button
              className="tcip-btn-primary"
              onClick={buildMapping}
              disabled={building || !datasetRoot}
            >
              {building ? "Building…" : "Build + save mapping"}
            </button>
            {buildMsg && <div className="text-[11px] text-tcip-muted">{buildMsg}</div>}
          </div>
        </div>
        {buildSummary && (
          <div className="mt-2 text-[11px] text-tcip-muted tabular-nums">
            {Object.entries(buildSummary).map(([d, s]) => (
              <div key={d}>
                {d}: {s.n_mapped}/{s.n_images} mapped · avg {s.avg_distance_m.toFixed(1)} m
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="tcip-panel p-4">
        <div className="tcip-heading mb-3">Per-plant phenology curves</div>
        <div className="grid grid-cols-[1fr_180px] gap-3">
          <div className="flex flex-col gap-1">
            <label className="tcip-label">Predictions by date</label>
            {datesError && (
              <div className="text-[11px] text-tcip-fp mb-1">
                {datesError}{" "}
                <button className="tcip-btn text-[11px] ml-1" onClick={refreshDatasetTree}>
                  Retry
                </button>
              </div>
            )}
            {dates.length === 0 ? (
              !datesError && (
                <div className="text-[11px] text-tcip-muted">No dates in this dataset yet.</div>
              )
            ) : (
              <div className="max-h-40 overflow-auto rounded border border-tcip-border">
                <table className="w-full text-[11px]">
                  <tbody>
                    {dates.map((d) => {
                      const opts = modelsByDate[d] ?? [];
                      return (
                        <tr key={d} className="border-t border-tcip-border first:border-t-0">
                          <td className="py-1 pl-2 pr-2 font-mono tabular-nums">{d}</td>
                          <td className="py-1 pr-2">
                            <select
                              className="tcip-select text-[11px] w-full"
                              value={dateModel[d] ?? ""}
                              onChange={(e) =>
                                setDateModel((prev) => ({ ...prev, [d]: e.target.value }))
                              }
                              disabled={opts.length === 0}
                              title={
                                opts.length === 0
                                  ? "No model has predictions for this date"
                                  : "Model whose predictions to use for this date"
                              }
                            >
                              <option value="">
                                {opts.length === 0 ? "no predictions" : "(skip)"}
                              </option>
                              {opts.map((m) => (
                                <option key={m} value={m}>
                                  {m}
                                </option>
                              ))}
                            </select>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          <div className="flex flex-col gap-2">
            <p className="text-[11px] text-tcip-muted">
              The positive-state fraction is the share of a plant's detected objects that are in the
              trait's positive state. That state is a class from the validated classifier, not a
              bbox measurement; predictions must be classified for it.
            </p>
            {positiveClassUnassessed && (
              <div className="text-[11px] text-tcip-fp border border-tcip-fp/40 rounded p-2">
                These predictions carry no positive-state class, so the curves below are not a valid
                phenology measurement and CSV export is disabled. Run the classifier first.
              </div>
            )}
            {unvalidatedRefusal && (
              <div className="text-[11px] text-tcip-fp border border-tcip-fp/40 rounded p-2 flex flex-col gap-2">
                <div>
                  These predictions have no validated operating point on disk, so this is not yet a
                  deliverable phenology measurement. Calibrate first, or look at the numbers as
                  provisional, which will not let you export them.
                </div>
                <div className="text-tcip-muted">{unvalidatedRefusal}</div>
                <div className="flex gap-2">
                  <button
                    className="tcip-btn text-[11px] self-start"
                    onClick={() => void compute(true)}
                    disabled={loading}
                  >
                    Show provisional numbers
                  </button>
                  <button
                    className="tcip-btn-primary text-[11px] self-start"
                    onClick={() =>
                      useStore
                        .getState()
                        .sendToAgentTerminal(calibrationRequest(unvalidatedRefusal))
                    }
                  >
                    Ask the agent to calibrate this
                  </button>
                </div>
              </div>
            )}
            {provisional && (
              <div className="text-[11px] text-tcip-fp border border-tcip-fp/40 rounded p-2 flex flex-col gap-2">
                <div>
                  Provisional: shown for inspection only, not a deliverable phenotype. Unvalidated:{" "}
                  {unvalidatedDims.join(", ") || "unknown"}. CSV export stays disabled until both
                  dimensions are validated on disk.
                </div>
                <button
                  className="tcip-btn-primary text-[11px] self-start"
                  onClick={() => useStore.getState().sendToAgentTerminal(calibrationRequest(null))}
                >
                  Ask the agent to calibrate this
                </button>
              </div>
            )}
            <button className="tcip-btn-primary" onClick={() => void compute()} disabled={loading}>
              {loading ? "Computing…" : "Compute curves + onset dates"}
            </button>
            <div className="flex gap-1">
              <button
                className="tcip-btn flex-1 text-[11px]"
                onClick={downloadCurvesCsv}
                disabled={curves.length === 0 || exportBlocked}
              >
                Curves CSV
              </button>
              <button
                className="tcip-btn-primary flex-1 text-[11px]"
                onClick={downloadOnsetCsv}
                disabled={onset.length === 0 || exportBlocked}
              >
                Onset CSV
              </button>
            </div>
          </div>
        </div>
        {error && <div className="mt-2 text-[11px] text-tcip-fp">{error}</div>}
      </div>

      <div className="tcip-panel p-4 h-80">
        <div className="tcip-heading mb-3">
          Positive-state fraction over time, per plant
          {plantKeys.length > 30
            ? ` (showing 30 of ${plantKeys.length} plants, the onset table below has all)`
            : ` (${plantKeys.length} plants)`}
        </div>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={chartData}>
              <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" />
              <XAxis dataKey="date" stroke={CHART.axis} style={{ fontSize: 11 }} />
              <YAxis stroke={CHART.axis} domain={[0, 1]} style={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: CHART.tooltipBg,
                  border: `1px solid ${CHART.tooltipBorder}`,
                  borderRadius: 4,
                  fontSize: 11,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: CHART.legendText }} />
              {plantKeys.slice(0, 30).map((pid, i) => (
                <Line
                  key={pid}
                  type="monotone"
                  dataKey={pid}
                  stroke={CHART_LINE_COLORS[i % CHART_LINE_COLORS.length]}
                  dot={false}
                  strokeWidth={1}
                  isAnimationActive={false}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-full text-tcip-muted text-[12px]">
            No data. Configure mapping + predictions above, then compute.
          </div>
        )}
      </div>

      <div className="tcip-panel p-4">
        <div className="tcip-heading mb-3">
          Phenology milestones for {trait || "the selected trait"} (onset + percentile crossings per
          plant): {onset.length} rows
        </div>
        {onset.length > 0 ? (
          <div className="overflow-auto max-h-96">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-tcip-panel">
                <tr className="border-b border-tcip-border">
                  <th className="tcip-th">Plant ID</th>
                  <th className="tcip-th">Accession</th>
                  <th className="tcip-th">N points</th>
                  <th className="tcip-th">Validity</th>
                  {milestoneColumns.map((c) => (
                    <th key={c} className="tcip-th">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {onset.map((r) => {
                  // Gate the derivation itself (matching the setOnset([]) pattern used above)
                  // rather than a banner, so a plant with any unclassified/missing date shows as
                  // such, not silently blank milestone cells with no explanation.
                  const rowValid = r.n_dates_unclassified === 0 && r.n_dates_missing_images === 0;
                  // "Valid" alone doesn't distinguish real detection data from a plant that was fully
                  // classified/observed but never had a single detection (before emergence, or a
                  // genuinely empty scene): that reads as no observations, not blank cells next
                  // to a reassuring "valid".
                  const neverObserved = rowValid && r.n_observed_dates === 0;
                  return (
                    <tr key={r.plant_id} className="border-t border-tcip-border first:border-t-0">
                      <td className="py-1.5 pr-3 font-mono">{r.plant_id}</td>
                      <td className="pr-3">{r.accession ?? "—"}</td>
                      <td className="pr-3 tabular-nums">{r.n_dates}</td>
                      <td className="pr-3">
                        {neverObserved ? (
                          <span
                            className="text-tcip-muted"
                            title="Fully classified and fully observed, but no detections on any date, so there is nothing to derive milestones from."
                          >
                            no observations
                          </span>
                        ) : rowValid && provisional ? (
                          // Coverage is complete, but the measurement behind these dates has no
                          // validated operating point. The banner announcing that sits two panels
                          // up and scrolls out of view, so the row must say so where it is read:
                          // a phenology date beside a plain "valid" would be an unearned precision claim.
                          <span
                            className="text-tcip-fp"
                            title="Coverage is complete, but the operating point behind these dates is not validated on disk: provisional, not a deliverable phenotype."
                          >
                            provisional
                          </span>
                        ) : rowValid ? (
                          <span className="text-tcip-muted">valid</span>
                        ) : (
                          <span
                            className="text-tcip-fp"
                            title={`${r.n_dates_unclassified} unclassified date(s), ${r.n_dates_missing_images} missing-image date(s)`}
                          >
                            incomplete
                          </span>
                        )}
                      </td>
                      {milestoneColumns.map((c) => {
                        const date = r[c] as string | null;
                        const bound = r[`${c}_bound`] as string | null;
                        // A left-censored crossing means the first observation already met the
                        // target, so the true date is only an upper bound; a right-censored one
                        // means the last observation still hadn't, so the true date (if any) is
                        // after this one, a lower bound. Rendering either as a plain date is a
                        // precision claim the data does not support.
                        const marker =
                          bound === "left_censored"
                            ? {
                                symbol: "≤",
                                className: "text-tcip-fp",
                                title:
                                  "Left-censored: the first observation already met this target, so the true date is at or before this one.",
                              }
                            : bound === "right_censored"
                              ? {
                                  symbol: ">",
                                  className: "text-tcip-fp",
                                  title:
                                    "Right-censored: the last observation still hadn't met this target, so the true date, if any, is after this one.",
                                }
                              : bound === "interpolated"
                                ? {
                                    symbol: "~",
                                    className: "text-tcip-muted",
                                    title: "Interpolated between two observed dates.",
                                  }
                                : null;
                        return (
                          <td key={c} className="pr-3 tabular-nums">
                            {date ?? "—"}
                            {date && marker && (
                              <span className={`ml-1 ${marker.className}`} title={marker.title}>
                                {marker.symbol}
                              </span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-[11px] text-tcip-muted">Run the compute step to populate.</div>
        )}
      </div>
    </div>
  );
}
