import { useEffect, useMemo, useState } from "react";
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

// K4/K5: the trait a delivery is computed for is now a required, threaded parameter everywhere —
// hardcoded to catkin here since a second trait's own UI affordance (a picker) is separate,
// deferred GUI-design work, not this fix's scope (see Group A's design doc, Commit 3).
const TRAIT = "catkin";

export function ResultsTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const projectRoot = dataset.project_root;
  const datasetRoot = dataset.dataset_root;

  const [mappingPath, setMappingPath] = useState(
    projectRoot ? `${projectRoot}/.tcip/state/plant_mapping.json` : "",
  );
  // True unless a computed run reported that its predictions carried no elongation class.
  const [elongationUnclassified, setElongationUnclassified] = useState(false);

  // Dataset tree (dates + which models actually have predictions per date) drives the structured
  // per-date picker below — never a hand-edited JSON blob (K15 #10: models_with_predictions is
  // the same primitive the backend already computes this from, via api.dataset.tree).
  const [dates, setDates] = useState<string[]>([]);
  const [modelsByDate, setModelsByDate] = useState<Record<string, string[]>>({});
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
  // The last-computed predictions_by_date, kept so CSV export can send it for the classifier/
  // count-operating-point reconciliation (K15: exportCsv now needs real bucket evidence, not a
  // caller-asserted row string).
  const [lastPredsMap, setLastPredsMap] = useState<Record<string, string>>({});

  useEffect(() => {
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
      })
      .catch(() => {});
  }, [datasetRoot]);

  // Same prediction-dir convention prefillPreds always used — kept as one place, now fed by the
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

  async function compute() {
    if (!projectRoot) return;
    setLoading(true);
    setError(null);
    try {
      const predsMap: Record<string, string> = {};
      for (const d of dates) {
        const dir = predDirFor(d, dateModel[d] ?? "");
        if (dir) predsMap[d] = dir;
      }
      setLastPredsMap(predsMap);
      const curveRes = await resultsApi.perPlantCurves({
        project_root: projectRoot,
        mapping_path: mappingPath,
        predictions_by_date: predsMap,
        trait: TRAIT,
      });
      const unclassified = curveRes.elongation_classified === false;
      setElongationUnclassified(unclassified);
      setCurves(curveRes.rows ?? []);
      if (unclassified) {
        // No elongation class → the fraction is not a bloom measurement. Don't derive
        // milestones from it at all, so there is nothing to export (belt-and-braces with the
        // disabled export buttons + the compute_phenology MCP tool's hard refusal).
        setOnset([]);
      } else {
        const onsetRes = await resultsApi.onsetDates(curveRes.rows ?? [], TRAIT);
        setOnset(onsetRes.rows ?? []);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function downloadCsv(
    rows: unknown[],
    filename: string,
    exportKind: "phenology" | "diagnostic",
  ) {
    if (rows.length === 0) return;
    try {
      const blob = await resultsApi.exportCsv(rows, filename, exportKind, lastPredsMap);
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

  // Measurement-integrity guard: never export a bloom CSV built on predictions that carry
  // no elongation class. Mirrors the compute_phenology MCP tool, which hard-refuses the same
  // case, so the GUI and the agent surface behave identically (see CLAUDE.md invariant).
  const downloadOnsetCsv = () => {
    if (elongationUnclassified) return;
    void downloadCsv(onset, "catkin_phenology.csv", "phenology");
  };
  const downloadCurvesCsv = () => {
    if (elongationUnclassified) return;
    // A curve export is the same delivered bloom measurement as the milestone one, just
    // un-summarised — it declares itself phenology and takes the identical gate.
    void downloadCsv(curves, "catkin_curves.csv", "phenology");
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

  // K4/K5: milestone columns are read generically off whatever the (threaded) trait's spec
  // returned — never hardcoded to catkin's own column names, so a second trait's rows render
  // instead of showing empty (round-2 finding RC-NEW-3's frontend half).
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
        if (!known.has(k) && k.endsWith("_date")) cols.add(k);
      });
    });
    return Array.from(cols).sort();
  }, [onset]);

  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      {/* Plant mapping — build (from geolocated images + plant CSVs) or point at an existing file */}
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
            <label className="tcip-label mt-1">Plant CSV path(s) — one per line</label>
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
            {dates.length === 0 ? (
              <div className="text-[11px] text-tcip-muted">No dates in this dataset yet.</div>
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
                                {opts.length === 0 ? "no predictions" : "— skip —"}
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
              Bloom = the fraction of a plant's detected catkins that are elongated. Elongation is a
              class from the validated classifier, not a bbox measurement — predictions must be
              elongation-classified.
            </p>
            {elongationUnclassified && (
              <div className="text-[11px] text-tcip-fp border border-tcip-fp/40 rounded p-2">
                These predictions carry no elongation class — the curves below are not a valid bloom
                measurement, so CSV export is disabled. Run the elongation classifier first.
              </div>
            )}
            <button className="tcip-btn-primary" onClick={compute} disabled={loading}>
              {loading ? "Computing…" : "Compute curves + onset dates"}
            </button>
            <div className="flex gap-1">
              <button
                className="tcip-btn flex-1 text-[11px]"
                onClick={downloadCurvesCsv}
                disabled={curves.length === 0 || elongationUnclassified}
              >
                Curves CSV
              </button>
              <button
                className="tcip-btn-primary flex-1 text-[11px]"
                onClick={downloadOnsetCsv}
                disabled={onset.length === 0 || elongationUnclassified}
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
          Elongated / total ratio over time, per plant
          {plantKeys.length > 30
            ? ` (showing 30 of ${plantKeys.length} plants — the onset table below has all)`
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
          Phenology milestones (elongation + catkin_05 / 50 / 95 per plant) — {onset.length} rows
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
                  // K15 finding #9: gate the DERIVATION (matching the setOnset([]) pattern used
                  // above) rather than a banner — a plant with any unclassified/missing date shows
                  // as such, not silently blank milestone cells with no explanation.
                  const rowValid = r.n_dates_unclassified === 0 && r.n_dates_missing_images === 0;
                  // Stage-6 review N6: "valid" alone doesn't distinguish real bloom data from a
                  // plant that was fully classified/observed but never had a single detection
                  // (before emergence, or a genuinely empty scene) — that reads as no observations,
                  // not blank cells next to a reassuring "valid".
                  const neverObserved = rowValid && r.n_observed_dates === 0;
                  return (
                    <tr key={r.plant_id} className="border-t border-tcip-border first:border-t-0">
                      <td className="py-1.5 pr-3 font-mono">{r.plant_id}</td>
                      <td className="pr-3">{r.accession ?? "—"}</td>
                      <td className="pr-3 tabular-nums">{r.n_datapoints}</td>
                      <td className="pr-3">
                        {neverObserved ? (
                          <span
                            className="text-tcip-muted"
                            title="Fully classified and fully observed, but no detections on any date — nothing to derive milestones from."
                          >
                            no observations
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
                      {milestoneColumns.map((c) => (
                        <td key={c} className="pr-3 tabular-nums">
                          {(r[c] as string | null) ?? "—"}
                        </td>
                      ))}
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
