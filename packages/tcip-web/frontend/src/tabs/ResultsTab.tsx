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

interface DateRow {
  date: string;
  [plantId: string]: number | string;
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
  // No baked-in dates: derived from the dataset (Prefill), or edited by hand.
  const [predsByDate, setPredsByDate] = useState<string>("{}");
  const [elongationHeight, setElongationHeight] = useState<number>(0.02);

  // Dataset tree (dates + prediction-model dir names) used for prefill.
  const [dates, setDates] = useState<string[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [predModel, setPredModel] = useState("");

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

  useEffect(() => {
    if (!datasetRoot) return;
    void api.dataset
      .tree(datasetRoot)
      .then((t) => {
        setDates(t.dates_with_images);
        setModels(t.model_names);
        setPredModel((m) => m || t.model_names[0] || "");
      })
      .catch(() => {});
  }, [datasetRoot]);

  function prefillPreds() {
    if (!datasetRoot || dates.length === 0) return;
    const map: Record<string, string> = {};
    for (const d of dates) {
      map[d] = predModel ? `${datasetRoot}/predictions/${predModel}/${d}/detect` : "";
    }
    setPredsByDate(JSON.stringify(map, null, 2));
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
      const predsMap: Record<string, string> = JSON.parse(predsByDate);
      // Drop empty entries
      for (const k of Object.keys(predsMap)) {
        if (!predsMap[k]) delete predsMap[k];
      }
      const curveRes = await resultsApi.perPlantCurves({
        project_root: projectRoot,
        mapping_path: mappingPath,
        predictions_by_date: predsMap,
        elongation_height: elongationHeight,
      });
      setCurves(curveRes.rows ?? []);
      const onsetRes = await resultsApi.onsetDates(curveRes.rows ?? []);
      setOnset(onsetRes.rows ?? []);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function downloadCsv(rows: unknown[], filename: string) {
    if (rows.length === 0) return;
    try {
      const blob = await resultsApi.exportCsv(rows, filename);
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

  const downloadOnsetCsv = () => downloadCsv(onset, "catkin_phenology.csv");
  const downloadCurvesCsv = () => downloadCsv(curves, "catkin_curves.csv");

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
            <div className="flex items-center gap-2">
              <label className="tcip-label flex-1">
                Predictions by date (JSON: date → detect/ dir)
              </label>
              <select
                className="tcip-select text-[11px]"
                value={predModel}
                onChange={(e) => setPredModel(e.target.value)}
                title="Prediction model dir under predictions/"
              >
                {models.length === 0 && <option value="">no models</option>}
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <button
                className="tcip-btn text-[11px]"
                onClick={prefillPreds}
                disabled={dates.length === 0}
                title="Fill from the dataset's dates + this model's prediction dirs"
              >
                Prefill from dataset
              </button>
            </div>
            <textarea
              className="tcip-input h-24 font-mono text-[11px] leading-4"
              value={predsByDate}
              onChange={(e) => setPredsByDate(e.target.value)}
              spellCheck={false}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="tcip-label">Elongation bbox-height threshold</label>
            <input
              className="tcip-input"
              type="number"
              step="0.005"
              min="0"
              max="1"
              value={elongationHeight}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setElongationHeight(Number.isFinite(v) ? v : 0.02);
              }}
            />
            <button className="tcip-btn-primary" onClick={compute} disabled={loading}>
              {loading ? "Computing…" : "Compute curves + onset dates"}
            </button>
            <div className="flex gap-1">
              <button
                className="tcip-btn flex-1 text-[11px]"
                onClick={downloadCurvesCsv}
                disabled={curves.length === 0}
              >
                Curves CSV
              </button>
              <button
                className="tcip-btn-primary flex-1 text-[11px]"
                onClick={downloadOnsetCsv}
                disabled={onset.length === 0}
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
          Elongated / total ratio over time, per plant ({plantKeys.length} plants)
        </div>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={chartData}>
              <CartesianGrid stroke="#3A3A3A" strokeDasharray="3 3" />
              <XAxis dataKey="date" stroke="#8A8A8A" style={{ fontSize: 11 }} />
              <YAxis stroke="#8A8A8A" domain={[0, 1]} style={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: "#242424",
                  border: "1px solid #3A3A3A",
                  borderRadius: 4,
                  fontSize: 11,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: "#E0E0E0" }} />
              {plantKeys.slice(0, 30).map((pid, i) => (
                <Line
                  key={pid}
                  type="monotone"
                  dataKey={pid}
                  stroke={`hsl(${(i * 137) % 360}, 60%, 60%)`}
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
          Onset dates (catkin_05 / 50 / 95 per plant) — {onset.length} rows
        </div>
        {onset.length > 0 ? (
          <div className="overflow-auto max-h-96">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-tcip-panel">
                <tr className="border-b border-tcip-border">
                  <th className="tcip-th">Plant ID</th>
                  <th className="tcip-th">Accession</th>
                  <th className="tcip-th">N points</th>
                  <th className="tcip-th">05per date</th>
                  <th className="tcip-th">50per date</th>
                  <th className="tcip-th">95per date</th>
                </tr>
              </thead>
              <tbody>
                {onset.map((r) => (
                  <tr key={r.plant_id} className="border-t border-tcip-border first:border-t-0">
                    <td className="py-1.5 pr-3 font-mono">{r.plant_id}</td>
                    <td className="pr-3">{r.accession ?? "—"}</td>
                    <td className="pr-3 tabular-nums">{r.n_datapoints}</td>
                    <td className="pr-3 tabular-nums">{r.catkin_05per_date ?? "—"}</td>
                    <td className="pr-3 tabular-nums">{r.catkin_50per_date ?? "—"}</td>
                    <td className="pr-3 tabular-nums">{r.catkin_95per_date ?? "—"}</td>
                  </tr>
                ))}
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
