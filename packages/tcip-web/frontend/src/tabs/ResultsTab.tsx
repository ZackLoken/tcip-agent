import { useMemo, useState } from "react";
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

import { resultsApi, type OnsetRow, type PerPlantRow } from "@/api/inference";
import { useStore } from "@/store";

interface DateRow {
  date: string;
  [plantId: string]: number | string;
}

export function ResultsTab() {
  const dataset = useStore((s) => s.gui.dataset);

  const [mappingPath, setMappingPath] = useState(
    dataset.project_root ? `${dataset.project_root}/.tcip/state/plant_mapping.json` : "",
  );
  const [predsByDate, setPredsByDate] = useState<string>(() =>
    JSON.stringify(
      {
        "2-11-26": "",
        "3-2-26": "",
        "3-9-26": "",
        "3-18-26": "",
        "3-24-26": "",
      },
      null,
      2,
    ),
  );
  const [elongationHeight, setElongationHeight] = useState<number>(0.02);

  const [curves, setCurves] = useState<PerPlantRow[]>([]);
  const [onset, setOnset] = useState<OnsetRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  async function compute() {
    if (!dataset.project_root) return;
    setLoading(true);
    setError(null);
    try {
      const predsMap: Record<string, string> = JSON.parse(predsByDate);
      // Drop empty entries
      for (const k of Object.keys(predsMap)) {
        if (!predsMap[k]) delete predsMap[k];
      }
      const curveRes = await resultsApi.perPlantCurves({
        project_root: dataset.project_root,
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

  async function downloadOnsetCsv() {
    if (onset.length === 0) return;
    const blob = await resultsApi.exportCsv(onset, "catkin_phenology.csv");
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "catkin_phenology.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function downloadCurvesCsv() {
    if (curves.length === 0) return;
    const blob = await resultsApi.exportCsv(curves, "catkin_curves.csv");
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "catkin_curves.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  const chartData: DateRow[] = useMemo(() => {
    const byDate: Record<string, DateRow> = {};
    for (const r of curves) {
      byDate[r.date] ??= { date: r.date };
      byDate[r.date][r.plant_id] = r.ratio;
    }
    return Object.values(byDate).sort((a, b) => (a.date < b.date ? -1 : 1));
  }, [curves]);

  const plantKeys = useMemo(() => {
    const set = new Set<string>();
    curves.forEach((r) => set.add(r.plant_id));
    return Array.from(set);
  }, [curves]);

  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      <div className="tcip-panel p-3">
        <div className="font-semibold text-[13px] mb-2">Per-plant phenology curves</div>
        <div className="grid grid-cols-[1fr_1fr_180px] gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-[11px] text-tcip-muted">Plant mapping path</label>
            <input
              className="tcip-input"
              value={mappingPath}
              onChange={(e) => setMappingPath(e.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[11px] text-tcip-muted">
              Predictions by date (JSON: date → detect/ dir)
            </label>
            <textarea
              className="tcip-input h-24 font-mono text-[11px] leading-4"
              value={predsByDate}
              onChange={(e) => setPredsByDate(e.target.value)}
              spellCheck={false}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="text-[11px] text-tcip-muted">Elongation bbox-height threshold</label>
            <input
              className="tcip-input"
              type="number"
              step="0.005"
              min="0"
              max="1"
              value={elongationHeight}
              onChange={(e) => setElongationHeight(parseFloat(e.target.value))}
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

      <div className="tcip-panel p-3 h-80">
        <div className="text-[12px] text-tcip-muted mb-2">
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

      <div className="tcip-panel p-3">
        <div className="text-[12px] text-tcip-muted mb-2">
          Onset dates (catkin_05 / 50 / 95 per plant) — {onset.length} rows
        </div>
        {onset.length > 0 ? (
          <div className="overflow-auto max-h-96">
            <table className="w-full text-[11px]">
              <thead className="text-tcip-muted text-left sticky top-0 bg-tcip-panel">
                <tr>
                  <th className="py-1 pr-3">Plant ID</th>
                  <th className="pr-3">Accession</th>
                  <th className="pr-3">N points</th>
                  <th className="pr-3">05per date</th>
                  <th className="pr-3">50per date</th>
                  <th className="pr-3">95per date</th>
                </tr>
              </thead>
              <tbody>
                {onset.map((r) => (
                  <tr key={r.plant_id} className="border-t border-tcip-border">
                    <td className="py-1 font-mono">{r.plant_id}</td>
                    <td className="pr-3">{r.accession ?? "—"}</td>
                    <td className="pr-3">{r.n_datapoints}</td>
                    <td className="pr-3">{r.catkin_05per_date ?? "—"}</td>
                    <td className="pr-3">{r.catkin_50per_date ?? "—"}</td>
                    <td className="pr-3">{r.catkin_95per_date ?? "—"}</td>
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
