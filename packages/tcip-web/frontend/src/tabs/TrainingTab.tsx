import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

import { openTrainingStream, trainingApi } from "@/api/training";
import type { MetricRow, TrainingRunSummary } from "@/api/training";
import { useStore } from "@/store";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";
import { mergeMetric } from "@/tabs/trainingMetrics";

// Runs can only be stopped while still active; terminal/historical runs show no button.
const TRAINING_CANCELLABLE: ReadonlySet<string> = new Set(["created", "running"]);

// Training is configured and launched by the agent (it writes the model + train loop). This tab
// tracks those runs and their live metrics; the human does not hand-author a model here.
export function TrainingTab() {
  const projectRoot = useStore((s) => s.gui.dataset.project_root);

  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const streamRef = useRef<(() => void) | null>(null);

  const refreshRuns = useCallback(async () => {
    try {
      const r = await trainingApi.listRuns();
      setRuns(r.runs ?? []);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
    const t = setInterval(refreshRuns, 4000);
    return () => clearInterval(t);
  }, [refreshRuns]);

  useEffect(() => {
    if (!selectedRun || !projectRoot) return;
    // Clear the previous run's curve; the stream replays this run from the start, so a
    // seed GET would just double-load the same rows. The WS is the single source now.
    setMetrics([]);
    streamRef.current?.();
    streamRef.current = openTrainingStream(projectRoot, selectedRun, (msg) => {
      if (msg.type === "metric" && msg.row) {
        setMetrics((prev) => mergeMetric(prev, msg.row as MetricRow));
      } else if (msg.type === "status") {
        // Terminal frame: surface completion/failure/cancellation + refresh the list.
        const st = msg.status?.status;
        if (st) useStore.getState().pushToast(`Training ${selectedRun}: ${st}`, "info");
        void refreshRuns();
      } else if (msg.type === "error") {
        useStore.getState().pushToast(`Training stream error: ${msg.error ?? "unknown run"}`);
      }
    });
    return () => streamRef.current?.();
  }, [selectedRun, projectRoot, refreshRuns]);

  async function onCancel(runId: string) {
    try {
      await trainingApi.cancel(runId);
      void refreshRuns();
    } catch (e) {
      useStore.getState().pushToast(`Cancel failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  const chartData = useMemo(() => {
    return metrics.map((m, i) => ({ step: m.epoch ?? m.step ?? i, ...m }));
  }, [metrics]);

  const metricKeys = useMemo(() => {
    const keys = new Set<string>();
    metrics.forEach((row) => {
      Object.entries(row).forEach(([k, v]) => {
        if (typeof v === "number" && k !== "epoch" && k !== "step") keys.add(k);
      });
    });
    return Array.from(keys);
  }, [metrics]);

  return (
    <div className="flex-1 grid grid-cols-[400px_1fr] overflow-hidden">
      {/* Left sidebar: runs (configured + launched by the agent) */}
      <div className="border-r border-tcip-border flex flex-col overflow-hidden">
        <div className="flex-1 overflow-auto p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="tcip-heading">Runs</span>
            <span className="flex-1" />
            <button className="tcip-btn text-[11px]" onClick={refreshRuns}>
              ↻&nbsp;&nbsp;Refresh
            </button>
          </div>
          {runs.length === 0 ? (
            <div className="text-[11px] text-tcip-muted">
              No runs yet. The agent configures and launches training.
            </div>
          ) : (
            <ul className="space-y-1">
              {runs.map((r) => (
                <li
                  key={r.run_id}
                  className={`p-2 rounded border cursor-pointer transition-colors ${
                    selectedRun === r.run_id
                      ? "border-tcip-accent bg-tcip-accent/10"
                      : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
                  }`}
                  onClick={() => setSelectedRun(r.run_id)}
                >
                  <div className="font-mono text-[11px]">{r.run_id}</div>
                  <div className="text-[10px] text-tcip-muted flex justify-between">
                    <span>
                      {r.status}
                      {r.external && r.status === "running" ? " · agent" : ""}
                    </span>
                    {r.best_metric !== undefined && r.best_metric !== null && (
                      <span className="tabular-nums">best: {Number(r.best_metric).toFixed(3)}</span>
                    )}
                  </div>
                  {TRAINING_CANCELLABLE.has(r.status) && !r.external && (
                    <button
                      className="tcip-btn text-[10px] mt-1"
                      onClick={(e) => {
                        e.stopPropagation();
                        void onCancel(r.run_id);
                      }}
                    >
                      Cancel
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Right pane: live curves */}
      <div className="flex flex-col overflow-hidden">
        <div className="p-4 border-b border-tcip-border flex items-center gap-2">
          {selectedRun ? (
            <>
              <span className="tcip-heading">Live metrics</span>
              <span className="font-mono text-[12px] text-tcip-fg">{selectedRun}</span>
            </>
          ) : (
            <span className="tcip-heading">Select a run to view metrics</span>
          )}
        </div>
        <div className="flex-1 p-4 overflow-hidden">
          {selectedRun && chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" />
                <XAxis
                  dataKey="step"
                  stroke={CHART.axis}
                  style={{ fontSize: 11 }}
                  label={{
                    value: "epoch/step",
                    position: "insideBottom",
                    offset: -5,
                    fill: CHART.axis,
                  }}
                />
                <YAxis stroke={CHART.axis} style={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: CHART.tooltipBg,
                    border: `1px solid ${CHART.tooltipBorder}`,
                    borderRadius: 4,
                    fontSize: 11,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: CHART.legendText }} />
                {metricKeys.map((key, i) => (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stroke={CHART_LINE_COLORS[i % CHART_LINE_COLORS.length]}
                    dot={false}
                    strokeWidth={1.5}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-tcip-muted text-[12px]">
              {selectedRun ? "Waiting for metrics…" : "No run selected."}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
