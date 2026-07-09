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
import { mergeMetric } from "@/tabs/trainingMetrics";

// Runs can only be stopped while still active; terminal/historical runs show no button.
const TRAINING_CANCELLABLE: ReadonlySet<string> = new Set(["created", "running"]);

const DEFAULT_CONFIG = `{
  "model_spec": {
    "backbone": {"name": "resnet50", "pretrained": true},
    "neck": {"name": "fpn"},
    "heads": [{"name": "detection_head", "task": "detection", "num_classes": 1}],
    "loss": {"name": "focal_loss"}
  },
  "data": {
    "images_dir": "",
    "labels_dir": "",
    "task": "detection"
  },
  "training": {
    "batch_size": 4,
    "num_workers": 0,
    "mixed_precision": true,
    "stages": [
      {"freeze_to": -1, "lr": 1e-3, "epochs": 5, "batch_size": 8},
      {"freeze_to": 2, "lr": 1e-4, "epochs": 10, "batch_size": 4}
    ]
  },
  "augmentation": {},
  "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4}
}
`;

interface ValidateResult {
  valid: boolean;
  issues: string[];
}

export function TrainingTab() {
  const projectRoot = useStore((s) => s.gui.dataset.project_root);
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  const annDetectDir = useStore((s) => s.gui.dataset.annotations_detect_dir);

  const [configText, setConfigText] = useState(() => {
    const saved = localStorage.getItem("tcip.training.config");
    if (saved) return saved;
    if (!datasetRoot) return DEFAULT_CONFIG;
    // Pre-fill images_dir / labels_dir from current dataset selection
    try {
      const cfg = JSON.parse(DEFAULT_CONFIG);
      const date = useStore.getState().gui.dataset.date;
      cfg.data.images_dir = date ? `${datasetRoot}/images/${date}` : "";
      cfg.data.labels_dir = annDetectDir ?? "";
      return JSON.stringify(cfg, null, 2);
    } catch {
      return DEFAULT_CONFIG;
    }
  });

  const [validate, setValidate] = useState<ValidateResult | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchMsg, setLaunchMsg] = useState<string | null>(null);
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
    // Clear the previous run's curve; the stream replays THIS run from the start, so a
    // seed GET would just double-load the same rows. The WS is the single source now.
    setMetrics([]);
    streamRef.current?.();
    streamRef.current = openTrainingStream(projectRoot, selectedRun, (msg) => {
      if (msg.type === "metric" && msg.row) {
        setMetrics((prev) => mergeMetric(prev, msg.row as MetricRow));
      } else if (msg.type === "status") {
        // Terminal frame — surface completion/failure/cancellation + refresh the list.
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

  async function onValidate() {
    try {
      const cfg = JSON.parse(configText);
      localStorage.setItem("tcip.training.config", configText);
      const res = (await trainingApi.validate(cfg)) as ValidateResult;
      setValidate(res);
    } catch (e) {
      setValidate({ valid: false, issues: [String(e)] });
    }
  }

  async function onLaunch() {
    if (!projectRoot) return;
    setLaunching(true);
    setLaunchMsg(null);
    try {
      const cfg = JSON.parse(configText);
      localStorage.setItem("tcip.training.config", configText);
      const outputDir = `${projectRoot}/.tcip/experiments`;
      const res = await trainingApi.launch(cfg, outputDir);
      if (res.run_id) {
        setLaunchMsg(`Launched ${res.run_id}`);
        setSelectedRun(res.run_id);
        void refreshRuns();
      } else {
        setLaunchMsg(`Error: ${JSON.stringify(res)}`);
      }
    } catch (e) {
      setLaunchMsg(String(e));
    } finally {
      setLaunching(false);
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

  const chartColors = [
    "#4CAF50",
    "#E6976B",
    "#00BFFF",
    "#FFD700",
    "#EF5350",
    "#81C784",
    "#BA68C8",
    "#FFB74D",
  ];

  return (
    <div className="flex-1 grid grid-cols-[400px_1fr] overflow-hidden">
      {/* Left sidebar: config + runs */}
      <div className="border-r border-tcip-border flex flex-col overflow-hidden">
        <div className="p-3 border-b border-tcip-border">
          <div className="flex items-center gap-2 mb-2">
            <span className="font-semibold text-[13px]">Training config</span>
            <span className="flex-1" />
            <button className="tcip-btn text-[11px]" onClick={onValidate}>
              Validate
            </button>
            <button
              className="tcip-btn-primary text-[11px]"
              onClick={onLaunch}
              disabled={launching || !projectRoot}
              title={projectRoot ? "Launch training run" : "Select a dataset first"}
            >
              {launching ? "Launching…" : "▶ Launch"}
            </button>
          </div>
          <textarea
            className="w-full h-[340px] tcip-input font-mono text-[11px] leading-4 resize-none"
            spellCheck={false}
            value={configText}
            onChange={(e) => setConfigText(e.target.value)}
          />
          {validate && (
            <div className={`mt-2 text-[11px] ${validate.valid ? "text-tcip-tp" : "text-tcip-fp"}`}>
              {validate.valid ? "✓ Config is valid" : "✕ " + validate.issues.join("; ")}
            </div>
          )}
          {launchMsg && <div className="mt-1 text-[11px] text-tcip-muted">{launchMsg}</div>}
        </div>

        <div className="flex-1 overflow-auto p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="font-semibold text-[13px]">Runs</span>
            <span className="flex-1" />
            <button className="tcip-btn text-[11px]" onClick={refreshRuns}>
              ↻ Refresh
            </button>
          </div>
          {runs.length === 0 ? (
            <div className="text-[11px] text-tcip-muted">No runs yet.</div>
          ) : (
            <ul className="space-y-1">
              {runs.map((r) => (
                <li
                  key={r.run_id}
                  className={`p-2 rounded border cursor-pointer ${
                    selectedRun === r.run_id
                      ? "border-tcip-accent bg-tcip-accent/10"
                      : "border-tcip-border hover:border-tcip-muted"
                  }`}
                  onClick={() => setSelectedRun(r.run_id)}
                >
                  <div className="font-mono text-[11px]">{r.run_id}</div>
                  <div className="text-[10px] text-tcip-muted flex justify-between">
                    <span>{r.status}</span>
                    {r.best_metric !== undefined && r.best_metric !== null && (
                      <span>best: {Number(r.best_metric).toFixed(3)}</span>
                    )}
                  </div>
                  {TRAINING_CANCELLABLE.has(r.status) && (
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
        <div className="p-3 border-b border-tcip-border flex items-center gap-2">
          <span className="font-semibold text-[13px]">
            {selectedRun ? `Live metrics — ${selectedRun}` : "Select a run to view metrics"}
          </span>
        </div>
        <div className="flex-1 p-4 overflow-hidden">
          {selectedRun && chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid stroke="#3A3A3A" strokeDasharray="3 3" />
                <XAxis
                  dataKey="step"
                  stroke="#8A8A8A"
                  style={{ fontSize: 11 }}
                  label={{
                    value: "epoch/step",
                    position: "insideBottom",
                    offset: -5,
                    fill: "#8A8A8A",
                  }}
                />
                <YAxis stroke="#8A8A8A" style={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: "#242424",
                    border: "1px solid #3A3A3A",
                    fontSize: 11,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: "#E0E0E0" }} />
                {metricKeys.map((key, i) => (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stroke={chartColors[i % chartColors.length]}
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
