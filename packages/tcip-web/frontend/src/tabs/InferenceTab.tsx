import { useEffect, useRef, useState } from "react";

import {
  inferenceApi,
  openInferenceStream,
  resultsApi,
  type InferenceJob,
  type InferenceStatus,
  type RegisteredModel,
} from "@/api/inference";
import { useStore } from "@/store";

// A job can still be stopped only while it is pending/running.
const CANCELLABLE: ReadonlySet<InferenceStatus> = new Set(["pending", "running"]);

function statusBadgeClass(status: InferenceStatus): string {
  if (status === "completed") return "bg-tcip-tp/20 text-tcip-tp";
  if (status === "failed" || status === "cancelled") return "bg-tcip-fp/20 text-tcip-fp";
  if (status === "interrupted") return "bg-tcip-border text-tcip-muted";
  return "bg-tcip-fn/20 text-tcip-fn"; // pending / running
}

export function InferenceTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const projectRoot = dataset.project_root;
  const datasetRoot = dataset.dataset_root;

  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [modelPath, setModelPath] = useState<string>("");
  const [imagesDir, setImagesDir] = useState<string>("");
  const [outputDir, setOutputDir] = useState<string>("");
  const [sahi, setSahi] = useState<boolean>(true);
  const [postprocess, setPostprocess] = useState<"nms" | "nmm">("nms");
  const [conf, setConf] = useState<number>(0.25);
  const [iou, setIou] = useState<number>(0.7);
  const [sliceH, setSliceH] = useState<number>(640);
  const [sliceW, setSliceW] = useState<number>(640);
  const [overlap, setOverlap] = useState<number>(0.2);
  const [jobs, setJobs] = useState<InferenceJob[]>([]);
  const [activeJob, setActiveJob] = useState<InferenceJob | null>(null);
  const streamRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!projectRoot) return;
    void resultsApi
      .registeredModels(projectRoot)
      .then((r) => setModels(r.models ?? []))
      .catch(() => setModels([]));
  }, [projectRoot]);

  useEffect(() => {
    const refresh = () =>
      inferenceApi
        .listJobs()
        .then((r) => setJobs(r.jobs))
        .catch(() => {});
    void refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  const activeJobId = activeJob?.job_id;
  useEffect(() => {
    if (!activeJobId) return;
    streamRef.current?.();
    streamRef.current = openInferenceStream(activeJobId, (msg) => {
      if (msg.type === "progress" || msg.type === "final") {
        // The "final" frame omits done/total; Number(undefined) is NaN and `?? `
        // does not catch NaN, so guard on Number.isFinite to keep the last value.
        const asNum = (v: unknown, fallback: number) => {
          const n = Number(v);
          return Number.isFinite(n) ? n : fallback;
        };
        setActiveJob((prev) =>
          prev && prev.job_id === activeJobId
            ? ({
                ...prev,
                done: asNum(msg.done, prev.done),
                total: asNum(msg.total, prev.total),
                status: (msg.status as InferenceJob["status"]) ?? prev.status,
              } as InferenceJob)
            : prev,
        );
      }
    });
    return () => streamRef.current?.();
  }, [activeJobId]);

  function prefillFromDataset() {
    if (!datasetRoot || !dataset.date) return;
    setImagesDir(`${datasetRoot}/images/${dataset.date}`);
    setOutputDir(`${datasetRoot}/predictions/live/${dataset.date}/detect`);
  }

  async function onLaunch() {
    if (!modelPath || !imagesDir || !outputDir) return;
    try {
      const res = await inferenceApi.launch({
        checkpoint_path: modelPath,
        images_dir: imagesDir,
        output_dir: outputDir,
        sahi,
        conf,
        iou,
        slice_h: sliceH,
        slice_w: sliceW,
        overlap,
        postprocess,
      });
      if (res.job_id) {
        const stub: InferenceJob = {
          job_id: res.job_id,
          status: "pending",
          done: 0,
          total: 0,
          images_dir: imagesDir,
          output_dir: outputDir,
          error: null,
        };
        setJobs((prev) => [stub, ...prev]);
        setActiveJob(stub);
      }
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Inference launch failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function onCancel(jobId: string) {
    // Optimistically flip the row so the button disappears immediately; the poll +
    // the worker's next-image-boundary stop will confirm the terminal state.
    setJobs((prev) => prev.map((j) => (j.job_id === jobId ? { ...j, status: "cancelled" } : j)));
    try {
      await inferenceApi.cancel(jobId);
    } catch (e) {
      useStore.getState().pushToast(`Cancel failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <div className="flex-1 grid grid-cols-[440px_1fr] overflow-hidden">
      <div className="border-r border-tcip-border p-4 overflow-auto">
        <div className="tcip-heading mb-3">Inference config</div>

        <label className="tcip-label mb-1">Model checkpoint</label>
        {models.length > 0 ? (
          <select
            className="tcip-select w-full mb-2"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
          >
            <option value="">Select a registered model…</option>
            {models.map((m) => (
              <option key={m.checkpoint_path} value={m.checkpoint_path}>
                {m.name} {m.tags?.length ? `(${m.tags.join(", ")})` : ""} —{" "}
                {m.checkpoint_path.split(/[/\\]/).slice(-3).join("/")}
              </option>
            ))}
          </select>
        ) : (
          <input
            className="tcip-input w-full mb-2"
            placeholder="Path to .pt checkpoint"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
          />
        )}

        <label className="tcip-label mb-1">Images directory</label>
        <input
          className="tcip-input w-full mb-1"
          value={imagesDir}
          onChange={(e) => setImagesDir(e.target.value)}
          placeholder="…/Valley_Farm/images/2-11-26"
        />
        <button className="tcip-btn text-[11px] mb-3" onClick={prefillFromDataset}>
          Prefill from current dataset
        </button>

        <label className="tcip-label mb-1">Output directory (YOLO txt)</label>
        <input
          className="tcip-input w-full mb-3"
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
          placeholder="…/Valley_Farm/predictions/live/2-11-26/detect"
        />

        <div className="flex items-center gap-3 mb-3">
          <label className="flex items-center gap-2 text-[12px]">
            <input type="checkbox" checked={sahi} onChange={(e) => setSahi(e.target.checked)} />
            SAHI tiled inference
          </label>
          <label className="flex items-center gap-1 text-[12px] text-tcip-muted">
            Tile merge
            <select
              className="tcip-select text-[12px]"
              value={postprocess}
              disabled={!sahi}
              title="How boxes from adjacent tiles are combined. NMM unions a box split across a seam; NMS suppresses overlaps."
              onChange={(e) => setPostprocess(e.target.value === "nmm" ? "nmm" : "nms")}
            >
              <option value="nms">NMS (suppress)</option>
              <option value="nmm">NMM (merge seams)</option>
            </select>
          </label>
        </div>

        <div className="grid grid-cols-2 gap-2 mb-3 text-[11px] text-tcip-muted">
          <label className="flex flex-col gap-1">
            Conf
            <input
              className="tcip-input w-full"
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={conf}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setConf(Number.isFinite(v) ? v : 0.25);
              }}
            />
          </label>
          <label className="flex flex-col gap-1">
            IoU
            <input
              className="tcip-input w-full"
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={iou}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setIou(Number.isFinite(v) ? v : 0.7);
              }}
            />
          </label>
          <label className="flex flex-col gap-1">
            Slice H
            <input
              className="tcip-input w-full"
              type="number"
              value={sliceH}
              onChange={(e) => setSliceH(parseInt(e.target.value, 10) || 640)}
            />
          </label>
          <label className="flex flex-col gap-1">
            Slice W
            <input
              className="tcip-input w-full"
              type="number"
              value={sliceW}
              onChange={(e) => setSliceW(parseInt(e.target.value, 10) || 640)}
            />
          </label>
          <label className="col-span-2 flex flex-col gap-1">
            Overlap
            <input
              className="tcip-input w-full"
              type="number"
              step="0.05"
              min="0"
              max="0.9"
              value={overlap}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setOverlap(Number.isFinite(v) ? v : 0.2);
              }}
            />
          </label>
        </div>

        <button
          className="tcip-btn-primary w-full"
          onClick={onLaunch}
          disabled={!modelPath || !imagesDir || !outputDir}
        >
          ▶&nbsp;&nbsp;Launch inference
        </button>
      </div>

      <div className="p-4 overflow-auto">
        <div className="tcip-heading mb-3">Jobs</div>
        {jobs.length === 0 ? (
          <div className="text-[11px] text-tcip-muted">No jobs yet.</div>
        ) : (
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-tcip-border">
                <th className="tcip-th">Job</th>
                <th className="tcip-th">Status</th>
                <th className="tcip-th">Progress</th>
                <th className="tcip-th">Images dir</th>
                <th className="tcip-th"></th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.job_id} className="border-t border-tcip-border">
                  <td className="py-1.5 pr-3 font-mono">{j.job_id}</td>
                  <td className="pr-3">
                    <span className={`tcip-badge ${statusBadgeClass(j.status)}`}>{j.status}</span>
                  </td>
                  <td className="pr-3 tabular-nums">
                    {j.total > 0 ? `${j.done} / ${j.total}` : j.done}
                    {j.total > 0 && (
                      <div className="h-1 mt-1 bg-tcip-border rounded overflow-hidden">
                        <div
                          className="h-full bg-tcip-accent"
                          style={{ width: `${(j.done / j.total) * 100}%` }}
                        />
                      </div>
                    )}
                  </td>
                  <td className="pr-3 truncate max-w-xs font-mono text-tcip-muted">
                    {j.images_dir}
                  </td>
                  <td>
                    <div className="flex gap-1">
                      <button className="tcip-btn text-[11px]" onClick={() => setActiveJob(j)}>
                        Watch
                      </button>
                      {CANCELLABLE.has(j.status) && (
                        <button className="tcip-btn text-[11px]" onClick={() => onCancel(j.job_id)}>
                          Cancel
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {activeJob && (
          <div className="mt-4 tcip-panel p-4">
            <div className="mb-1 flex items-center gap-2">
              <span className="tcip-heading">Active</span>
              <span className="font-mono text-[12px]">{activeJob.job_id}</span>
            </div>
            <div className="text-[11px] text-tcip-muted">
              Output: <span className="font-mono">{activeJob.output_dir}</span>
            </div>
            <div className="text-[11px] mt-1 tabular-nums">
              Status: {activeJob.status} · {activeJob.done} / {activeJob.total}
            </div>
            {activeJob.error && (
              <div className="text-[11px] text-tcip-fp mt-1">Error: {activeJob.error}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
