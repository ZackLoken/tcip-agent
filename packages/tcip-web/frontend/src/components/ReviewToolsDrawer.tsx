/**
 * Review-management drawer: build a curated training set from recorded verdicts
 * (materialize) and rank unreviewed images for the next batch (active-learning queue).
 * Kept out of ReviewTab (which owns the canvas walk) so neither grows unwieldy.
 */

import { useEffect, useRef, useState } from "react";

import { api, type MaterializeResult, type ReviewQueueResult } from "@/api/client";
import { resultsApi, type RegisteredModel } from "@/api/inference";
import { useStore } from "@/store";

function basename(path: string): string {
  return path.split(/[/\\]/).pop() ?? path;
}

export function ReviewToolsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dataset = useStore((s) => s.gui.dataset);
  const patchGui = useStore((s) => s.patchGui);
  const projectRoot = dataset.project_root;
  const datasetRoot = dataset.dataset_root;
  const date = dataset.date;

  // ── Materialize ──
  const [outputDir, setOutputDir] = useState("");
  const [onlyCompleted, setOnlyCompleted] = useState(false);
  const [hardNegatives, setHardNegatives] = useState(true);
  const [materializing, setMaterializing] = useState(false);
  const [materializeResult, setMaterializeResult] = useState<MaterializeResult | null>(null);

  // ── Review queue ──
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [checkpoint, setCheckpoint] = useState("");
  const [imagesDir, setImagesDir] = useState("");
  const [method, setMethod] = useState("combined");
  const [budget, setBudget] = useState(50);
  const [queueJobId, setQueueJobId] = useState<string | null>(null);
  const [queueStatus, setQueueStatus] = useState<string>("");
  const [queueResult, setQueueResult] = useState<ReviewQueueResult | null>(null);
  const pollRef = useRef<number | null>(null);

  // Prefill dir defaults when the drawer opens for a dataset/date.
  useEffect(() => {
    if (!open) return;
    if (datasetRoot && date) {
      setImagesDir((v) => v || `${datasetRoot}/images/${date}`);
      setOutputDir((v) => v || `${datasetRoot}/curated/${date}`);
    }
    if (projectRoot) {
      void resultsApi
        .registeredModels(projectRoot)
        .then((r) => setModels(r.models ?? []))
        .catch(() => setModels([]));
    }
  }, [open, datasetRoot, date, projectRoot]);

  // Poll the queue job until it terminates.
  useEffect(() => {
    if (!queueJobId) return;
    const tick = () =>
      api.review
        .getQueue(queueJobId)
        .then((job) => {
          setQueueStatus(job.status);
          if (job.status === "completed") {
            setQueueResult((job.result as ReviewQueueResult) ?? null);
            if (pollRef.current) window.clearInterval(pollRef.current);
          } else if (job.status === "failed") {
            useStore.getState().pushToast(`Review queue failed: ${job.error ?? "unknown error"}`);
            if (pollRef.current) window.clearInterval(pollRef.current);
          }
        })
        .catch(() => {});
    void tick();
    pollRef.current = window.setInterval(tick, 2000);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [queueJobId]);

  async function runMaterialize() {
    if (!projectRoot || !datasetRoot || !date || !outputDir) return;
    setMaterializing(true);
    setMaterializeResult(null);
    try {
      const res = await api.review.materialize({
        project_root: projectRoot,
        source_images_dir: `${datasetRoot}/images/${date}`,
        output_dir: outputDir,
        include_hard_negatives: hardNegatives,
        only_completed: onlyCompleted,
      });
      setMaterializeResult(res);
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Materialize failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setMaterializing(false);
    }
  }

  async function runQueue() {
    if (!projectRoot || !checkpoint || !imagesDir) return;
    setQueueResult(null);
    setQueueStatus("launching");
    try {
      const res = await api.review.launchQueue({
        project_root: projectRoot,
        checkpoint_path: checkpoint,
        images_dir: imagesDir,
        method,
        budget,
      });
      setQueueJobId(res.job_id);
    } catch (e) {
      setQueueStatus("");
      useStore
        .getState()
        .pushToast(`Could not launch review queue: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function jumpTo(imagePath: string) {
    const name = basename(imagePath);
    const idx = dataset.image_list.indexOf(name);
    if (idx < 0) {
      useStore.getState().pushToast(`${name} isn't in the current image list.`);
      return;
    }
    patchGui({ dataset: { ...dataset, current_image_index: idx } });
    onClose();
  }

  if (!open) return null;

  return (
    <div className="absolute inset-0 z-20 flex justify-end">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative w-[380px] h-full bg-tcip-panel border-l border-tcip-border overflow-auto p-4 flex flex-col gap-5 text-[12px]">
        <div className="flex items-center">
          <span className="font-semibold text-[13px]">Review tools</span>
          <span className="flex-1" />
          <button className="tcip-btn text-[11px]" onClick={onClose}>
            Close
          </button>
        </div>

        {/* ── Materialize ── */}
        <section className="flex flex-col gap-2">
          <div className="font-semibold">Build training set from verdicts</div>
          <p className="text-[11px] text-tcip-muted">
            Accepted/edited GT → positives; rejected-only images → hard negatives. Chains into split
            + train.
          </p>
          <label className="text-[11px] text-tcip-muted">Output directory</label>
          <input
            className="tcip-input"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            placeholder="…/curated/2-11-26"
          />
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={onlyCompleted}
              onChange={(e) => setOnlyCompleted(e.target.checked)}
            />
            Only fully-reviewed images
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={hardNegatives}
              onChange={(e) => setHardNegatives(e.target.checked)}
            />
            Include hard negatives
          </label>
          <button
            className="tcip-btn-primary"
            onClick={runMaterialize}
            disabled={materializing || !projectRoot || !date || !outputDir}
          >
            {materializing ? "Building…" : "Build training set"}
          </button>
          {materializeResult && (
            <div className="text-[11px] text-tcip-muted">
              ✓ {materializeResult.positive} positive · {materializeResult.hard_negative} hard-neg ·{" "}
              {materializeResult.total_boxes} boxes → {materializeResult.output_dir}
            </div>
          )}
        </section>

        <div className="border-t border-tcip-border" />

        {/* ── Review queue ── */}
        <section className="flex flex-col gap-2">
          <div className="font-semibold">Prioritize unreviewed images</div>
          <p className="text-[11px] text-tcip-muted">
            Ranks unreviewed images by model informativeness (needs a trained checkpoint).
          </p>
          <label className="text-[11px] text-tcip-muted">Model checkpoint</label>
          {models.length > 0 ? (
            <select
              className="tcip-select"
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
            >
              <option value="">Select a registered model…</option>
              {models.map((m) => (
                <option key={m.checkpoint_path} value={m.checkpoint_path}>
                  {m.name} — {basename(m.checkpoint_path)}
                </option>
              ))}
            </select>
          ) : (
            <input
              className="tcip-input"
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
              placeholder="Path to .pt checkpoint"
            />
          )}
          <label className="text-[11px] text-tcip-muted">Images directory</label>
          <input
            className="tcip-input"
            value={imagesDir}
            onChange={(e) => setImagesDir(e.target.value)}
          />
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-tcip-muted">Method</span>
              <select
                className="tcip-select"
                value={method}
                onChange={(e) => setMethod(e.target.value)}
              >
                <option value="combined">Combined</option>
                <option value="uncertainty">Uncertainty</option>
                <option value="diversity">Diversity</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-tcip-muted">Budget</span>
              <input
                className="tcip-input"
                type="number"
                min={1}
                max={500}
                value={budget}
                onChange={(e) => setBudget(parseInt(e.target.value, 10) || 50)}
              />
            </label>
          </div>
          <button
            className="tcip-btn-primary"
            onClick={runQueue}
            disabled={
              !projectRoot ||
              !checkpoint ||
              !imagesDir ||
              queueStatus === "running" ||
              queueStatus === "launching"
            }
          >
            {queueStatus === "running" || queueStatus === "launching"
              ? "Ranking…"
              : "Rank unreviewed"}
          </button>
          {queueResult && (
            <div className="flex flex-col gap-1">
              <div className="text-[11px] text-tcip-muted">
                {queueResult.selected_count} of {queueResult.total_candidates} candidates (
                {queueResult.reviewed_skipped} already reviewed) — click to jump:
              </div>
              <ul className="flex flex-col gap-0.5 max-h-72 overflow-auto">
                {queueResult.queue.map((entry, i) => (
                  <li key={entry.image}>
                    <button
                      className="w-full text-left px-2 py-1 rounded hover:bg-tcip-border font-mono text-[11px] flex justify-between"
                      onClick={() => jumpTo(entry.image)}
                    >
                      <span className="truncate">
                        {i + 1}. {basename(entry.image)}
                      </span>
                      <span className="text-tcip-muted ml-2">{entry.score.toFixed(3)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
