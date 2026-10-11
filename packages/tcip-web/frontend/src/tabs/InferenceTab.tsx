import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import {
  bucketRefusalOf,
  inferenceApi,
  openInferenceStream,
  resultsApi,
  type BucketExistsRefusal,
  type InferenceJob,
  type InferenceStatus,
} from "@/api/inference";
import type { RegisteredModel } from "@/api/types.generated";
import { TabHeading } from "@/components/TabHeading";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";

// A job can still be stopped only while it is pending/running.
const CANCELLABLE: ReadonlySet<InferenceStatus> = new Set(["pending", "running"]);

type StatedField = "conf" | "cross_tile_nms";

/** The execution values a run without an assessment states, one input each, every one a
 *  fraction in [0, 1]. */
const STATED_INPUTS: { field: StatedField; label: string; placeholder: string }[] = [
  {
    field: "conf",
    label: "Confidence threshold (a detector run without an assessment)",
    placeholder: "the score a box is kept at",
  },
  {
    field: "cross_tile_nms",
    label: "Cross-tile merge threshold (a tiled run without an assessment)",
    placeholder: "the IoU above which two tiles' detections are one object",
  },
];

/** A launch refused for one date, holding what it was refused for. */
interface RefusedLaunch {
  date: string;
  refusal: BucketExistsRefusal;
}

/** One refused-launch entry: the facts the response carried, watching the job still writing the
 *  bucket, and a dismissal. Every accessible name below is date-qualified. */
function RefusedLaunchEntry({
  entry,
  onWatch,
  onDismiss,
}: {
  entry: RefusedLaunch;
  onWatch: (refusal: BucketExistsRefusal) => void;
  onDismiss: (date: string) => void;
}) {
  const { refusal, date } = entry;
  return (
    <li className="border border-tcip-border rounded p-2 flex flex-col gap-1 text-[11px]">
      <p>
        {date}: job {refusal.job_id} is still writing the bucket{" "}
        <span className="font-mono">{refusal.requested_bucket}</span>. A bucket is published once,
        so nothing was launched; name another bucket to publish this run.
      </p>
      <button
        className="tcip-btn text-[11px] self-start"
        aria-label={`Watch job ${refusal.job_id} for ${date}`}
        onClick={() => onWatch(refusal)}
      >
        {`Watch job ${refusal.job_id}`}
      </button>
      <button
        className="tcip-btn text-[11px] self-start"
        aria-label={`Dismiss refusal for ${date}`}
        onClick={() => onDismiss(date)}
      >
        Dismiss
      </button>
    </li>
  );
}

function statusBadgeClass(status: InferenceStatus): string {
  if (status === "completed") return "bg-tcip-tp/20 text-tcip-tp";
  if (status === "failed" || status === "canceled") return "bg-tcip-fp/20 text-tcip-fp";
  return "bg-tcip-fn/20 text-tcip-fn"; // pending / running
}

export function InferenceTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const projectRoot = useStore(selectProjectRoot);
  const datasetRoot = dataset.dataset_root;

  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelPath, setModelPath] = useState<string>("");
  const [dates, setDates] = useState<string[]>([]);
  const [datesError, setDatesError] = useState<string | null>(null);
  const [selectedDates, setSelectedDates] = useState<string[]>([]);
  // The name the run's buckets publish under, one ``<name>/<date>`` per date; the breeder names it.
  const [bucketName, setBucketName] = useState("");
  // The assessment whose execution record the run takes and publishes under; empty for none.
  const [assessmentId, setAssessmentId] = useState("");
  const [statedInputs, setStatedInputs] = useState<Record<StatedField, string>>({
    conf: "",
    cross_tile_nms: "",
  });
  const [jobs, setJobs] = useState<InferenceJob[]>([]);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<InferenceJob | null>(null);
  // Whether the watched job still appeared in the last poll: false once a job is delisted, so
  // the panel shows its last error alone rather than a status line frozen on a job that is gone.
  const [activeJobListed, setActiveJobListed] = useState(true);
  // A bucket_exists refusal, keyed by the date it was refused for.
  const [refusedLaunches, setRefusedLaunches] = useState<Record<string, RefusedLaunch>>({});
  const [launching, setLaunching] = useState(false);
  const streamRef = useRef<(() => void) | null>(null);
  const launchButtonRef = useRef<HTMLButtonElement>(null);
  const prevRefusedDatesRef = useRef<string[]>([]);
  const activeJobId = activeJob?.job_id ?? null;

  // A refused entry's control can unmount while focused (Dismiss, or its action succeeding),
  // dropping focus to body; restore it to the launch button instead of stranding it there.
  useEffect(() => {
    const dates = Object.keys(refusedLaunches);
    const anyRemoved = prevRefusedDatesRef.current.some((d) => !dates.includes(d));
    prevRefusedDatesRef.current = dates;
    if (anyRemoved && document.activeElement === document.body) {
      launchButtonRef.current?.focus();
    }
  }, [refusedLaunches]);

  // A refused entry names a bucket of the previous choice: neither survives a model or dataset
  // change.
  useEffect(() => {
    setRefusedLaunches({});
  }, [modelPath, datasetRoot]);

  const refreshModels = useCallback(() => {
    if (!projectRoot) return;
    void resultsApi
      .registeredModels()
      .then((r) => {
        setModels(r.models ?? []);
        setModelsError(null);
      })
      .catch((e) => {
        setModels([]);
        setModelsError(
          `Could not load registered models: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [projectRoot]);

  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

  const refreshDates = useCallback(() => {
    if (!datasetRoot) return;
    void api.dataset
      .tree(datasetRoot)
      .then((t) => {
        setDates(t.dates_with_images);
        setSelectedDates((prev) => prev.filter((d) => t.dates_with_images.includes(d)));
        setDatesError(null);
      })
      .catch((e) => {
        setDates([]);
        setDatesError(
          `Could not load this dataset's dates: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [datasetRoot]);

  useEffect(() => {
    refreshDates();
  }, [refreshDates]);

  const refreshJobs = useCallback(
    () =>
      inferenceApi
        .listJobs()
        .then((r) => {
          setJobs(r.jobs);
          setJobsError(null);
          if (activeJobId) {
            const row = r.jobs.find((j) => j.job_id === activeJobId);
            setActiveJobListed(row !== undefined);
            if (row) {
              setActiveJob((prev) =>
                prev && prev.job_id === activeJobId
                  ? { ...row, error: row.error ?? prev.error }
                  : prev,
              );
            }
          }
        })
        .catch((e) => {
          setJobsError(
            `Could not load inference jobs: ${e instanceof Error ? e.message : String(e)}`,
          );
        }),
    [activeJobId],
  );

  useEffect(() => {
    void refreshJobs();
    const t = setInterval(refreshJobs, 3000);
    return () => clearInterval(t);
  }, [refreshJobs]);

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
            ? {
                ...prev,
                done: asNum(msg.done, prev.done),
                total: asNum(msg.total, prev.total),
                status: (msg.status as InferenceJob["status"]) ?? prev.status,
                // A frame's presence of the key decides, including error: null.
                error: "error" in msg ? (msg.error as string | null) : prev.error,
              }
            : prev,
        );
      }
    });
    return () => streamRef.current?.();
  }, [activeJobId]);

  function toggleDate(date: string) {
    setSelectedDates((prev) =>
      prev.includes(date) ? prev.filter((d) => d !== date) : [...prev, date],
    );
  }

  function dropRefusal(date: string) {
    setRefusedLaunches((prev) => {
      const next = { ...prev };
      delete next[date];
      return next;
    });
  }

  // One date's launch body: its bucket is the named bucket's ``<name>/<date>``.
  async function launchOne(checkpointPath: string, date: string) {
    if (!datasetRoot) return;
    try {
      const res = await inferenceApi.launch({
        checkpoint_path: checkpointPath,
        dataset_root: datasetRoot,
        date,
        bucket: `${bucketName.replace(/\/+$/, "")}/${date}`,
        stated: Object.fromEntries(
          Object.entries(statedInputs)
            .filter(([, value]) => value.trim() !== "")
            .map(([field, value]) => [field, Number(value)]),
        ),
        assessment_id: assessmentId.trim() || null,
        user: useStore.getState().user,
      });
      if (res.job_id) {
        const stub: InferenceJob = {
          job_id: res.job_id,
          status: "pending",
          done: 0,
          total: 0,
          images_dir: res.images_dir,
          dataset_root: res.dataset_root,
          bucket: res.bucket,
          error: null,
        };
        setJobs((prev) => [stub, ...prev]);
        setActiveJob(stub);
        setActiveJobListed(true);
        dropRefusal(date);
      }
    } catch (e) {
      const refusal = bucketRefusalOf(e);
      if (refusal) {
        setRefusedLaunches((prev) => ({ ...prev, [date]: { date, refusal } }));
        return;
      }
      dropRefusal(date);
      useStore
        .getState()
        .pushToast(
          `Inference launch failed for ${date}: ${e instanceof Error ? e.message : String(e)}`,
        );
    }
  }

  async function onLaunch() {
    const model = models.find((m) => m.checkpoint_path === modelPath);
    if (!model || !datasetRoot || !bucketName || selectedDates.length === 0) return;
    setRefusedLaunches((prev) => {
      const next = { ...prev };
      for (const date of selectedDates) delete next[date];
      return next;
    });
    setLaunching(true);
    try {
      // One job per date: each date is its own prediction bucket, one job row per date.
      for (const date of selectedDates) {
        await launchOne(model.checkpoint_path, date);
      }
    } finally {
      setLaunching(false);
    }
  }

  function onWatchRefusedJob(refusal: BucketExistsRefusal) {
    const row = jobs.find((j) => j.job_id === refusal.job_id);
    setActiveJob(
      row ?? {
        job_id: refusal.job_id,
        status: "running",
        done: 0,
        total: 0,
        // Nothing the refusal carries names the images dir; left empty rather than fabricated.
        images_dir: "",
        dataset_root: datasetRoot ?? "",
        bucket: refusal.requested_bucket ?? "",
        error: null,
      },
    );
    setActiveJobListed(row !== undefined);
  }

  async function onCancel(jobId: string) {
    // Optimistically flip the row so the button disappears immediately; the poll +
    // the worker's next-image-boundary stop will confirm the terminal state.
    setJobs((prev) => prev.map((j) => (j.job_id === jobId ? { ...j, status: "canceled" } : j)));
    try {
      await inferenceApi.cancel(jobId, useStore.getState().user);
    } catch (e) {
      useStore.getState().pushToast(`Cancel failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <div className="flex-1 grid grid-cols-[440px_1fr] overflow-hidden">
      <TabHeading tab="inference" />
      <div className="border-r border-tcip-border p-4 overflow-auto">
        <div className="tcip-heading mb-3">Inference config</div>

        <label className="tcip-label mb-1" htmlFor="inference-model-select">
          Model checkpoint
        </label>
        {modelsError && (
          <div className="text-[11px] text-tcip-fp mb-1">
            {modelsError}{" "}
            <button className="tcip-btn text-[11px] ml-1" onClick={refreshModels}>
              Retry
            </button>
          </div>
        )}
        {models.length > 0 ? (
          <select
            id="inference-model-select"
            className="tcip-select w-full mb-3"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
          >
            <option value="">Select a registered model…</option>
            {models.map((m) => (
              <option key={m.checkpoint_path} value={m.checkpoint_path}>
                {m.name}
                {m.experiment_id ? ` [${m.experiment_id}]` : ""}{" "}
                {m.tags?.length ? `(${m.tags.join(", ")})` : ""} -{" "}
                {m.checkpoint_path.split(/[/\\]/).slice(-3).join("/")}
              </option>
            ))}
          </select>
        ) : (
          !modelsError &&
          projectRoot && (
            <p className="text-[11px] text-tcip-muted mb-3">
              No model is registered for this project yet. Train one, or ask the agent in the
              terminal to register a checkpoint, and it will appear here.
            </p>
          )
        )}

        <div className="tcip-label mb-1">Capture dates</div>
        {datesError && (
          <div className="text-[11px] text-tcip-fp mb-1">
            {datesError}{" "}
            <button className="tcip-btn text-[11px] ml-1" onClick={refreshDates}>
              Retry
            </button>
          </div>
        )}
        {!datasetRoot ? (
          <p className="text-[11px] text-tcip-muted mb-3">
            No project is open. Open one from the top bar to choose which dates to run on.
          </p>
        ) : dates.length === 0 ? (
          !datesError && (
            <p className="text-[11px] text-tcip-muted mb-3">
              This dataset has no capture dates yet. Ingest images (or ask the agent in the terminal
              to) before running inference.
            </p>
          )
        ) : (
          <div className="mb-3 max-h-48 overflow-auto border border-tcip-border rounded p-2 flex flex-col gap-1">
            {dates.map((d) => (
              <label key={d} className="flex items-center gap-2 text-[12px]">
                <input
                  type="checkbox"
                  checked={selectedDates.includes(d)}
                  onChange={() => toggleDate(d)}
                />
                {d}
              </label>
            ))}
          </div>
        )}

        <label className="tcip-label mb-1" htmlFor="inference-bucket-root">
          Bucket name
        </label>
        <input
          id="inference-bucket-root"
          className="tcip-input w-full mb-3 font-mono"
          value={bucketName}
          onChange={(e) => setBucketName(e.target.value)}
          placeholder="a name no bucket of this dataset is published under"
        />

        <label className="tcip-label mb-1" htmlFor="inference-assessment">
          Assessment (optional)
        </label>
        <input
          id="inference-assessment"
          className="tcip-input w-full mb-3 font-mono"
          value={assessmentId}
          onChange={(e) => setAssessmentId(e.target.value)}
          placeholder="the assessment id this checkpoint earned"
        />

        {STATED_INPUTS.map(({ field, label, placeholder }) => (
          <div key={field}>
            <label className="tcip-label mb-1" htmlFor={`inference-${field}`}>
              {label}
            </label>
            <input
              id={`inference-${field}`}
              className="tcip-input w-full mb-3 font-mono"
              type="number"
              min={0}
              max={1}
              step="any"
              value={statedInputs[field]}
              onChange={(e) => setStatedInputs((prev) => ({ ...prev, [field]: e.target.value }))}
              placeholder={placeholder}
            />
          </div>
        ))}

        <p className="text-[11px] text-tcip-muted mb-3">
          Every execution value (conf, object density, tiling, cross-tile merge) is the named
          assessment&apos;s own. Without one, a detector&apos;s run states its conf above, and a
          tiled run its merge threshold, since no reference stands behind either; the density is the
          one this checkpoint recorded, each image keeping at most its density times its pixels,
          rounded up, and the tiling follows this checkpoint as the agent-facing door resolves it.
          Only predictions published under an assessment can deliver validated numbers. Each date
          publishes once, as its own bucket under the name above.
        </p>

        <button
          ref={launchButtonRef}
          className="tcip-btn-primary w-full"
          onClick={() => void onLaunch()}
          disabled={launching || !modelPath || !bucketName || selectedDates.length === 0}
        >
          ▶&nbsp;&nbsp;Launch inference
        </button>

        <div className="mt-3">
          {Object.keys(refusedLaunches).length > 0 && (
            <div className="tcip-heading mb-1">Refused launches</div>
          )}
          <ul aria-live="polite" className="flex flex-col gap-1">
            {Object.values(refusedLaunches).map((entry) => (
              <RefusedLaunchEntry
                key={entry.date}
                entry={entry}
                onWatch={onWatchRefusedJob}
                onDismiss={dropRefusal}
              />
            ))}
          </ul>
        </div>
      </div>

      <div className="p-4 overflow-auto">
        <div className="tcip-heading mb-3">Jobs</div>
        {jobsError && (
          <div className="text-[11px] text-tcip-fp mb-2">
            {jobsError}{" "}
            <button className="tcip-btn text-[11px] ml-1" onClick={() => void refreshJobs()}>
              Retry
            </button>
          </div>
        )}
        {jobs.length === 0 ? (
          !jobsError && <div className="text-[11px] text-tcip-muted">No jobs yet.</div>
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
                      <button
                        className="tcip-btn text-[11px]"
                        onClick={() => {
                          setActiveJob(j);
                          setActiveJobListed(true);
                        }}
                      >
                        Watch
                      </button>
                      {CANCELLABLE.has(j.status) && (
                        <button
                          className="tcip-btn text-[11px]"
                          onClick={() => void onCancel(j.job_id)}
                        >
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
              Bucket: <span className="font-mono">{activeJob.bucket}</span>
            </div>
            {activeJobListed && (
              <div className="text-[11px] mt-1 tabular-nums">
                Status: {activeJob.status} · {activeJob.done} / {activeJob.total}
              </div>
            )}
            {activeJob.error && (
              <div className="text-[11px] text-tcip-fp mt-1">Error: {activeJob.error}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
