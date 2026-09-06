import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import {
  bucketRefusalOf,
  inferenceApi,
  openInferenceStream,
  resultsApi,
  type BucketRefusal,
  type InferenceJob,
  type InferenceStatus,
  type RegisteredModel,
} from "@/api/inference";
import { TabHeading } from "@/components/TabHeading";
import { useStore } from "@/store";

// A job can still be stopped only while it is pending/running.
const CANCELLABLE: ReadonlySet<InferenceStatus> = new Set(["pending", "running"]);

/** A launch refused for one date, holding what it was refused for so its own remediation
 *  action re-posts the same checkpoint and date rather than the select's current choice. */
interface RefusedLaunch {
  date: string;
  modelName: string;
  checkpointPath: string;
  refusal: BucketRefusal;
}

/** One refused-launch entry: the facts the response carried, its one remediation action (a
 *  fresh-bucket relaunch or watching the in-flight job), and a dismissal. Every accessible name
 *  below is date-qualified so two entries' controls are distinguishable. */
function RefusedLaunchEntry({
  entry,
  onRunSuggestion,
  onWatch,
  onDismiss,
}: {
  entry: RefusedLaunch;
  onRunSuggestion: (entry: RefusedLaunch, suggestedModelName: string) => void;
  onWatch: (jobId: string) => void;
  onDismiss: (date: string) => void;
}) {
  const { refusal, date } = entry;
  return (
    <li className="border border-tcip-border rounded p-2 flex flex-col gap-1 text-[11px]">
      {refusal.kind === "bucket_holds_documents" ? (
        <>
          <p>
            {date}: {refusal.requested_output_dir} already holds {refusal.document_stem_count}{" "}
            prediction document(s).
          </p>
          {refusal.suggested_model_name ? (
            <button
              className="tcip-btn text-[11px] self-start"
              onClick={() => onRunSuggestion(entry, refusal.suggested_model_name as string)}
            >
              {`Run into ${refusal.suggested_model_name} instead for ${date}`}
            </button>
          ) : (
            <p>
              Every {refusal.requested_model_name}@r&lt;n&gt; variant up to the ceiling already
              holds a verdict or a document: ask the agent to publish this model under another
              bucket name through run_inference&apos;s own output_dir.
            </p>
          )}
        </>
      ) : (
        <>
          <p>
            {date}: job {refusal.job_id} is already writing to {refusal.requested_output_dir}.
          </p>
          <button
            className="tcip-btn text-[11px] self-start"
            onClick={() => onWatch(refusal.job_id)}
          >
            {`Watch job ${refusal.job_id} for ${date}`}
          </button>
        </>
      )}
      <button className="tcip-btn text-[11px] self-start" onClick={() => onDismiss(date)}>
        {`Dismiss refusal for ${date}`}
      </button>
    </li>
  );
}

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
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelPath, setModelPath] = useState<string>("");
  const [dates, setDates] = useState<string[]>([]);
  const [datesError, setDatesError] = useState<string | null>(null);
  const [selectedDates, setSelectedDates] = useState<string[]>([]);
  const [jobs, setJobs] = useState<InferenceJob[]>([]);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<InferenceJob | null>(null);
  // Whether the watched job still appeared in the last poll: false once a job is delisted, so
  // the panel shows its last error alone rather than a status line frozen on a job that is gone.
  const [activeJobListed, setActiveJobListed] = useState(true);
  // A bucket_holds_documents or bucket_in_flight refusal, keyed by the date it was refused for.
  const [refusedLaunches, setRefusedLaunches] = useState<Record<string, RefusedLaunch>>({});
  const [launching, setLaunching] = useState(false);
  const streamRef = useRef<(() => void) | null>(null);
  const activeJobId = activeJob?.job_id ?? null;

  // A refused entry names a bucket of the previous choice: neither survives a model or dataset
  // change.
  useEffect(() => {
    setRefusedLaunches({});
  }, [modelPath, datasetRoot]);

  const refreshModels = useCallback(() => {
    if (!projectRoot) return;
    void resultsApi
      .registeredModels(projectRoot)
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
            ? ({
                ...prev,
                done: asNum(msg.done, prev.done),
                total: asNum(msg.total, prev.total),
                status: (msg.status as InferenceJob["status"]) ?? prev.status,
                warning: (msg.warning as string | null | undefined) ?? prev.warning,
                // A progress frame carries no error key at all and must not clear one already
                // shown; a final frame's presence of the key decides, including error: null.
                error: "error" in msg ? (msg.error as string | null) : prev.error,
              } as InferenceJob)
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

  // One date's launch body: onLaunch supplies the select's own model, a refused entry's own
  // action its own checkpoint and the suggested model name, so the action never reads the select.
  async function launchOne(checkpointPath: string, modelName: string, date: string) {
    if (!datasetRoot) return;
    try {
      const res = await inferenceApi.launch({
        checkpoint_path: checkpointPath,
        dataset_root: datasetRoot,
        model_name: modelName,
        date,
      });
      if (res.job_id) {
        const stub: InferenceJob = {
          job_id: res.job_id,
          status: "pending",
          done: 0,
          total: 0,
          images_dir: res.images_dir,
          output_dir: res.output_dir,
          error: null,
          warning: null,
        };
        setJobs((prev) => [stub, ...prev]);
        setActiveJob(stub);
        setActiveJobListed(true);
        dropRefusal(date);
        if (res.bucket_redirected) {
          useStore
            .getState()
            .pushToast(
              `${date}: the requested bucket has review verdicts, so this run writes to ${res.output_dir}.`,
              "info",
            );
        }
      }
    } catch (e) {
      const refusal = bucketRefusalOf(e);
      if (refusal) {
        setRefusedLaunches((prev) => ({
          ...prev,
          [date]: { date, modelName, checkpointPath, refusal },
        }));
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
    if (!model || !datasetRoot || selectedDates.length === 0) return;
    setRefusedLaunches((prev) => {
      const next = { ...prev };
      for (const date of selectedDates) delete next[date];
      return next;
    });
    setLaunching(true);
    try {
      // One job per date: each date is its own prediction bucket, one job row per date.
      for (const date of selectedDates) {
        await launchOne(model.checkpoint_path, model.name, date);
      }
    } finally {
      setLaunching(false);
    }
  }

  function onRunSuggestion(entry: RefusedLaunch, suggestedModelName: string) {
    void launchOne(entry.checkpointPath, suggestedModelName, entry.date);
  }

  function onWatchRefusedJob(jobId: string) {
    const row = jobs.find((j) => j.job_id === jobId);
    setActiveJob(
      row ??
        ({
          job_id: jobId,
          status: "running",
          done: 0,
          total: 0,
          images_dir: "",
          output_dir: "",
          error: null,
          warning: null,
        } as InferenceJob),
    );
    setActiveJobListed(row !== undefined);
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
      <TabHeading tab="inference" />
      <div className="border-r border-tcip-border p-4 overflow-auto">
        <div className="tcip-heading mb-3">Inference config</div>

        <label className="tcip-label mb-1">Model checkpoint</label>
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

        <label className="tcip-label mb-1">Capture dates</label>
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

        <p className="text-[11px] text-tcip-muted mb-3">
          Conf/IoU come from the platform&apos;s own defaults, and whether this run tiles (and at
          what size/overlap) comes from this checkpoint&apos;s own training geometry, and
          predictions land in this dataset&apos;s prediction dir for the model and date: the same
          operating point and layout the agent-facing door resolves, so a run here and a run there
          cannot diverge.
        </p>

        <button
          className="tcip-btn-primary w-full"
          onClick={onLaunch}
          disabled={launching || !modelPath || selectedDates.length === 0}
        >
          ▶&nbsp;&nbsp;Launch inference
        </button>

        {Object.keys(refusedLaunches).length > 0 && (
          <div className="mt-3">
            <div className="tcip-heading mb-1">Refused launches</div>
            <ul aria-live="polite" className="flex flex-col gap-1">
              {Object.values(refusedLaunches).map((entry) => (
                <RefusedLaunchEntry
                  key={entry.date}
                  entry={entry}
                  onRunSuggestion={onRunSuggestion}
                  onWatch={onWatchRefusedJob}
                  onDismiss={dropRefusal}
                />
              ))}
            </ul>
          </div>
        )}
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
            {activeJobListed && (
              <div className="text-[11px] mt-1 tabular-nums">
                Status: {activeJob.status} · {activeJob.done} / {activeJob.total}
              </div>
            )}
            {activeJob.error && (
              <div className="text-[11px] text-tcip-fp mt-1">Error: {activeJob.error}</div>
            )}
            {activeJob.warning && (
              <div className="text-[11px] text-tcip-warn mt-1">Warning: {activeJob.warning}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
