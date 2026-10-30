import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import { messageOf, refusalOf } from "@/api/http";
import { inferenceApi, openInferenceStream, resultsApi } from "@/api/inference";
import {
  BUCKET_EXISTS,
  CROSS_TILE_MERGE_DESCRIPTIONS,
  type AssessmentSummary,
  type Assessments,
  type BucketExistsRefusal,
  type DatasetTree,
  type InferenceJob,
  type JobStatus,
  type Listed,
  type MaskCut,
  NOTHING_STATED,
  type RegisteredModels,
  type Stated,
} from "@/api/types.generated";
import { TabHeading } from "@/components/TabHeading";
import { useStore } from "@/store";
import { selectDatasetRoot, selectProjectRoot } from "@/store/slices/gui";

type StatedField = "conf" | "cross_tile_nms" | keyof MaskCut;

/** The execution values a run without an assessment states, one input each, their ranges the
 *  server's to refuse; the two mask cut inputs are stated together as its one ``mask_cut``. */
const STATED_INPUTS: { field: StatedField; label: string }[] = [
  { field: "conf", label: "Confidence" },
  { field: "mask_binarize", label: "Mask threshold" },
  { field: "vertex_spacing", label: "Ring spacing" },
  { field: "cross_tile_nms", label: "Merge threshold" },
];

/** One date's refused launch: the facts the response carried, watching the job still writing the
 *  bucket, and a dismissal. Every accessible name below is date-qualified. */
function RefusedLaunchEntry({
  date,
  refusal,
  onWatch,
  onDismiss,
}: {
  date: string;
  refusal: BucketExistsRefusal;
  onWatch: (refusal: BucketExistsRefusal) => void;
  onDismiss: (date: string) => void;
}) {
  return (
    <li className="border border-tcip-border rounded p-2 flex flex-col gap-1">
      <p>
        {date}: {refusal.message}
      </p>
      <div className="flex gap-1">
        <button
          className="tcip-btn"
          aria-label={`Watch job ${refusal.job.job_id} for ${date}`}
          onClick={() => onWatch(refusal)}
        >
          Watch
        </button>
        <button
          className="tcip-btn"
          aria-label={`Dismiss refusal for ${date}`}
          onClick={() => onDismiss(date)}
        >
          Dismiss
        </button>
      </div>
    </li>
  );
}

function statusBadgeClass(status: JobStatus): string {
  if (status === "completed") return "bg-tcip-tp/20 text-tcip-tp";
  if (status === "failed" || status === "canceled") return "bg-tcip-fp/20 tcip-problem";
  return "bg-tcip-fn/20 text-tcip-fn"; // pending / running
}

/** The assessments among `assessments` that judged `model`'s checkpoint. */
function assessmentsOf(assessments: AssessmentSummary[], model: Listed | undefined) {
  return model
    ? assessments.filter((a) => a.assessment.producer?.checkpoint_sha256 === model.entry.sha256)
    : [];
}

export function InferenceTab() {
  const projectRoot = useStore(selectProjectRoot);
  const datasetRoot = useStore(selectDatasetRoot);

  const [registered, setRegistered] = useState<RegisteredModels | null>(null);
  const models = registered?.models ?? [];
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelPath, setModelPath] = useState<string>("");
  const [assessed, setAssessed] = useState<Assessments | null>(null);
  const [assessmentsError, setAssessmentsError] = useState<string | null>(null);
  const [tree, setTree] = useState<DatasetTree | null>(null);
  const dates = (tree?.captures ?? []).map((c) => c.name);
  const [datesError, setDatesError] = useState<string | null>(null);
  const [selectedDates, setSelectedDates] = useState<string[]>([]);
  // The named bucket each date's own bucket is published beneath; the breeder names it.
  const [bucketName, setBucketName] = useState("");
  // The assessment whose execution record the run takes and publishes under; null for a run
  // stating its own values.
  const [assessmentId, setAssessmentId] = useState<string | null>(null);
  const [statedInputs, setStatedInputs] = useState<Record<StatedField, string>>({
    conf: "",
    mask_binarize: "",
    vertex_spacing: "",
    cross_tile_nms: "",
  });
  // The cross-tile merge a tiled run without an assessment states, and whether an instance
  // segmentation run without one collects masks; null for none stated.
  const [merge, setMerge] = useState<Stated["postprocess"]>(null);
  const [masks, setMasks] = useState<Stated["masks"]>(null);
  // Why the run states the values above, carried on its execution record.
  const [rationale, setRationale] = useState("");
  // Each job's one record by its id: the listed jobs', each launched or watched since, and the
  // watched one's once a poll delisted it.
  const [jobs, setJobs] = useState<Record<string, InferenceJob>>({});
  // The watched job the last poll did not list: it shows its last error alone rather than a status
  // line frozen on a job that is gone; null while the watched job is listed.
  const [delisted, setDelisted] = useState<string | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  // Each bucket_exists refusal, keyed by the date it was refused for.
  const [refusedLaunches, setRefusedLaunches] = useState<Record<string, BucketExistsRefusal>>({});
  const [launching, setLaunching] = useState(false);
  const streamRef = useRef<(() => void) | null>(null);
  const launchButtonRef = useRef<HTMLButtonElement>(null);
  const prevRefusedDatesRef = useRef<string[]>([]);
  const activeJob = activeJobId === null ? undefined : jobs[activeJobId];
  const listedJobs = Object.values(jobs).filter((j) => j.job_id !== delisted);
  const model = models.find((m) => m.entry.checkpoint_path === modelPath);
  const modelAssessments = assessmentsOf(assessed?.assessments ?? [], model);

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

  // The chosen model's first passed assessment answers for its run unless the person chooses
  // otherwise; a model with none states its own values.
  useEffect(() => {
    setAssessmentId(modelAssessments.find((a) => a.passed)?.assessment.assessment_id ?? null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPath, assessed]);

  const refreshModels = useCallback(() => {
    if (!projectRoot) return;
    void resultsApi
      .registeredModels()
      .then((r) => {
        setRegistered(r);
        setModelsError(null);
      })
      .catch((e) => {
        setRegistered(null);
        setModelsError(`Could not load registered models: ${messageOf(e)}`);
      });
    void resultsApi
      .assessments()
      .then((r) => {
        setAssessed(r);
        setAssessmentsError(null);
      })
      .catch((e) => {
        setAssessed(null);
        setAssessmentsError(`Could not load assessments: ${messageOf(e)}`);
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
        setTree(t);
        setSelectedDates((prev) => prev.filter((d) => t.captures.some((c) => c.name === d)));
        setDatesError(null);
      })
      .catch((e) => {
        setTree(null);
        setDatesError(`Could not load this dataset's dates: ${messageOf(e)}`);
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
          const polled = Object.fromEntries(r.jobs.map((j) => [j.job_id, j]));
          const watchedGone = activeJobId !== null && !(activeJobId in polled);
          setJobs((prev) =>
            watchedGone && activeJobId in prev
              ? { ...polled, [activeJobId]: prev[activeJobId] }
              : polled,
          );
          setDelisted(watchedGone ? activeJobId : null);
          setJobsError(null);
        })
        .catch((e) => {
          setJobsError(`Could not load inference jobs: ${messageOf(e)}`);
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
    streamRef.current = openInferenceStream(activeJobId, (frame) => {
      setJobs((prev) => ({
        ...prev,
        [activeJobId]: "job" in frame ? frame.job : { ...prev[activeJobId], error: frame.error },
      }));
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

  const own = assessmentId === null;
  const filled = (field: StatedField) => {
    const value = statedInputs[field].trim();
    return own && value !== "" ? Number(value) : null;
  };
  const maskBinarize = filled("mask_binarize");
  const vertexSpacing = filled("vertex_spacing");
  // A mask cut is its two values together: one typed without the other states no cut.
  const halfCut = (maskBinarize === null) !== (vertexSpacing === null);

  /** What the run states of its execution record: each filled value, the mask cut once both of
   *  its values are, the chosen merge, and the reason; nothing under an assessment. */
  function statement(): Stated {
    return {
      ...NOTHING_STATED,
      postprocess: own ? merge : null,
      cross_tile_nms: filled("cross_tile_nms"),
      conf: filled("conf"),
      masks: own ? masks : null,
      mask_cut:
        maskBinarize !== null && vertexSpacing !== null
          ? { mask_binarize: maskBinarize, vertex_spacing: vertexSpacing }
          : null,
      rationale: own && rationale.trim() ? rationale.trim() : null,
    };
  }

  // One date's launch: the server publishes the date's own bucket beneath the named one.
  async function launchOne(checkpointPath: string, date: string) {
    if (!datasetRoot) return;
    try {
      const launched = await inferenceApi.launch({
        checkpoint_path: checkpointPath,
        dataset_root: datasetRoot,
        date,
        bucket_name: bucketName,
        stated: statement(),
        assessment_id: assessmentId,
        user: useStore.getState().user,
      });
      setJobs((prev) => ({ [launched.job_id]: launched, ...prev }));
      setActiveJobId(launched.job_id);
      dropRefusal(date);
    } catch (e) {
      const refusal = refusalOf<BucketExistsRefusal>(e, BUCKET_EXISTS);
      if (refusal) {
        setRefusedLaunches((prev) => ({ ...prev, [date]: refusal }));
        return;
      }
      dropRefusal(date);
      useStore.getState().pushToast(`Inference launch failed for ${date}: ${messageOf(e)}`);
    }
  }

  async function onLaunch() {
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
        await launchOne(model.entry.checkpoint_path, date);
      }
    } finally {
      setLaunching(false);
    }
  }

  // The refusal's record is the job's until the tab holds one of its own.
  function onWatchRefusedJob({ job }: BucketExistsRefusal) {
    setJobs((prev) => (job.job_id in prev ? prev : { ...prev, [job.job_id]: job }));
    setActiveJobId(job.job_id);
  }

  // Jobs whose cancel was requested and not yet reported terminal by the backend; the row reads
  // "canceling" until the poll reports the state the worker ended in.
  const [cancelRequested, setCancelRequested] = useState<ReadonlySet<string>>(new Set());

  async function onCancel(jobId: string) {
    setCancelRequested((prev) => new Set(prev).add(jobId));
    try {
      await inferenceApi.cancel(jobId, { user: useStore.getState().user });
    } catch (e) {
      setCancelRequested((prev) => {
        const next = new Set(prev);
        next.delete(jobId);
        return next;
      });
      useStore.getState().pushToast(`Cancel failed: ${messageOf(e)}`);
    }
  }

  /** A job's state as the row reads it: the backend's, or "canceling" while a cancel it has
   *  not yet honored is requested. */
  function shownStatus(j: InferenceJob): string {
    return j.live && cancelRequested.has(j.job_id) ? "canceling" : j.status;
  }

  return (
    <div className="flex-1 grid grid-cols-[440px_1fr] overflow-hidden">
      <TabHeading tab="inference" />
      <div className="border-r border-tcip-border p-4 overflow-auto">
        <label className="tcip-label mb-1" htmlFor="inference-model-select">
          Model
        </label>
        {modelsError && (
          <div className="tcip-problem mb-1">
            {modelsError}{" "}
            <button className="tcip-btn ml-1" onClick={refreshModels}>
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
            <option value="">Choose a model</option>
            {models.map(({ entry }) => (
              <option key={entry.checkpoint_path} value={entry.checkpoint_path}>
                {entry.name}
              </option>
            ))}
          </select>
        ) : (
          !modelsError &&
          projectRoot && <p className="text-tcip-muted mb-3">No model is registered.</p>
        )}

        <div className="tcip-label mb-1">Dates</div>
        {datesError && (
          <div className="tcip-problem mb-1">
            {datesError}{" "}
            <button className="tcip-btn ml-1" onClick={refreshDates}>
              Retry
            </button>
          </div>
        )}
        {!datasetRoot ? (
          <p className="text-tcip-muted mb-3">No project is open.</p>
        ) : dates.length === 0 ? (
          !datesError && <p className="text-tcip-muted mb-3">No dates with images.</p>
        ) : (
          <div className="mb-3 max-h-48 overflow-auto border border-tcip-border rounded p-2 flex flex-col gap-1">
            {dates.map((d) => (
              <label key={d} className="flex items-center gap-2">
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
          Predictions
        </label>
        <input
          id="inference-bucket-root"
          className="tcip-input w-full mb-3 font-mono"
          value={bucketName}
          onChange={(e) => setBucketName(e.target.value)}
        />

        <label className="tcip-label mb-1" htmlFor="inference-assessment">
          Assessment
        </label>
        {assessmentsError && (
          <div className="tcip-problem mb-1">
            {assessmentsError}{" "}
            <button className="tcip-btn ml-1" onClick={refreshModels}>
              Retry
            </button>
          </div>
        )}
        <select
          id="inference-assessment"
          className="tcip-select w-full mb-3"
          value={assessmentId ?? ""}
          onChange={(e) => setAssessmentId(e.target.value || null)}
        >
          <option value="">None</option>
          {modelAssessments.map((a) => (
            <option key={a.assessment.assessment_id} value={a.assessment.assessment_id}>
              {a.assessment.assessment_id}
              {a.passed ? "" : " (failed)"}
            </option>
          ))}
        </select>

        {own && (
          <>
            {STATED_INPUTS.map(({ field, label }) => (
              <div key={field}>
                <label className="tcip-label mb-1" htmlFor={`inference-${field}`}>
                  {label}
                </label>
                <input
                  id={`inference-${field}`}
                  className="tcip-input w-full mb-3 font-mono"
                  type="number"
                  step="any"
                  value={statedInputs[field]}
                  onChange={(e) =>
                    setStatedInputs((prev) => ({ ...prev, [field]: e.target.value }))
                  }
                />
              </div>
            ))}

            <label className="tcip-label mb-1" htmlFor="inference-merge">
              Merge
            </label>
            <select
              id="inference-merge"
              className="tcip-select w-full mb-3"
              value={merge ?? ""}
              onChange={(e) => setMerge(e.target.value || null)}
            >
              <option value="">None</option>
              {Object.entries(CROSS_TILE_MERGE_DESCRIPTIONS).map(([name, description]) => (
                <option key={name} value={name}>
                  {description}
                </option>
              ))}
            </select>

            <label className="tcip-label mb-1" htmlFor="inference-masks">
              Masks
            </label>
            <select
              id="inference-masks"
              className="tcip-select w-full mb-3"
              value={masks === null ? "" : String(masks)}
              onChange={(e) => setMasks(e.target.value === "" ? null : e.target.value === "true")}
            >
              <option value="">None</option>
              <option value="true">Collect masks</option>
              <option value="false">Boxes only</option>
            </select>

            <label className="tcip-label mb-1" htmlFor="inference-rationale">
              Reason
            </label>
            <input
              id="inference-rationale"
              className="tcip-input w-full mb-3"
              value={rationale}
              onChange={(e) => setRationale(e.target.value)}
            />
            {halfCut && (
              <p className="tcip-problem mb-3" role="status">
                State both the mask threshold and the ring spacing, or neither.
              </p>
            )}
          </>
        )}

        <button
          ref={launchButtonRef}
          className="tcip-btn-primary w-full"
          onClick={() => void onLaunch()}
          disabled={launching || !modelPath || !bucketName || selectedDates.length === 0 || halfCut}
        >
          Run
        </button>

        <div className="mt-3">
          {Object.keys(refusedLaunches).length > 0 && (
            <div className="tcip-heading mb-1">Refused</div>
          )}
          <ul aria-live="polite" className="flex flex-col gap-1">
            {Object.entries(refusedLaunches).map(([date, refusal]) => (
              <RefusedLaunchEntry
                key={date}
                date={date}
                refusal={refusal}
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
          <div className="tcip-problem mb-2">
            {jobsError}{" "}
            <button className="tcip-btn ml-1" onClick={() => void refreshJobs()}>
              Retry
            </button>
          </div>
        )}
        {listedJobs.length === 0 ? (
          !jobsError && <div className="text-tcip-muted">No jobs.</div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-tcip-border">
                <th className="tcip-th">Job</th>
                <th className="tcip-th">State</th>
                <th className="tcip-th">Progress</th>
                <th className="tcip-th"></th>
              </tr>
            </thead>
            <tbody>
              {listedJobs.map((j) => (
                <tr key={j.job_id} className="border-t border-tcip-border">
                  <td className="py-1.5 pr-3 font-mono">{j.job_id}</td>
                  <td className="pr-3">
                    <span className={`tcip-badge ${statusBadgeClass(j.status)}`}>
                      {shownStatus(j)}
                    </span>
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
                  <td>
                    <div className="flex gap-1">
                      <button className="tcip-btn" onClick={() => setActiveJobId(j.job_id)}>
                        Watch
                      </button>
                      {j.live && !cancelRequested.has(j.job_id) && (
                        <button className="tcip-btn" onClick={() => void onCancel(j.job_id)}>
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
              <span className="font-mono">{activeJob.job_id}</span>
            </div>
            {activeJob.bucket && (
              <div className="text-tcip-muted">
                Prediction set <span className="font-mono">{activeJob.bucket}</span>
              </div>
            )}
            {activeJob.job_id !== delisted && (
              <div className="mt-1 tabular-nums">
                {activeJob.status}, {activeJob.done} of {activeJob.total}
              </div>
            )}
            {activeJob.error && <div className="tcip-problem mt-1">{activeJob.error}</div>}
          </div>
        )}
      </div>
    </div>
  );
}
