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
import {
  CROSS_TILE_MERGE_DESCRIPTIONS,
  type AssessmentSummary,
  type MaskCut,
  type RegisteredModel,
  type Stated,
} from "@/api/types.generated";
import { TabHeading } from "@/components/TabHeading";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";

// A job can still be stopped only while it is pending/running.
const CANCELLABLE: ReadonlySet<InferenceStatus> = new Set(["pending", "running"]);

type StatedField = "conf" | "cross_tile_nms" | keyof MaskCut;

const MASK_CUT_FIELDS: ReadonlySet<StatedField> = new Set(["mask_binarize", "vertex_spacing"]);

/** The execution values a run without an assessment states, one input each, their ranges the
 *  server's to refuse; the two mask cut inputs are stated together as its one ``mask_cut``. */
const STATED_INPUTS: { field: StatedField; label: string }[] = [
  { field: "conf", label: "Confidence" },
  { field: "mask_binarize", label: "Mask threshold" },
  { field: "vertex_spacing", label: "Ring spacing" },
  { field: "cross_tile_nms", label: "Merge threshold" },
];

/** The assessment select's value for a run that states its own values. */
const NO_ASSESSMENT = "";

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
    <li className="border border-tcip-border rounded p-2 flex flex-col gap-1">
      <p>
        {date}: job {refusal.job_id} is still writing{" "}
        <span className="font-mono">{refusal.requested_bucket}</span>. Choose another name.
      </p>
      <div className="flex gap-1">
        <button
          className="tcip-btn"
          aria-label={`Watch job ${refusal.job_id} for ${date}`}
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

function statusBadgeClass(status: InferenceStatus): string {
  if (status === "completed") return "bg-tcip-tp/20 text-tcip-tp";
  if (status === "failed" || status === "canceled") return "bg-tcip-fp/20 tcip-problem";
  return "bg-tcip-fn/20 text-tcip-fn"; // pending / running
}

/** The assessments among `assessments` that judged `model`'s checkpoint. */
function assessmentsOf(assessments: AssessmentSummary[], model: RegisteredModel | undefined) {
  return model ? assessments.filter((a) => a.checkpoint_sha256 === model.sha256) : [];
}

export function InferenceTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const projectRoot = useStore(selectProjectRoot);
  const datasetRoot = dataset.dataset_root;

  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelPath, setModelPath] = useState<string>("");
  const [assessments, setAssessments] = useState<AssessmentSummary[]>([]);
  const [assessmentsError, setAssessmentsError] = useState<string | null>(null);
  const [dates, setDates] = useState<string[]>([]);
  const [datesError, setDatesError] = useState<string | null>(null);
  const [selectedDates, setSelectedDates] = useState<string[]>([]);
  // The name the run's buckets publish under, one ``<name>/<date>`` per date; the breeder names it.
  const [bucketName, setBucketName] = useState("");
  // The assessment whose execution record the run takes and publishes under; NO_ASSESSMENT for a
  // run stating its own values.
  const [assessmentId, setAssessmentId] = useState(NO_ASSESSMENT);
  const [statedInputs, setStatedInputs] = useState<Record<StatedField, string>>({
    conf: "",
    mask_binarize: "",
    vertex_spacing: "",
    cross_tile_nms: "",
  });
  // The cross-tile merge a tiled run without an assessment states; empty for none stated.
  const [merge, setMerge] = useState("");
  // Whether an instance segmentation run without an assessment collects masks; empty for none
  // stated.
  const [masks, setMasks] = useState("");
  // Why the run states the values above, carried on its execution record.
  const [rationale, setRationale] = useState("");
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
  const model = models.find((m) => m.checkpoint_path === modelPath);
  const modelAssessments = assessmentsOf(assessments, model);

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
    setAssessmentId(modelAssessments.find((a) => a.passed)?.assessment_id ?? NO_ASSESSMENT);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPath, assessments]);

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
    void resultsApi
      .assessments()
      .then((r) => {
        setAssessments(r.assessments);
        setAssessmentsError(null);
      })
      .catch((e) => {
        setAssessments([]);
        setAssessmentsError(
          `Could not load assessments: ${e instanceof Error ? e.message : String(e)}`,
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

  /** What the run states of its execution record: each filled value, the mask cut's filled
   *  values as its one block, the chosen merge, and the reason; nothing under an assessment. */
  function statement(): Stated {
    if (assessmentId !== NO_ASSESSMENT) return {};
    const filled = Object.entries(statedInputs)
      .filter(([, value]) => value.trim() !== "")
      .map(([field, value]) => [field, Number(value)] as const);
    const stated: Stated = Object.fromEntries(
      filled.filter(([field]) => !MASK_CUT_FIELDS.has(field as StatedField)),
    );
    const cut = filled.filter(([field]) => MASK_CUT_FIELDS.has(field as StatedField));
    if (cut.length) stated.mask_cut = Object.fromEntries(cut) as unknown as MaskCut;
    if (merge) stated.postprocess = merge;
    if (masks) stated.masks = masks === "true";
    if (rationale.trim()) stated.rationale = rationale.trim();
    return stated;
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
        stated: statement(),
        assessment_id: assessmentId === NO_ASSESSMENT ? null : assessmentId,
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

  // Jobs whose cancel was requested and not yet reported terminal by the backend; the row reads
  // "canceling" until the poll reports the state the worker ended in.
  const [cancelRequested, setCancelRequested] = useState<ReadonlySet<string>>(new Set());

  async function onCancel(jobId: string) {
    setCancelRequested((prev) => new Set(prev).add(jobId));
    try {
      await inferenceApi.cancel(jobId, useStore.getState().user);
    } catch (e) {
      setCancelRequested((prev) => {
        const next = new Set(prev);
        next.delete(jobId);
        return next;
      });
      useStore.getState().pushToast(`Cancel failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  /** A job's state as the row reads it: the backend's, or "canceling" while a cancel it has
   *  not yet honored is requested. */
  function shownStatus(j: InferenceJob): string {
    return CANCELLABLE.has(j.status) && cancelRequested.has(j.job_id) ? "canceling" : j.status;
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
            {models.map((m) => (
              <option key={m.checkpoint_path} value={m.checkpoint_path}>
                {m.name}
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
          value={assessmentId}
          onChange={(e) => setAssessmentId(e.target.value)}
        >
          <option value={NO_ASSESSMENT}>None</option>
          {modelAssessments.map((a) => (
            <option key={a.assessment_id} value={a.assessment_id}>
              {a.assessment_id}
              {a.passed ? "" : " (failed)"}
            </option>
          ))}
        </select>

        {assessmentId === NO_ASSESSMENT && (
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
              value={merge}
              onChange={(e) => setMerge(e.target.value)}
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
              value={masks}
              onChange={(e) => setMasks(e.target.value)}
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
          </>
        )}

        <button
          ref={launchButtonRef}
          className="tcip-btn-primary w-full"
          onClick={() => void onLaunch()}
          disabled={launching || !modelPath || !bucketName || selectedDates.length === 0}
        >
          Run
        </button>

        <div className="mt-3">
          {Object.keys(refusedLaunches).length > 0 && (
            <div className="tcip-heading mb-1">Refused</div>
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
          <div className="tcip-problem mb-2">
            {jobsError}{" "}
            <button className="tcip-btn ml-1" onClick={() => void refreshJobs()}>
              Retry
            </button>
          </div>
        )}
        {jobs.length === 0 ? (
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
              {jobs.map((j) => (
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
                      <button
                        className="tcip-btn"
                        onClick={() => {
                          setActiveJob(j);
                          setActiveJobListed(true);
                        }}
                      >
                        Watch
                      </button>
                      {CANCELLABLE.has(j.status) && !cancelRequested.has(j.job_id) && (
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
            {activeJobListed && (
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
