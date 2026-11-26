import { Fragment, useCallback, useEffect, useRef, useState } from "react";

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
  type Listed,
  type MaskCut,
  NOTHING_STATED,
  type Stated,
} from "@/api/types.generated";
import { ErrorRetry } from "@/components/ErrorRetry";
import { ModelPicker } from "@/components/ModelPicker";
import { NotificationItem } from "@/components/Notifications";
import { TabHeading } from "@/components/TabHeading";
import { unended } from "@/hooks/jobPoll";
import { tell } from "@/lib/tell";
import { useStore } from "@/store";
import { selectDatasetRoot, selectProjectRoot } from "@/store/slices/gui";
import { selectStanding } from "@/store/slices/notifications";

type StatedField = "conf" | "cross_tile_nms" | keyof MaskCut;

/** The execution values a run without an assessment states, one input each, their ranges the
 *  server's to refuse; the two mask cut inputs are stated together as its one ``mask_cut``. */
const STATED_INPUTS: { field: StatedField; label: string }[] = [
  { field: "conf", label: "Confidence" },
  { field: "mask_binarize", label: "Mask threshold" },
  { field: "vertex_spacing", label: "Ring spacing" },
  { field: "cross_tile_nms", label: "Merge threshold" },
];

/** The key a failed or refused launch for `date` of the dataset at `datasetRoot` stands on in the
 *  project `projectId`. */
const launchKey = (projectId: string | undefined, datasetRoot: string, date: string) =>
  `inference-launch:${projectId}:${datasetRoot}:${date}`;

/** The key a failed cancel of the job `jobId` stands on in the project `projectId`. */
const cancelKey = (projectId: string | undefined, jobId: string) =>
  `inference-cancel:${projectId}:${jobId}`;

/** The id of the Watch control of the job `jobId`, where focus returns once its cancel's failure
 *  is dismissed. */
const watchControlId = (jobId: string) => `watch-${jobId}`;

/** Ask the shell's reader to list the inference jobs again now. */
const listJobsNow = () => useStore.getState().askListing("inference");

/** The job the tab watches, as its latest record came to the tab (the listing's, its launch's
 *  answer, the refusal naming it, its stream's last frame), and the listing as it stood then. */
interface Watched {
  record: InferenceJob;
  listedAt: Record<string, InferenceJob>;
}

/** The failure standing on the last cancel of the job `jobId`, in place as a row under that job's
 *  own, spanning the table's `columns`. */
function CancelFailureRow({
  entryKey,
  jobId,
  columns,
}: {
  entryKey: string;
  jobId: string;
  columns: number;
}) {
  const failed = useStore((s) => selectStanding(entryKey)(s) !== null);
  if (!failed) return null;
  return (
    <tr>
      <td colSpan={columns} className="pb-1.5">
        <NotificationItem
          placement="in place"
          entryKey={entryKey}
          afterDismiss={() => document.getElementById(watchControlId(jobId))?.focus()}
        />
      </td>
    </tr>
  );
}

function statusBadgeClass(job: InferenceJob): string {
  if (unended(job.state)) return "bg-tcip-fn/20 text-tcip-fn";
  return job.state.status === "completed"
    ? "bg-tcip-tp/20 text-tcip-tp"
    : "bg-tcip-fp/20 tcip-problem";
}

/** The assessments among `assessments` that judged `model`'s checkpoint. */
function assessmentsOf(assessments: AssessmentSummary[], model: Listed | null) {
  return model
    ? assessments.filter(
        (a) =>
          a.assessment.kind === "checkpoint" &&
          a.assessment.producer.checkpoint_sha256 === model.entry.sha256,
      )
    : [];
}

export function InferenceTab() {
  const projectRoot = useStore(selectProjectRoot);
  const datasetRoot = useStore(selectDatasetRoot);
  const projectId = useStore((s) => s.openProject?.id);

  const [model, setModel] = useState<Listed | null>(null);
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
  const listed = useStore((s) => s.jobListings.inference.answer);
  const listFailure = useStore((s) => s.jobListings.inference.failure);
  const jobsError = listFailure && `Could not load inference jobs: ${listFailure}`;
  const [watched, setWatched] = useState<Watched | null>(null);
  const activeJobId = watched?.record.job_id ?? null;
  // What each watched job's stream said once the backend held no record of it, by its id.
  const [gone, setGone] = useState<Record<string, string>>({});
  // Whether a listing answered after the tab held the watched job's record: whichever of the two
  // the tab held later shows, however long before its answer the listing was asked for.
  const relisted = watched !== null && listed !== watched.listedAt;
  // The watched job that listing does not list: it shows its last error alone rather than a status
  // line frozen on a job that is gone; null while it is listed.
  const delisted = relisted && !(watched.record.job_id in listed) ? watched.record.job_id : null;
  const activeJob =
    watched && (relisted && !delisted ? listed[watched.record.job_id] : watched.record);
  // Each job's one record by its id: the listed jobs', and the watched one's in its place.
  const jobs = activeJob ? { ...listed, [activeJob.job_id]: activeJob } : listed;
  // The job still writing the bucket of each refused launch, by the key its entry stands on: what
  // Watch, offered in place beside the entry, watches.
  const [refusedJobs, setRefusedJobs] = useState<Record<string, InferenceJob>>({});
  const refusedJobsRef = useRef(refusedJobs);
  const [launching, setLaunching] = useState(false);
  const streamRef = useRef<(() => void) | null>(null);
  const launchButtonRef = useRef<HTMLButtonElement>(null);
  const listedJobs = Object.values(jobs).filter((j) => j.job_id !== delisted);
  const modelAssessments = assessmentsOf(assessed?.assessments ?? [], model);

  useEffect(() => {
    refusedJobsRef.current = refusedJobs;
  }, [refusedJobs]);

  // A refusal names a bucket of the model and dataset it was made under: neither their change nor
  // the tab's leaving keeps it in place, and its entry stays listed in the area.
  useEffect(
    () => () => {
      const { releaseNotification } = useStore.getState();
      for (const key of Object.keys(refusedJobsRef.current)) releaseNotification(key);
      setRefusedJobs({});
    },
    [model, datasetRoot],
  );

  // The chosen model's first passed assessment answers for its run unless the person chooses
  // otherwise; a model with none states its own values.
  useEffect(() => {
    setAssessmentId(modelAssessments.find((a) => a.passed)?.assessment.assessment_id ?? null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, assessed]);

  const refreshAssessments = useCallback(() => {
    if (!projectRoot) return;
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
    refreshAssessments();
  }, [refreshAssessments]);

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

  useEffect(() => {
    if (!activeJobId) return;
    streamRef.current?.();
    streamRef.current = openInferenceStream(activeJobId, (frame) => {
      if (frame.type === "missing") {
        setGone((prev) => ({ ...prev, [activeJobId]: frame.error }));
      } else {
        const { answer } = useStore.getState().jobListings.inference;
        setWatched((prev) =>
          prev?.record.job_id === activeJobId ? { record: frame.job, listedAt: answer } : prev,
        );
      }
    });
    return () => streamRef.current?.();
  }, [activeJobId]);

  /** Watch the job `record` is the latest record of, unless it is the one watched; the listing is
   *  read again, so a job it no longer lists is known as gone. */
  function watch(record: InferenceJob) {
    if (record.job_id === activeJobId) return;
    setWatched({ record, listedAt: useStore.getState().jobListings.inference.answer });
    listJobsNow();
  }

  function toggleDate(date: string) {
    setSelectedDates((prev) =>
      prev.includes(date) ? prev.filter((d) => d !== date) : [...prev, date],
    );
  }

  /** Forget the job each refused launch standing on one of `keys` named. */
  function dropRefusals(keys: string[]) {
    setRefusedJobs((prev) => {
      const next = { ...prev };
      for (const key of keys) delete next[key];
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
    const key = launchKey(projectId, datasetRoot, date);
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
      watch(launched);
    } catch (e) {
      const refusal = refusalOf<BucketExistsRefusal>(e, BUCKET_EXISTS);
      if (refusal) {
        setRefusedJobs((prev) => ({ ...prev, [key]: refusal.job }));
        tell("failure", "error", refusal.message, date, key);
        return;
      }
      tell("failure", "error", `Inference launch failed for ${date}: ${messageOf(e)}`, date, key);
    }
  }

  async function onLaunch() {
    if (!model || !datasetRoot || !bucketName || selectedDates.length === 0) return;
    const keys = selectedDates.map((date) => launchKey(projectId, datasetRoot, date));
    const { releaseNotification } = useStore.getState();
    for (const key of keys) releaseNotification(key);
    dropRefusals(keys);
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

  // Jobs whose cancel was requested and not yet reported terminal by the backend; the row reads
  // "canceling" until the poll reports the state the worker ended in.
  const [cancelRequested, setCancelRequested] = useState<ReadonlySet<string>>(new Set());

  async function onCancel(jobId: string) {
    const key = cancelKey(projectId, jobId);
    setCancelRequested((prev) => new Set(prev).add(jobId));
    useStore.getState().releaseNotification(key);
    try {
      await inferenceApi.cancel(jobId, { user: useStore.getState().user });
    } catch (e) {
      setCancelRequested((prev) => {
        const next = new Set(prev);
        next.delete(jobId);
        return next;
      });
      tell("failure", "error", `Cancel failed: ${messageOf(e)}`, jobId, key);
    }
  }

  /** A job's state as the row reads it: the backend's, or "canceling" while a cancel it has
   *  not yet honored is requested. */
  function shownStatus(j: InferenceJob): string {
    return unended(j.state) && cancelRequested.has(j.job_id) ? "canceling" : j.state.status;
  }

  return (
    <div className="flex-1 grid grid-cols-[440px_1fr] overflow-hidden">
      <TabHeading tab="inference" />
      <div className="border-r border-tcip-border p-4 overflow-auto">
        <ModelPicker id="inference-model-select" value={model} onChange={setModel} />

        <div className="tcip-label mb-1">Dates</div>
        {datesError && <ErrorRetry message={datesError} onRetry={refreshDates} className="mb-1" />}
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
          <ErrorRetry message={assessmentsError} onRetry={refreshAssessments} className="mb-1" />
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
          disabled={launching || !model || !bucketName || selectedDates.length === 0 || halfCut}
        >
          Run
        </button>

        {datasetRoot && (
          <div className="flex flex-col gap-1 has-[>:not(:empty)]:mt-3">
            {dates.map((date) => {
              const key = launchKey(projectId, datasetRoot, date);
              const job = refusedJobs[key];
              return (
                <div
                  key={date}
                  role="group"
                  aria-label={`Run for ${date}`}
                  className="empty:hidden"
                >
                  <NotificationItem
                    placement="in place"
                    entryKey={key}
                    afterDismiss={() => {
                      dropRefusals([key]);
                      launchButtonRef.current?.focus();
                    }}
                    action={job && { label: "Watch", run: () => watch(jobs[job.job_id] ?? job) }}
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="p-4 overflow-auto">
        <div className="tcip-heading mb-3">Jobs</div>
        {jobsError && <ErrorRetry message={jobsError} onRetry={listJobsNow} className="mb-2" />}
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
                <Fragment key={j.job_id}>
                  <tr className="border-t border-tcip-border">
                    <td className="py-1.5 pr-3 font-mono">{j.job_id}</td>
                    <td className="pr-3">
                      <span className={`tcip-badge ${statusBadgeClass(j)}`}>{shownStatus(j)}</span>
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
                          id={watchControlId(j.job_id)}
                          className="tcip-btn"
                          onClick={() => watch(j)}
                        >
                          Watch
                        </button>
                        {unended(j.state) && !cancelRequested.has(j.job_id) && (
                          <button className="tcip-btn" onClick={() => void onCancel(j.job_id)}>
                            Cancel
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                  <CancelFailureRow
                    entryKey={cancelKey(projectId, j.job_id)}
                    jobId={j.job_id}
                    columns={4}
                  />
                </Fragment>
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
                {activeJob.state.status}, {activeJob.done} of {activeJob.total}
              </div>
            )}
            {activeJob.state.status === "failed" && (
              <div className="tcip-problem mt-1">{activeJob.state.error}</div>
            )}
            {gone[activeJob.job_id] && (
              <div className="tcip-problem mt-1">{gone[activeJob.job_id]}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
