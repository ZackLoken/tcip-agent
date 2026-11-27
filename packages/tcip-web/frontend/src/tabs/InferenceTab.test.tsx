import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { inferenceApi, openInferenceStream, resultsApi } from "@/api/inference";
import { type InferenceJob, type InferenceJobs, NOTHING_STATED } from "@/api/types.generated";
import { NotificationsLiveRegion, NotificationsPane } from "@/components/Notifications";
import { useStore } from "@/store";
import { InferenceTab } from "@/tabs/InferenceTab";
import { mockDatasetTree } from "@/test/datasetTree";
import {
  FAILED_ASSESSMENTS,
  PRODUCED_ASSESSMENTS,
  REGISTERED_MODELS,
} from "@/test/deliveryRecords";
import { INFERENCE_ANSWERS } from "@/test/inference";
import { renderBesideListings } from "@/test/jobListings";
import { areaEntrySaying, entriesSaying, metaLine } from "@/test/notifications";
import { thrown } from "@/test/refusals";
import {
  DATASET_TREES,
  DS_DATE,
  DS_UNPUBLISHED_DATE,
  GROUPED_SELECTION,
  openTestProject,
  RAW_SELECTION,
  TEST_PROJECT,
} from "@/test/store";

// The live job stream owns a real WebSocket; only its frame-to-state mapping is under test here,
// so the transport is replaced while the rest of the module stays real.
vi.mock("@/api/inference", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/inference")>();
  return { ...actual, openInferenceStream: vi.fn(() => () => {}) };
});

const initialStoreState = useStore.getState();

const [MODEL, , OTHER_MODEL] = REGISTERED_MODELS.models;
const [PASSED, FAILED] = PRODUCED_ASSESSMENTS.assessments;
const {
  launched: LAUNCHED,
  failing_started: FAILING_STARTED,
  both_started: BOTH_STARTED,
  failing_final: FAILING_FINAL,
  held_started: HELD_STARTED,
  running: RUNNING,
  cancel: CANCEL,
  canceled: CANCELED,
} = INFERENCE_ANSWERS;
const HELD = LAUNCHED.job_id;
const [FAILED_JOB] = RUNNING.jobs;

function setupDataset() {
  openTestProject(RAW_SELECTION);
}

/** The watched job panel's counts line of `job`. */
function countsOf(job: InferenceJob): string {
  return `${job.state.status}, ${job.done} of ${job.total}`;
}

/** The error the produced `job` failed on; a job in any other state throws. */
function errorOf(job: InferenceJob): string {
  if (job.state.status !== "failed") throw new Error("the produced job did not fail");
  return job.state.error;
}

/** Answer the dataset tree with the produced photographs' dataset, or with one holding no
 *  capture. */
function mockTree(captured = true) {
  mockDatasetTree(captured ? DATASET_TREES.ds : DATASET_TREES.empty);
}

/** Answer the registered models with the produced listing. */
function mockModels() {
  vi.spyOn(resultsApi, "registeredModels").mockResolvedValue(REGISTERED_MODELS);
}

/** Answer every launch with the produced launch answer. */
function mockLaunches() {
  return vi.spyOn(inferenceApi, "launch").mockResolvedValue(LAUNCHED);
}

function selectBaseline() {
  fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
    target: { value: MODEL.entry.checkpoint_path },
  });
  fireEvent.change(screen.getByLabelText("Predictions"), {
    target: { value: "baseline" },
  });
}

/** Click the Watch control on the row of the job `jobId` names. */
function watch(jobId: string) {
  const row = screen.getByText(jobId).closest("tr");
  if (!row) throw new Error(`no row lists ${jobId}`);
  fireEvent.click(within(row).getByRole("button", { name: `Watch ${jobId}` }));
}

// The refusal's bucket renders in its own <span>, so the sentence's default per-node text (which
// excludes nested elements) never carries it whole; match on a <p>'s full textContent instead.
function paragraphHolding(sentence: string) {
  return (_content: string, element: Element | null) =>
    element?.tagName === "P" && (element.textContent ?? "").includes(sentence);
}

/** The produced refusal of a launch naming the bucket a live job is still writing. */
const REFUSED = INFERENCE_ANSWERS.bucket_exists.body.detail;

const existsRefusal = () => thrown(INFERENCE_ANSWERS.bucket_exists);

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  setupDataset();
  vi.spyOn(inferenceApi, "listJobs").mockResolvedValue({ jobs: [] });
  vi.spyOn(resultsApi, "assessments").mockResolvedValue({ assessments: [] });
});

afterEach(cleanup);

describe("InferenceTab date selection", () => {
  it("launches one job per selected date, each its own bucket under the named one", async () => {
    mockTree();
    mockModels();
    const launchSpy = mockLaunches();
    useStore.setState({ user: "jordan" });

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());

    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("checkbox", { name: DS_UNPUBLISHED_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(2));
    expect(launchSpy.mock.calls.map(([body]) => body)).toEqual([
      {
        checkpoint_path: MODEL.entry.checkpoint_path,
        dataset_root: RAW_SELECTION.dataset_root,
        date: DS_DATE,
        bucket_name: "baseline",
        stated: NOTHING_STATED,
        assessment_id: null,
        user: "jordan",
      },
      {
        checkpoint_path: MODEL.entry.checkpoint_path,
        dataset_root: RAW_SELECTION.dataset_root,
        date: DS_UNPUBLISHED_DATE,
        bucket_name: "baseline",
        stated: NOTHING_STATED,
        assessment_id: null,
        user: "jordan",
      },
    ]);
  });

  it("states the conf, merge and merge threshold the breeder enters with their reason, and no cap", async () => {
    mockTree();
    mockModels();
    const launchSpy = mockLaunches();

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());

    selectBaseline();
    fireEvent.change(screen.getByLabelText("Confidence"), { target: { value: "0.35" } });
    expect(screen.queryByLabelText(/detection cap/i)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Merge threshold"), { target: { value: "0.45" } });
    fireEvent.change(screen.getByLabelText("Merge"), { target: { value: "nmm" } });
    fireEvent.change(screen.getByLabelText("Reason"), {
      target: { value: "the breeder's own reason" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(1));
    expect(launchSpy.mock.calls[0][0].stated).toEqual({
      ...NOTHING_STATED,
      conf: 0.35,
      cross_tile_nms: 0.45,
      postprocess: "nmm",
      rationale: "the breeder's own reason",
    });
  });

  it("shows a launched job's counts when a listing without it answered during the launch", async () => {
    mockTree();
    mockModels();
    const launch = Promise.withResolvers<InferenceJob>();
    vi.spyOn(inferenceApi, "launch").mockReturnValue(launch.promise);
    vi.mocked(inferenceApi.listJobs)
      .mockResolvedValueOnce({ jobs: [] })
      .mockResolvedValueOnce({ jobs: [] })
      .mockReturnValue(new Promise(() => {}));

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await act(() => Promise.resolve(useStore.getState().askListing("inference")));
    await act(() => Promise.resolve(launch.resolve(LAUNCHED)));

    expect(screen.getByText(countsOf(LAUNCHED))).toBeInTheDocument();
  });

  it("says a dataset has no dates rather than showing an empty picker", async () => {
    mockTree(false);
    mockModels();

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("No dates with images.")).toBeInTheDocument());
  });
});

describe("InferenceTab assessment", () => {
  beforeEach(() => {
    mockTree();
    mockModels();
  });

  it("runs under the model's first passed assessment with no stated inputs", async () => {
    vi.mocked(resultsApi.assessments).mockResolvedValue(PRODUCED_ASSESSMENTS);
    const launchSpy = mockLaunches();

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();

    const select = screen.getByRole("combobox", { name: "Assessment" });
    const passed = PASSED.assessment.assessment_id;
    await waitFor(() => expect(select).toHaveValue(passed));
    expect(
      screen.getByRole("option", { name: `${FAILED.assessment.assessment_id} (failed)` }),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText("Confidence")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Reason")).not.toBeInTheDocument();
    expect(screen.queryByText(/Every execution value/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(1));
    expect(launchSpy.mock.calls[0][0]).toMatchObject({
      assessment_id: passed,
      stated: NOTHING_STATED,
    });
  });

  it("offers no assessment of another model's checkpoint", async () => {
    vi.mocked(resultsApi.assessments).mockResolvedValue(PRODUCED_ASSESSMENTS);

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
      target: { value: OTHER_MODEL.entry.checkpoint_path },
    });

    const select = screen.getByRole("combobox", { name: "Assessment" });
    expect(
      within(select)
        .getAllByRole("option")
        .map((o) => o.textContent),
    ).toEqual(["None"]);
  });

  it("shows the stated inputs when no assessment passed", async () => {
    vi.mocked(resultsApi.assessments).mockResolvedValue(FAILED_ASSESSMENTS);

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();

    expect(screen.getByRole("combobox", { name: "Assessment" })).toHaveValue("");
    expect(screen.getByLabelText("Confidence")).toBeInTheDocument();
    expect(screen.getByLabelText("Mask threshold")).toBeInTheDocument();
    expect(screen.getByLabelText("Ring spacing")).toBeInTheDocument();
    expect(screen.getByLabelText("Masks")).toBeInTheDocument();
  });

  it("refuses a mask cut stated by one of its two values and sends it once both are", async () => {
    vi.mocked(resultsApi.assessments).mockResolvedValue({ assessments: [] });
    const launchSpy = mockLaunches();

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.change(screen.getByLabelText("Mask threshold"), { target: { value: "0.5" } });

    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    expect(screen.getByText(/State both the mask threshold and the ring spacing/)).toBeVisible();

    fireEvent.change(screen.getByLabelText("Ring spacing"), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(1));
    expect(launchSpy.mock.calls[0][0].stated.mask_cut).toEqual({
      mask_binarize: 0.5,
      vertex_spacing: 2,
    });
  });
});

describe("InferenceTab job table", () => {
  beforeEach(() => {
    mockTree(false);
    mockModels();
  });

  it("names each row's Watch and Cancel for its job, so no two rows' controls share a name", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(BOTH_STARTED);

    renderBesideListings(<InferenceTab />);
    expect(await screen.findByText(HELD)).toBeInTheDocument();

    expect(BOTH_STARTED.jobs).toHaveLength(2);
    for (const { job_id } of BOTH_STARTED.jobs) {
      const row = screen.getByText(job_id).closest("tr") as HTMLElement;
      expect(within(row).getByRole("button", { name: `Watch ${job_id}` })).toHaveTextContent(
        /^Watch$/,
      );
      expect(within(row).getByRole("button", { name: `Cancel ${job_id}` })).toHaveTextContent(
        /^Cancel$/,
      );
    }
  });

  it("offers a stop control only for a job still in flight, and cancels that job by id", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(RUNNING);
    const cancelSpy = vi.spyOn(inferenceApi, "cancel").mockResolvedValue(CANCEL);
    useStore.setState({ user: "jordan" });

    renderBesideListings(<InferenceTab />);
    expect(await screen.findByText(HELD)).toBeInTheDocument();
    expect(screen.getByText(FAILED_JOB.job_id)).toBeInTheDocument();

    expect(screen.queryByRole("button", { name: `Cancel ${FAILED_JOB.job_id}` })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: `Cancel ${HELD}` }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith(HELD, { user: "jordan" }));
  });

  it("shows a failed job cancel in place in that job's row, until its next cancel", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(RUNNING);
    // What fetch rejects with when the backend cannot be reached.
    vi.spyOn(inferenceApi, "cancel")
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockReturnValue(new Promise(() => {}));
    const sentence = "Cancel failed: Failed to fetch";

    renderBesideListings(<InferenceTab />);
    expect(await screen.findByText(HELD)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: `Cancel ${HELD}` }));

    const line = await screen.findByTestId("entry-in-place");
    expect(line).toHaveTextContent(sentence);
    expect(line.closest("tr")?.previousElementSibling).toBe(screen.getByText(HELD).closest("tr"));
    expect(entriesSaying(sentence)).toEqual([
      expect.objectContaining({
        kind: "failure",
        level: "error",
        context: { project: TEST_PROJECT.id, target: HELD },
      }),
    ]);

    fireEvent.click(screen.getByRole("button", { name: `Cancel ${HELD}` }));

    await waitFor(() => expect(screen.queryByTestId("entry-in-place")).toBeNull());
  });

  it("lists a failed job cancel as resolved once the next cancel of that job lands", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(RUNNING);
    // What fetch rejects with when the backend cannot be reached.
    const cancel = vi
      .spyOn(inferenceApi, "cancel")
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValue(CANCEL);
    const sentence = "Cancel failed: Failed to fetch";

    renderBesideListings(<InferenceTab />);
    render(<NotificationsPane />);
    expect(await screen.findByText(HELD)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: `Cancel ${HELD}` }));
    await screen.findByTestId("entry-in-place");
    expect(metaLine(areaEntrySaying(sentence))).not.toMatch(/resolved/);

    fireEvent.click(screen.getByRole("button", { name: `Cancel ${HELD}` }));

    await waitFor(() => expect(cancel).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(metaLine(areaEntrySaying(sentence))).toMatch(
        new RegExp(`^Error, failure, ${HELD}, .+, resolved$`),
      ),
    );
  });

  it("shows canceling until the poll reports the job ended", async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(inferenceApi.listJobs).mockResolvedValue(RUNNING);
      vi.spyOn(inferenceApi, "cancel").mockResolvedValue(CANCEL);

      renderBesideListings(<InferenceTab />);
      await vi.waitFor(() => expect(screen.getByText(HELD)).toBeInTheDocument());
      fireEvent.click(screen.getByRole("button", { name: `Cancel ${HELD}` }));

      await vi.waitFor(() => expect(screen.getByText("canceling")).toBeInTheDocument());
      expect(screen.queryByRole("button", { name: `Cancel ${HELD}` })).not.toBeInTheDocument();

      // The next poll still reports running: the row keeps canceling rather than flipping back.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });
      expect(screen.getByText("canceling")).toBeInTheDocument();

      vi.mocked(inferenceApi.listJobs).mockResolvedValue(CANCELED);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });
      expect(screen.getByText("canceled")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows a failed job's reason, not just its status badge", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(RUNNING);

    renderBesideListings(<InferenceTab />);
    await screen.findByText(FAILED_JOB.job_id);
    watch(FAILED_JOB.job_id);
    expect(await screen.findByText(errorOf(FAILED_JOB))).toBeInTheDocument();
  });

  /** Render the tab listing `listing`, watch the job `jobId` names and return its frame handler. */
  async function watching(listing: InferenceJobs, jobId: string) {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(listing);
    renderBesideListings(<InferenceTab />);
    await screen.findByText(jobId);
    watch(jobId);
    await waitFor(() => expect(vi.mocked(openInferenceStream)).toHaveBeenCalled());
    return vi.mocked(openInferenceStream).mock.calls[0][1];
  }

  it("carries a live frame's counts into the watched job panel", async () => {
    const onFrame = await watching(HELD_STARTED, HELD);
    const started = HELD_STARTED.jobs.find((job) => job.job_id === HELD)!;
    expect(screen.getByText(countsOf(started))).toBeInTheDocument();

    act(() => onFrame(INFERENCE_ANSWERS.progress));

    expect(await screen.findByText(countsOf(INFERENCE_ANSWERS.progress.job))).toBeInTheDocument();
  });

  it("carries a final frame's error into the watched job panel", async () => {
    const failing = FAILING_FINAL.job;
    const onFrame = await watching(FAILING_STARTED, failing.job_id);
    expect(screen.queryByText(errorOf(failing))).not.toBeInTheDocument();

    act(() => onFrame(FAILING_FINAL));

    expect(await screen.findByText(errorOf(failing))).toBeInTheDocument();
  });

  it("shows the frame's error alone once the poll no longer lists the watched job", async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(inferenceApi.listJobs).mockResolvedValueOnce(RUNNING);

      renderBesideListings(<InferenceTab />);
      await vi.waitFor(() => expect(screen.getByText(HELD)).toBeInTheDocument());
      watch(HELD);

      const missing = INFERENCE_ANSWERS.unknown_job;
      const onFrame = vi.mocked(openInferenceStream).mock.calls[0][1];
      act(() => onFrame(missing));
      expect(screen.getByText(missing.error)).toBeInTheDocument();
      expect(screen.getByText(/^running, /)).toBeInTheDocument();

      vi.mocked(inferenceApi.listJobs).mockResolvedValue({ jobs: [] });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });

      expect(screen.getByText(missing.error)).toBeInTheDocument();
      expect(screen.queryByText(/^running, /)).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("InferenceTab bucket refusals", () => {
  beforeEach(() => {
    mockTree();
    mockModels();
  });

  async function launchOneRefused() {
    const rendered = renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    return rendered;
  }

  /** The mount, under Run, of what the launch for `date` was answered. */
  const mountOf = (date: string) => screen.getByRole("group", { name: `Run for ${date}` });

  /** The entry shown in place in the mount of `date`'s launch, once it is shown. */
  const lineOf = (date: string) =>
    waitFor(() => within(mountOf(date)).getByTestId("entry-in-place"));

  /** The Watch the refusal shown in place as `line` offers there. */
  const watchIn = (line: HTMLElement) => within(line).getByRole("button", { name: /^Watch\b/ });

  /** The dismiss control of the entry shown in place, `line`. */
  const dismissIn = (line: HTMLElement) => within(line).getByRole("button", { name: /^Dismiss: / });

  /** Every listed entry of the refusal, read off the store. */
  const refusalEntries = () => entriesSaying(REFUSED.message);

  it("raises the refusal as one entry about its date, shown in place in that date's mount with Watch there", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    const line = await lineOf(DS_DATE);
    expect(line).toHaveTextContent(REFUSED.message);
    expect(watchIn(line)).toHaveAccessibleDescription(REFUSED.message);
    expect(within(mountOf(DS_UNPUBLISHED_DATE)).queryByTestId("entry-in-place")).toBeNull();
    expect(refusalEntries()).toEqual([
      expect.objectContaining({
        kind: "failure",
        level: "error",
        context: { project: TEST_PROJECT.id, target: DS_DATE },
        action: undefined,
      }),
    ]);
  });

  it("names the job still writing the bucket and watches it", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(RUNNING);
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    fireEvent.click(watchIn(await screen.findByTestId("entry-in-place")));
    const watched = RUNNING.jobs.find((job) => job.job_id === REFUSED.job.job_id)!;
    expect(await screen.findByText(countsOf(watched))).toBeInTheDocument();
  });

  it("watches the job the refusal carries when the poll hasn't listed it yet", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    fireEvent.click(watchIn(await screen.findByTestId("entry-in-place")));

    // The watched job's row names its bucket, beside the refused entry's own sentence.
    expect(await screen.findByText(REFUSED.job.bucket)).toBeInTheDocument();
    expect(screen.getByText(paragraphHolding(REFUSED.message))).toBeInTheDocument();
  });

  it("keeps one refused line and one job row when one of two dates launches and the other refuses", async () => {
    mockTree();
    const refusal = await existsRefusal();
    vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      body.date === DS_DATE ? Promise.resolve(LAUNCHED) : Promise.reject(refusal),
    );

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("checkbox", { name: DS_UNPUBLISHED_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(await screen.findByText(HELD)).toBeInTheDocument();
    expect(watchIn(await lineOf(DS_UNPUBLISHED_DATE))).toBeInTheDocument();
    expect(within(mountOf(DS_DATE)).queryByTestId("entry-in-place")).toBeNull();
    expect(screen.getAllByTestId("entry-in-place")).toHaveLength(1);
    expect(refusalEntries()).toEqual([
      expect.objectContaining({
        context: { project: TEST_PROJECT.id, target: DS_UNPUBLISHED_DATE },
      }),
    ]);
  });

  it("keeps each of two refused dates its own line and dismiss name, one dismissed leaving the other with Watch", async () => {
    mockTree();
    const { bucket_exists: first, failing_bucket_exists: second } = INFERENCE_ANSWERS;
    const refusals: Record<string, Error> = {
      [DS_DATE]: await thrown(first),
      [DS_UNPUBLISHED_DATE]: await thrown(second),
    };
    vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      Promise.reject(refusals[body.date]),
    );

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("checkbox", { name: DS_UNPUBLISHED_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    const dismissed = await lineOf(DS_DATE);
    const kept = await lineOf(DS_UNPUBLISHED_DATE);
    expect(dismissed).toHaveTextContent(first.body.detail.message);
    expect(kept).toHaveTextContent(second.body.detail.message);
    expect(dismissIn(dismissed).getAttribute("aria-label")).not.toBe(
      dismissIn(kept).getAttribute("aria-label"),
    );
    fireEvent.click(dismissIn(dismissed));

    expect(within(mountOf(DS_DATE)).queryByTestId("entry-in-place")).toBeNull();
    expect(watchIn(await lineOf(DS_UNPUBLISHED_DATE))).toBeInTheDocument();
    expect(entriesSaying(first.body.detail.message)).toEqual([]);
    expect(entriesSaying(second.body.detail.message)).toEqual([
      expect.objectContaining({
        context: { project: TEST_PROJECT.id, target: DS_UNPUBLISHED_DATE },
      }),
    ]);
  });

  /** Render the tab, select both dates and Run, each date's launch refused naming another job;
   *  answer each date's refused job by that date. */
  async function refuseBothDates(): Promise<Record<string, string>> {
    mockTree();
    const { bucket_exists: first, failing_bucket_exists: second } = INFERENCE_ANSWERS;
    const refusals: Record<string, Error> = {
      [DS_DATE]: await thrown(first),
      [DS_UNPUBLISHED_DATE]: await thrown(second),
    };
    vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      Promise.reject(refusals[body.date]),
    );

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("checkbox", { name: DS_UNPUBLISHED_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    return {
      [DS_DATE]: first.body.detail.job.job_id,
      [DS_UNPUBLISHED_DATE]: second.body.detail.job.job_id,
    };
  }

  it("names each of two refused dates' Watch by its label, its job and its date, the two names apart", async () => {
    const jobOf = await refuseBothDates();
    const [first, second] = [DS_DATE, DS_UNPUBLISHED_DATE];
    expect(jobOf[first]).not.toBe(jobOf[second]);

    for (const [date, other] of [
      [first, second],
      [second, first],
    ]) {
      const watch = watchIn(await lineOf(date));
      expect(watch).toHaveTextContent(/^Watch$/);
      expect(watch).toHaveAccessibleName(expect.stringMatching(/^Watch /));
      expect(watch).toHaveAccessibleName(expect.stringContaining(jobOf[date]));
      expect(watch).toHaveAccessibleName(expect.stringContaining(date));
      expect(watch).not.toHaveAccessibleName(expect.stringContaining(jobOf[other]));
    }
  });

  it("shows on each of two refused dates' lines the date it is for, in words on the screen", async () => {
    await refuseBothDates();

    for (const date of [DS_DATE, DS_UNPUBLISHED_DATE]) {
      expect(within(await lineOf(date)).getByText(date)).toBeVisible();
    }
  });

  it("offers Watch on a refusal that landed while the tab was away, once the person returns", async () => {
    const launch = Promise.withResolvers<InferenceJob>();
    vi.spyOn(inferenceApi, "launch").mockReturnValue(launch.promise);
    const refusal = await existsRefusal();
    const { unmount } = await launchOneRefused();
    await waitFor(() => expect(inferenceApi.launch).toHaveBeenCalledTimes(1));

    unmount();
    await act(() => Promise.resolve(launch.reject(refusal)));
    const { unmount: leaveAgain } = render(<InferenceTab />);

    fireEvent.click(watchIn(await lineOf(DS_DATE)));
    expect(await screen.findByText(REFUSED.job.bucket)).toBeInTheDocument();

    leaveAgain();
    expect(refusalEntries()).toEqual([expect.objectContaining({ key: undefined })]);
  });

  it("removes a dismissed refusal from the tab and the area, focus on Run", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    const dismiss = dismissIn(await screen.findByTestId("entry-in-place"));
    dismiss.focus();
    fireEvent.click(dismiss);

    expect(screen.queryByTestId("entry-in-place")).toBeNull();
    expect(refusalEntries()).toEqual([]);
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Run" }));
  });

  it("shows a launch refused again after a dismissal as a new entry, Watch offered again", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();
    fireEvent.click(dismissIn(await screen.findByTestId("entry-in-place")));

    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    const line = await screen.findByTestId("entry-in-place");
    expect(watchIn(line)).toBeInTheDocument();
    expect(refusalEntries()).toEqual([expect.objectContaining({ count: 1 })]);
  });

  it("shows a launch refused again while its earlier entry is listed as that entry, counted", async () => {
    const launch = vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();
    await screen.findByTestId("entry-in-place");

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(launch).toHaveBeenCalledTimes(2));

    const line = await screen.findByTestId("entry-in-place");
    expect(line).toHaveTextContent(`${REFUSED.message} (2 times)`);
    expect(watchIn(line)).toBeInTheDocument();
    expect(refusalEntries()).toEqual([expect.objectContaining({ count: 2 })]);
  });

  it("clears each selected date's refusal in place at a Run, its entry kept with no action", async () => {
    mockTree();
    const launch = vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("checkbox", { name: DS_UNPUBLISHED_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(screen.getAllByTestId("entry-in-place")).toHaveLength(2));

    launch.mockReturnValue(new Promise(() => {}));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(screen.queryByTestId("entry-in-place")).toBeNull();
    expect(refusalEntries()).toEqual([
      expect.objectContaining({ key: undefined, action: undefined }),
      expect.objectContaining({ key: undefined, action: undefined }),
    ]);
  });

  it("keeps a refusal listed, standing on no key, once the person leaves the tab", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    const { unmount } = await launchOneRefused();
    await screen.findByTestId("entry-in-place");

    unmount();

    expect(refusalEntries()).toEqual([expect.objectContaining({ key: undefined })]);
  });

  it("shows a launch failure of another kind in place under its date, said once, with no Watch, until that date launches again", async () => {
    const missing = INFERENCE_ANSWERS.missing_checkpoint;
    vi.spyOn(inferenceApi, "launch")
      .mockRejectedValueOnce(await thrown(missing))
      .mockResolvedValue(LAUNCHED);
    const sentence = `Inference launch failed: ${missing.body.detail}`;
    await launchOneRefused();

    const line = await lineOf(DS_DATE);
    expect(within(line).getByText(DS_DATE)).toBeVisible();
    expect(line.textContent?.split(DS_DATE)).toHaveLength(2);
    expect(line).toHaveTextContent(sentence);
    expect(within(line).queryByRole("button", { name: /^Watch\b/ })).toBeNull();
    expect(entriesSaying(sentence)).toEqual([
      expect.objectContaining({
        kind: "failure",
        level: "error",
        context: { project: TEST_PROJECT.id, target: DS_DATE },
      }),
    ]);

    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(await screen.findByText(HELD)).toBeInTheDocument();
    expect(screen.queryByTestId("entry-in-place")).toBeNull();
    expect(entriesSaying(sentence)).toHaveLength(1);
  });

  it("lists a failed launch as resolved once its date launches", async () => {
    const missing = INFERENCE_ANSWERS.missing_checkpoint;
    vi.spyOn(inferenceApi, "launch")
      .mockRejectedValueOnce(await thrown(missing))
      .mockResolvedValue(LAUNCHED);
    const sentence = `Inference launch failed: ${missing.body.detail}`;
    render(<NotificationsPane />);
    await launchOneRefused();
    await lineOf(DS_DATE);
    expect(metaLine(areaEntrySaying(sentence))).not.toMatch(/resolved/);

    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(await screen.findByText(HELD)).toBeInTheDocument();
    expect(metaLine(areaEntrySaying(sentence))).toMatch(
      new RegExp(`^Error, failure, ${DS_DATE}, .+, resolved$`),
    );
  });

  it("clears a refusal in place on a model change, its entry kept on no key", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();
    await screen.findByTestId("entry-in-place");

    fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
      target: { value: OTHER_MODEL.entry.checkpoint_path },
    });
    expect(screen.queryByTestId("entry-in-place")).toBeNull();
    expect(refusalEntries()).toEqual([expect.objectContaining({ key: undefined })]);
  });

  it("clears a refusal in place on a dataset change, its entry kept on no key", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();
    await screen.findByTestId("entry-in-place");

    act(() => openTestProject(GROUPED_SELECTION));
    expect(screen.queryByTestId("entry-in-place")).toBeNull();
    expect(refusalEntries()).toEqual([expect.objectContaining({ key: undefined })]);
  });

  it("names each of two dates' launch failures of one reason by its date, under Run and in the area, the names apart", async () => {
    mockTree();
    const missing = INFERENCE_ANSWERS.missing_checkpoint;
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await thrown(missing));
    const sentence = `Inference launch failed: ${missing.body.detail}`;
    renderBesideListings(<InferenceTab />);
    render(<NotificationsPane />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("checkbox", { name: DS_UNPUBLISHED_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    const dates = [DS_DATE, DS_UNPUBLISHED_DATE];
    const lines = await Promise.all(dates.map(lineOf));
    expect(entriesSaying(sentence)).toEqual([
      expect.objectContaining({ context: { project: TEST_PROJECT.id, target: DS_DATE } }),
      expect.objectContaining({
        context: { project: TEST_PROJECT.id, target: DS_UNPUBLISHED_DATE },
      }),
    ]);
    const area = within(screen.getByRole("region", { name: "Notifications" }));
    const areaNames = area
      .getAllByRole("button", { name: /^Dismiss: / })
      .map((b) => b.getAttribute("aria-label"));
    for (const [names, where] of [
      [lines.map((line) => dismissIn(line).getAttribute("aria-label")), "under Run"],
      [areaNames, "in the area"],
    ] as const) {
      expect(new Set(names).size, where).toBe(2);
      for (const date of dates) {
        expect(
          names.filter((name) => name?.includes(date)),
          where,
        ).toHaveLength(1);
      }
    }
  });

  it.each([
    [
      "a change of model",
      () =>
        fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
          target: { value: OTHER_MODEL.entry.checkpoint_path },
        }),
    ],
    [
      "leaving the tab and returning",
      (unmount: () => void) => {
        unmount();
        render(<InferenceTab />);
      },
    ],
  ])(
    "takes a launch failure of another kind off its place under Run at %s, its entry kept on no key",
    async (_event, happen) => {
      const missing = INFERENCE_ANSWERS.missing_checkpoint;
      vi.spyOn(inferenceApi, "launch").mockRejectedValue(await thrown(missing));
      const { unmount } = await launchOneRefused();
      await lineOf(DS_DATE);

      happen(unmount);

      expect(await screen.findByRole("checkbox", { name: DS_DATE })).toBeInTheDocument();
      expect(screen.queryByTestId("entry-in-place")).toBeNull();
      expect(useStore.getState().notifications).toEqual([
        expect.objectContaining({
          kind: "failure",
          key: undefined,
          context: { project: TEST_PROJECT.id, target: DS_DATE },
        }),
      ]);
    },
  );

  it("disables the launch button while onLaunch's loop is in flight", async () => {
    const launch = Promise.withResolvers<InferenceJob>();
    vi.spyOn(inferenceApi, "launch").mockReturnValue(launch.promise);

    renderBesideListings(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    const launchButton = screen.getByRole("button", { name: "Run" });
    fireEvent.click(launchButton);

    await waitFor(() => expect(launchButton).toBeDisabled());
    act(() => launch.resolve(LAUNCHED));
    await waitFor(() => expect(launchButton).not.toBeDisabled());
  });

  it("keeps no live list of its own, the area's live region announcing a refusal as an alert", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    render(<NotificationsLiveRegion />);
    const { container } = await launchOneRefused();
    await screen.findByTestId("entry-in-place");

    expect(within(container).queryByRole("list")).toBeNull();
    expect(container.querySelector("[aria-live]")).toBeNull();
    expect(within(screen.getByRole("alert")).getByText(REFUSED.message)).toBeInTheDocument();
  });
});

describe("InferenceTab model select", () => {
  it("associates the model label with the select by accessible name", async () => {
    mockTree(false);
    mockModels();

    renderBesideListings(<InferenceTab />);

    expect(await screen.findByRole("combobox", { name: "Model" })).toBeInTheDocument();
  });
});

describe("InferenceTab load errors", () => {
  it("shows each failed read's error with its own Retry, which reads that one again", async () => {
    const tree = vi
      .spyOn(api.dataset, "tree")
      .mockRejectedValueOnce(new Error("dates down"))
      .mockResolvedValue(DATASET_TREES.ds);
    const assessments = vi
      .spyOn(resultsApi, "assessments")
      .mockRejectedValueOnce(new Error("assessments down"))
      .mockResolvedValue(PRODUCED_ASSESSMENTS);
    const listJobs = vi
      .spyOn(inferenceApi, "listJobs")
      .mockRejectedValueOnce(new Error("jobs down"))
      .mockResolvedValue(RUNNING);
    const models = vi
      .spyOn(resultsApi, "registeredModels")
      .mockRejectedValueOnce(new Error("models down"))
      .mockResolvedValue(REGISTERED_MODELS);

    renderBesideListings(<InferenceTab />);

    const reads = [
      {
        message: "Could not load this dataset's dates: dates down",
        retry: "Retry this dataset's dates",
        read: tree,
      },
      {
        message: "Could not load assessments: assessments down",
        retry: "Retry assessments",
        read: assessments,
      },
      {
        message: "Could not load inference jobs: jobs down",
        retry: "Retry inference jobs",
        read: listJobs,
      },
      {
        message: "Could not load registered models: models down",
        retry: "Retry registered models",
        read: models,
      },
    ];
    for (const { message } of reads) await screen.findByText(message);
    for (const { message, retry } of reads) {
      expect(
        within(screen.getByText(message)).getByRole("button", { name: retry }),
      ).toHaveTextContent(/^Retry$/);
    }
    for (const { message, retry, read } of reads) {
      const block = await screen.findByText(message);
      expect(screen.getAllByText(message)).toHaveLength(1);
      expect(read).toHaveBeenCalledTimes(1);

      fireEvent.click(within(block).getByRole("button", { name: retry }));

      await waitFor(() => expect(screen.queryByText(message)).not.toBeInTheDocument());
      expect(read).toHaveBeenCalledTimes(2);
    }
  });
});

describe("InferenceTab heading", () => {
  it("renders exactly one top-level heading naming the tab", () => {
    renderBesideListings(<InferenceTab />);
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Inference");
  });
});
