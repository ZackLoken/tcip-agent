import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { inferenceApi, openInferenceStream, resultsApi } from "@/api/inference";
import { type InferenceJob, type InferenceJobs, NOTHING_STATED } from "@/api/types.generated";
import { useStore } from "@/store";
import { InferenceTab } from "@/tabs/InferenceTab";
import { mockDatasetTree } from "@/test/datasetTree";
import {
  FAILED_ASSESSMENTS,
  PRODUCED_ASSESSMENTS,
  REGISTERED_MODELS,
} from "@/test/deliveryRecords";
import { INFERENCE_ANSWERS } from "@/test/inference";
import { thrown } from "@/test/refusals";
import {
  DATASET_TREES,
  DS_DATE,
  DS_UNPUBLISHED_DATE,
  GROUPED_SELECTION,
  openTestProject,
  RAW_SELECTION,
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
  return `${job.status}, ${job.done} of ${job.total}`;
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
  fireEvent.click(within(row).getByRole("button", { name: "Watch" }));
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

    render(<InferenceTab />);
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

    render(<InferenceTab />);
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

  it("says a dataset has no dates rather than showing an empty picker", async () => {
    mockTree(false);
    mockModels();

    render(<InferenceTab />);
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

    render(<InferenceTab />);
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

    render(<InferenceTab />);
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

    render(<InferenceTab />);
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

    render(<InferenceTab />);
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

  it("offers a stop control only for a job still in flight, and cancels that job by id", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(RUNNING);
    const cancelSpy = vi.spyOn(inferenceApi, "cancel").mockResolvedValue(CANCEL);
    useStore.setState({ user: "jordan" });

    render(<InferenceTab />);
    expect(await screen.findByText(HELD)).toBeInTheDocument();
    expect(screen.getByText(FAILED_JOB.job_id)).toBeInTheDocument();

    const stops = screen.getAllByRole("button", { name: "Cancel" });
    expect(stops).toHaveLength(1);
    fireEvent.click(stops[0]);
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith(HELD, { user: "jordan" }));
  });

  it("shows canceling until the poll reports the job ended", async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(inferenceApi.listJobs).mockResolvedValue(RUNNING);
      vi.spyOn(inferenceApi, "cancel").mockResolvedValue(CANCEL);

      render(<InferenceTab />);
      await vi.waitFor(() => expect(screen.getByText(HELD)).toBeInTheDocument());
      fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

      await vi.waitFor(() => expect(screen.getByText("canceling")).toBeInTheDocument());
      expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();

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

    render(<InferenceTab />);
    await screen.findByText(FAILED_JOB.job_id);
    watch(FAILED_JOB.job_id);
    expect(await screen.findByText(String(FAILED_JOB.error))).toBeInTheDocument();
  });

  /** Render the tab listing `listing`, watch the job `jobId` names and return its frame handler. */
  async function watching(listing: InferenceJobs, jobId: string) {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(listing);
    render(<InferenceTab />);
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
    expect(screen.queryByText(String(failing.error))).not.toBeInTheDocument();

    act(() => onFrame(FAILING_FINAL));

    expect(await screen.findByText(String(failing.error))).toBeInTheDocument();
  });

  it("shows the frame's error alone once the poll no longer lists the watched job", async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(inferenceApi.listJobs).mockResolvedValueOnce(RUNNING);

      render(<InferenceTab />);
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
    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
  }

  it("renders the refusal's own sentence under its date, with no toast", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    const sentence = `${DS_DATE}: ${REFUSED.message}`;
    expect(await screen.findByText(paragraphHolding(sentence))).toBeInTheDocument();
    expect(useStore.getState().toasts).toHaveLength(0);
  });

  it("names the job still writing the bucket and watches it", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue(RUNNING);
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    fireEvent.click(
      await screen.findByRole("button", {
        name: `Watch job ${REFUSED.job.job_id} for ${DS_DATE}`,
      }),
    );
    const watched = RUNNING.jobs.find((job) => job.job_id === REFUSED.job.job_id)!;
    expect(await screen.findByText(countsOf(watched))).toBeInTheDocument();
  });

  it("watches the job the refusal carries when the poll hasn't listed it yet", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    fireEvent.click(
      await screen.findByRole("button", {
        name: `Watch job ${REFUSED.job.job_id} for ${DS_DATE}`,
      }),
    );

    // The watched job's row names its bucket, beside the refused entry's own sentence.
    expect(await screen.findByText(REFUSED.job.bucket)).toBeInTheDocument();
    expect(screen.getByText(paragraphHolding(REFUSED.message))).toBeInTheDocument();
  });

  it("keeps one refused entry and one job row when one of two dates launches and the other refuses", async () => {
    mockTree();
    const refusal = await existsRefusal();
    vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      body.date === DS_DATE ? Promise.resolve(LAUNCHED) : Promise.reject(refusal),
    );

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("checkbox", { name: DS_UNPUBLISHED_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(await screen.findByText(HELD)).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: `Dismiss refusal for ${DS_UNPUBLISHED_DATE}` }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });

  it("keeps two refused dates' controls distinguishable by accessible name", async () => {
    mockTree();
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("checkbox", { name: DS_UNPUBLISHED_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(
      await screen.findByRole("button", { name: `Dismiss refusal for ${DS_DATE}` }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: `Dismiss refusal for ${DS_UNPUBLISHED_DATE}` }),
    ).toBeInTheDocument();
  });

  it("removes the entry on Dismiss", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    fireEvent.click(await screen.findByRole("button", { name: `Dismiss refusal for ${DS_DATE}` }));
    expect(screen.queryByText("Refused")).not.toBeInTheDocument();
  });

  it("still toasts, and renders no entry, for a launch failure of another kind", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(
      await thrown(INFERENCE_ANSWERS.missing_checkpoint),
    );
    await launchOneRefused();

    await waitFor(() => expect(useStore.getState().toasts).toHaveLength(1));
    expect(screen.queryByText("Refused")).not.toBeInTheDocument();
  });

  it("drops refused entries on a model change", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();
    expect(await screen.findByText("Refused")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
      target: { value: OTHER_MODEL.entry.checkpoint_path },
    });
    expect(screen.queryByText("Refused")).not.toBeInTheDocument();
  });

  it("drops refused entries on a dataset change", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();
    expect(await screen.findByText("Refused")).toBeInTheDocument();

    act(() => openTestProject(GROUPED_SELECTION));
    expect(screen.queryByText("Refused")).not.toBeInTheDocument();
  });

  it("disables the launch button while onLaunch's loop is in flight", async () => {
    const launch = Promise.withResolvers<InferenceJob>();
    vi.spyOn(inferenceApi, "launch").mockReturnValue(launch.promise);

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    const launchButton = screen.getByRole("button", { name: "Run" });
    fireEvent.click(launchButton);

    await waitFor(() => expect(launchButton).toBeDisabled());
    act(() => launch.resolve(LAUNCHED));
    await waitFor(() => expect(launchButton).not.toBeDisabled());
  });

  it("announces the refused-launches list and labels it without a level-one heading", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    const list = await screen.findByRole("list");
    expect(list).toHaveAttribute("aria-live", "polite");
    expect(screen.getByText("Refused").tagName).not.toBe("H1");
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });

  it("mounts the aria-live region before any refusal, with the entry appearing inside it once refused", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());

    const list = screen.getByRole("list");
    expect(list).toHaveAttribute("aria-live", "polite");
    expect(screen.queryAllByRole("listitem")).toHaveLength(0);

    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: DS_DATE }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(screen.getAllByRole("listitem")).toHaveLength(1));
    expect(screen.getByRole("list")).toHaveAttribute("aria-live", "polite");
  });

  it("moves focus to the launch button when Dismiss unmounts the entry that held it", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(await existsRefusal());
    await launchOneRefused();

    const dismiss = await screen.findByRole("button", { name: `Dismiss refusal for ${DS_DATE}` });
    dismiss.focus();
    fireEvent.click(dismiss);

    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole("button", { name: "Run" })),
    );
  });
});

describe("InferenceTab model select", () => {
  it("associates the model label with the select by accessible name", async () => {
    mockTree(false);
    mockModels();

    render(<InferenceTab />);

    expect(await screen.findByRole("combobox", { name: "Model" })).toBeInTheDocument();
  });
});

describe("InferenceTab heading", () => {
  it("renders exactly one top-level heading naming the tab", () => {
    render(<InferenceTab />);
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Inference");
  });
});
