import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { StructuredRefusalError } from "@/api/http";
import { inferenceApi, openInferenceStream, resultsApi } from "@/api/inference";
import type {
  AssessmentSummary,
  BucketExistsRefusal,
  InferenceFrame,
  InferenceJob,
  RegisteredModel,
} from "@/api/types.generated";
import { useStore } from "@/store";
import { InferenceTab } from "@/tabs/InferenceTab";
import { mockDatasetTree } from "@/test/datasetTree";
import { GROUPED_SELECTION, openTestProject, RAW_SELECTION } from "@/test/store";

// The live job stream owns a real WebSocket; only its frame-to-state mapping is under test here,
// so the transport is replaced while the rest of the module stays real.
vi.mock("@/api/inference", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/inference")>();
  return { ...actual, openInferenceStream: vi.fn(() => () => {}) };
});

const initialStoreState = useStore.getState();

const BASELINE_MODEL: RegisteredModel = {
  name: "baseline",
  checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt",
  sha256: "baseline-digest",
  tags: [],
  experiment_id: null,
};

const PASSED_ASSESSMENT: AssessmentSummary = {
  assessment_id: "assessment-passed",
  checkpoint_sha256: "baseline-digest",
  experiment_id: null,
  delivery_kind: "per_image_count",
  passed: true,
};

function job(overrides: Partial<InferenceJob> & { job_id: string }): InferenceJob {
  return {
    project: "C:/proj",
    checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt",
    images_dir: "C:/data/images/2026-01-01",
    bucket: "baseline/2026-01-01",
    actor: "user:jordan",
    stated: {},
    assessment_id: null,
    total: 9,
    done: 1,
    status: "running",
    error: null,
    ...overrides,
  };
}

function setupDataset() {
  openTestProject(RAW_SELECTION);
}

function mockTree(dates: string[]) {
  mockDatasetTree({ dates_with_images: dates, buckets_by_date: {} });
}

function selectBaseline() {
  fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
    target: { value: "C:/proj/.tcip/models/baseline/best.pt" },
  });
  fireEvent.change(screen.getByLabelText("Predictions"), {
    target: { value: "baseline" },
  });
}

// The refusal's bucket renders in its own <span>, so the sentence's default per-node text (which
// excludes nested elements) never carries it whole; match on a <p>'s full textContent instead.
function paragraphMatching(pattern: RegExp) {
  return (_content: string, element: Element | null) =>
    element?.tagName === "P" && pattern.test(element.textContent ?? "");
}

function existsRefusal(live: InferenceJob = job({ job_id: "inf-live" })): StructuredRefusalError {
  const detail: BucketExistsRefusal = {
    kind: "bucket_exists",
    message: `bucket '${live.bucket}' under C:/data is the bucket job ${live.job_id} is still writing.`,
    job: live,
  };
  return new StructuredRefusalError(
    detail as unknown as Record<string, unknown>,
    409,
    detail.message,
  );
}

function launched(date: string, bucket: string) {
  return job({ job_id: `inf-${date}`, status: "pending", done: 0, total: 0, bucket });
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  setupDataset();
  vi.spyOn(inferenceApi, "listJobs").mockResolvedValue({ jobs: [] });
  vi.spyOn(resultsApi, "assessments").mockResolvedValue({ assessments: [] });
});

afterEach(cleanup);

describe("InferenceTab date selection", () => {
  it("launches one job per selected date, each its own bucket under the named one", async () => {
    mockTree(["2026-01-01", "2026-01-08"]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [BASELINE_MODEL],
    });
    const launchSpy = vi
      .spyOn(inferenceApi, "launch")
      .mockImplementation((body) => Promise.resolve(launched(body.date, body.bucket)));
    useStore.setState({ user: "jordan" });

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());

    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-08" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(2));
    expect(launchSpy.mock.calls.map(([body]) => body)).toEqual([
      {
        checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt",
        dataset_root: RAW_SELECTION.dataset_root,
        date: "2026-01-01",
        bucket: "baseline/2026-01-01",
        stated: {},
        assessment_id: null,
        user: "jordan",
      },
      {
        checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt",
        dataset_root: RAW_SELECTION.dataset_root,
        date: "2026-01-08",
        bucket: "baseline/2026-01-08",
        stated: {},
        assessment_id: null,
        user: "jordan",
      },
    ]);
  });

  it("states the conf, merge and merge threshold the breeder enters with their reason, and no cap", async () => {
    mockTree(["2026-01-01"]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [BASELINE_MODEL],
    });
    const launchSpy = vi
      .spyOn(inferenceApi, "launch")
      .mockImplementation((body) => Promise.resolve(launched(body.date, body.bucket)));

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());

    selectBaseline();
    fireEvent.change(screen.getByLabelText("Confidence"), { target: { value: "0.35" } });
    expect(screen.queryByLabelText(/detection cap/i)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Merge threshold"), { target: { value: "0.45" } });
    fireEvent.change(screen.getByLabelText("Merge"), { target: { value: "nmm" } });
    fireEvent.change(screen.getByLabelText("Reason"), {
      target: { value: "the breeder's own reason" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(1));
    expect(launchSpy.mock.calls[0][0].stated).toEqual({
      conf: 0.35,
      cross_tile_nms: 0.45,
      postprocess: "nmm",
      rationale: "the breeder's own reason",
    });
  });

  it("says a dataset has no dates rather than showing an empty picker", async () => {
    mockTree([]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({ models: [] });

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("No dates with images.")).toBeInTheDocument());
  });
});

describe("InferenceTab assessment", () => {
  beforeEach(() => {
    mockTree(["2026-01-01"]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({ models: [BASELINE_MODEL] });
  });

  it("runs under the model's first passed assessment with no stated inputs", async () => {
    vi.mocked(resultsApi.assessments).mockResolvedValue({
      assessments: [
        { ...PASSED_ASSESSMENT, assessment_id: "assessment-failed", passed: false },
        PASSED_ASSESSMENT,
        { ...PASSED_ASSESSMENT, assessment_id: "assessment-other", checkpoint_sha256: "other" },
      ],
    });
    const launchSpy = vi
      .spyOn(inferenceApi, "launch")
      .mockImplementation((body) => Promise.resolve(launched(body.date, body.bucket)));

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();

    const select = screen.getByRole("combobox", { name: "Assessment" });
    await waitFor(() => expect(select).toHaveValue("assessment-passed"));
    expect(screen.getByRole("option", { name: "assessment-failed (failed)" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /assessment-other/ })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Confidence")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Reason")).not.toBeInTheDocument();
    expect(screen.queryByText(/Every execution value/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(1));
    expect(launchSpy.mock.calls[0][0]).toMatchObject({
      assessment_id: "assessment-passed",
      stated: {},
    });
  });

  it("shows the stated inputs when no assessment passed", async () => {
    vi.mocked(resultsApi.assessments).mockResolvedValue({
      assessments: [{ ...PASSED_ASSESSMENT, passed: false }],
    });

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();

    expect(screen.getByRole("combobox", { name: "Assessment" })).toHaveValue("");
    expect(screen.getByLabelText("Confidence")).toBeInTheDocument();
    expect(screen.getByLabelText("Mask threshold")).toBeInTheDocument();
    expect(screen.getByLabelText("Ring spacing")).toBeInTheDocument();
    expect(screen.getByLabelText("Masks")).toBeInTheDocument();
  });
});

describe("InferenceTab job table", () => {
  beforeEach(() => {
    mockTree([]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({ models: [] });
  });

  it("offers a stop control only for a job still in flight, and cancels that job by id", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({
      jobs: [
        job({ job_id: "inf-live" }),
        job({ job_id: "inf-done", status: "completed", done: 9 }),
      ],
    });
    const cancelSpy = vi
      .spyOn(inferenceApi, "cancel")
      .mockResolvedValue(job({ job_id: "inf-live", status: "canceled" }));
    useStore.setState({ user: "jordan" });

    render(<InferenceTab />);
    expect(await screen.findByText("inf-live")).toBeInTheDocument();
    expect(screen.getByText("inf-done")).toBeInTheDocument();

    const stops = screen.getAllByRole("button", { name: "Cancel" });
    expect(stops).toHaveLength(1);
    fireEvent.click(stops[0]);
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("inf-live", "jordan"));
  });

  it("shows canceling until the poll reports the job ended", async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(inferenceApi.listJobs).mockResolvedValue({ jobs: [job({ job_id: "inf-live" })] });
      vi.spyOn(inferenceApi, "cancel").mockResolvedValue(job({ job_id: "inf-live" }));

      render(<InferenceTab />);
      await vi.waitFor(() => expect(screen.getByText("inf-live")).toBeInTheDocument());
      fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

      await vi.waitFor(() => expect(screen.getByText("canceling")).toBeInTheDocument());
      expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();

      // The next poll still reports running: the row keeps canceling rather than flipping back.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });
      expect(screen.getByText("canceling")).toBeInTheDocument();

      vi.mocked(inferenceApi.listJobs).mockResolvedValue({
        jobs: [job({ job_id: "inf-live", status: "canceled" })],
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });
      expect(screen.getByText("canceled")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows a failed job's reason, not just its status badge", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({
      jobs: [
        job({
          job_id: "inf-broken",
          status: "failed",
          done: 2,
          error: "checkpoint has 4 input channels, images have 3",
        }),
      ],
    });

    render(<InferenceTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Watch" }));
    expect(
      await screen.findByText(/checkpoint has 4 input channels, images have 3/),
    ).toBeInTheDocument();
  });

  it("carries a live frame's counts into the watched job panel", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({ jobs: [job({ job_id: "inf-live" })] });

    render(<InferenceTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Watch" }));
    await waitFor(() => expect(vi.mocked(openInferenceStream)).toHaveBeenCalled());

    const onFrame = vi.mocked(openInferenceStream).mock.calls[0][1];
    act(() => onFrame({ type: "progress", job: job({ job_id: "inf-live", done: 4 }) }));

    expect(await screen.findByText("running, 4 of 9")).toBeInTheDocument();
  });

  it("carries a final frame's error into the watched job panel", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({ jobs: [job({ job_id: "inf-live" })] });

    render(<InferenceTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Watch" }));
    await waitFor(() => expect(vi.mocked(openInferenceStream)).toHaveBeenCalled());

    const onFrame = vi.mocked(openInferenceStream).mock.calls[0][1];
    const failed: InferenceFrame = {
      type: "final",
      job: job({ job_id: "inf-live", status: "failed", error: "disk full" }),
    };
    act(() => onFrame(failed));

    expect(await screen.findByText("disk full")).toBeInTheDocument();
  });

  it("shows the frame's error alone once the poll no longer lists the watched job", async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(inferenceApi.listJobs).mockResolvedValueOnce({
        jobs: [job({ job_id: "inf-gone" })],
      });

      render(<InferenceTab />);
      await vi.waitFor(() => expect(screen.getByText("inf-gone")).toBeInTheDocument());
      fireEvent.click(screen.getByRole("button", { name: "Watch" }));

      const onFrame = vi.mocked(openInferenceStream).mock.calls[0][1];
      act(() => onFrame({ type: "final", error: "job not found" }));
      expect(screen.getByText("job not found")).toBeInTheDocument();
      expect(screen.getByText(/^running, /)).toBeInTheDocument();

      vi.mocked(inferenceApi.listJobs).mockResolvedValue({ jobs: [] });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });

      expect(screen.getByText("job not found")).toBeInTheDocument();
      expect(screen.queryByText(/^running, /)).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("InferenceTab bucket refusals", () => {
  beforeEach(() => {
    mockTree(["2026-01-01"]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [BASELINE_MODEL],
    });
  });

  async function launchOneRefused() {
    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
  }

  it("renders the job still writing the bucket and its name, with no toast", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();

    expect(
      await screen.findByText(
        paragraphMatching(/2026-01-01: job inf-live is still writing baseline\/2026-01-01\./),
      ),
    ).toBeInTheDocument();
    expect(useStore.getState().toasts).toHaveLength(0);
  });

  it("names the job still writing the bucket and watches it", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({
      jobs: [job({ job_id: "inf-live", status: "running" })],
    });
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();

    fireEvent.click(
      await screen.findByRole("button", { name: "Watch job inf-live for 2026-01-01" }),
    );
    expect(await screen.findByText("running, 1 of 9")).toBeInTheDocument();
  });

  it("watches the job the refusal carries when the poll hasn't listed it yet", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();

    fireEvent.click(
      await screen.findByRole("button", { name: "Watch job inf-live for 2026-01-01" }),
    );

    // Once in the refused entry, once as the watched job's bucket.
    expect(await screen.findAllByText("baseline/2026-01-01")).toHaveLength(2);
  });

  it("keeps one refused entry and one job row when one of two dates launches and the other refuses", async () => {
    mockTree(["2026-01-01", "2026-01-08"]);
    vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      body.date === "2026-01-01"
        ? Promise.resolve(launched(body.date, body.bucket))
        : Promise.reject(existsRefusal()),
    );

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-08" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(await screen.findByText("inf-2026-01-01")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: "Dismiss refusal for 2026-01-08" }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });

  it("keeps two refused dates' controls distinguishable by accessible name", async () => {
    mockTree(["2026-01-01", "2026-01-08"]);
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-08" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(
      await screen.findByRole("button", { name: "Dismiss refusal for 2026-01-01" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Dismiss refusal for 2026-01-08" }),
    ).toBeInTheDocument();
  });

  it("removes the entry on Dismiss", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();

    fireEvent.click(await screen.findByRole("button", { name: "Dismiss refusal for 2026-01-01" }));
    expect(screen.queryByText("Refused")).not.toBeInTheDocument();
  });

  it("still toasts, and renders no entry, for a launch failure of another kind", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(new Error("checkpoint not found: C:/x"));
    await launchOneRefused();

    await waitFor(() => expect(useStore.getState().toasts).toHaveLength(1));
    expect(screen.queryByText("Refused")).not.toBeInTheDocument();
  });

  it("drops refused entries on a model change", async () => {
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [
        BASELINE_MODEL,
        {
          name: "other",
          checkpoint_path: "C:/proj/.tcip/models/other/best.pt",
          sha256: "other-digest",
          tags: [],
          experiment_id: null,
        },
      ],
    });
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();
    expect(await screen.findByText("Refused")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
      target: { value: "C:/proj/.tcip/models/other/best.pt" },
    });
    expect(screen.queryByText("Refused")).not.toBeInTheDocument();
  });

  it("drops refused entries on a dataset change", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();
    expect(await screen.findByText("Refused")).toBeInTheDocument();

    act(() => openTestProject(GROUPED_SELECTION));
    expect(screen.queryByText("Refused")).not.toBeInTheDocument();
  });

  it("disables the launch button while onLaunch's loop is in flight", async () => {
    const launch = Promise.withResolvers<ReturnType<typeof launched>>();
    vi.spyOn(inferenceApi, "launch").mockReturnValue(launch.promise);

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    const launchButton = screen.getByRole("button", { name: "Run" });
    fireEvent.click(launchButton);

    await waitFor(() => expect(launchButton).toBeDisabled());
    act(() => launch.resolve(launched("2026-01-01", "baseline")));
    await waitFor(() => expect(launchButton).not.toBeDisabled());
  });

  it("announces the refused-launches list and labels it without a level-one heading", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();

    const list = await screen.findByRole("list");
    expect(list).toHaveAttribute("aria-live", "polite");
    expect(screen.getByText("Refused").tagName).not.toBe("H1");
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });

  it("mounts the aria-live region before any refusal, with the entry appearing inside it once refused", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());

    const list = screen.getByRole("list");
    expect(list).toHaveAttribute("aria-live", "polite");
    expect(screen.queryAllByRole("listitem")).toHaveLength(0);

    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(screen.getAllByRole("listitem")).toHaveLength(1));
    expect(screen.getByRole("list")).toHaveAttribute("aria-live", "polite");
  });

  it("moves focus to the launch button when Dismiss unmounts the entry that held it", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();

    const dismiss = await screen.findByRole("button", { name: "Dismiss refusal for 2026-01-01" });
    dismiss.focus();
    fireEvent.click(dismiss);

    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole("button", { name: "Run" })),
    );
  });
});

describe("InferenceTab model select", () => {
  it("associates the model label with the select by accessible name", async () => {
    mockTree([]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [BASELINE_MODEL],
    });

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
