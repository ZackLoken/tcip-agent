import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import {
  inferenceApi,
  openInferenceStream,
  resultsApi,
  type BucketExistsRefusal,
  type InferenceJob,
} from "@/api/inference";
import { StructuredRefusalError } from "@/api/http";
import { useStore } from "@/store";
import { InferenceTab } from "@/tabs/InferenceTab";

// The live job stream owns a real WebSocket; only its frame-to-state mapping is under test here,
// so the transport is replaced while the rest of the module stays real.
vi.mock("@/api/inference", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/inference")>();
  return { ...actual, openInferenceStream: vi.fn(() => () => {}) };
});

const initialStoreState = useStore.getState();

function job(overrides: Partial<InferenceJob> & { job_id: string }): InferenceJob {
  return {
    status: "running",
    done: 1,
    total: 9,
    images_dir: "C:/data/images/2026-01-01",
    output_dir: "C:/data/predictions/baseline/2026-01-01",
    error: null,
    audit_warning: null,
    ...overrides,
  };
}

function setupDataset() {
  useStore.setState((s) => ({
    gui: {
      ...s.gui,
      dataset: { ...s.gui.dataset, dataset_root: "C:/data" },
    },
    openProject: { id: "a1b2c3d4e5f6", path: "C:/proj" },
  }));
}

function mockTree(dates: string[]) {
  vi.spyOn(api.dataset, "tree").mockResolvedValue({
    dataset_root: "C:/data",
    dates_with_images: dates,
    subjects: ["subject_a"],
    subjects_by_date: {},
    prediction_dirs: {},
    label_problem: null,
  });
}

function selectBaseline() {
  fireEvent.change(screen.getByRole("combobox"), {
    target: { value: "C:/proj/.tcip/models/baseline/best.pt" },
  });
  fireEvent.change(screen.getByLabelText("Bucket directory"), {
    target: { value: "C:/data/predictions/baseline" },
  });
}

// The refusal's path renders in its own <span>, so the sentence's default per-node text (which
// excludes nested elements) never carries it whole; match on a <p>'s full textContent instead.
function paragraphMatching(pattern: RegExp) {
  return (_content: string, element: Element | null) =>
    element?.tagName === "P" && pattern.test(element.textContent ?? "");
}

function existsRefusal(overrides: Partial<BucketExistsRefusal> = {}): StructuredRefusalError {
  const detail: BucketExistsRefusal = {
    kind: "bucket_exists",
    message:
      "C:/data/predictions/baseline/2026-01-01 already exists: a bucket is published once, and " +
      "a new run names a new bucket.",
    date: "2026-01-01",
    requested_output_dir: "C:/data/predictions/baseline/2026-01-01",
    job_id: "inf-live",
    ...overrides,
  };
  return new StructuredRefusalError(
    detail as unknown as Record<string, unknown>,
    409,
    detail.message,
  );
}

function launched(date: string, outputDir: string) {
  return {
    status: "launched",
    job_id: `inf-${date}`,
    images_dir: `C:/data/images/${date}`,
    output_dir: outputDir,
  };
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  setupDataset();
  vi.spyOn(inferenceApi, "listJobs").mockResolvedValue({ jobs: [] });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("InferenceTab date selection", () => {
  it("launches one job per selected date, each into its own child of the named bucket directory", async () => {
    mockTree(["2026-01-01", "2026-01-08"]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [{ name: "baseline", checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt" }],
    });
    const launchSpy = vi
      .spyOn(inferenceApi, "launch")
      .mockImplementation((body) => Promise.resolve(launched(body.date, body.output_dir)));
    useStore.setState({ user: "jordan" });

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());

    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-08" }));
    fireEvent.click(screen.getByRole("button", { name: /launch inference/i }));

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(2));
    expect(launchSpy.mock.calls.map(([body]) => body)).toEqual([
      {
        checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt",
        dataset_root: "C:/data",
        date: "2026-01-01",
        output_dir: "C:/data/predictions/baseline/2026-01-01",
        stated: {},
        assessment_id: null,
        user: "jordan",
      },
      {
        checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt",
        dataset_root: "C:/data",
        date: "2026-01-08",
        output_dir: "C:/data/predictions/baseline/2026-01-08",
        stated: {},
        assessment_id: null,
        user: "jordan",
      },
    ]);
  });

  it("says a dataset has no dates rather than showing an empty picker", async () => {
    mockTree([]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({ models: [] });

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(/no capture dates yet/i)).toBeInTheDocument());
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
    const cancelSpy = vi.spyOn(inferenceApi, "cancel").mockResolvedValue({
      job_id: "inf-live",
      status: "canceled",
      cancel_requested: true,
    });
    useStore.setState({ user: "jordan" });

    render(<InferenceTab />);
    expect(await screen.findByText("inf-live")).toBeInTheDocument();
    expect(screen.getByText("inf-done")).toBeInTheDocument();

    const stops = screen.getAllByRole("button", { name: "Cancel" });
    expect(stops).toHaveLength(1);
    fireEvent.click(stops[0]);
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("inf-live", "jordan"));
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
    act(() =>
      onFrame({
        type: "progress",
        done: 4,
        total: 9,
        status: "running",
        error: null,
      }),
    );

    expect(await screen.findByText(/Status: running · 4 \/ 9/)).toBeInTheDocument();
  });

  it("carries a final frame's audit_warning into the watched job panel", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({ jobs: [job({ job_id: "inf-live" })] });

    render(<InferenceTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Watch" }));
    await waitFor(() => expect(vi.mocked(openInferenceStream)).toHaveBeenCalled());

    const onFrame = vi.mocked(openInferenceStream).mock.calls[0][1];
    act(() =>
      onFrame({
        type: "final",
        status: "completed",
        error: null,
        audit_warning:
          "prediction_bucket_published completed and its audit entry could not be written",
      }),
    );

    expect(await screen.findByText(/its audit entry could not be written/)).toBeInTheDocument();
  });

  it("carries a final frame's error into the watched job panel", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({ jobs: [job({ job_id: "inf-live" })] });

    render(<InferenceTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Watch" }));
    await waitFor(() => expect(vi.mocked(openInferenceStream)).toHaveBeenCalled());

    const onFrame = vi.mocked(openInferenceStream).mock.calls[0][1];
    act(() => onFrame({ type: "final", status: "failed", error: "job not found" }));

    expect(await screen.findByText(/Error: job not found/)).toBeInTheDocument();
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
      expect(screen.getByText(/Error: job not found/)).toBeInTheDocument();
      expect(screen.getByText(/Status: running/)).toBeInTheDocument();

      vi.mocked(inferenceApi.listJobs).mockResolvedValue({ jobs: [] });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });

      expect(screen.getByText(/Error: job not found/)).toBeInTheDocument();
      expect(screen.queryByText(/Status:/)).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("InferenceTab bucket refusals", () => {
  beforeEach(() => {
    mockTree(["2026-01-01"]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [{ name: "baseline", checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt" }],
    });
  });

  async function launchOneRefused() {
    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("button", { name: /launch inference/i }));
  }

  it("renders the job still writing the bucket and its path, with no toast", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();

    expect(
      await screen.findByText(
        paragraphMatching(
          /2026-01-01: job inf-live is still writing the bucket at C:\/data\/predictions\/baseline\/2026-01-01\./,
        ),
      ),
    ).toBeInTheDocument();
    expect(useStore.getState().toasts).toHaveLength(0);
  });

  it("names the job still writing the bucket and watches it", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({
      jobs: [job({ job_id: "inf-live", status: "running" })],
    });
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal({ job_id: "inf-live" }));
    await launchOneRefused();

    fireEvent.click(
      await screen.findByRole("button", { name: "Watch job inf-live for 2026-01-01" }),
    );
    expect(await screen.findByText(/Status: running/)).toBeInTheDocument();
  });

  it("seeds the watched stub's output_dir from the refusal when the poll hasn't listed the job yet", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal({ job_id: "inf-live" }));
    await launchOneRefused();

    fireEvent.click(
      await screen.findByRole("button", { name: "Watch job inf-live for 2026-01-01" }),
    );

    // Once in the refused entry, once as the watched job's output.
    expect(await screen.findAllByText("C:/data/predictions/baseline/2026-01-01")).toHaveLength(2);
  });

  it("keeps one refused entry and one job row when one of two dates launches and the other refuses", async () => {
    mockTree(["2026-01-01", "2026-01-08"]);
    vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      body.date === "2026-01-01"
        ? Promise.resolve(launched(body.date, body.output_dir))
        : Promise.reject(existsRefusal({ date: body.date })),
    );

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-08" }));
    fireEvent.click(screen.getByRole("button", { name: /launch inference/i }));

    expect(await screen.findByText("inf-2026-01-01")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: "Dismiss refusal for 2026-01-08" }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });

  it("keeps two refused dates' controls distinguishable by accessible name", async () => {
    mockTree(["2026-01-01", "2026-01-08"]);
    vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      Promise.reject(existsRefusal({ date: body.date ?? undefined })),
    );

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-08" }));
    fireEvent.click(screen.getByRole("button", { name: /launch inference/i }));

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
    expect(screen.queryByText("Refused launches")).not.toBeInTheDocument();
  });

  it("still toasts, and renders no entry, for a launch failure of another kind", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(new Error("checkpoint not found: C:/x"));
    await launchOneRefused();

    await waitFor(() => expect(useStore.getState().toasts).toHaveLength(1));
    expect(screen.queryByText("Refused launches")).not.toBeInTheDocument();
  });

  it("drops refused entries on a model change", async () => {
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [
        { name: "baseline", checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt" },
        { name: "other", checkpoint_path: "C:/proj/.tcip/models/other/best.pt" },
      ],
    });
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();
    expect(await screen.findByText("Refused launches")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "C:/proj/.tcip/models/other/best.pt" },
    });
    expect(screen.queryByText("Refused launches")).not.toBeInTheDocument();
  });

  it("drops refused entries on a dataset change", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();
    expect(await screen.findByText("Refused launches")).toBeInTheDocument();

    act(() => {
      useStore.setState((s) => ({
        gui: { ...s.gui, dataset: { ...s.gui.dataset, dataset_root: "C:/data2" } },
      }));
    });
    expect(screen.queryByText("Refused launches")).not.toBeInTheDocument();
  });

  it("disables the launch button while onLaunch's loop is in flight", async () => {
    let resolveLaunch: (value: ReturnType<typeof launched>) => void = () => {};
    vi.spyOn(inferenceApi, "launch").mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveLaunch = resolve;
        }),
    );

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    const launchButton = screen.getByRole("button", { name: /launch inference/i });
    fireEvent.click(launchButton);

    await waitFor(() => expect(launchButton).toBeDisabled());
    act(() => resolveLaunch(launched("2026-01-01", "baseline")));
    await waitFor(() => expect(launchButton).not.toBeDisabled());
  });

  it("announces the refused-launches list and labels it without a level-one heading", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(existsRefusal());
    await launchOneRefused();

    const list = await screen.findByRole("list");
    expect(list).toHaveAttribute("aria-live", "polite");
    expect(screen.getByText("Refused launches").tagName).not.toBe("H1");
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
    fireEvent.click(screen.getByRole("button", { name: /launch inference/i }));

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
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: /launch inference/i }),
      ),
    );
  });
});

describe("InferenceTab model select", () => {
  it("associates the model checkpoint label with the select by accessible name", async () => {
    mockTree([]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [{ name: "baseline", checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt" }],
    });

    render(<InferenceTab />);

    expect(await screen.findByRole("combobox", { name: "Model checkpoint" })).toBeInTheDocument();
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
