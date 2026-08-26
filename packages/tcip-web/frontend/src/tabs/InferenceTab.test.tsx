import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { inferenceApi, openInferenceStream, resultsApi, type InferenceJob } from "@/api/inference";
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
    warning: null,
    ...overrides,
  };
}

function setupDataset() {
  useStore.setState((s) => ({
    gui: {
      ...s.gui,
      dataset: { ...s.gui.dataset, project_root: "C:/proj", dataset_root: "C:/data" },
    },
  }));
}

function mockTree(dates: string[]) {
  vi.spyOn(api.dataset, "tree").mockResolvedValue({
    dataset_root: "C:/data",
    dates_with_images: dates,
    subjects: ["subject_a"],
    model_names: [],
    subjects_by_date: {},
    models_by_date: {},
    prediction_dirs: {},
    label_problem: null,
  });
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
  it("launches one job per selected date, naming the bucket instead of spelling a path", async () => {
    mockTree(["2026-01-01", "2026-01-08"]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [{ name: "baseline", checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt" }],
    });
    const launchSpy = vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      Promise.resolve({
        status: "launched",
        job_id: `inf-${body.date}`,
        images_dir: `C:/data/images/${body.date}`,
        output_dir: `C:/data/predictions/${body.model_name}/${body.date}`,
        bucket_redirected: false,
        requested_output_dir: null,
      }),
    );

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "C:/proj/.tcip/models/baseline/best.pt" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-08" }));
    fireEvent.click(screen.getByRole("button", { name: /launch inference/i }));

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(2));
    expect(launchSpy.mock.calls.map(([body]) => body)).toEqual([
      expect.objectContaining({
        dataset_root: "C:/data",
        model_name: "baseline",
        date: "2026-01-01",
      }),
      expect.objectContaining({
        dataset_root: "C:/data",
        model_name: "baseline",
        date: "2026-01-08",
      }),
    ]);
  });

  it("says a dataset has no dates rather than showing an empty picker", async () => {
    mockTree([]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({ models: [] });

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText(/no capture dates yet/i)).toBeInTheDocument());
  });

  it("names the bucket a redirected run actually wrote to", async () => {
    mockTree(["2026-01-01"]);
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [{ name: "baseline", checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt" }],
    });
    vi.spyOn(inferenceApi, "launch").mockResolvedValue({
      status: "launched",
      job_id: "inf-1",
      images_dir: "C:/data/images/2026-01-01",
      output_dir: "C:/data/predictions/baseline/2026-01-01__rerun",
      bucket_redirected: true,
      requested_output_dir: "C:/data/predictions/baseline/2026-01-01",
    });

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "C:/proj/.tcip/models/baseline/best.pt" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("button", { name: /launch inference/i }));

    await waitFor(() => expect(useStore.getState().toasts).toHaveLength(1));
    const notice = useStore.getState().toasts[0];
    expect(notice.level).toBe("info");
    expect(notice.message).toContain("C:/data/predictions/baseline/2026-01-01__rerun");
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
      status: "cancelled",
      cancel_requested: true,
    });

    render(<InferenceTab />);
    expect(await screen.findByText("inf-live")).toBeInTheDocument();
    expect(screen.getByText("inf-done")).toBeInTheDocument();

    const stops = screen.getAllByRole("button", { name: "Cancel" });
    expect(stops).toHaveLength(1);
    fireEvent.click(stops[0]);
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("inf-live"));
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

  it("carries a live frame's warning and counts into the watched job panel", async () => {
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
        warning: "3 images carried no readable capture date",
      }),
    );

    expect(
      await screen.findByText(/3 images carried no readable capture date/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Status: running · 4 \/ 9/)).toBeInTheDocument();
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
