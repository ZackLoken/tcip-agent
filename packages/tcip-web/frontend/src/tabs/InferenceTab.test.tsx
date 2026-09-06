import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import {
  inferenceApi,
  openInferenceStream,
  resultsApi,
  type BucketHoldsDocumentsRefusal,
  type BucketInFlightRefusal,
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

function selectBaseline() {
  fireEvent.change(screen.getByRole("combobox"), {
    target: { value: "C:/proj/.tcip/models/baseline/best.pt" },
  });
}

function documentsRefusal(
  overrides: Partial<BucketHoldsDocumentsRefusal> = {},
): StructuredRefusalError {
  const detail: BucketHoldsDocumentsRefusal = {
    kind: "bucket_holds_documents",
    message: "prediction bucket 'baseline' already holds 1 prediction document(s).",
    date: "2026-01-01",
    requested_model_name: "baseline",
    requested_output_dir: "C:/data/predictions/baseline/2026-01-01",
    document_stem_count: 1,
    suggested_model_name: "baseline@r2",
    suggested_output_dir: "C:/data/predictions/baseline@r2/2026-01-01",
    ...overrides,
  };
  return new StructuredRefusalError(
    detail as unknown as Record<string, unknown>,
    409,
    detail.message,
  );
}

function inFlightRefusal(overrides: Partial<BucketInFlightRefusal> = {}): StructuredRefusalError {
  const detail: BucketInFlightRefusal = {
    kind: "bucket_in_flight",
    message: "job inf-live is already writing to C:/data/predictions/baseline/2026-01-01.",
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

  it("renders the requested path, the count and a date- and suggestion-named action, with no toast", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(documentsRefusal());
    await launchOneRefused();

    expect(
      await screen.findByText(
        /C:\/data\/predictions\/baseline\/2026-01-01 already holds 1 prediction document/,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Run into baseline@r2 instead for 2026-01-01" }),
    ).toBeInTheDocument();
    expect(useStore.getState().toasts).toHaveLength(0);
  });

  it("re-posts the suggested bucket from the entry's own action, clearing it on success", async () => {
    const launchSpy = vi
      .spyOn(inferenceApi, "launch")
      .mockRejectedValueOnce(documentsRefusal())
      .mockResolvedValueOnce({
        status: "launched",
        job_id: "inf-r2",
        images_dir: "C:/data/images/2026-01-01",
        output_dir: "C:/data/predictions/baseline@r2/2026-01-01",
        bucket_redirected: false,
        requested_output_dir: null,
      });
    await launchOneRefused();

    fireEvent.click(
      await screen.findByRole("button", { name: "Run into baseline@r2 instead for 2026-01-01" }),
    );

    await waitFor(() => expect(launchSpy).toHaveBeenCalledTimes(2));
    expect(launchSpy.mock.calls[1][0]).toEqual(
      expect.objectContaining({
        checkpoint_path: "C:/proj/.tcip/models/baseline/best.pt",
        dataset_root: "C:/data",
        model_name: "baseline@r2",
        date: "2026-01-01",
      }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Run into baseline@r2 instead for 2026-01-01" }),
      ).not.toBeInTheDocument(),
    );
    expect(await screen.findByText("inf-r2")).toBeInTheDocument();
  });

  it("replaces the entry with a fresh suggestion on a second refusal from its own action", async () => {
    vi.spyOn(inferenceApi, "launch")
      .mockRejectedValueOnce(documentsRefusal())
      .mockRejectedValueOnce(
        documentsRefusal({
          requested_model_name: "baseline@r2",
          requested_output_dir: "C:/data/predictions/baseline@r2/2026-01-01",
          suggested_model_name: "baseline@r3",
          suggested_output_dir: "C:/data/predictions/baseline@r3/2026-01-01",
        }),
      );
    await launchOneRefused();

    fireEvent.click(
      await screen.findByRole("button", { name: "Run into baseline@r2 instead for 2026-01-01" }),
    );

    expect(
      await screen.findByRole("button", { name: "Run into baseline@r3 instead for 2026-01-01" }),
    ).toBeInTheDocument();
  });

  it("removes the entry and toasts when its own action fails a different way", async () => {
    vi.spyOn(inferenceApi, "launch")
      .mockRejectedValueOnce(documentsRefusal())
      .mockRejectedValueOnce(new Error("checkpoint not found"));
    await launchOneRefused();

    fireEvent.click(
      await screen.findByRole("button", { name: "Run into baseline@r2 instead for 2026-01-01" }),
    );

    await waitFor(() => expect(useStore.getState().toasts).toHaveLength(1));
    expect(useStore.getState().toasts[0].message).toContain("checkpoint not found");
    expect(screen.queryByText("Refused launches")).not.toBeInTheDocument();
  });

  it("renders the agent's own remedy and no launch action when no fresh bucket exists", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(
      documentsRefusal({ suggested_model_name: null, suggested_output_dir: null }),
    );
    await launchOneRefused();

    expect(await screen.findByText(/run_inference/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Run into/ })).not.toBeInTheDocument();
  });

  it("renders an in-flight refusal's job id and watches it", async () => {
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({
      jobs: [job({ job_id: "inf-live", status: "running" })],
    });
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(inFlightRefusal());
    await launchOneRefused();

    expect(await screen.findByText(/inf-live/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Watch job inf-live for 2026-01-01" }));
    expect(await screen.findByText(/Status: running/)).toBeInTheDocument();
  });

  it("keeps one refused entry and one job row when one of two dates launches and the other refuses", async () => {
    mockTree(["2026-01-01", "2026-01-08"]);
    vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      body.date === "2026-01-01"
        ? Promise.resolve({
            status: "launched",
            job_id: "inf-1",
            images_dir: `C:/data/images/${body.date}`,
            output_dir: `C:/data/predictions/${body.model_name}/${body.date}`,
            bucket_redirected: false,
            requested_output_dir: null,
          })
        : Promise.reject(documentsRefusal({ date: body.date ?? undefined })),
    );

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-08" }));
    fireEvent.click(screen.getByRole("button", { name: /launch inference/i }));

    expect(await screen.findByText("inf-1")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: "Run into baseline@r2 instead for 2026-01-08" }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });

  it("keeps two refused dates' controls distinguishable by accessible name", async () => {
    mockTree(["2026-01-01", "2026-01-08"]);
    vi.spyOn(inferenceApi, "launch").mockImplementation((body) =>
      Promise.reject(documentsRefusal({ date: body.date ?? undefined })),
    );

    render(<InferenceTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    selectBaseline();
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-01" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2026-01-08" }));
    fireEvent.click(screen.getByRole("button", { name: /launch inference/i }));

    expect(
      await screen.findByRole("button", { name: "Run into baseline@r2 instead for 2026-01-01" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Run into baseline@r2 instead for 2026-01-08" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Dismiss refusal for 2026-01-01" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Dismiss refusal for 2026-01-08" }),
    ).toBeInTheDocument();
  });

  it("removes the entry on Dismiss", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(documentsRefusal());
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
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(documentsRefusal());
    await launchOneRefused();
    expect(await screen.findByText("Refused launches")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "C:/proj/.tcip/models/other/best.pt" },
    });
    expect(screen.queryByText("Refused launches")).not.toBeInTheDocument();
  });

  it("drops refused entries on a dataset change", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(documentsRefusal());
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
    let resolveLaunch: (value: {
      status: string;
      job_id: string;
      images_dir: string;
      output_dir: string;
      bucket_redirected: boolean;
      requested_output_dir: string | null;
    }) => void = () => {};
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
    act(() =>
      resolveLaunch({
        status: "launched",
        job_id: "inf-1",
        images_dir: "C:/data/images/2026-01-01",
        output_dir: "C:/data/predictions/baseline/2026-01-01",
        bucket_redirected: false,
        requested_output_dir: null,
      }),
    );
    await waitFor(() => expect(launchButton).not.toBeDisabled());
  });

  it("announces the refused-launches list and labels it without a level-one heading", async () => {
    vi.spyOn(inferenceApi, "launch").mockRejectedValue(documentsRefusal());
    await launchOneRefused();

    const list = await screen.findByRole("list");
    expect(list).toHaveAttribute("aria-live", "polite");
    expect(screen.getByText("Refused launches").tagName).not.toBe("H1");
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
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
