import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { inferenceApi, resultsApi } from "@/api/inference";
import { useStore } from "@/store";
import { InferenceTab } from "@/tabs/InferenceTab";

const initialStoreState = useStore.getState();

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
    subjects: ["catkin"],
    model_names: [],
    subjects_by_date: {},
    models_by_date: {},
    prediction_dirs: {},
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
});
