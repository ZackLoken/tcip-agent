import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { StructuredRefusalError } from "@/api/http";
import { trainingApi, type LaunchableConfig, type TrainingRunSummary } from "@/api/training";
import { useStore } from "@/store";
import { TrainingTab } from "@/tabs/TrainingTab";

// The live metrics stream owns a real WebSocket; only the run list and its controls are under
// test here, so the transport is replaced while the rest of the module stays real.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return { ...actual, openTrainingStream: vi.fn(() => () => {}) };
});

const initialStoreState = useStore.getState();

function run(overrides: Partial<TrainingRunSummary> & { run_id: string }): TrainingRunSummary {
  return {
    status: "running",
    ...overrides,
  };
}

function config(
  overrides: Partial<LaunchableConfig> & { experiment_id: string },
): LaunchableConfig {
  return {
    builder: "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
    task: "regression",
    images_dir: "/data/images",
    subject: "burr",
    created: "2026-08-01T00:00:00Z",
    state: "created",
    parent_experiment: null,
    ...overrides,
  };
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("TrainingTab run list", () => {
  it("offers a stop control for a run the platform reconstructed from another process, and cancels it by id", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-agent-1", status: "running", external: true })],
    });
    const cancelSpy = vi
      .spyOn(trainingApi, "cancel")
      .mockResolvedValue({ run_id: "train-agent-1", status: "running", cancel_requested: true });

    render(<TrainingTab />);
    expect(await screen.findByText("train-agent-1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("train-agent-1"));
  });

  it("offers no stop control for a run in a terminal status, external or not", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({ run_id: "train-done", status: "completed", external: false }),
        run({ run_id: "train-done-agent", status: "failed", external: true }),
      ],
    });

    render(<TrainingTab />);
    expect(await screen.findByText("train-done")).toBeInTheDocument();
    expect(screen.getByText("train-done-agent")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
  });
});

describe("TrainingTab config picker", () => {
  it("opens the picker, selects a config, and starts it through the relaunch door", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: [] });
    vi.spyOn(trainingApi, "listConfigs").mockResolvedValue({
      configs: [config({ experiment_id: "exp-pristine-1" })],
    });
    const relaunchSpy = vi
      .spyOn(trainingApi, "relaunch")
      .mockResolvedValue({ run_id: "run-new-1", experiment_id: "exp-pristine-1" });

    render(<TrainingTab />);
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    expect(await screen.findByText("exp-pristine-1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("exp-pristine-1"));
    expect(screen.getByText("Its first run, on the data paths it names")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(relaunchSpy).toHaveBeenCalledWith("exp-pristine-1", null));
  });

  it("shows a refused start's issues under the picked config's row", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: [] });
    vi.spyOn(trainingApi, "listConfigs").mockResolvedValue({
      configs: [config({ experiment_id: "exp-refused-1" })],
    });
    vi.spyOn(trainingApi, "relaunch").mockRejectedValue(
      new StructuredRefusalError({ issues: ["batch_size must be positive"] }, 422, ""),
    );

    render(<TrainingTab />);
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    expect(await screen.findByText("exp-refused-1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("exp-refused-1"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(screen.getByText("batch_size must be positive")).toBeInTheDocument(),
    );
  });
});
