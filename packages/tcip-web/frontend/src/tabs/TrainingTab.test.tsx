import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { trainingApi, type TrainingRunSummary } from "@/api/training";
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
