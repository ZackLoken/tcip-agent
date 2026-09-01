import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { tuningApi, type Sweep } from "@/api/tuning";
import { useStore } from "@/store";
import { TuningTab } from "@/tabs/TuningTab";

const initialStoreState = useStore.getState();

function sweep(overrides: Partial<Sweep> & { sweep_id: string }): Sweep {
  return {
    status: "running",
    error: null,
    has_result: false,
    relaunchable: false,
    reason: null,
    cancel_requested: false,
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

describe("TuningTab sweep row actions", () => {
  it("shows Cancel on a running sweep row without expanding it", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-running-1", status: "running" })],
    });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-running-1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
  });

  it("shows no action on a finished, non-relaunchable sweep", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({
          sweep_id: "hpo-done-1",
          status: "completed",
          relaunchable: false,
          reason: "this sweep's record holds no base config",
        }),
      ],
    });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-done-1")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run again" })).not.toBeInTheDocument();
  });

  it("shows Run again on a relaunchable terminal sweep, and relaunches it on click", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-done-2", status: "completed", relaunchable: true })],
    });
    const relaunchSpy = vi
      .spyOn(tuningApi, "relaunch")
      .mockResolvedValue({ status: "launched", sweep_id: "hpo-new-1" });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-done-2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run again" }));
    await waitFor(() => expect(relaunchSpy).toHaveBeenCalledWith("hpo-done-2"));
  });

  it("renders the not-relaunchable reason in the collapsed row header, without expanding", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({
          sweep_id: "hpo-noreason",
          status: "completed",
          relaunchable: false,
          reason: "this sweep's record holds no base config",
        }),
      ],
    });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-noreason")).toBeInTheDocument();
    expect(screen.getByText("this sweep's record holds no base config")).toBeInTheDocument();
  });

  it("renders a sweep's error under its status line, whatever the status", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({
          sweep_id: "hpo-failed-1",
          status: "failed",
          error: "the sweep's base config fails preflight",
        }),
      ],
    });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-failed-1")).toBeInTheDocument();
    expect(screen.getByText("the sweep's base config fails preflight")).toBeInTheDocument();
  });

  it("shows the stop-requested state as status text after a cancel, and no more Cancel button", async () => {
    vi.spyOn(tuningApi, "listSweeps")
      .mockResolvedValueOnce({
        sweeps: [sweep({ sweep_id: "hpo-cancelme", status: "running", cancel_requested: false })],
      })
      .mockResolvedValue({
        sweeps: [sweep({ sweep_id: "hpo-cancelme", status: "running", cancel_requested: true })],
      });
    const cancelSpy = vi
      .spyOn(tuningApi, "cancel")
      .mockResolvedValue({ study_name: "hpo-cancelme", status: "running", cancel_requested: true });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-cancelme")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("hpo-cancelme"));

    expect(await screen.findByText(/stop requested/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
  });

  it("keeps offering Cancel on an external sweep even after stop requested", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({
          sweep_id: "hpo-external-1",
          status: "running",
          external: true,
          cancel_requested: true,
        }),
      ],
    });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-external-1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
  });

  it("shows the sweep's search shape in the expanded region", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({
          sweep_id: "hpo-search-1",
          status: "running",
          n_trials: 5,
          search_alg: "bayesopt",
          scheduler: "median",
          param_space_keys: ["lr", "batch_size"],
        }),
      ],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue({
      sweep_id: "hpo-search-1",
      status: "running",
      result: {},
    });
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-search-1", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-search-1"));
    expect(await screen.findByText(/bayesopt/)).toBeInTheDocument();
    expect(screen.getByText(/median/)).toBeInTheDocument();
    expect(screen.getByText(/lr, batch_size/)).toBeInTheDocument();
  });

  it("shows one cancelled line, not the manifest stub, for a cancelled sweep's detail", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-cxl-detail", status: "cancelled" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue({
      sweep_id: "hpo-cxl-detail",
      status: "cancelled",
      error: "the sweep was cancelled by request before it could finish",
      result: { status: "cancelled", study_name: "hpo-cxl-detail" },
    });
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-cxl-detail", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-cxl-detail"));
    expect(
      await screen.findByText(
        "Cancelled: the sweep was cancelled by request before it could finish",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/"status": "cancelled"/)).not.toBeInTheDocument();
  });
});
