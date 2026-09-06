import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { StructuredRefusalError } from "@/api/http";
import { tuningApi, type Sweep, type SweepDetail } from "@/api/tuning";
import { EMBEDDED_TOOL_RETRY_MS } from "@/hooks/useEmbeddedToolRetry";
import { useStore } from "@/store";
import { TuningTab } from "@/tabs/TuningTab";
import { RUN_REFRESH_MS } from "@/tabs/trainingMetrics";

const initialStoreState = useStore.getState();

function sweep(overrides: Partial<Sweep> & { sweep_id: string }): Sweep {
  return {
    status: "running",
    error: null,
    has_result: false,
    has_manifest: true,
    relaunchable: false,
    reason: null,
    cancel_requested: false,
    ...overrides,
  };
}

function sweepDetail(overrides: Partial<SweepDetail> & { sweep_id: string }): SweepDetail {
  return {
    status: "running",
    result: {},
    has_manifest: true,
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
    expect(screen.getByRole("button", { name: "Cancel hpo-running-1" })).toBeInTheDocument();
  });

  it("names each sweep's Cancel and Run again controls with that sweep's own id", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({ sweep_id: "hpo-a", status: "running" }),
        sweep({ sweep_id: "hpo-b", status: "completed", relaunchable: true }),
      ],
    });

    render(<TuningTab />);
    await screen.findByText("hpo-a");

    expect(screen.getByRole("button", { name: "Cancel hpo-a" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run again hpo-b" })).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: "Run again hpo-done-2" }));
    await waitFor(() => expect(relaunchSpy).toHaveBeenCalledWith("hpo-done-2"));
  });

  it("shows no Cancel control on an interrupted sweep row, only Run again when relaunchable", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-interrupted-1", status: "interrupted", relaunchable: true })],
    });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-interrupted-1")).toBeInTheDocument();
    expect(screen.getByText("interrupted")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Cancel hpo-interrupted-1" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run again hpo-interrupted-1" })).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: "Cancel hpo-cancelme" }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("hpo-cancelme"));

    expect(await screen.findByText(/stop requested/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel hpo-cancelme" })).not.toBeInTheDocument();
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
    expect(screen.getByRole("button", { name: "Cancel hpo-external-1" })).toBeInTheDocument();
  });

  it("shows which sweep a relaunched row was relaunched from", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({
          sweep_id: "hpo-relaunched-1",
          status: "running",
          relaunched_from: "hpo-source-1",
        }),
      ],
    });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-relaunched-1")).toBeInTheDocument();
    expect(screen.getByText(/relaunched from hpo-source-1/)).toBeInTheDocument();
  });

  it("shows no relaunched-from line for a sweep that was not a relaunch", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-not-relaunched-1", status: "running" })],
    });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-not-relaunched-1")).toBeInTheDocument();
    expect(screen.queryByText(/relaunched from/)).not.toBeInTheDocument();
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
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-search-1", status: "running" }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-search-1", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-search-1"));
    expect(await screen.findByText(/bayesopt/)).toBeInTheDocument();
    expect(screen.getByText(/median/)).toBeInTheDocument();
    expect(screen.getByText(/lr, batch_size/)).toBeInTheDocument();
    expect(screen.getByText(/5 trials planned/)).toBeInTheDocument();
  });

  it("states the draw count on the header line when the manifest carries split_draws", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-draws-1", status: "running", n_trials: 4, split_draws: 3 })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-draws-1", status: "running" }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-draws-1", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-draws-1"));
    expect(await screen.findByText(/4 trials planned, 3 draws each/)).toBeInTheDocument();
  });

  it("names the bound manifest on the draws line when the base config redraws inside it", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({
          sweep_id: "hpo-redraw-1",
          status: "running",
          n_trials: 4,
          split_draws: 2,
          redraws_within_manifest: true,
        }),
      ],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-redraw-1", status: "running" }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-redraw-1", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-redraw-1"));
    expect(
      await screen.findByText(/4 trials planned, 2 draws each inside the bound manifest/),
    ).toBeInTheDocument();
  });

  it("states only the trial count when split_draws is 1 or absent", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-no-draws-1", status: "running", n_trials: 4 })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-no-draws-1", status: "running" }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-no-draws-1", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-no-draws-1"));
    expect(await screen.findByText(/4 trials planned/)).toBeInTheDocument();
    expect(screen.queryByText(/draws each/)).not.toBeInTheDocument();
  });

  it("shows one cancelled line, not the manifest stub, for a cancelled sweep's detail", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-cxl-detail", status: "cancelled" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({
        sweep_id: "hpo-cxl-detail",
        status: "cancelled",
        error: "the sweep was cancelled by request before it could finish",
        result: { status: "cancelled", study_name: "hpo-cxl-detail" },
      }),
    );
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

  it("names a cancelled sweep's own missing reason the same way in the row and the detail", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-cxl-noreason", status: "cancelled", error: null })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-cxl-noreason", status: "cancelled", error: null }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({
      sweep_id: "hpo-cxl-noreason",
      trials: [],
    });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-cxl-noreason")).toBeInTheDocument();
    expect(screen.getByText("no reason recorded")).toBeInTheDocument();

    fireEvent.click(screen.getByText("hpo-cxl-noreason"));
    expect(await screen.findByText("Cancelled: no reason recorded")).toBeInTheDocument();
  });

  it("carries aria-expanded, not aria-pressed, on a sweep row's own disclosure toggle", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-toggle-1", status: "running" })],
    });

    render(<TuningTab />);
    const toggle = (await screen.findByText("hpo-toggle-1")).closest("button") as HTMLElement;
    expect(toggle).toHaveAttribute("aria-expanded");
    expect(toggle).not.toHaveAttribute("aria-pressed");
  });

  it("names the row's disclosure toggle with the id and status alone, and describes the relaunch source and the error separately", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({
          sweep_id: "hpo-described-1",
          status: "failed",
          error: "the sweep's base config fails preflight",
          relaunched_from: "hpo-source-1",
        }),
      ],
    });

    render(<TuningTab />);
    const toggle = await screen.findByRole("button", { name: "hpo-described-1 failed" });
    const source = screen.getByText("relaunched from hpo-source-1");
    const error = screen.getByText("the sweep's base config fails preflight");
    expect(toggle).toHaveAttribute("aria-describedby", `${source.id} ${error.id}`);
  });

  it("shows an in-flight Run again as a disabled, pending row control, and a failure in the row", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-relaunch-flight", status: "completed", relaunchable: true })],
    });
    let resolveRelaunch: (v: { sweep_id?: string }) => void = () => {};
    vi.spyOn(tuningApi, "relaunch").mockReturnValue(
      new Promise((resolve) => {
        resolveRelaunch = resolve;
      }),
    );

    render(<TuningTab />);
    const button = await screen.findByRole("button", { name: "Run again hpo-relaunch-flight" });
    fireEvent.click(button);

    expect(await screen.findByText("Starting…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run again hpo-relaunch-flight" })).toBeDisabled();

    resolveRelaunch({ sweep_id: "hpo-new-2" });
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Run again hpo-relaunch-flight" }),
      ).not.toBeDisabled(),
    );
  });

  it("shows a failed Cancel in the row, not only as a toast", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-cancel-flight", status: "running" })],
    });
    vi.spyOn(tuningApi, "cancel").mockRejectedValue(new Error("network error"));

    render(<TuningTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel hpo-cancel-flight" }));

    expect(await screen.findByText("Cancel failed: network error")).toBeInTheDocument();
  });
});

describe("TuningTab sweeps loader", () => {
  it("keeps and shows a failed sweeps fetch, with a Retry control, in place of the empty line", async () => {
    const listSweepsSpy = vi
      .spyOn(tuningApi, "listSweeps")
      .mockRejectedValueOnce(new Error("network error"))
      .mockResolvedValue({ sweeps: [] });

    render(<TuningTab />);
    expect(await screen.findByText("Could not load sweeps: network error")).toBeInTheDocument();
    expect(screen.queryByText('No sweeps yet. Use "Start a sweep" above.')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(listSweepsSpy).toHaveBeenCalledTimes(2));
    expect(
      await screen.findByText('No sweeps yet. Use "Start a sweep" above.'),
    ).toBeInTheDocument();
  });
});

describe("TuningTab sweep detail before a manifest exists", () => {
  it("keys the not-yet-recorded state on has_manifest, then clears once the manifest appears", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-pending", status: "running" })],
    });
    const getSweepSpy = vi
      .spyOn(tuningApi, "getSweep")
      .mockResolvedValueOnce(
        sweepDetail({ sweep_id: "hpo-pending", status: "running", has_manifest: false }),
      )
      .mockResolvedValue(
        sweepDetail({ sweep_id: "hpo-pending", status: "running", has_manifest: true }),
      );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-pending", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    vi.useFakeTimers();
    try {
      render(<TuningTab />);
      await vi.waitFor(() => expect(screen.getByText("hpo-pending")).toBeInTheDocument());
      fireEvent.click(screen.getByText("hpo-pending"));

      await vi.waitFor(() =>
        expect(screen.getByText("This sweep's record is not written yet.")).toBeInTheDocument(),
      );
      // Once, in the detail's own polite status region, never duplicated under the row too.
      const notWritten = screen.getByText("This sweep's record is not written yet.");
      expect(screen.getAllByText("This sweep's record is not written yet.")).toHaveLength(1);
      expect(notWritten.closest('[role="status"]')).not.toBeNull();
      expect(getSweepSpy).toHaveBeenCalledTimes(1);

      // The shared poll (the listing's own cadence) picks the manifest up on its own; no
      // separate detail timer and no Try again for the breeder to press.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(RUN_REFRESH_MS);
      });

      expect(getSweepSpy).toHaveBeenCalledTimes(2);
      expect(screen.queryByText("This sweep's record is not written yet.")).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("retries the sweep TensorBoard panel on its own timer rather than holding off entirely", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-pending", status: "running" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-pending", status: "running", has_manifest: false }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-pending", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    const launchTbSpy = vi
      .spyOn(tuningApi, "launchSweepTensorboard")
      .mockRejectedValueOnce(
        new StructuredRefusalError(
          { error: "sweep not found: hpo-pending" },
          404,
          "sweep not found: hpo-pending",
        ),
      )
      .mockResolvedValue({ url: "http://localhost:6006" });

    vi.useFakeTimers();
    try {
      render(<TuningTab />);
      await vi.waitFor(() => expect(screen.getByText("hpo-pending")).toBeInTheDocument());
      fireEvent.click(screen.getByText("hpo-pending"));

      await vi.waitFor(() => expect(launchTbSpy).toHaveBeenCalledTimes(1));

      await act(async () => {
        await vi.advanceTimersByTimeAsync(EMBEDDED_TOOL_RETRY_MS);
      });

      await vi.waitFor(() => expect(launchTbSpy).toHaveBeenCalledTimes(2));
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows the sweep TensorBoard panel's calm loading state in the pre-manifest window, never the raw refusal or a Try again", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-pending", status: "running" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-pending", status: "running", has_manifest: false }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-pending", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockRejectedValue(
      new StructuredRefusalError(
        { error: "sweep not found: hpo-pending" },
        404,
        "sweep not found: hpo-pending",
      ),
    );

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-pending"));

    expect(await screen.findByRole("heading", { name: "Sweep TensorBoard" })).toBeInTheDocument();
    expect(screen.queryByText(/sweep not found/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again: Sweep TensorBoard" })).toBeNull();
  });
});

describe("TuningTab settled sweep TensorBoard panel", () => {
  it("never shows Starting… once the sweep has settled, even with an attempt still in flight", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-settled", status: "cancelled" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({
        sweep_id: "hpo-settled",
        status: "cancelled",
        error: "the sweep was cancelled by request before it could finish",
        result: { status: "cancelled" },
      }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-settled", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    // Never resolves: the settle outran the panel's own in-flight attempt.
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockReturnValue(new Promise(() => {}));

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-settled"));

    await screen.findByText("Cancelled: the sweep was cancelled by request before it could finish");
    expect(screen.queryByText("Starting…")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Sweep TensorBoard" })).not.toBeInTheDocument();
  });
});

describe("TuningTab selection change", () => {
  it("clears the previous sweep's detail and trials rather than showing them under a new selection", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({ sweep_id: "hpo-a", status: "completed" }),
        sweep({ sweep_id: "hpo-b", status: "running" }),
      ],
    });
    // hpo-b's own getSweep never resolves in this test.
    const pendingB = new Promise<never>(() => {});
    vi.spyOn(tuningApi, "getSweep").mockImplementation((id: string) =>
      id === "hpo-a"
        ? Promise.resolve(sweepDetail({ sweep_id: "hpo-a", status: "completed" }))
        : pendingB,
    );
    vi.spyOn(tuningApi, "listTrials").mockImplementation((id: string) =>
      id === "hpo-a"
        ? Promise.resolve({
            sweep_id: "hpo-a",
            trials: [
              { trial_id: "trial_a1", has_metrics: false, params: {}, unconsumed_params: [] },
            ],
          })
        : new Promise(() => {}),
    );
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-a"));
    expect(await screen.findByText("trial_a1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("hpo-b"));
    // hpo-b's own getSweep never resolves, so this proves the clear happens on selection,
    // not merely once hpo-b's own records eventually arrive.
    await waitFor(() => expect(screen.queryByText("trial_a1")).not.toBeInTheDocument());
    expect(screen.queryByText("(completed)")).not.toBeInTheDocument();
  });

  it("launches a fresh TensorBoard for the newly selected sweep, not the previous one's, on a direct switch", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({ sweep_id: "hpo-a", status: "completed" }),
        sweep({ sweep_id: "hpo-b", status: "completed" }),
      ],
    });
    vi.spyOn(tuningApi, "getSweep").mockImplementation((id: string) =>
      Promise.resolve(sweepDetail({ sweep_id: id, status: "completed" })),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-a", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    const launchTbSpy = vi
      .spyOn(tuningApi, "launchSweepTensorboard")
      .mockImplementation((id) => Promise.resolve({ url: `http://localhost:6006/${id}` }));

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-a"));
    await waitFor(() => expect(launchTbSpy).toHaveBeenCalledWith("hpo-a"));

    fireEvent.click(screen.getByText("hpo-b"));
    await waitFor(() => expect(launchTbSpy).toHaveBeenCalledWith("hpo-b"));
    expect(screen.getByTitle("Sweep TensorBoard")).toHaveAttribute(
      "src",
      "http://localhost:6006/hpo-b",
    );
  });
});

describe("TuningTab sweep detail layout", () => {
  it("places the result summary above the Ray and TensorBoard panels", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-layout", status: "completed" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({
        sweep_id: "hpo-layout",
        status: "completed",
        result: { best_trial: "trial_1" },
      }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-layout", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-layout"));

    const result = await screen.findByText(/"best_trial": "trial_1"/);
    const rayHeading = screen.getByRole("heading", { name: "Ray dashboard" });
    expect(
      result.compareDocumentPosition(rayHeading) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});

describe("TuningTab row accessible name", () => {
  it("carries the stop-requested state in the row's own name, not only in its visible text", async () => {
    vi.spyOn(tuningApi, "listSweeps")
      .mockResolvedValueOnce({
        sweeps: [sweep({ sweep_id: "hpo-stopping", status: "running", cancel_requested: false })],
      })
      .mockResolvedValue({
        sweeps: [sweep({ sweep_id: "hpo-stopping", status: "running", cancel_requested: true })],
      });
    vi.spyOn(tuningApi, "cancel").mockResolvedValue({
      study_name: "hpo-stopping",
      status: "running",
      cancel_requested: true,
    });

    render(<TuningTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel hpo-stopping" }));

    expect(
      await screen.findByRole("button", { name: "hpo-stopping running, stop requested" }),
    ).toBeInTheDocument();
  });
});

describe("TuningTab trial list pending state", () => {
  it("says nothing about disk while the sweep's own manifest is still unknown", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-x", status: "running" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockImplementation(() => new Promise(() => {}));
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-x", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-x"));

    await waitFor(() =>
      expect(screen.getAllByText("Reading this sweep's record…").length).toBeGreaterThan(0),
    );
    expect(screen.queryByText("No trials on disk yet.")).not.toBeInTheDocument();
    expect(screen.queryByText("This sweep's record is not written yet.")).not.toBeInTheDocument();
  });
});

describe("TuningTab sweep detail failures", () => {
  it("shows a non-404 failure as an error and keeps polling, on the listing's own cadence, instead of swallowing it", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-flaky", status: "running" })],
    });
    const serverError = new StructuredRefusalError(
      { error: "internal error" },
      500,
      "internal error",
    );
    const getSweepSpy = vi.spyOn(tuningApi, "getSweep").mockRejectedValue(serverError);
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-flaky", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    vi.useFakeTimers();
    try {
      render(<TuningTab />);
      await vi.waitFor(() => expect(screen.getByText("hpo-flaky")).toBeInTheDocument());
      fireEvent.click(screen.getByText("hpo-flaky"));

      await vi.waitFor(() => expect(getSweepSpy).toHaveBeenCalledTimes(1));
      await vi.waitFor(() => expect(screen.getByText("internal error")).toBeInTheDocument());
      expect(screen.queryByText("This sweep's record is not written yet.")).not.toBeInTheDocument();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(RUN_REFRESH_MS);
      });
      expect(getSweepSpy).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("TuningTab detail promise", () => {
  it("shows the future-tense promise only for a running sweep; a failed one shows its error alone", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-failed-detail", status: "failed" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({
        sweep_id: "hpo-failed-detail",
        status: "failed",
        error: "the sweep's base config fails preflight",
      }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({
      sweep_id: "hpo-failed-detail",
      trials: [],
    });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-failed-detail"));

    expect(
      await screen.findByText(
        "No result recorded; the sweep failed: the sweep's base config fails preflight",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/The best config appears here once the sweep finishes/),
    ).not.toBeInTheDocument();
  });

  it("says no result was recorded, with the sweep's own status, for a terminal sweep the record never gave one", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-interrupted-detail", status: "interrupted" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-interrupted-detail", status: "interrupted", error: null }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({
      sweep_id: "hpo-interrupted-detail",
      trials: [],
    });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-interrupted-detail"));

    expect(
      await screen.findByText("No result recorded; the sweep interrupted."),
    ).toBeInTheDocument();
  });

  it("keeps the future-tense promise for a running sweep with no result yet", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-running-detail", status: "running" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-running-detail", status: "running" }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({
      sweep_id: "hpo-running-detail",
      trials: [],
    });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-running-detail"));

    expect(
      await screen.findByText(/The best config appears here once the sweep finishes/),
    ).toBeInTheDocument();
  });
});

describe("TuningTab sweep summary line", () => {
  it("labels the search shape's own tokens and renders an absent scheduler as no scheduler, not none", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [
        sweep({
          sweep_id: "hpo-labeled-1",
          status: "running",
          n_trials: 1,
          search_alg: "random",
          scheduler: "none",
          param_space_keys: ["weight_decay"],
        }),
      ],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-labeled-1", status: "running" }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-labeled-1", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-labeled-1"));

    expect(
      await screen.findByText("1 trial planned · search random · no scheduler · axes weight_decay"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/scheduler none/)).not.toBeInTheDocument();
    expect(screen.queryByText(/scheduler no scheduler/)).not.toBeInTheDocument();
  });

  it("shows the trial line's own parameter value exactly as recorded, with no rounding or title", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-trial-exact", status: "running" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-trial-exact", status: "running" }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({
      sweep_id: "hpo-trial-exact",
      trials: [
        {
          trial_id: "trial_1",
          has_metrics: true,
          params: { weight_decay: 0.0016999999 },
          unconsumed_params: [],
        },
      ],
    });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: null });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({ error: "no cluster" });

    render(<TuningTab />);
    fireEvent.click(await screen.findByText("hpo-trial-exact"));

    const exact = await screen.findByText("weight_decay=0.0016999999");
    expect(exact).not.toHaveAttribute("title");
    expect(screen.getByText(/metrics recorded/)).toBeInTheDocument();
  });
});

describe("TuningTab heading", () => {
  it("renders exactly one top-level heading naming the tab", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({ sweeps: [] });
    render(<TuningTab />);
    await waitFor(() => expect(tuningApi.listSweeps).toHaveBeenCalled());
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Tuning");
  });

  it("carries the sidebar and the embedded panels' own section titles as level-2 headings", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-heading", status: "completed" })],
    });
    vi.spyOn(tuningApi, "getSweep").mockResolvedValue(
      sweepDetail({ sweep_id: "hpo-heading", status: "completed" }),
    );
    vi.spyOn(tuningApi, "listTrials").mockResolvedValue({ sweep_id: "hpo-heading", trials: [] });
    vi.spyOn(tuningApi, "getRayDashboard").mockResolvedValue({ url: "http://localhost:8265" });
    vi.spyOn(tuningApi, "launchSweepTensorboard").mockResolvedValue({
      url: "http://localhost:6006",
    });

    render(<TuningTab />);
    expect(await screen.findByRole("heading", { level: 2, name: "Sweeps" })).toBeInTheDocument();
    fireEvent.click(screen.getByText("hpo-heading"));
    expect(
      await screen.findByRole("heading", { level: 2, name: "Ray dashboard" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Sweep TensorBoard" }),
    ).toBeInTheDocument();
  });
});

describe("TuningTab list order", () => {
  it("states how the sweep list is ordered", async () => {
    vi.spyOn(tuningApi, "listSweeps").mockResolvedValue({
      sweeps: [sweep({ sweep_id: "hpo-a", status: "running" })],
    });

    render(<TuningTab />);
    expect(await screen.findByText("hpo-a")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Sweeps this running process itself launched come first, in launch order; every other recorded sweep follows/,
      ),
    ).toBeInTheDocument();
  });
});
