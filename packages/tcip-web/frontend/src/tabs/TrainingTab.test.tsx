import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { StructuredRefusalError } from "@/api/http";
import {
  openTrainingStream,
  trainingApi,
  type SplitChoices,
  type TrainingStreamMsg,
} from "@/api/training";
import {
  NOT_FINITE_SUFFIX,
  type RunRow,
  type SweepGroup,
  type TrainingMetricFrame,
  type TrainingStatusFrame,
} from "@/api/types.generated";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { useStore } from "@/store";
import { TrainingTab, dataPickerFor } from "@/tabs/TrainingTab";
import { RUN_REFRESH_MS } from "@/tabs/trainingMetrics";
import { openTestProject } from "@/test/store";

// The live metrics stream owns a real WebSocket; only the run list and its controls are under
// test here, so the transport is replaced while the rest of the module stays real.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return { ...actual, openTrainingStream: vi.fn(() => () => {}) };
});

const initialStoreState = useStore.getState();

function run(overrides: Partial<RunRow> & { experiment_id: string }): RunRow {
  return {
    state: "running",
    created: "2026-08-01T00:00:00Z",
    relaunched_from: null,
    sweep: null,
    trial_params: null,
    builder: "m:build_detector",
    task: "detection",
    images_dir: "/data/images",
    current_epoch: null,
    best_metric: null,
    best_metric_name: null,
    output_dir: `/proj/.tcip/experiments/${overrides.experiment_id}`,
    error: null,
    heartbeat: "",
    launch: null,
    ...overrides,
  };
}

function sweep(overrides: Partial<SweepGroup> & { sweep_id: string }): SweepGroup {
  return {
    state: "running",
    error: null,
    input: { n_trials: 2, search_alg: "random" },
    objective: { selection_metric: "map50", higher_is_better: true },
    cancel_requested: false,
    heartbeat: "",
    trials: [],
    outcome: { best_params: null, best_value: null },
    ...overrides,
  };
}

/** The run list answers ``runs`` and ``sweeps`` on every poll. */
function listing(runs: RunRow[], sweeps: SweepGroup[] = []) {
  return vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs, sweeps });
}

/** One run's own detail, with no TensorBoard serving it. */
function detailOf(row: RunRow) {
  vi.spyOn(trainingApi, "getRun").mockResolvedValue({
    run: row,
    sweep: null,
    tensorboard_url: null,
  });
}

type Frame =
  Omit<TrainingMetricFrame, "experiment_id"> | Omit<TrainingStatusFrame, "experiment_id">;

/** Every stream the tab opens delivers ``frames`` at once, each under the id it was opened for. */
function streamFrames(...frames: Frame[]) {
  vi.mocked(openTrainingStream).mockImplementation((experimentId, onMessage) => {
    frames.forEach((frame) =>
      onMessage({ ...frame, experiment_id: experimentId } as TrainingStreamMsg),
    );
    return () => {};
  });
}

function metricFrame(row: Record<string, unknown>): Frame {
  return { type: "metric", row };
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.setState({ user: "jordan" });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("TrainingTab run list", () => {
  it("offers a stop control for a running run, and cancels it by id", async () => {
    listing([run({ experiment_id: "train-agent-1" })]);
    const cancelSpy = vi.spyOn(trainingApi, "cancel").mockResolvedValue({
      experiment_id: "train-agent-1",
      state: "running",
      cancel_requested: true,
    });

    render(<TrainingTab />);
    expect(await screen.findByText("train-agent-1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Cancel train-agent-1" }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("train-agent-1", "jordan"));
  });

  it("shows a running row's own heartbeat age, the one liveness signal a record with no process id can offer", async () => {
    listing([
      run({
        experiment_id: "train-stale",
        heartbeat: new Date(Date.now() - 3 * 60_000).toISOString(),
      }),
    ]);

    render(<TrainingTab />);
    expect(await screen.findByText(/running, last heartbeat 3 min ago/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /train-stale running, last heartbeat 3 min ago/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel train-stale" })).toHaveAttribute(
      "title",
      "Reaches a live process only; a stale running row keeps this control until its heartbeat window lapses.",
    );
  });

  it("offers no stop control for a run in a terminal state", async () => {
    listing([
      run({ experiment_id: "train-done", state: "completed" }),
      run({ experiment_id: "train-done-agent", state: "failed" }),
    ]);

    render(<TrainingTab />);
    expect(await screen.findByText("train-done")).toBeInTheDocument();
    expect(screen.getByText("train-done-agent")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Cancel/ })).not.toBeInTheDocument();
  });

  it("names the best value's own metric exactly as the record carries it, and shows nothing when the row carries no name", async () => {
    listing([
      run({
        experiment_id: "train-named",
        state: "completed",
        best_metric: 0.812,
        best_metric_name: "map50",
      }),
      run({
        experiment_id: "train-loss-only",
        state: "completed",
        best_metric: 0.907,
        best_metric_name: "loss",
      }),
      run({
        experiment_id: "train-unnamed",
        state: "completed",
        best_metric: 0.5,
        best_metric_name: null,
      }),
    ]);

    render(<TrainingTab />);
    // The record names the metric bare; the row never adds a val_ qualifier the record
    // itself never stated (a no-validation run's best can be the training loss).
    expect(await screen.findByText("best map50 0.812")).toBeInTheDocument();
    expect(await screen.findByText("best loss 0.907")).toBeInTheDocument();
    expect(screen.queryByText(/val_/)).not.toBeInTheDocument();
    expect(screen.queryByText(/best.*0\.500/)).not.toBeInTheDocument();
  });

  it("states how the run list is ordered", async () => {
    listing([run({ experiment_id: "train-a" })]);

    render(<TrainingTab />);
    expect(await screen.findByText("train-a")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Every recorded run, sorted by experiment id, then every sweep with its trials.",
      ),
    ).toBeInTheDocument();
  });

  it("names the row's own select control with the id and state", async () => {
    listing([run({ experiment_id: "train-named-row" })]);

    render(<TrainingTab />);
    expect(
      await screen.findByRole("button", {
        name: "train-named-row running, no launch event recorded",
      }),
    ).toBeInTheDocument();
  });

  it("shows the row's best value exactly as the record carries it, with no rounding or title", async () => {
    listing([
      run({
        experiment_id: "train-exact",
        state: "completed",
        best_metric: 0.4130041,
        best_metric_name: "loss",
      }),
    ]);

    render(<TrainingTab />);
    const value = await screen.findByText("best loss 0.4130041");
    expect(value).not.toHaveAttribute("title");
  });

  it("names the row's accessible name with its best value and metric when the record carries one", async () => {
    listing([
      run({
        experiment_id: "train-named-value",
        state: "completed",
        best_metric: 0.4130041,
        best_metric_name: "loss",
      }),
    ]);

    render(<TrainingTab />);
    expect(
      await screen.findByRole("button", {
        name: "train-named-value completed, no launch event recorded, best loss 0.4130041",
      }),
    ).toBeInTheDocument();
  });

  it("shows an in-flight Cancel as a disabled, pending row control, and a failure in the row", async () => {
    listing([run({ experiment_id: "train-cancel-flight" })]);
    let resolveCancel: (v: {
      experiment_id: string;
      state: string;
      cancel_requested: boolean;
    }) => void = () => {};
    vi.spyOn(trainingApi, "cancel").mockReturnValue(
      new Promise((resolve) => {
        resolveCancel = resolve;
      }),
    );

    render(<TrainingTab />);
    const button = await screen.findByRole("button", { name: "Cancel train-cancel-flight" });
    fireEvent.click(button);

    expect(await screen.findByText("Canceling…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel train-cancel-flight" })).toBeDisabled();

    resolveCancel({
      experiment_id: "train-cancel-flight",
      state: "running",
      cancel_requested: true,
    });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Cancel train-cancel-flight" })).not.toBeDisabled(),
    );
  });

  it("shows a failed Cancel in the row, not only as a toast", async () => {
    listing([run({ experiment_id: "train-cancel-fail" })]);
    vi.spyOn(trainingApi, "cancel").mockRejectedValue(new Error("network error"));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel train-cancel-fail" }));

    expect(await screen.findByText("Cancel failed: network error")).toBeInTheDocument();
  });
});

describe("TrainingTab sweeps", () => {
  const trialA = run({
    experiment_id: "sweep-1_aaaa",
    sweep: "sweep-1",
    state: "completed",
    trial_params: { lr: 0.01 },
  });
  const trialB = run({ experiment_id: "sweep-1_bbbb", sweep: "sweep-1" });

  it("groups a sweep's trial rows under the sweep, with its state and outcome", async () => {
    listing(
      [run({ experiment_id: "train-solo" })],
      [
        sweep({
          sweep_id: "sweep-1",
          trials: [trialA, trialB],
          outcome: { best_params: { lr: 0.01 }, best_value: 0.7 },
        }),
      ],
    );

    render(<TrainingTab />);
    const group = await screen.findByRole("group", { name: "Sweep sweep-1" });
    expect(within(group).getByText("sweep-1_aaaa")).toBeInTheDocument();
    expect(within(group).getByText("sweep-1_bbbb")).toBeInTheDocument();
    expect(within(group).getByText(/sweep · running · best map50 0.7/)).toBeInTheDocument();
    expect(within(group).queryByText("train-solo")).not.toBeInTheDocument();
  });

  it("cancels a trial by its own id and the whole sweep by the sweep's", async () => {
    listing([], [sweep({ sweep_id: "sweep-1", trials: [trialB] })]);
    const cancelSpy = vi.spyOn(trainingApi, "cancel").mockResolvedValue({
      experiment_id: "x",
      state: "running",
      cancel_requested: true,
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel sweep-1_bbbb" }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("sweep-1_bbbb", "jordan"));
    fireEvent.click(screen.getByRole("button", { name: "Cancel sweep-1" }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("sweep-1", "jordan"));
  });

  it("streams a selected trial's metrics, and shows a selected sweep's board without a stream", async () => {
    openTestProject();
    listing([], [sweep({ sweep_id: "sweep-1", trials: [trialA] })]);
    const getRun = vi
      .spyOn(trainingApi, "getRun")
      .mockResolvedValue({ run: trialA, sweep: null, tensorboard_url: "http://127.0.0.1:6006" });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("sweep-1_aaaa"));
    await waitFor(() =>
      expect(openTrainingStream).toHaveBeenCalledWith("sweep-1_aaaa", expect.any(Function)),
    );

    vi.mocked(openTrainingStream).mockClear();
    fireEvent.click(screen.getByRole("button", { name: /^sweep-1\s/ }));
    await waitFor(() => expect(getRun).toHaveBeenCalledWith("sweep-1"));
    expect(openTrainingStream).not.toHaveBeenCalled();
  });
});

describe("TrainingTab run launcher mark", () => {
  it("states who launched the run from its launch event alone", async () => {
    listing([
      run({
        experiment_id: "train-agent",
        launch: { agent_client_name: "claude-code", agent_client_version: "2.1.238" },
      }),
      run({ experiment_id: "train-undeclared", launch: {} }),
      run({ experiment_id: "train-no-event", state: "completed", launch: null }),
    ]);

    render(<TrainingTab />);
    await screen.findByText("train-agent");

    expect(
      screen.getByRole("button", { name: "train-agent running, started by the agent" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: "train-undeclared running, started with no agent declared",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "train-no-event completed, no launch event recorded" }),
    ).toBeInTheDocument();
  });

  it("names the declared client in the agent row's accessible description", async () => {
    listing([
      run({
        experiment_id: "train-agent-desc",
        launch: { agent_client_name: "claude-code", agent_client_version: "2.1.238" },
      }),
    ]);

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: "train-agent-desc running, started by the agent",
    });

    const describedBy = button.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent("claude-code 2.1.238");
  });

  it("makes a missing launch event's explanation reachable by assistive technology", async () => {
    listing([run({ experiment_id: "train-no-launcher" })]);

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: "train-no-launcher running, no launch event recorded",
    });

    const describedBy = button.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      "No launch event in this project's audit log names this run.",
    );
  });
});

describe("TrainingTab heading", () => {
  it("renders exactly one top-level heading naming the tab", async () => {
    listing([]);
    render(<TrainingTab />);
    await waitFor(() => expect(trainingApi.listRuns).toHaveBeenCalled());
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Training");
  });

  it("carries the section titles as level-2 headings under that h1", async () => {
    const row = run({ experiment_id: "train-heading", state: "completed" });
    listing([row]);
    vi.spyOn(trainingApi, "getRun").mockResolvedValue({
      run: row,
      sweep: null,
      tensorboard_url: "http://127.0.0.1:6006",
    });

    render(<TrainingTab />);
    expect(await screen.findByRole("heading", { level: 2, name: "Runs" })).toBeInTheDocument();
    fireEvent.click(screen.getByText("train-heading"));
    expect(
      await screen.findByRole("heading", { level: 2, name: "TensorBoard" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Metrics" })).toBeInTheDocument();
  });
});

describe("TrainingTab tensorboard panel", () => {
  it("says a run produced no logs and offers no Try again, rather than a raw refusal", async () => {
    const row = run({ experiment_id: "train-nologs", state: "failed" });
    listing([row]);
    detailOf(row);
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(
      new StructuredRefusalError(
        { error: "run produced no logs: train-nologs", no_logs: true },
        404,
        "run produced no logs: train-nologs",
      ),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-nologs"));

    expect(await screen.findByText("This run produced no logs.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
  });

  it("keeps Try again for an ordinary failed launch", async () => {
    const row = run({ experiment_id: "train-broken", state: "failed" });
    listing([row]);
    detailOf(row);
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(new Error("connection refused"));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-broken"));

    expect(await screen.findByText("connection refused")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again: TensorBoard" })).toBeInTheDocument();
  });

  it("keeps the run's own recorded crash reason for a no-logs run that failed with one", async () => {
    const crash =
      "[WinError 183] Cannot create a file when that file already exists: 'tensorboard'";
    const row = run({ experiment_id: "train-nologs-crashed", state: "failed", error: crash });
    listing([row]);
    detailOf(row);
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(
      new StructuredRefusalError({ error: crash, no_logs: true }, 404, "run produced no logs"),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-nologs-crashed"));

    expect(
      await screen.findByText(`This run failed: ${crash}. It produced no logs.`),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Try again: TensorBoard" }),
    ).not.toBeInTheDocument();
  });

  it("launches a fresh TensorBoard for the newly selected run, not the previous one's", async () => {
    listing([
      run({ experiment_id: "train-x", state: "completed" }),
      run({ experiment_id: "train-y", state: "completed" }),
    ]);
    vi.spyOn(trainingApi, "getRun").mockImplementation((experimentId: string) =>
      Promise.resolve({
        run: run({ experiment_id: experimentId, state: "completed" }),
        sweep: null,
        tensorboard_url: null,
      }),
    );
    const launchSpy = vi
      .spyOn(trainingApi, "launchTensorboard")
      .mockImplementation((experimentId: string) =>
        Promise.resolve({ url: `http://127.0.0.1:6006/${experimentId}` }),
      );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-x"));
    await waitFor(() => expect(launchSpy).toHaveBeenCalledWith("train-x"));

    fireEvent.click(screen.getByText("train-y"));
    await waitFor(() => expect(launchSpy).toHaveBeenCalledWith("train-y"));
    expect(screen.getByTitle("TensorBoard")).toHaveAttribute(
      "src",
      "http://127.0.0.1:6006/train-y",
    );
  });
});

describe("TrainingTab epoch table", () => {
  beforeEach(() => {
    openTestProject();
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
  });

  it("says a terminal run recorded no metrics, rather than waiting on one that will never arrive", async () => {
    listing([run({ experiment_id: "train-terminal-empty", state: "interrupted" })]);

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-terminal-empty"));

    expect(await screen.findByText("This run recorded no metrics.")).toBeInTheDocument();
    expect(screen.queryByText("Waiting for metrics…")).not.toBeInTheDocument();
  });

  it("marks the waiting placeholder as a live region on a run with no metrics yet", async () => {
    listing([run({ experiment_id: "train-waiting-live" })]);

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-waiting-live"));

    expect(await screen.findByRole("status")).toHaveTextContent("Waiting for metrics…");
  });

  it("shows every streamed epoch as a table row, outright", async () => {
    listing([run({ experiment_id: "train-table" })]);
    streamFrames(metricFrame({ epoch: 1, loss: 0.5 }), metricFrame({ epoch: 2, loss: 0.3 }));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-table"));

    const table = await screen.findByRole("table", { name: "train-table metrics by epoch" });
    expect(within(table).getAllByRole("row")).toHaveLength(3);
    expect(within(table).getByRole("columnheader", { name: "epoch" })).toBeInTheDocument();
    expect(within(table).getByText("0.5")).toBeInTheDocument();
    expect(within(table).getByText("0.3")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "as table" })).not.toBeInTheDocument();
  });

  it("renders the unset glyph for a row carrying no epoch, and for a non-finite value", async () => {
    listing([run({ experiment_id: "train-unset" })]);
    streamFrames(metricFrame({ loss: 0.5 }), metricFrame({ epoch: 2, loss: null }));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-unset"));

    const rows = within(await screen.findByRole("table")).getAllByRole("row");
    expect(within(rows[1]).getByText(UNSET_GLYPH)).toBeInTheDocument();
    expect(within(rows[1]).getByText("0.5")).toBeInTheDocument();
    expect(within(rows[2]).getByText(UNSET_GLYPH)).toBeInTheDocument();
  });

  it("gives a bookkeeping field or a non-finite state companion no column", async () => {
    listing([run({ experiment_id: "train-bookkeeping" })]);
    streamFrames(
      metricFrame({
        epoch: 1,
        loss: 0.5,
        [`loss${NOT_FINITE_SUFFIX}`]: "nan",
        timestamp: "2026-01-01T00:00:00Z",
      }),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-bookkeeping"));

    const headers = within(await screen.findByRole("table"))
      .getAllByRole("columnheader")
      .map((h) => h.textContent);
    expect(headers).toEqual(["epoch", "loss"]);
  });
});

describe("TrainingTab status toast", () => {
  beforeEach(() => {
    openTestProject();
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
  });

  function completedFrame(experimentId: string): Frame {
    return {
      type: "status",
      status: run({ experiment_id: experimentId, state: "completed" }),
      error: null,
    };
  }

  it("does not toast on selecting a run already known to be terminal, only rediscovering its state", async () => {
    listing([run({ experiment_id: "train-old-done", state: "completed" })]);
    streamFrames(completedFrame("train-old-done"));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-old-done"));

    await waitFor(() => expect(trainingApi.listRuns).toHaveBeenCalled());
    expect(pushToast).not.toHaveBeenCalledWith("Training train-old-done: completed", "info");
  });

  it("toasts a transition to terminal observed while watching a live run", async () => {
    listing([run({ experiment_id: "train-live-done" })]);
    streamFrames(completedFrame("train-live-done"));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-live-done"));

    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith("Training train-live-done: completed", "info"),
    );
  });

  it("toasts a stream error once, then still shows a metric the reconnect delivers", async () => {
    listing([run({ experiment_id: "train-just-launched" })]);
    const unknown = {
      type: "status" as const,
      status: null,
      error: "unknown run: train-just-launched",
    };
    streamFrames(unknown, unknown, metricFrame({ epoch: 1, loss: 0.5 }));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-just-launched"));

    expect(within(await screen.findByRole("table")).getByText("0.5")).toBeInTheDocument();
    expect(pushToast).toHaveBeenCalledWith(
      "Training stream error: unknown run: train-just-launched",
    );
    expect(pushToast).toHaveBeenCalledTimes(1);
  });
});

describe("TrainingTab stream lifecycle", () => {
  it("closes the stream and clears the detail panel when the selected run leaves the next poll", async () => {
    openTestProject();
    const listRuns = vi.spyOn(trainingApi, "listRuns");
    listRuns.mockResolvedValueOnce({
      runs: [run({ experiment_id: "train-vanishing" })],
      sweeps: [],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    const stop = vi.fn();
    vi.mocked(openTrainingStream).mockImplementation(() => stop);

    vi.useFakeTimers();
    try {
      render(<TrainingTab />);
      await vi.waitFor(() => expect(screen.getByText("train-vanishing")).toBeInTheDocument());
      fireEvent.click(screen.getByText("train-vanishing"));
      await vi.waitFor(() => expect(openTrainingStream).toHaveBeenCalledTimes(1));

      listRuns.mockResolvedValueOnce({ runs: [], sweeps: [] });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(RUN_REFRESH_MS);
      });

      expect(stop).toHaveBeenCalled();
      expect(screen.getByText("No run selected.")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("TrainingTab launch picker", () => {
  const recordedChoices: SplitChoices = {
    as_recorded: {
      case: "bound",
      line: "on the partition it bound",
      compatible: true,
      reason: null,
    },
    selections: [],
  };

  it("opens the picker, selects a recorded run, and starts it through the relaunch door", async () => {
    listing([run({ experiment_id: "exp-pristine-1", state: "completed" })]);
    vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue(recordedChoices);
    const relaunchSpy = vi
      .spyOn(trainingApi, "relaunch")
      .mockResolvedValue({ experiment_id: "exp-pristine-2" });

    render(<TrainingTab />);
    await screen.findByText("exp-pristine-1");
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;

    fireEvent.click(within(picker).getByText("exp-pristine-1"));
    expect(
      screen.getByText(
        "A new run of this config on the data paths it names, as they are now, with the recorded seed",
      ),
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(relaunchSpy).toHaveBeenCalledWith("exp-pristine-1", "jordan", null));
  });

  it("starts a recorded sweep again by its id, with no data choice to fetch", async () => {
    listing([], [sweep({ sweep_id: "sweep-1", state: "completed" })]);
    const choicesSpy = vi.spyOn(trainingApi, "listSplitChoices");
    const relaunchSpy = vi.spyOn(trainingApi, "relaunch").mockResolvedValue({
      sweep_id: "sweep-2",
      status: "launched",
    });

    render(<TrainingTab />);
    await screen.findByRole("group", { name: "Sweep sweep-1" });
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;

    fireEvent.click(within(picker).getByText("sweep-1"));
    expect(
      screen.getByText("A new sweep over this sweep's recorded config, search and seed"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(relaunchSpy).toHaveBeenCalledWith("sweep-1", "jordan", null));
    expect(choicesSpy).not.toHaveBeenCalled();
  });

  it("shows a refused start's issues under the picked row", async () => {
    listing([run({ experiment_id: "exp-refused-1", state: "completed" })]);
    vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue(recordedChoices);
    vi.spyOn(trainingApi, "relaunch").mockRejectedValue(
      new StructuredRefusalError({ issues: ["batch_size must be positive"] }, 422, ""),
    );

    render(<TrainingTab />);
    await screen.findByText("exp-refused-1");
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;

    fireEvent.click(within(picker).getByText("exp-refused-1"));
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(screen.getByText("batch_size must be positive")).toBeInTheDocument(),
    );
  });

  it("opens a row's Data choices only once selected, and starts with the chosen manifest", async () => {
    listing([run({ experiment_id: "exp-1", state: "completed" })]);
    const listSplitChoicesSpy = vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue({
      as_recorded: {
        case: "drawn",
        line: "draws its split again with seed 42 over the labels as they are now",
        compatible: true,
        reason: null,
      },
      selections: [
        {
          selection_dir: "/data/splits",
          enabled: true,
          reason: null,
          seed: 7,
          group_by: "tile_prefix",
          train: 4,
          val: 2,
          calibration: 1,
          replaced_split_keys: [],
        },
      ],
    });
    const relaunchSpy = vi
      .spyOn(trainingApi, "relaunch")
      .mockResolvedValue({ experiment_id: "exp-2" });

    render(<TrainingTab />);
    await screen.findByText("exp-1");
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;
    expect(listSplitChoicesSpy).not.toHaveBeenCalled();

    fireEvent.click(within(picker).getByText("exp-1"));
    expect(
      await screen.findByText(/draws its split again with seed 42 over the labels as they are now/),
    ).toBeInTheDocument();
    expect(listSplitChoicesSpy).toHaveBeenCalledWith("exp-1");

    fireEvent.click(await screen.findByRole("radio", { name: /\/data\/splits/ }));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(relaunchSpy).toHaveBeenCalledWith("exp-1", "jordan", "/data/splits"),
    );
  });

  it("disables Start while a selected row's data choices are still loading", async () => {
    listing([run({ experiment_id: "exp-1", state: "completed" })]);
    let resolveChoices: (v: SplitChoices) => void = () => {};
    vi.spyOn(trainingApi, "listSplitChoices").mockReturnValue(
      new Promise<SplitChoices>((resolve) => {
        resolveChoices = resolve;
      }),
    );

    render(<TrainingTab />);
    await screen.findByText("exp-1");
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;
    fireEvent.click(within(picker).getByText("exp-1"));

    expect(await screen.findByRole("button", { name: "Start" })).toBeDisabled();

    resolveChoices({
      as_recorded: { case: "drawn", line: "draws its split again", compatible: true, reason: null },
      selections: [],
    });
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
  });

  it("shows a per-row split-choices fetch failure as text on the row", async () => {
    listing([run({ experiment_id: "exp-1", state: "completed" })]);
    vi.spyOn(trainingApi, "listSplitChoices").mockRejectedValue(new Error("network error"));

    render(<TrainingTab />);
    await screen.findByText("exp-1");
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;
    fireEvent.click(within(picker).getByText("exp-1"));

    expect(
      await screen.findByText(/Could not load its data choices: network error/),
    ).toBeInTheDocument();
  });

  it("shows a failed listing in the picker, with a Retry control, in place of the empty line", async () => {
    const listRuns = vi
      .spyOn(trainingApi, "listRuns")
      .mockRejectedValueOnce(new Error("network error"))
      .mockResolvedValue({ runs: [], sweeps: [] });

    render(<TrainingTab />);
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;

    expect(
      await within(picker).findByText("Could not load training runs: network error"),
    ).toBeInTheDocument();
    expect(
      within(picker).queryByText("No run or sweep exists in this project yet."),
    ).not.toBeInTheDocument();

    fireEvent.click(within(picker).getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(listRuns).toHaveBeenCalledTimes(2));
    expect(
      await within(picker).findByText("No run or sweep exists in this project yet."),
    ).toBeInTheDocument();
  });
});

describe("dataPickerFor", () => {
  const choices: SplitChoices = {
    as_recorded: {
      case: "bound",
      line: "on the partition it bound",
      compatible: false,
      reason: "Not found: data.images_dir = '/moved'",
    },
    selections: [
      {
        selection_dir: "/data/splits",
        enabled: true,
        reason: null,
        seed: 7,
        group_by: "tile_prefix",
        train: 4,
        val: 2,
        calibration: 1,
        replaced_split_keys: ["seed"],
      },
      {
        selection_dir: "/data/other",
        enabled: false,
        reason: "the selection at /data/other records no subject",
        seed: null,
        group_by: null,
        train: 0,
        val: 0,
        calibration: 0,
        replaced_split_keys: [],
      },
    ],
  };

  it("maps the snapshot's own compatibility into the disabled/reason pair the picker renders", () => {
    const picker = dataPickerFor(choices);
    expect(picker?.asRecordedLine).toBe("on the partition it bound");
    expect(picker?.asRecordedDisabled).toBe(true);
    expect(picker?.asRecordedReason).toBe("Not found: data.images_dir = '/moved'");
    expect(picker?.absenceMessage).toMatch(/this listing found no other recorded partition/);
  });

  it("maps each selection's own enabled flag and reason onto the choice the picker renders", () => {
    const picker = dataPickerFor(choices);
    expect(picker?.choices[0]).toMatchObject({
      selectionDir: "/data/splits",
      disabled: false,
      replacedSplitKeys: ["seed"],
    });
    expect(picker?.choices[1]).toMatchObject({
      selectionDir: "/data/other",
      disabled: true,
      reason: "the selection at /data/other records no subject",
    });
  });

  it("renders the seed/group_by/counts line from the selection's own recorded fields", () => {
    const picker = dataPickerFor(choices);
    render(<div>{picker?.choices[0].label}</div>);
    expect(
      screen.getByText(/seed 7 · tile_prefix · train 4 · val 2 · calibration 1/),
    ).toBeInTheDocument();
  });

  it("answers undefined for a row with no choices fetched yet", () => {
    expect(dataPickerFor(undefined)).toBeUndefined();
  });
});

describe("TrainingTab compare", () => {
  // The comparison's own columns repeat a marked run's id in its detail region (table headers),
  // and the single-run header repeats the selected one; the sidebar's own row is always first.
  function rowFor(id: string): HTMLElement {
    return screen.getAllByText(id)[0].closest("li") as HTMLElement;
  }

  function compareAnswers(ids: string[]) {
    vi.spyOn(trainingApi, "compare").mockResolvedValue({
      experiments: ids.map((experiment_id) => ({ experiment_id })),
      count: ids.length,
      same_dataset_fingerprint: null,
    });
  }

  it("marking two runs switches the detail region and closes the single-run stream", async () => {
    openTestProject();
    listing([run({ experiment_id: "run-a" }), run({ experiment_id: "run-b" })]);
    const stopSingle = vi.fn();
    vi.mocked(openTrainingStream).mockReturnValueOnce(stopSingle);
    compareAnswers(["run-a", "run-b"]);

    render(<TrainingTab />);
    await screen.findByText("run-a");
    fireEvent.click(within(rowFor("run-a")).getByText("run-a"));
    await waitFor(() =>
      expect(openTrainingStream).toHaveBeenCalledWith("run-a", expect.any(Function)),
    );

    fireEvent.click(within(rowFor("run-a")).getByRole("button", { name: "Compare run-a" }));
    fireEvent.click(within(rowFor("run-b")).getByRole("button", { name: "Compare run-b" }));

    expect(await screen.findByText("Comparing")).toBeInTheDocument();
    expect(stopSingle).toHaveBeenCalled();
  });

  it("every row's Compare toggle stays live: one id, no resolution to wait on", async () => {
    listing([run({ experiment_id: "run-fresh" })]);

    render(<TrainingTab />);
    await screen.findByText("run-fresh");

    expect(
      within(rowFor("run-fresh")).getByRole("button", { name: "Compare run-fresh" }),
    ).not.toBeDisabled();
  });

  it("caps the marked set and names the reason on a fifth toggle", async () => {
    openTestProject();
    const ids = ["run-1", "run-2", "run-3", "run-4", "run-5"];
    listing(ids.map((id) => run({ experiment_id: id })));
    compareAnswers([]);
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    await screen.findByText("run-1");
    for (const id of ["run-1", "run-2", "run-3", "run-4"]) {
      fireEvent.click(within(rowFor(id)).getByRole("button", { name: `Compare ${id}` }));
    }
    fireEvent.click(within(rowFor("run-5")).getByRole("button", { name: "Compare run-5" }));

    expect(pushToast).toHaveBeenCalledWith(expect.stringContaining("at most 4 runs"));
    // The fifth toggle stayed off: the fourth (last accepted) row is still the marked one.
    expect(within(rowFor("run-4")).getByRole("button", { name: "Compare run-4" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(within(rowFor("run-5")).getByRole("button", { name: "Compare run-5" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("groups Compare and Cancel as one action group", async () => {
    listing([run({ experiment_id: "run-grouped" })]);

    render(<TrainingTab />);
    await screen.findByText("run-grouped");

    const group = within(rowFor("run-grouped")).getByRole("group", { name: "Run actions" });
    expect(within(group).getByRole("button", { name: "Compare run-grouped" })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: "Cancel run-grouped" })).toBeInTheDocument();
  });

  it("selects the sole remaining run when unmarking drops the marked set to one, instead of stranding it unselected", async () => {
    openTestProject();
    listing([run({ experiment_id: "run-a" }), run({ experiment_id: "run-b" })]);
    compareAnswers(["run-a", "run-b"]);

    render(<TrainingTab />);
    await screen.findByText("run-a");
    fireEvent.click(within(rowFor("run-a")).getByRole("button", { name: "Compare run-a" }));
    fireEvent.click(within(rowFor("run-b")).getByRole("button", { name: "Compare run-b" }));
    expect(await screen.findByText("Comparing")).toBeInTheDocument();

    fireEvent.click(within(rowFor("run-a")).getByRole("button", { name: "Compare run-a" }));

    expect(screen.queryByText("Comparing")).not.toBeInTheDocument();
    expect(await screen.findByText("Waiting for metrics…")).toBeInTheDocument();
    expect(screen.queryByText("No run selected.")).not.toBeInTheDocument();
  });
});
