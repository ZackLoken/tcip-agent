import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { openTrainingStream, trainingApi, type SplitChoices } from "@/api/training";
import {
  NOT_FINITE_SUFFIX,
  TENSORBOARD_RELOAD_SELECTOR,
  type RunRow,
  type SweepGroup,
  type TrainingRowFrame,
  type TrainingStatusFrame,
} from "@/api/types.generated";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { useStore } from "@/store";
import { RUN_REFRESH_MS } from "@/tabs/trainingMetrics";
import { TrainingTab, dataPickerFor } from "@/tabs/TrainingTab";
import { openTestProject } from "@/test/store";
import SERVINGS from "@/test/tensorboardServings.json";
import {
  RUN_RELAUNCH,
  SWEEP_RELAUNCH,
  SWEEP_SELECTION_METRIC,
  UNKNOWN_RUN_FRAME,
  asRecordedChoice,
  batchFrame,
  cancelAnswer,
  compareEntry,
  compareResult,
  metricFrame,
  producedRefusal,
  refusalDetail,
  runRow,
  selectionChoice,
  splitChoices,
  statusFrame,
  sweepGroup,
  trainingDetail,
  trainingListing,
  trialRow,
} from "@/test/training";

// The live metrics stream owns a real WebSocket; only the run list and its controls are under
// test here, so the transport is replaced while the rest of the module stays real.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return { ...actual, openTrainingStream: vi.fn(() => () => {}) };
});

const initialStoreState = useStore.getState();

/** The run list answers the produced listing holding ``runs`` and ``sweeps`` on every poll. */
function listing(runs: RunRow[], sweeps: SweepGroup[] = []) {
  return vi.spyOn(trainingApi, "listRuns").mockResolvedValue(trainingListing({ runs, sweeps }));
}

/** One run's produced detail, holding ``row``, with no TensorBoard serving it. */
function detailOf(row: RunRow) {
  vi.spyOn(trainingApi, "getRun").mockResolvedValue(trainingDetail({ run: row }));
}

type Frame = TrainingRowFrame | TrainingStatusFrame;

/** Every stream the tab opens delivers ``frames`` at once, each under the id it was opened for. */
function streamFrames(...frames: Frame[]) {
  vi.mocked(openTrainingStream).mockImplementation((experimentId, onMessage) => {
    frames.forEach((frame) => onMessage({ ...frame, experiment_id: experimentId }));
    return () => {};
  });
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.setState({ user: "jordan" });
});

afterEach(cleanup);

describe("TrainingTab run list", () => {
  it("offers a stop control for a running run, and cancels it by id", async () => {
    listing([runRow({ experiment_id: "train-agent-1" })]);
    const cancelSpy = vi
      .spyOn(trainingApi, "cancel")
      .mockResolvedValue(cancelAnswer({ experiment_id: "train-agent-1" }));

    render(<TrainingTab />);
    expect(await screen.findByText("train-agent-1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Cancel train-agent-1" }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("train-agent-1", "jordan"));
  });

  it("shows a running row's own heartbeat age, the one liveness signal a record with no process id can offer", async () => {
    listing([
      runRow({
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

  it("shows a run's recorded error in its row", async () => {
    listing([
      runRow({
        experiment_id: "train-crashed",
        state: "failed",
        status_error: "CUDA out of memory at epoch 3",
      }),
    ]);

    render(<TrainingTab />);
    expect(await screen.findByText("CUDA out of memory at epoch 3")).toBeInTheDocument();
  });

  it("offers no stop control for a run in a terminal state", async () => {
    listing([
      runRow({ experiment_id: "train-done", state: "completed" }),
      runRow({ experiment_id: "train-done-agent", state: "failed" }),
    ]);

    render(<TrainingTab />);
    expect(await screen.findByText("train-done")).toBeInTheDocument();
    expect(screen.getByText("train-done-agent")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Cancel/ })).not.toBeInTheDocument();
  });

  it("names the best value's own metric exactly as the record carries it, and shows nothing when the row carries no name", async () => {
    listing([
      runRow({
        experiment_id: "train-named",
        state: "completed",
        best_metric: 0.812,
        best_metric_name: "map50",
      }),
      runRow({
        experiment_id: "train-loss-only",
        state: "completed",
        best_metric: 0.907,
        best_metric_name: "loss",
      }),
      runRow({
        experiment_id: "train-unnamed",
        state: "completed",
        best_metric: 0.5,
        best_metric_name: null,
      }),
    ]);

    render(<TrainingTab />);
    // The row names the metric as bare as the record does (a no-validation run's best can be the
    // training loss).
    expect(await screen.findByText("best map50 0.812")).toBeInTheDocument();
    expect(await screen.findByText("best loss 0.907")).toBeInTheDocument();
    expect(screen.queryByText(/val_/)).not.toBeInTheDocument();
    expect(screen.queryByText(/best.*0\.500/)).not.toBeInTheDocument();
  });

  it("says No runs. for an empty listing", async () => {
    const listRuns = listing([]);

    render(<TrainingTab />);
    expect(await screen.findByText("No runs.")).toBeInTheDocument();

    cleanup();
    listRuns.mockResolvedValue(
      trainingListing({ runs: [runRow({ experiment_id: "train-a" })], sweeps: [] }),
    );
    render(<TrainingTab />);
    expect(await screen.findByText("train-a")).toBeInTheDocument();
    expect(screen.queryByText(/sorted by experiment id/)).not.toBeInTheDocument();
  });

  it("names the row's own select control with the id and state", async () => {
    listing([runRow({ experiment_id: "train-named-row" })]);

    render(<TrainingTab />);
    expect(
      await screen.findByRole("button", {
        name: "train-named-row running, no launch event recorded",
      }),
    ).toBeInTheDocument();
  });

  it("shows the row's best value at four significant figures, with no title", async () => {
    listing([
      runRow({
        experiment_id: "train-exact",
        state: "completed",
        best_metric: 0.4130041,
        best_metric_name: "loss",
      }),
    ]);

    render(<TrainingTab />);
    const value = await screen.findByText("best loss 0.413");
    expect(value).not.toHaveAttribute("title");
  });

  it("names the row's accessible name with its best value and metric when the record carries one", async () => {
    listing([
      runRow({
        experiment_id: "train-named-value",
        state: "completed",
        best_metric: 0.4130041,
        best_metric_name: "loss",
      }),
    ]);

    render(<TrainingTab />);
    expect(
      await screen.findByRole("button", {
        name: "train-named-value completed, no launch event recorded, best loss 0.413",
      }),
    ).toBeInTheDocument();
  });

  it("shows an in-flight Cancel as a disabled, pending row control, and a failure in the row", async () => {
    listing([runRow({ experiment_id: "train-cancel-flight" })]);
    const cancel = Promise.withResolvers<Awaited<ReturnType<typeof trainingApi.cancel>>>();
    vi.spyOn(trainingApi, "cancel").mockReturnValue(cancel.promise);

    render(<TrainingTab />);
    const button = await screen.findByRole("button", { name: "Cancel train-cancel-flight" });
    fireEvent.click(button);

    expect(await screen.findByText("Canceling…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel train-cancel-flight" })).toBeDisabled();

    cancel.resolve(cancelAnswer({ experiment_id: "train-cancel-flight" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Cancel train-cancel-flight" })).not.toBeDisabled(),
    );
  });

  it("shows a failed Cancel in the row, not only as a toast", async () => {
    listing([runRow({ experiment_id: "train-cancel-fail" })]);
    vi.spyOn(trainingApi, "cancel").mockRejectedValue(new Error("network error"));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel train-cancel-fail" }));

    expect(await screen.findByText("Cancel failed: network error")).toBeInTheDocument();
  });
});

describe("TrainingTab sweeps", () => {
  const trialA = trialRow({
    experiment_id: "train-x",
    sweep: "sweep-1",
    state: "completed",
    trial_params: { lr: 0.01 },
  });
  const trialB = trialRow({ experiment_id: "sweep-1_bbbb", sweep: "sweep-1" });

  it("groups a sweep's trial rows under the sweep, with its state and outcome", async () => {
    const produced = sweepGroup({});
    listing(
      [runRow({ experiment_id: "train-solo" })],
      [
        sweepGroup({
          sweep_id: "sweep-1",
          trials: [trialA, trialB],
          outcome: { ...produced.outcome, best_params: { lr: 0.01 }, best_value: 0.7 },
        }),
      ],
    );

    render(<TrainingTab />);
    const group = await screen.findByRole("group", { name: "Sweep sweep-1" });
    expect(within(group).getByText("train-x")).toBeInTheDocument();
    expect(within(group).getByText("sweep-1_bbbb")).toBeInTheDocument();
    expect(
      within(group).getByText(
        new RegExp(`sweep · ${produced.state} · best ${SWEEP_SELECTION_METRIC} 0.7`),
      ),
    ).toBeInTheDocument();
    expect(within(group).queryByText("train-solo")).not.toBeInTheDocument();
  });

  it("cancels a trial by its own id and the whole sweep by the sweep's", async () => {
    listing([], [sweepGroup({ sweep_id: "sweep-1", trials: [trialB] })]);
    const cancelSpy = vi.spyOn(trainingApi, "cancel").mockResolvedValue(cancelAnswer({}));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel sweep-1_bbbb" }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("sweep-1_bbbb", "jordan"));
    fireEvent.click(screen.getByRole("button", { name: "Cancel sweep-1" }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith("sweep-1", "jordan"));
  });

  it("streams a selected trial's metrics, and shows a selected sweep's board without a stream", async () => {
    openTestProject();
    listing([], [sweepGroup({ sweep_id: "sweep-1", trials: [trialA] })]);
    const getRun = vi
      .spyOn(trainingApi, "getRun")
      .mockResolvedValue(trainingDetail({ run: trialA, tensorboard: SERVINGS["train-x"] }));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-x"));
    await waitFor(() =>
      expect(openTrainingStream).toHaveBeenCalledWith("train-x", expect.any(Function)),
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
      runRow({
        experiment_id: "train-agent",
        launch: { agent_client_name: "claude-code", agent_client_version: "2.1.238" },
      }),
      runRow({ experiment_id: "train-undeclared", launch: {} }),
      runRow({ experiment_id: "train-no-event", state: "completed", launch: null }),
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
      runRow({
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
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      "Launched by the agent claude-code 2.1.238.",
    );
  });

  it("describes a launch with no agent named", async () => {
    listing([runRow({ experiment_id: "train-undeclared-desc", launch: {} })]);

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: "train-undeclared-desc running, started with no agent declared",
    });

    const describedBy = button.getAttribute("aria-describedby");
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      "Launched with no agent named.",
    );
  });

  it("makes a missing launch event's explanation reachable by assistive technology", async () => {
    listing([runRow({ experiment_id: "train-no-launcher" })]);

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: "train-no-launcher running, no launch event recorded",
    });

    const describedBy = button.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      "No launch event names this run.",
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
    const row = runRow({ experiment_id: "train-x", state: "completed" });
    listing([row]);
    vi.spyOn(trainingApi, "getRun").mockResolvedValue(
      trainingDetail({ run: row, tensorboard: SERVINGS["train-x"] }),
    );

    render(<TrainingTab />);
    expect(await screen.findByRole("heading", { level: 2, name: "Runs" })).toBeInTheDocument();
    fireEvent.click(screen.getByText("train-x"));
    expect(
      await screen.findByRole("heading", { level: 2, name: "TensorBoard" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Metrics" })).toBeInTheDocument();
  });
});

describe("TrainingTab tensorboard panel", () => {
  it("says a run produced no logs and offers no Try again, rather than a raw refusal", async () => {
    const row = runRow({ experiment_id: "train-nologs", state: "failed" });
    listing([row]);
    detailOf(row);
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(
      await producedRefusal("no_logs_refusal"),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-nologs"));

    expect(await screen.findByText("This run produced no logs.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
  });

  it("keeps Try again for an ordinary failed launch", async () => {
    const row = runRow({ experiment_id: "train-broken", state: "failed" });
    listing([row]);
    detailOf(row);
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(new Error("connection refused"));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-broken"));

    expect(await screen.findByText("connection refused")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again: TensorBoard" })).toBeInTheDocument();
  });

  it("keeps the run's own recorded crash reason for a no-logs run that failed with one", async () => {
    const crash = refusalDetail("crashed_no_logs_refusal").message;
    const row = runRow({
      experiment_id: "train-nologs-crashed",
      state: "failed",
      status_error: crash,
    });
    listing([row]);
    detailOf(row);
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(
      await producedRefusal("crashed_no_logs_refusal"),
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
      runRow({ experiment_id: "train-x", state: "completed" }),
      runRow({ experiment_id: "train-y", state: "completed" }),
    ]);
    vi.spyOn(trainingApi, "getRun").mockImplementation((experimentId: string) =>
      Promise.resolve(
        trainingDetail({ run: runRow({ experiment_id: experimentId, state: "completed" }) }),
      ),
    );
    const launchSpy = vi
      .spyOn(trainingApi, "launchTensorboard")
      .mockImplementation((experimentId: string) =>
        Promise.resolve(SERVINGS[experimentId as keyof typeof SERVINGS]),
      );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-x"));
    await waitFor(() => expect(launchSpy).toHaveBeenCalledWith("train-x"));

    fireEvent.click(screen.getByText("train-y"));
    await waitFor(() => expect(launchSpy).toHaveBeenCalledWith("train-y"));
    expect(screen.getByTitle("TensorBoard")).toHaveAttribute("src", SERVINGS["train-y"].url);
  });

  it("reloads TensorBoard in place once per new epoch row, never remounting its frame", async () => {
    openTestProject();
    const row = runRow({ experiment_id: "train-x" });
    listing([row]);
    vi.spyOn(trainingApi, "getRun").mockResolvedValue(
      trainingDetail({ run: row, tensorboard: SERVINGS["train-x"] }),
    );
    let deliver: (frame: Frame) => void = () => {};
    vi.mocked(openTrainingStream).mockImplementation((experimentId, onMessage) => {
      deliver = (frame) => onMessage({ ...frame, experiment_id: experimentId });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-x"));
    const frame = (await screen.findByTitle("TensorBoard")) as HTMLIFrameElement;
    const page = frame.contentDocument as Document;
    const header = page.createElement("app-header-reload");
    header.innerHTML = '<button class="reload-button">reload</button>';
    page.append(header);
    const button = page.querySelector<HTMLButtonElement>(TENSORBOARD_RELOAD_SELECTOR)!;
    const reloads = vi.fn();
    button.addEventListener("click", reloads);
    const srcWrites: MutationRecord[] = [];
    new MutationObserver((records) => srcWrites.push(...records)).observe(frame, {
      attributeFilter: ["src"],
    });

    act(() => deliver(metricFrame({ epoch: 1, loss: 0.5 })));
    expect(reloads).toHaveBeenCalledTimes(1);
    act(() => deliver(metricFrame({ epoch: 1, loss: 0.4 })));
    expect(reloads).toHaveBeenCalledTimes(1);
    act(() => deliver(metricFrame({ epoch: 2, loss: 0.3 })));
    expect(reloads).toHaveBeenCalledTimes(2);
    button.disabled = true;
    act(() => deliver(metricFrame({ epoch: 3, loss: 0.2 })));
    expect(reloads).toHaveBeenCalledTimes(2);

    expect(screen.getByTitle("TensorBoard")).toBe(frame);
    await Promise.resolve();
    expect(srcWrites).toHaveLength(0);
  });
});

describe("TrainingTab epoch table", () => {
  beforeEach(() => {
    openTestProject();
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
  });

  it("says a terminal run recorded no metrics, rather than waiting on one that will never arrive", async () => {
    listing([runRow({ experiment_id: "train-terminal-empty", state: "interrupted" })]);

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-terminal-empty"));

    expect(await screen.findByText("This run recorded no metrics.")).toBeInTheDocument();
    expect(screen.queryByText("Waiting for metrics…")).not.toBeInTheDocument();
  });

  it("marks the waiting placeholder as a live region on a run with no metrics yet", async () => {
    listing([runRow({ experiment_id: "train-waiting-live" })]);

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-waiting-live"));

    expect(await screen.findByRole("status")).toHaveTextContent("Waiting for metrics…");
  });

  it("shows every streamed epoch as a table row, outright", async () => {
    listing([runRow({ experiment_id: "train-table" })]);
    streamFrames(metricFrame({ epoch: 1, loss: 0.5 }), metricFrame({ epoch: 2, loss: 0.3 }));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-table"));

    const table = await screen.findByRole("table", { name: "train-table metrics by epoch" });
    expect(within(table).getAllByRole("row")).toHaveLength(3);
    expect(within(table).getByRole("columnheader", { name: "epoch" })).toBeInTheDocument();
    expect(table.querySelector("thead")).toHaveClass("sticky", "top-0");
    expect(within(table).getByText("0.5")).toBeInTheDocument();
    expect(within(table).getByText("0.3")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "as table" })).not.toBeInTheDocument();
  });

  it("renders the unset glyph for a row carrying no epoch, and for a non-finite value", async () => {
    listing([runRow({ experiment_id: "train-unset" })]);
    streamFrames(metricFrame({ loss: 0.5 }), metricFrame({ epoch: 2, loss: null }));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-unset"));

    const rows = within(await screen.findByRole("table")).getAllByRole("row");
    expect(within(rows[1]).getByText(UNSET_GLYPH)).toBeInTheDocument();
    expect(within(rows[1]).getByText("0.5")).toBeInTheDocument();
    expect(within(rows[2]).getByText(UNSET_GLYPH)).toBeInTheDocument();
  });

  it("spells a metric key's underscores as spaces and gives bookkeeping no column", async () => {
    listing([runRow({ experiment_id: "train-bookkeeping" })]);
    streamFrames(
      metricFrame({
        epoch: 1,
        loss: 0.5,
        val_map50: 0.4,
        [`loss${NOT_FINITE_SUFFIX}`]: "nan",
        timestamp: "2026-01-01T00:00:00Z",
      }),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-bookkeeping"));

    const headers = within(await screen.findByRole("table"))
      .getAllByRole("columnheader")
      .map((h) => h.textContent);
    expect(headers).toEqual(["epoch", "loss", "val map50"]);
  });

  it("shows the latest batch row as the in-progress epoch's line, with no column of its own", async () => {
    listing([runRow({ experiment_id: "train-batches" })]);
    streamFrames(
      metricFrame({ epoch: 1, loss: 0.5 }),
      batchFrame({ epoch: 2, step: 7, batch_loss: 0.25 }),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-batches"));

    const table = await screen.findByRole("table", { name: "train-batches metrics by epoch" });
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(3);
    expect(within(rows[2]).getByText("in progress · step 7 · batch_loss 0.25")).toBeInTheDocument();
    expect(
      within(table)
        .getAllByRole("columnheader")
        .map((h) => h.textContent),
    ).toEqual(["epoch", "loss"]);
  });

  it("drops the in-progress line once its epoch's own row arrives", async () => {
    listing([runRow({ experiment_id: "train-batch-done" })]);
    streamFrames(
      batchFrame({ epoch: 1, step: 3, batch_loss: 0.25 }),
      metricFrame({ epoch: 1, loss: 0.5 }),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-batch-done"));

    const table = await screen.findByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(2);
    expect(within(table).queryByText(/in progress/)).not.toBeInTheDocument();
  });

  it("names the last batch line of a run that ended mid-epoch as where it stopped", async () => {
    listing([runRow({ experiment_id: "train-batch-failed", state: "failed" })]);
    streamFrames(batchFrame({ epoch: 2, step: 5, batch_loss: 0.25 }));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-batch-failed"));

    const table = await screen.findByRole("table");
    expect(within(table).getByText("stopped at step 5 · batch_loss 0.25")).toBeInTheDocument();
    expect(within(table).queryByText(/in progress/)).not.toBeInTheDocument();
  });
});

describe("TrainingTab status toast", () => {
  beforeEach(() => {
    openTestProject();
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
  });

  function completedFrame(experimentId: string): Frame {
    return statusFrame({ experiment_id: experimentId, state: "completed" });
  }

  it("does not toast on selecting a run already known to be terminal, only rediscovering its state", async () => {
    listing([runRow({ experiment_id: "train-old-done", state: "completed" })]);
    streamFrames(completedFrame("train-old-done"));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-old-done"));

    await waitFor(() => expect(trainingApi.listRuns).toHaveBeenCalled());
    expect(pushToast).not.toHaveBeenCalledWith("Training train-old-done: completed", "info");
  });

  it("toasts a transition to terminal observed while watching a live run", async () => {
    listing([runRow({ experiment_id: "train-live-done" })]);
    streamFrames(completedFrame("train-live-done"));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-live-done"));

    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith("Training train-live-done: completed", "info"),
    );
  });

  it("toasts a stream error once, then still shows a metric the reconnect delivers", async () => {
    listing([runRow({ experiment_id: "train-just-launched" })]);
    streamFrames(UNKNOWN_RUN_FRAME, UNKNOWN_RUN_FRAME, metricFrame({ epoch: 1, loss: 0.5 }));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-just-launched"));

    expect(within(await screen.findByRole("table")).getByText("0.5")).toBeInTheDocument();
    expect(pushToast).toHaveBeenCalledWith(`Training stream error: ${UNKNOWN_RUN_FRAME.error}`);
    expect(pushToast).toHaveBeenCalledTimes(1);
  });
});

describe("TrainingTab stream lifecycle", () => {
  it("closes the stream and clears the detail panel when the selected run leaves the next poll", async () => {
    openTestProject();
    const listRuns = vi.spyOn(trainingApi, "listRuns");
    listRuns.mockResolvedValueOnce(
      trainingListing({ runs: [runRow({ experiment_id: "train-vanishing" })], sweeps: [] }),
    );
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    const stop = vi.fn();
    vi.mocked(openTrainingStream).mockImplementation(() => stop);

    vi.useFakeTimers();
    try {
      render(<TrainingTab />);
      await vi.waitFor(() => expect(screen.getByText("train-vanishing")).toBeInTheDocument());
      fireEvent.click(screen.getByText("train-vanishing"));
      await vi.waitFor(() => expect(openTrainingStream).toHaveBeenCalledTimes(1));

      listRuns.mockResolvedValueOnce(trainingListing({ runs: [], sweeps: [] }));
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
  it("opens the picker, selects a recorded run, and starts it through the relaunch door", async () => {
    listing([runRow({ experiment_id: "exp-pristine-1", state: "completed" })]);
    vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue(splitChoices({}));
    const relaunchSpy = vi.spyOn(trainingApi, "relaunch").mockResolvedValue(RUN_RELAUNCH);

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
    listing([], [sweepGroup({ sweep_id: "sweep-1", state: "completed" })]);
    const choicesSpy = vi.spyOn(trainingApi, "listSplitChoices");
    const relaunchSpy = vi.spyOn(trainingApi, "relaunch").mockResolvedValue(SWEEP_RELAUNCH);

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
    listing([runRow({ experiment_id: "exp-refused-1", state: "completed" })]);
    vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue(splitChoices({}));
    vi.spyOn(trainingApi, "relaunch").mockRejectedValue(await producedRefusal("refused_relaunch"));

    render(<TrainingTab />);
    await screen.findByText("exp-refused-1");
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;

    fireEvent.click(within(picker).getByText("exp-refused-1"));
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    const [issue] = refusalDetail("refused_relaunch").issues;
    await waitFor(() => expect(screen.getByText(issue)).toBeInTheDocument());
  });

  it("opens a row's Data choices only once selected, and starts with the chosen manifest", async () => {
    listing([runRow({ experiment_id: "exp-1", state: "completed" })]);
    const choices = splitChoices({});
    const [selection] = choices.selections;
    const listSplitChoicesSpy = vi
      .spyOn(trainingApi, "listSplitChoices")
      .mockResolvedValue(choices);
    const relaunchSpy = vi.spyOn(trainingApi, "relaunch").mockResolvedValue(RUN_RELAUNCH);

    render(<TrainingTab />);
    await screen.findByText("exp-1");
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;
    expect(listSplitChoicesSpy).not.toHaveBeenCalled();

    fireEvent.click(within(picker).getByText("exp-1"));
    expect(
      await screen.findByText((text) => text.includes(choices.as_recorded.line)),
    ).toBeInTheDocument();
    expect(listSplitChoicesSpy).toHaveBeenCalledWith("exp-1");

    fireEvent.click(
      await screen.findByRole("radio", {
        name: (name) => name.includes(selection.selection_dir),
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(relaunchSpy).toHaveBeenCalledWith("exp-1", "jordan", selection.selection_dir),
    );
  });

  it("disables Start while a selected row's data choices are still loading", async () => {
    listing([runRow({ experiment_id: "exp-1", state: "completed" })]);
    const choices = Promise.withResolvers<SplitChoices>();
    vi.spyOn(trainingApi, "listSplitChoices").mockReturnValue(choices.promise);

    render(<TrainingTab />);
    await screen.findByText("exp-1");
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;
    fireEvent.click(within(picker).getByText("exp-1"));

    expect(await screen.findByRole("button", { name: "Start" })).toBeDisabled();

    choices.resolve(splitChoices({}));
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
  });

  it("shows a per-row split-choices fetch failure as text on the row", async () => {
    listing([runRow({ experiment_id: "exp-1", state: "completed" })]);
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
      .mockResolvedValue(trainingListing({ runs: [], sweeps: [] }));

    render(<TrainingTab />);
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    const picker = screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;

    expect(
      await within(picker).findByText("Could not load training runs: network error"),
    ).toBeInTheDocument();
    expect(within(picker).queryByText("No runs.")).not.toBeInTheDocument();

    fireEvent.click(within(picker).getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(listRuns).toHaveBeenCalledTimes(2));
    expect(await within(picker).findByText("No runs.")).toBeInTheDocument();
  });
});

describe("dataPickerFor", () => {
  const recorded = selectionChoice({});
  const choices = splitChoices({
    as_recorded: asRecordedChoice({
      compatible: false,
      reason: "Not found: data.images_dir = '/moved'",
    }),
    selections: [
      recorded,
      selectionChoice({
        selection_dir: "/data/other",
        enabled: false,
        reason: "the selection at /data/other records no subject",
        seed: null,
        group_by: null,
        train: 0,
        val: 0,
        calibration: 0,
      }),
    ],
  });

  it("maps the snapshot's own compatibility into the disabled/reason pair the picker renders", () => {
    const picker = dataPickerFor(choices);
    expect(picker?.asRecordedLine).toBe(choices.as_recorded.line);
    expect(picker?.asRecordedDisabled).toBe(true);
    expect(picker?.asRecordedReason).toBe("Not found: data.images_dir = '/moved'");
    expect(picker?.absenceMessage).toMatch(/this listing found no other recorded partition/);
  });

  it("maps each selection's own enabled flag and reason onto the choice the picker renders", () => {
    const picker = dataPickerFor(choices);
    expect(picker?.choices[0]).toMatchObject({
      selectionDir: recorded.selection_dir,
      disabled: false,
      replacedSplitKeys: recorded.replaced_split_keys,
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
      screen.getByText(
        `seed ${recorded.seed} · ${recorded.group_by} · train ${recorded.train} · val ${recorded.val} · calibration ${recorded.calibration}`,
      ),
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

  /** The comparison answers the produced comparison with one produced column per id. */
  function compareAnswers(ids: string[]) {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      compareResult({ experiments: ids.map((experiment_id) => compareEntry({ experiment_id })) }),
    );
  }

  it("marking two runs switches the detail region and closes the single-run stream", async () => {
    openTestProject();
    listing([runRow({ experiment_id: "run-a" }), runRow({ experiment_id: "run-b" })]);
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
    listing([runRow({ experiment_id: "run-fresh" })]);

    render(<TrainingTab />);
    await screen.findByText("run-fresh");

    expect(
      within(rowFor("run-fresh")).getByRole("button", { name: "Compare run-fresh" }),
    ).not.toBeDisabled();
  });

  it("caps the marked set and names the reason on a fifth toggle", async () => {
    openTestProject();
    const ids = ["run-1", "run-2", "run-3", "run-4", "run-5"];
    listing(ids.map((id) => runRow({ experiment_id: id })));
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
    listing([runRow({ experiment_id: "run-grouped" })]);

    render(<TrainingTab />);
    await screen.findByText("run-grouped");

    const group = within(rowFor("run-grouped")).getByRole("group", { name: "Run actions" });
    expect(within(group).getByRole("button", { name: "Compare run-grouped" })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: "Cancel run-grouped" })).toBeInTheDocument();
  });

  it("selects the sole remaining run when unmarking drops the marked set to one, instead of stranding it unselected", async () => {
    openTestProject();
    listing([runRow({ experiment_id: "run-a" }), runRow({ experiment_id: "run-b" })]);
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
