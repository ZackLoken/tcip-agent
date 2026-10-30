import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { openTrainingStream, trainingApi } from "@/api/training";
import {
  NOT_FINITE_SUFFIX,
  TENSORBOARD_RELOAD_SELECTOR,
  type SplitChoices,
  type TrainingListing,
  type TrainingRowFrame,
  type TrainingStatusFrame,
} from "@/api/types.generated";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { formatNumber } from "@/lib/numberFormat";
import { useStore } from "@/store";
import { numericMetricKeys, RUN_REFRESH_MS } from "@/tabs/trainingMetrics";
import { TrainingTab } from "@/tabs/TrainingTab";
import { thrown } from "@/test/refusals";
import { openTestProject } from "@/test/store";
import { servingOf } from "@/test/tensorboard";
import {
  AGENT_CLIENT,
  BATCH_FRAMES,
  CANCEL,
  COMPARISONS,
  LISTING_BEFORE_FAILURE,
  METRIC_FRAMES,
  RUN_RELAUNCH,
  RUNS,
  SPLIT_CHOICES,
  STATUS_FRAME,
  STREAM,
  SWEEP_RELAUNCH,
  SWEEPS,
  TRAINING_LISTING,
  TRAINING_REFUSALS,
  TRIALS,
  UNKNOWN_RUN_FRAME,
  detailOf,
} from "@/test/training";

// The live metrics stream owns a real WebSocket; only the run list and its controls are under
// test here, so the transport is replaced while the rest of the module stays real.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return { ...actual, openTrainingStream: vi.fn(() => () => {}) };
});

const initialStoreState = useStore.getState();

/** The run list answers `answer`, by default the produced listing, on every poll. */
function listing(answer: TrainingListing = TRAINING_LISTING) {
  return vi.spyOn(trainingApi, "listRuns").mockResolvedValue(answer);
}

/** Every detail read answers the produced detail of the id it names. */
function details() {
  return vi
    .spyOn(trainingApi, "getRun")
    .mockImplementation((experimentId: string) => Promise.resolve(detailOf(experimentId)));
}

/** The row of the run `id` names in `inside`, by default the run list's own. */
function rowFor(id: string, inside: HTMLElement = document.body): HTMLElement {
  return within(inside).getAllByText(id)[0].closest("li") as HTMLElement;
}

/** The produced best value as a row shows it, through the frontend's own number format. */
const BEST = `best ${RUNS.completed.best_metric_name} ${formatNumber(Number(RUNS.completed.best_metric))}`;

type Frame = TrainingRowFrame | TrainingStatusFrame;

/** Every stream the tab opens delivers the produced ``frames`` at once. */
function streamFrames(...frames: Frame[]) {
  vi.mocked(openTrainingStream).mockImplementation((_experimentId, onMessage) => {
    frames.forEach(onMessage);
    return () => {};
  });
}

/** Pin the clock `minutes` after the produced heartbeat of `run`. */
function atHeartbeat(run: { heartbeat: string }, minutes: number) {
  vi.setSystemTime(Date.parse(run.heartbeat) + minutes * 60_000);
}

/** Every running row's own heartbeat age as it reads with the clock at the first run's heartbeat:
 *  the produced heartbeats lie seconds apart. */
const FRESH = "last heartbeat under a minute ago";

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.setState({ user: "jordan" });
  vi.useFakeTimers({ toFake: ["Date"] });
  atHeartbeat(RUNS.unlaunched, 0);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("TrainingTab run list", () => {
  const { unlaunched, completed, unlogged } = RUNS;
  const cancelOf = (id: string) => ({ name: `Cancel ${id}` });

  it("offers a stop control for a running run, and cancels it by id", async () => {
    listing();
    const cancelSpy = vi.spyOn(trainingApi, "cancel").mockResolvedValue(CANCEL);

    render(<TrainingTab />);
    expect(await screen.findByText(unlaunched.experiment_id)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", cancelOf(unlaunched.experiment_id)));
    await waitFor(() =>
      expect(cancelSpy).toHaveBeenCalledWith(unlaunched.experiment_id, { user: "jordan" }),
    );
  });

  it("shows a running row's own heartbeat age, the one liveness signal a record with no process id can offer", async () => {
    atHeartbeat(unlaunched, 3);
    listing();

    render(<TrainingTab />);
    await screen.findByText(unlaunched.experiment_id);
    expect(
      within(rowFor(unlaunched.experiment_id)).getByText(/running, last heartbeat 3 min ago/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: new RegExp(`${unlaunched.experiment_id} running, last heartbeat 3 min ago`),
      }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", cancelOf(unlaunched.experiment_id))).toHaveAttribute(
      "title",
      "Reaches a live process only; a stale running row keeps this control until its heartbeat window lapses.",
    );
  });

  it("shows a run's recorded error in its row", async () => {
    listing();

    render(<TrainingTab />);
    await screen.findByText(unlogged.experiment_id);
    expect(
      within(rowFor(unlogged.experiment_id)).getByText(String(unlogged.status_error)),
    ).toBeInTheDocument();
  });

  it("offers no stop control for a run in a terminal state", async () => {
    listing();

    render(<TrainingTab />);
    await screen.findByText(completed.experiment_id);
    for (const { experiment_id } of [completed, unlogged]) {
      expect(
        within(rowFor(experiment_id)).queryByRole("button", { name: /^Cancel/ }),
      ).not.toBeInTheDocument();
    }
  });

  it("names the best value's own metric exactly as the record carries it, and shows nothing for a row carrying no best value", async () => {
    listing();

    render(<TrainingTab />);
    await screen.findByText(completed.experiment_id);
    expect(within(rowFor(completed.experiment_id)).getByText(BEST)).toBeInTheDocument();
    expect(within(rowFor(unlaunched.experiment_id)).queryByText(/^best /)).not.toBeInTheDocument();
    expect(screen.queryByText(/val_/)).not.toBeInTheDocument();
  });

  it("says No runs. for an empty listing", async () => {
    const listRuns = listing({ runs: [], sweeps: [] });

    render(<TrainingTab />);
    expect(await screen.findByText("No runs.")).toBeInTheDocument();

    cleanup();
    listRuns.mockResolvedValue(TRAINING_LISTING);
    render(<TrainingTab />);
    expect(await screen.findByText(unlaunched.experiment_id)).toBeInTheDocument();
    expect(screen.queryByText(/sorted by experiment id/)).not.toBeInTheDocument();
  });

  it("names the row's own select control with the id and state", async () => {
    listing();

    render(<TrainingTab />);
    expect(
      await screen.findByRole("button", {
        name: `${unlaunched.experiment_id} running, ${FRESH}, no launch event recorded`,
      }),
    ).toBeInTheDocument();
  });

  it("shows the row's best value at four significant figures, with no title", async () => {
    listing();

    render(<TrainingTab />);
    await screen.findByText(completed.experiment_id);
    const value = within(rowFor(completed.experiment_id)).getByText(BEST);
    expect(value).not.toHaveAttribute("title");
  });

  it("names the row's accessible name with its best value and metric when the record carries one", async () => {
    listing();

    render(<TrainingTab />);
    expect(
      await screen.findByRole("button", {
        name: `${completed.experiment_id} completed, no launch event recorded, ${BEST}`,
      }),
    ).toBeInTheDocument();
  });

  it("shows an in-flight Cancel as a disabled, pending row control, and a failure in the row", async () => {
    listing();
    const cancel = Promise.withResolvers<Awaited<ReturnType<typeof trainingApi.cancel>>>();
    vi.spyOn(trainingApi, "cancel").mockReturnValue(cancel.promise);

    render(<TrainingTab />);
    const button = await screen.findByRole("button", cancelOf(unlaunched.experiment_id));
    fireEvent.click(button);

    expect(await screen.findByText("Canceling…")).toBeInTheDocument();
    expect(screen.getByRole("button", cancelOf(unlaunched.experiment_id))).toBeDisabled();

    cancel.resolve(CANCEL);
    await waitFor(() =>
      expect(screen.getByRole("button", cancelOf(unlaunched.experiment_id))).not.toBeDisabled(),
    );
  });

  it("shows a failed Cancel in the row, not only as a toast", async () => {
    listing();
    vi.spyOn(trainingApi, "cancel").mockRejectedValue(new Error("network error"));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", cancelOf(unlaunched.experiment_id)));

    expect(await screen.findByText("Cancel failed: network error")).toBeInTheDocument();
  });
});

describe("TrainingTab sweeps", () => {
  const { searched } = SWEEPS;
  const sweepGroupNamed = { name: `Sweep ${searched.sweep_id}` };

  it("groups a sweep's trial rows under the sweep, with its state and outcome", async () => {
    const best = searched.outcome.best_value;
    expect(best).not.toBeNull();
    listing();

    render(<TrainingTab />);
    const group = await screen.findByRole("group", sweepGroupNamed);
    expect(within(group).getByText(TRIALS.completed.experiment_id)).toBeInTheDocument();
    expect(within(group).getByText(TRIALS.running.experiment_id)).toBeInTheDocument();
    expect(
      within(group).getByText(
        `sweep · ${searched.state}, ${FRESH} · best ${searched.selection_metric} ${formatNumber(best as number)}`,
        { exact: false },
      ),
    ).toBeInTheDocument();
    expect(within(group).queryByText(RUNS.unlaunched.experiment_id)).not.toBeInTheDocument();
  });

  it("cancels a trial by its own id and the whole sweep by the sweep's", async () => {
    listing();
    const cancelSpy = vi.spyOn(trainingApi, "cancel").mockResolvedValue(CANCEL);
    const trial = TRIALS.running.experiment_id;

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", { name: `Cancel ${trial}` }));
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith(trial, { user: "jordan" }));
    fireEvent.click(screen.getByRole("button", { name: `Cancel ${searched.sweep_id}` }));
    await waitFor(() =>
      expect(cancelSpy).toHaveBeenCalledWith(searched.sweep_id, { user: "jordan" }),
    );
  });

  it("streams a selected trial's metrics, and shows a selected sweep's board without a stream", async () => {
    openTestProject();
    listing();
    const getRun = details();
    const trial = TRIALS.completed.experiment_id;

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(trial));
    await waitFor(() =>
      expect(openTrainingStream).toHaveBeenCalledWith(trial, expect.any(Function)),
    );

    vi.mocked(openTrainingStream).mockClear();
    const group = screen.getByRole("group", sweepGroupNamed);
    fireEvent.click(
      within(group).getByRole("button", { name: new RegExp(`^${searched.sweep_id}\\s`) }),
    );
    await waitFor(() => expect(getRun).toHaveBeenCalledWith(searched.sweep_id));
    expect(openTrainingStream).not.toHaveBeenCalled();
  });
});

describe("TrainingTab run launcher mark", () => {
  const { agent, relaunched, unlaunched, completed } = RUNS;

  it("states who launched the run from its launch event alone", async () => {
    listing();

    render(<TrainingTab />);
    await screen.findByText(agent.experiment_id);

    expect(
      screen.getByRole("button", {
        name: `${agent.experiment_id} running, ${FRESH}, started by the agent`,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: `${relaunched.experiment_id} running, ${FRESH}, started with no agent declared`,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: `${completed.experiment_id} completed, no launch event recorded, ${BEST}`,
      }),
    ).toBeInTheDocument();
  });

  it("names the declared client in the agent row's accessible description", async () => {
    listing();

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: `${agent.experiment_id} running, ${FRESH}, started by the agent`,
    });

    const describedBy = button.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      `Launched by the agent ${AGENT_CLIENT}.`,
    );
  });

  it("describes a launch with no agent named", async () => {
    listing();

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: `${relaunched.experiment_id} running, ${FRESH}, started with no agent declared`,
    });

    const describedBy = button.getAttribute("aria-describedby");
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      "Launched with no agent named.",
    );
  });

  it("makes a missing launch event's explanation reachable by assistive technology", async () => {
    listing();

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: `${unlaunched.experiment_id} running, ${FRESH}, no launch event recorded`,
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
    listing();
    render(<TrainingTab />);
    await waitFor(() => expect(trainingApi.listRuns).toHaveBeenCalled());
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Training");
  });

  it("carries the section titles as level-2 headings under that h1", async () => {
    const id = RUNS.unlaunched.experiment_id;
    listing();
    details();
    vi.spyOn(trainingApi, "launchTensorboard").mockResolvedValue(servingOf(id));

    render(<TrainingTab />);
    expect(await screen.findByRole("heading", { level: 2, name: "Runs" })).toBeInTheDocument();
    fireEvent.click(await screen.findByText(id));
    expect(
      await screen.findByRole("heading", { level: 2, name: "TensorBoard" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Metrics" })).toBeInTheDocument();
  });
});

describe("TrainingTab tensorboard panel", () => {
  it("says a run produced no logs and offers no Try again, rather than a raw refusal", async () => {
    const id = RUNS.empty.experiment_id;
    listing();
    details();
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(
      await thrown(TRAINING_REFUSALS.no_logs_refusal),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(id));

    expect(
      await screen.findByText(TRAINING_REFUSALS.no_logs_refusal.body.detail.message),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
  });

  it("keeps Try again for an ordinary failed launch", async () => {
    const id = RUNS.unlogged.experiment_id;
    listing();
    details();
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(new Error("connection refused"));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(id));

    expect(await screen.findByText("connection refused")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again: TensorBoard" })).toBeInTheDocument();
  });

  it("keeps the run's own recorded crash reason for a no-logs run that failed with one", async () => {
    const { message } = TRAINING_REFUSALS.crashed_no_logs_refusal.body.detail;
    const id = RUNS.unlogged.experiment_id;
    expect(message).toContain(String(RUNS.unlogged.status_error));
    listing();
    details();
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(
      await thrown(TRAINING_REFUSALS.crashed_no_logs_refusal),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(id));

    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Try again: TensorBoard" }),
    ).not.toBeInTheDocument();
  });

  it("launches a fresh TensorBoard for the newly selected run, not the previous one's", async () => {
    const [first, second] = [RUNS.unlaunched, RUNS.streamed].map((r) => r.experiment_id);
    listing();
    details();
    const launchSpy = vi
      .spyOn(trainingApi, "launchTensorboard")
      .mockImplementation((experimentId: string) => Promise.resolve(servingOf(experimentId)));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(first));
    await waitFor(() => expect(launchSpy).toHaveBeenCalledWith(first));

    fireEvent.click(screen.getByText(second));
    await waitFor(() => expect(launchSpy).toHaveBeenCalledWith(second));
    expect(screen.getByTitle("TensorBoard")).toHaveAttribute("src", servingOf(second).url);
  });

  it("reloads TensorBoard in place once per new epoch row, never remounting its frame", async () => {
    openTestProject();
    const id = RUNS.streamed.experiment_id;
    listing();
    details();
    vi.spyOn(trainingApi, "launchTensorboard").mockResolvedValue(servingOf(id));
    let deliver: (frame: Frame) => void = () => {};
    vi.mocked(openTrainingStream).mockImplementation((_experimentId, onMessage) => {
      deliver = onMessage;
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(id));
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

    const [first, second, third] = METRIC_FRAMES;
    act(() => deliver(first));
    expect(reloads).toHaveBeenCalledTimes(1);
    act(() => deliver(first));
    expect(reloads).toHaveBeenCalledTimes(1);
    act(() => deliver(second));
    expect(reloads).toHaveBeenCalledTimes(2);
    button.disabled = true;
    act(() => deliver(third));
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

  const id = RUNS.streamed.experiment_id;
  const [, lastBatch] = BATCH_FRAMES;
  const unfinished = `step ${String(lastBatch.row.step)} · batch_loss ${String(lastBatch.row.batch_loss)}`;

  /** The tab over `answer`, the run `RUNS.streamed` names selected, its stream sending `frames`. */
  async function streamed(answer: TrainingListing, frames: Frame[]) {
    listing(answer);
    streamFrames(...frames);
    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(id));
    return screen.findByRole("table", { name: `${id} metrics by epoch` });
  }

  it("says a terminal run recorded no metrics, rather than waiting on one that will never arrive", async () => {
    listing();

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(RUNS.empty.experiment_id));

    expect(await screen.findByText("This run recorded no metrics.")).toBeInTheDocument();
    expect(screen.queryByText("Waiting for metrics…")).not.toBeInTheDocument();
  });

  it("marks the waiting placeholder as a live region on a run with no metrics yet", async () => {
    listing();

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(RUNS.unlaunched.experiment_id));

    expect(await screen.findByRole("status")).toHaveTextContent("Waiting for metrics…");
  });

  it("shows every streamed epoch as a table row, outright", async () => {
    const table = await streamed(LISTING_BEFORE_FAILURE, METRIC_FRAMES);

    expect(within(table).getAllByRole("row")).toHaveLength(METRIC_FRAMES.length + 1);
    expect(within(table).getByRole("columnheader", { name: "epoch" })).toBeInTheDocument();
    expect(table.querySelector("thead")).toHaveClass("sticky", "top-0");
    for (const { row } of METRIC_FRAMES) {
      if (typeof row.loss === "number") {
        expect(within(table).getByText(String(row.loss))).toBeInTheDocument();
      }
    }
    expect(screen.queryByRole("button", { name: "as table" })).not.toBeInTheDocument();
  });

  it("renders the unset glyph for a non-finite value", async () => {
    const rows = within(await streamed(LISTING_BEFORE_FAILURE, METRIC_FRAMES)).getAllByRole("row");
    const unset = METRIC_FRAMES.findIndex(({ row }) => row[`loss${NOT_FINITE_SUFFIX}`]);
    const [, loss] = within(rows[unset + 1]).getAllByRole("cell");

    expect(loss).toHaveTextContent(UNSET_GLYPH);
  });

  it("spells a metric key's underscores as spaces and gives bookkeeping no column", async () => {
    const table = await streamed(LISTING_BEFORE_FAILURE, METRIC_FRAMES);

    const headers = within(table)
      .getAllByRole("columnheader")
      .map((h) => h.textContent);
    const keys = numericMetricKeys(METRIC_FRAMES[0].row);
    expect(keys.some((k) => k.includes("_"))).toBe(true);
    expect(headers).toEqual(["epoch", ...keys.map((k) => k.replaceAll("_", " "))]);
  });

  it("shows the latest batch row as the in-progress epoch's line, with no column of its own", async () => {
    const table = await streamed(LISTING_BEFORE_FAILURE, STREAM);

    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(METRIC_FRAMES.length + 2);
    expect(within(rows.at(-1)!).getByText(`in progress · ${unfinished}`)).toBeInTheDocument();
    expect(within(table).queryByRole("columnheader", { name: "step" })).not.toBeInTheDocument();
    expect(
      within(table).queryByRole("columnheader", { name: "batch loss" }),
    ).not.toBeInTheDocument();
  });

  it("drops the in-progress line once its epoch's own row arrives", async () => {
    const [batch, epochRow] = STREAM;
    expect([batch.type, epochRow.type]).toEqual(["batch", "metric"]);
    const table = await streamed(LISTING_BEFORE_FAILURE, [batch, epochRow]);

    expect(within(table).getAllByRole("row")).toHaveLength(2);
    expect(within(table).queryByText(/in progress/)).not.toBeInTheDocument();
  });

  it("names the last batch line of a run that ended mid-epoch as where it stopped", async () => {
    const table = await streamed(TRAINING_LISTING, [...STREAM, STATUS_FRAME]);

    expect(within(table).getByText(`stopped at ${unfinished}`)).toBeInTheDocument();
    expect(within(table).queryByText(/in progress/)).not.toBeInTheDocument();
  });
});

describe("TrainingTab status toast", () => {
  beforeEach(() => {
    openTestProject();
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
  });

  const id = RUNS.streamed.experiment_id;
  const ended = `Training ${id}: ${String(STATUS_FRAME.status?.state)}`;

  it("does not toast on selecting a run already known to be terminal, only rediscovering its state", async () => {
    listing();
    streamFrames(STATUS_FRAME);
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(id));

    await waitFor(() => expect(trainingApi.listRuns).toHaveBeenCalled());
    expect(pushToast).not.toHaveBeenCalledWith(ended, "info");
  });

  it("toasts a transition to terminal observed while watching a live run", async () => {
    listing(LISTING_BEFORE_FAILURE);
    streamFrames(STATUS_FRAME);
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(id));

    await waitFor(() => expect(pushToast).toHaveBeenCalledWith(ended, "info"));
  });

  it("toasts a stream error once, then still shows a metric the reconnect delivers", async () => {
    const [first] = METRIC_FRAMES;
    listing(LISTING_BEFORE_FAILURE);
    streamFrames(UNKNOWN_RUN_FRAME, UNKNOWN_RUN_FRAME, first);
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText(id));

    expect(
      within(await screen.findByRole("table")).getByText(String(first.row.loss)),
    ).toBeInTheDocument();
    expect(pushToast).toHaveBeenCalledWith(`Training stream error: ${UNKNOWN_RUN_FRAME.error}`);
    expect(pushToast).toHaveBeenCalledTimes(1);
  });
});

describe("TrainingTab stream lifecycle", () => {
  it("closes the stream and clears the detail panel when the selected run leaves the next poll", async () => {
    openTestProject();
    const id = RUNS.unlaunched.experiment_id;
    const listRuns = vi.spyOn(trainingApi, "listRuns");
    listRuns.mockResolvedValueOnce(TRAINING_LISTING);
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    const stop = vi.fn();
    vi.mocked(openTrainingStream).mockImplementation(() => stop);

    vi.useFakeTimers();
    try {
      render(<TrainingTab />);
      await vi.waitFor(() => expect(screen.getByText(id)).toBeInTheDocument());
      fireEvent.click(screen.getByText(id));
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
  // The produced data choices are the unlaunched run's own.
  const id = RUNS.unlaunched.experiment_id;

  /** Render the tab over the produced listing and open the picker; return it. */
  async function openPicker(shown: string = id): Promise<HTMLElement> {
    listing();
    render(<TrainingTab />);
    await screen.findAllByText(shown);
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    return screen.getByText("Runs and sweeps in this project").parentElement as HTMLElement;
  }

  it("shows a recorded run's own creation instant, read as a date", async () => {
    const { created, state } = RUNS.unlaunched;
    expect(Number.isNaN(Date.parse(created))).toBe(false);
    const picker = await openPicker();
    expect(
      within(rowFor(id, picker)).getByText(
        new RegExp(`${new Date(created).toLocaleString()} · ${state}`),
      ),
    ).toBeInTheDocument();
  });

  it("opens the picker, selects a recorded run, and starts it through the relaunch door", async () => {
    vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue(SPLIT_CHOICES);
    const relaunchSpy = vi.spyOn(trainingApi, "relaunch").mockResolvedValue(RUN_RELAUNCH);

    const picker = await openPicker();
    fireEvent.click(within(picker).getByText(id));
    expect(
      screen.getByText(
        "A new run of this config on the data paths it names, as they are now, with the recorded seed",
      ),
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(relaunchSpy).toHaveBeenCalledWith({
        relaunched_from: id,
        selection_dir: null,
        user: "jordan",
      }),
    );
  });

  it("starts a recorded sweep again by its id, with no data choice to fetch", async () => {
    const sweep = SWEEPS.searched.sweep_id;
    const choicesSpy = vi.spyOn(trainingApi, "listSplitChoices");
    const relaunchSpy = vi.spyOn(trainingApi, "relaunch").mockResolvedValue(SWEEP_RELAUNCH);

    const picker = await openPicker(sweep);
    fireEvent.click(within(picker).getByText(sweep));
    expect(
      screen.getByText("A new sweep over this sweep's recorded config, search and seed"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(relaunchSpy).toHaveBeenCalledWith({
        relaunched_from: sweep,
        selection_dir: null,
        user: "jordan",
      }),
    );
    expect(choicesSpy).not.toHaveBeenCalled();
  });

  it("shows a refused start's issues under the picked row", async () => {
    vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue(SPLIT_CHOICES);
    vi.spyOn(trainingApi, "relaunch").mockRejectedValue(
      await thrown(TRAINING_REFUSALS.refused_relaunch),
    );

    const picker = await openPicker();
    fireEvent.click(within(picker).getByText(id));
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    const [issue] = TRAINING_REFUSALS.refused_relaunch.body.detail.issues;
    await waitFor(() => expect(screen.getByText(issue)).toBeInTheDocument());
  });

  it("opens a row's Data choices only once selected, and starts with the chosen manifest", async () => {
    const choices = SPLIT_CHOICES;
    const selection = choices.selections.find((choice) => choice.reason === null)!;
    const listSplitChoicesSpy = vi
      .spyOn(trainingApi, "listSplitChoices")
      .mockResolvedValue(choices);
    const relaunchSpy = vi.spyOn(trainingApi, "relaunch").mockResolvedValue(RUN_RELAUNCH);

    const picker = await openPicker();
    expect(listSplitChoicesSpy).not.toHaveBeenCalled();

    fireEvent.click(within(picker).getByText(id));
    expect(
      await screen.findByText((text) => text.includes(choices.as_recorded.line)),
    ).toBeInTheDocument();
    expect(listSplitChoicesSpy).toHaveBeenCalledWith(id);

    fireEvent.click(
      await screen.findByRole("radio", {
        name: (name) =>
          name.startsWith(selection.selection_dir) &&
          !/^[/\\]/.test(name.slice(selection.selection_dir.length)),
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(relaunchSpy).toHaveBeenCalledWith({
        relaunched_from: id,
        selection_dir: selection.selection_dir,
        user: "jordan",
      }),
    );
  });

  it("disables Start while a selected row's data choices are still loading", async () => {
    const choices = Promise.withResolvers<SplitChoices>();
    vi.spyOn(trainingApi, "listSplitChoices").mockReturnValue(choices.promise);

    const picker = await openPicker();
    fireEvent.click(within(picker).getByText(id));

    expect(await screen.findByRole("button", { name: "Start" })).toBeDisabled();

    choices.resolve(SPLIT_CHOICES);
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
  });

  it("shows a per-row split-choices fetch failure as text on the row", async () => {
    vi.spyOn(trainingApi, "listSplitChoices").mockRejectedValue(new Error("network error"));

    const picker = await openPicker();
    fireEvent.click(within(picker).getByText(id));

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
    expect(within(picker).queryByText("No runs.")).not.toBeInTheDocument();

    fireEvent.click(within(picker).getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(listRuns).toHaveBeenCalledTimes(2));
    expect(await within(picker).findByText("No runs.")).toBeInTheDocument();
  });
});

describe("TrainingTab compare", () => {
  // The produced comparison of the relaunch and the run it relaunched, both still running.
  const compared = COMPARISONS.bound;
  const [first, second] = [RUNS.relaunched, RUNS.unlaunched].map((r) => r.experiment_id);
  const toggle = (id: string) => within(rowFor(id)).getByRole("button", { name: `Compare ${id}` });

  it("marking two runs switches the detail region and closes the single-run stream", async () => {
    openTestProject();
    listing();
    const stopSingle = vi.fn();
    vi.mocked(openTrainingStream).mockReturnValueOnce(stopSingle);
    vi.spyOn(trainingApi, "compare").mockResolvedValue(compared);

    render(<TrainingTab />);
    await screen.findByText(first);
    fireEvent.click(within(rowFor(first)).getByText(first));
    await waitFor(() =>
      expect(openTrainingStream).toHaveBeenCalledWith(first, expect.any(Function)),
    );

    fireEvent.click(toggle(first));
    fireEvent.click(toggle(second));

    expect(await screen.findByText("Comparing")).toBeInTheDocument();
    expect(stopSingle).toHaveBeenCalled();
  });

  it("every row's Compare toggle stays live: one id, no resolution to wait on", async () => {
    listing();

    render(<TrainingTab />);
    await screen.findByText(first);

    expect(toggle(first)).not.toBeDisabled();
  });

  it("caps the marked set and names the reason on a fifth toggle", async () => {
    openTestProject();
    const ids = TRAINING_LISTING.runs.slice(0, 5).map((r) => r.experiment_id);
    expect(ids).toHaveLength(5);
    listing();
    vi.spyOn(trainingApi, "compare").mockResolvedValue(compared);
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    await screen.findByText(ids[0]);
    for (const id of ids.slice(0, 4)) fireEvent.click(toggle(id));
    fireEvent.click(toggle(ids[4]));

    expect(pushToast).toHaveBeenCalledWith(expect.stringContaining("at most 4 runs"));
    // The fifth toggle stayed off: the fourth (last accepted) row is still the marked one.
    expect(toggle(ids[3])).toHaveAttribute("aria-pressed", "true");
    expect(toggle(ids[4])).toHaveAttribute("aria-pressed", "false");
  });

  it("groups Compare and Cancel as one action group", async () => {
    listing();

    render(<TrainingTab />);
    await screen.findByText(first);

    const group = within(rowFor(first)).getByRole("group", { name: "Run actions" });
    expect(within(group).getByRole("button", { name: `Compare ${first}` })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: `Cancel ${first}` })).toBeInTheDocument();
  });

  it("selects the sole remaining run when unmarking drops the marked set to one, instead of stranding it unselected", async () => {
    openTestProject();
    listing();
    vi.spyOn(trainingApi, "compare").mockResolvedValue(compared);

    render(<TrainingTab />);
    await screen.findByText(first);
    fireEvent.click(toggle(first));
    fireEvent.click(toggle(second));
    expect(await screen.findByText("Comparing")).toBeInTheDocument();

    fireEvent.click(toggle(first));

    expect(screen.queryByText("Comparing")).not.toBeInTheDocument();
    expect(await screen.findByText("Waiting for metrics…")).toBeInTheDocument();
    expect(screen.queryByText("No run selected.")).not.toBeInTheDocument();
  });
});
