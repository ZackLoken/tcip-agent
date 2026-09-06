import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { StructuredRefusalError } from "@/api/http";
import {
  openTrainingStream,
  trainingApi,
  type LaunchableConfig,
  type SplitChoices,
  type TrainingRunSummary,
} from "@/api/training";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { useStore } from "@/store";
import { TrainingTab, dataPickerFor } from "@/tabs/TrainingTab";
import { RUN_REFRESH_MS } from "@/tabs/trainingMetrics";

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

    fireEvent.click(screen.getByRole("button", { name: "Cancel train-agent-1" }));
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

  it("shows the experiment id beside the run id, once when the two are the same", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({ run_id: "train-forked", status: "running", experiment_id: "exp-forked" }),
        run({ run_id: "train-same", status: "running", experiment_id: "train-same" }),
        run({ run_id: "train-unresolved", status: "running", experiment_id: null }),
      ],
    });

    render(<TrainingTab />);
    expect(await screen.findByText("train-forked")).toBeInTheDocument();
    expect(screen.getByText(/exp-forked/)).toBeInTheDocument();
    expect(screen.getAllByText("train-same")).toHaveLength(1);
    expect(screen.getByText("train-unresolved")).toBeInTheDocument();
    expect(screen.queryByText("null")).not.toBeInTheDocument();
  });

  it("names the best value's own metric exactly as the record carries it, and shows nothing when the row carries no name", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({
          run_id: "train-named",
          status: "completed",
          best_metric: 0.812,
          best_metric_name: "map50",
        }),
        run({
          run_id: "train-loss-only",
          status: "completed",
          best_metric: 0.907,
          best_metric_name: "loss",
        }),
        run({
          run_id: "train-unnamed",
          status: "completed",
          best_metric: 0.5,
          best_metric_name: null,
        }),
      ],
    });

    render(<TrainingTab />);
    // The record names the metric bare; the row never adds a val_ qualifier the record
    // itself never stated (a no-validation run's best can be the training loss).
    expect(await screen.findByText("best map50 0.812")).toBeInTheDocument();
    expect(await screen.findByText("best loss 0.907")).toBeInTheDocument();
    expect(screen.queryByText(/val_/)).not.toBeInTheDocument();
    expect(screen.queryByText(/best.*0\.500/)).not.toBeInTheDocument();
  });

  it("states how the run list is ordered", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-a", status: "running" })],
    });

    render(<TrainingTab />);
    expect(await screen.findByText("train-a")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Runs this running process itself launched come first, in launch order; every other recorded run follows/,
      ),
    ).toBeInTheDocument();
  });

  it("names the row's own select control with the id, its experiment id and status", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-named-row", status: "running", experiment_id: "exp-other" })],
    });

    render(<TrainingTab />);
    expect(
      await screen.findByRole("button", {
        name: "train-named-row · exp-other running, launcher not recorded",
      }),
    ).toBeInTheDocument();
  });

  it("shows the row's best value exactly as the record carries it, with no rounding or title", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({
          run_id: "train-exact",
          status: "completed",
          best_metric: 0.4130041,
          best_metric_name: "loss",
        }),
      ],
    });

    render(<TrainingTab />);
    const value = await screen.findByText("best loss 0.4130041");
    expect(value).not.toHaveAttribute("title");
  });

  it("names the row's accessible name with its best value and metric when the record carries one", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({
          run_id: "train-named-value",
          status: "completed",
          best_metric: 0.4130041,
          best_metric_name: "loss",
        }),
      ],
    });

    render(<TrainingTab />);
    expect(
      await screen.findByRole("button", {
        name: "train-named-value completed, launcher not recorded, best loss 0.4130041",
      }),
    ).toBeInTheDocument();
  });

  it("shows an in-flight Cancel as a disabled, pending row control, and a failure in the row", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-cancel-flight", status: "running" })],
    });
    let resolveCancel: (v: {
      run_id: string;
      status: string;
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

    expect(await screen.findByText("Cancelling…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel train-cancel-flight" })).toBeDisabled();

    resolveCancel({ run_id: "train-cancel-flight", status: "running", cancel_requested: true });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Cancel train-cancel-flight" })).not.toBeDisabled(),
    );
  });

  it("shows a failed Cancel in the row, not only as a toast", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-cancel-fail", status: "running" })],
    });
    vi.spyOn(trainingApi, "cancel").mockRejectedValue(new Error("network error"));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel train-cancel-fail" }));

    expect(await screen.findByText("Cancel failed: network error")).toBeInTheDocument();
  });
});

describe("TrainingTab run launcher mark", () => {
  it("states who launched the run from launched_by alone, not from external", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({ run_id: "train-gui", status: "running", launched_by: { launcher: "gui" } }),
        run({
          run_id: "train-agent",
          status: "running",
          launched_by: {
            launcher: "agent",
            agent_client_name: "claude-code",
            agent_client_version: "2.1.238",
          },
        }),
        run({ run_id: "train-process", status: "running", launched_by: { launcher: "process" } }),
        run({
          run_id: "train-done-unrecorded",
          status: "completed",
          experiment_id: "exp-done-unrecorded",
          launched_by: null,
        }),
        run({
          run_id: "train-other-value",
          status: "running",
          // external true carries no weight on the sentence any more: the record is the fact.
          external: true,
          launched_by: { launcher: "bespoke-cli" },
        }),
      ],
    });

    render(<TrainingTab />);
    await screen.findByText("train-gui");

    expect(
      screen.getByRole("button", { name: "train-gui running, started through this app" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "train-agent running, started by the agent" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "train-process running, started by another process" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: "train-done-unrecorded · exp-done-unrecorded completed, launcher not recorded",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "train-other-value running, started by bespoke-cli" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/started elsewhere/)).not.toBeInTheDocument();
  });

  it("names the declared client in the agent row's accessible description", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({
          run_id: "train-agent-desc",
          status: "running",
          launched_by: {
            launcher: "agent",
            agent_client_name: "claude-code",
            agent_client_version: "2.1.238",
          },
        }),
      ],
    });

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: "train-agent-desc running, started by the agent",
    });

    const describedBy = button.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent("claude-code 2.1.238");
  });

  it("makes an unrecorded launcher's explanation reachable by assistive technology", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-no-launcher", status: "running" })],
    });

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: "train-no-launcher running, launcher not recorded",
    });

    const describedBy = button.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      "This run's record carries no launcher",
    );
  });

  it("describes a process launch by what the record rules out, not a browser claim it can't carry", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({
          run_id: "train-process-desc",
          status: "running",
          launched_by: { launcher: "process" },
        }),
      ],
    });

    render(<TrainingTab />);
    const button = await screen.findByRole("button", {
      name: "train-process-desc running, started by another process",
    });

    const describedBy = button.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      "This run was launched outside this app's own route and outside the agent, by a script or a test.",
    );
  });
});

describe("TrainingTab heading", () => {
  it("renders exactly one top-level heading naming the tab", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: [] });
    render(<TrainingTab />);
    await waitFor(() => expect(trainingApi.listRuns).toHaveBeenCalled());
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Training");
  });

  it("carries the section titles as level-2 headings under that h1", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-heading", status: "completed" })],
    });
    vi.spyOn(trainingApi, "getRun").mockResolvedValue({
      run_id: "train-heading",
      status: "completed",
      tensorboard_url: "http://localhost:6006",
    });

    render(<TrainingTab />);
    expect(await screen.findByRole("heading", { level: 2, name: "Runs" })).toBeInTheDocument();
    fireEvent.click(screen.getByText("train-heading"));
    expect(
      await screen.findByRole("heading", { level: 2, name: "TensorBoard" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Live metrics" })).toBeInTheDocument();
  });
});

describe("TrainingTab tensorboard panel", () => {
  it("says a run produced no logs and offers no Try again, rather than a raw refusal", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-nologs", status: "failed" })],
    });
    vi.spyOn(trainingApi, "getRun").mockResolvedValue({
      run_id: "train-nologs",
      status: "failed",
      tensorboard_url: null,
    });
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(
      new StructuredRefusalError(
        { error: "run has no output directory: train-nologs", no_logs: true },
        404,
        "run has no output directory: train-nologs",
      ),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-nologs"));

    expect(await screen.findByText("This run produced no logs.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
  });

  it("keeps Try again for an ordinary failed launch", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-broken", status: "failed" })],
    });
    vi.spyOn(trainingApi, "getRun").mockResolvedValue({
      run_id: "train-broken",
      status: "failed",
      tensorboard_url: null,
    });
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(new Error("connection refused"));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-broken"));

    expect(await screen.findByText("connection refused")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again: TensorBoard" })).toBeInTheDocument();
  });

  it("keeps the run's own recorded crash reason for a no-logs run that failed with one", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-nologs-crashed", status: "failed" })],
    });
    vi.spyOn(trainingApi, "getRun").mockResolvedValue({
      run_id: "train-nologs-crashed",
      status: "failed",
      tensorboard_url: null,
      error: "[WinError 183] Cannot create a file when that file already exists: 'tensorboard'",
    });
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(
      new StructuredRefusalError(
        {
          error: "[WinError 183] Cannot create a file when that file already exists: 'tensorboard'",
          no_logs: true,
        },
        404,
        "run produced no logs",
      ),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-nologs-crashed"));

    expect(
      await screen.findByText(
        "This run failed: [WinError 183] Cannot create a file when that file already exists: 'tensorboard'. It produced no logs.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Try again: TensorBoard" }),
    ).not.toBeInTheDocument();
  });
});

describe("TrainingTab chart placeholder", () => {
  it("says a terminal run recorded no metrics, rather than waiting on one that will never arrive", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-terminal-empty", status: "interrupted" })],
    });
    vi.spyOn(trainingApi, "getRun").mockResolvedValue({
      run_id: "train-terminal-empty",
      status: "interrupted",
      tensorboard_url: null,
    });
    vi.spyOn(trainingApi, "launchTensorboard").mockRejectedValue(
      new StructuredRefusalError(
        { error: "run produced no logs: train-terminal-empty", no_logs: true },
        404,
        "run produced no logs",
      ),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-terminal-empty"));

    expect(await screen.findByText("This run recorded no metrics.")).toBeInTheDocument();
    expect(screen.queryByText("Waiting for metrics…")).not.toBeInTheDocument();
  });

  it("keeps waiting for metrics on a non-terminal run with none yet", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-live-empty", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockResolvedValue({
      run_id: "train-live-empty",
      status: "running",
      tensorboard_url: null,
    });
    vi.spyOn(trainingApi, "launchTensorboard").mockResolvedValue({ error: "not ready" });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-live-empty"));

    expect(await screen.findByText("Waiting for metrics…")).toBeInTheDocument();
  });
});

describe("TrainingTab chart accessibility", () => {
  it("names the chart as an image built from the run and its metrics, with the same values behind a table disclosure", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-chart", status: "running" })],
    });
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "metric", run_id: runId, row: { epoch: 1, loss: 0.5 } });
      onMessage({ type: "metric", run_id: runId, row: { epoch: 2, loss: 0.3 } });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-chart"));

    expect(
      await screen.findByRole("img", { name: "Live metrics for train-chart: loss" }),
    ).toBeInTheDocument();

    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "as table" }));
    const table = await screen.findByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(3);
    expect(within(table).getByText("0.5")).toBeInTheDocument();
    expect(within(table).getByText("0.3")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "as table" }));
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("names the table's own container from the as-table toggle", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-controls", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "metric", run_id: runId, row: { epoch: 1, loss: 0.5 } });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-controls"));
    const toggle = await screen.findByRole("button", { name: "as table" });
    fireEvent.click(toggle);

    const table = await screen.findByRole("table");
    expect(toggle.getAttribute("aria-controls")).toBe(table.parentElement?.id);
  });

  it("resolves aria-controls to an element even while the table disclosure is closed", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-controls-closed", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "metric", run_id: runId, row: { epoch: 1, loss: 0.5 } });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-controls-closed"));
    const toggle = await screen.findByRole("button", { name: "as table" });

    const controlsId = toggle.getAttribute("aria-controls");
    expect(controlsId).toBeTruthy();
    expect(document.getElementById(controlsId as string)).not.toBeNull();
  });

  it("marks the metrics placeholder as a live region so its swap to the chart is announced", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-waiting-live", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-waiting-live"));

    expect(await screen.findByRole("status")).toHaveTextContent("Waiting for metrics…");
  });

  it("names the table's first column the same words as the chart's own axis label", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-axis-label", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "metric", run_id: runId, row: { epoch: 1, loss: 0.5 } });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-axis-label"));
    fireEvent.click(await screen.findByRole("button", { name: "as table" }));

    const table = await screen.findByRole("table");
    expect(within(table).getByRole("columnheader", { name: "epoch/step" })).toBeInTheDocument();
  });

  it("prefers a row's own epoch over its step for the derived ordinal, even when both are present", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-both-ordinals", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "metric", run_id: runId, row: { epoch: 1, step: 99, loss: 0.5 } });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-both-ordinals"));
    fireEvent.click(await screen.findByRole("button", { name: "as table" }));

    const table = await screen.findByRole("table");
    expect(within(table).getByText("1")).toBeInTheDocument();
    expect(within(table).queryByText("99")).not.toBeInTheDocument();
  });

  it("renders the unset glyph, not the array index, for a row with no epoch or step ordinal", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-no-ordinal", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "metric", run_id: runId, row: { loss: 0.5 } });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-no-ordinal"));
    fireEvent.click(await screen.findByRole("button", { name: "as table" }));

    const table = await screen.findByRole("table");
    const dataRow = within(table).getAllByRole("row")[1];
    expect(within(dataRow).getByText(UNSET_GLYPH)).toBeInTheDocument();
    expect(within(dataRow).getByText("0.5")).toBeInTheDocument();
  });

  it("renders the unset glyph for a non-finite metric value, not an empty cell", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-nonfinite", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "metric", run_id: runId, row: { epoch: 1, loss: 0.5 } });
      onMessage({ type: "metric", run_id: runId, row: { epoch: 2, loss: null } });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-nonfinite"));
    fireEvent.click(await screen.findByRole("button", { name: "as table" }));

    const table = await screen.findByRole("table");
    const rows = within(table).getAllByRole("row");
    expect(within(rows[1]).queryByText(UNSET_GLYPH)).not.toBeInTheDocument();
    expect(within(rows[2]).getByText(UNSET_GLYPH)).toBeInTheDocument();
  });

  it("excludes bookkeeping fields the shared metric-key filter defines, not only epoch and step", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-bookkeeping", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({
        type: "metric",
        run_id: runId,
        row: { epoch: 1, loss: 0.5, loss_state: "nan", timestamp: 1735689600 },
      });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-bookkeeping"));

    expect(
      await screen.findByRole("img", { name: "Live metrics for train-bookkeeping: loss" }),
    ).toBeInTheDocument();
  });
});

describe("TrainingTab chart default series", () => {
  it("plots both loss lines plus the merged selection line for a validated run selecting on loss, with every key still in the table", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-validated", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({
        type: "metric",
        run_id: runId,
        row: {
          epoch: 1,
          train_loss: 0.9,
          val_loss: 0.4,
          lr: 0.001,
          selection: 0.4,
          selection_metric: "loss",
        },
      });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-validated"));

    expect(
      await screen.findByRole("img", {
        name: 'Live metrics for train-validated: train_loss, val_loss (selection); every other logged metric is behind the "as table" toggle',
      }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "as table" }));
    const table = await screen.findByRole("table");
    expect(within(table).getByRole("columnheader", { name: "lr" })).toBeInTheDocument();
  });

  it("plots every numeric key and says so when the log carries neither a loss key nor selection", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-no-loss-key", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "metric", run_id: runId, row: { epoch: 1, val_map50: 0.7, lr: 0.001 } });
      return () => {};
    });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-no-loss-key"));

    expect(
      await screen.findByRole("img", {
        name: "Live metrics for train-no-loss-key: all logged metrics: val_map50, lr",
      }),
    ).toBeInTheDocument();
  });
});

describe("TrainingTab status toast", () => {
  it("does not toast on selecting a run already known to be terminal, only rediscovering its status", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-old-done", status: "completed" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "status", run_id: runId, status: { status: "completed" }, error: null });
      return () => {};
    });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-old-done"));

    await waitFor(() => expect(trainingApi.listRuns).toHaveBeenCalled());
    expect(pushToast).not.toHaveBeenCalledWith("Training train-old-done: completed", "info");
  });

  it("toasts a transition to terminal observed while watching a live run", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
    }));
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "train-live-done", status: "running" })],
    });
    vi.spyOn(trainingApi, "getRun").mockReturnValue(new Promise(() => {}));
    vi.mocked(openTrainingStream).mockImplementation((_root, runId, onMessage) => {
      onMessage({ type: "status", run_id: runId, status: { status: "completed" }, error: null });
      return () => {};
    });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-live-done"));

    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith("Training train-live-done: completed", "info"),
    );
  });
});

describe("TrainingTab switching between runs directly", () => {
  it("launches a fresh TensorBoard for the newly selected run, not the previous one's", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({ run_id: "train-x", status: "completed" }),
        run({ run_id: "train-y", status: "completed" }),
      ],
    });
    vi.spyOn(trainingApi, "getRun").mockImplementation((runId: string) =>
      Promise.resolve({ run_id: runId, status: "completed", tensorboard_url: null }),
    );
    const launchSpy = vi
      .spyOn(trainingApi, "launchTensorboard")
      .mockImplementation((runId: string) =>
        Promise.resolve({ url: `http://localhost:6006/${runId}` }),
      );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByText("train-x"));
    await waitFor(() => expect(launchSpy).toHaveBeenCalledWith("train-x"));

    fireEvent.click(screen.getByText("train-y"));
    await waitFor(() => expect(launchSpy).toHaveBeenCalledWith("train-y"));
    expect(screen.getByTitle("TensorBoard")).toHaveAttribute(
      "src",
      "http://localhost:6006/train-y",
    );
  });
});

describe("TrainingTab config picker", () => {
  it("opens the picker, selects a config, and starts it through the relaunch door", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: [] });
    vi.spyOn(trainingApi, "listConfigs").mockResolvedValue({
      configs: [config({ experiment_id: "exp-pristine-1" })],
    });
    vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue({
      as_recorded: {
        case: "bound",
        line: "on the partition it bound",
        compatible: true,
        reason: null,
      },
      manifests: [],
    });
    const relaunchSpy = vi
      .spyOn(trainingApi, "relaunch")
      .mockResolvedValue({ run_id: "run-new-1", experiment_id: "exp-pristine-1" });

    render(<TrainingTab />);
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    expect(await screen.findByText("exp-pristine-1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("exp-pristine-1"));
    expect(screen.getByText("Its first run, on the data paths it names")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(relaunchSpy).toHaveBeenCalledWith("exp-pristine-1", null));
  });

  it("shows a refused start's issues under the picked config's row", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: [] });
    vi.spyOn(trainingApi, "listConfigs").mockResolvedValue({
      configs: [config({ experiment_id: "exp-refused-1" })],
    });
    vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue({
      as_recorded: {
        case: "bound",
        line: "on the partition it bound",
        compatible: true,
        reason: null,
      },
      manifests: [],
    });
    vi.spyOn(trainingApi, "relaunch").mockRejectedValue(
      new StructuredRefusalError({ issues: ["batch_size must be positive"] }, 422, ""),
    );

    render(<TrainingTab />);
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));
    expect(await screen.findByText("exp-refused-1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("exp-refused-1"));
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(screen.getByText("batch_size must be positive")).toBeInTheDocument(),
    );
  });

  it("opens a config row's Data choices only once selected, and starts with the chosen manifest", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: [] });
    vi.spyOn(trainingApi, "listConfigs").mockResolvedValue({
      configs: [
        {
          experiment_id: "exp-1",
          builder: "m:build_detector",
          task: "detection",
          images_dir: "/data/images",
          subject: "leaf",
          created: null,
          state: "created",
          parent_experiment: null,
        },
      ],
    });
    const listSplitChoicesSpy = vi.spyOn(trainingApi, "listSplitChoices").mockResolvedValue({
      as_recorded: {
        case: "drawn",
        line: "draws its split again with seed 42 over the labels as they are now",
        compatible: true,
        reason: null,
      },
      manifests: [
        {
          manifest_dir: "/data/splits",
          enabled: true,
          reason: null,
          seed: 7,
          group_by: "tile_prefix",
          train: 4,
          val: 2,
          calibration: 1,
          other_dates: 0,
          replaced_split_keys: [],
        },
      ],
    });
    const relaunchSpy = vi
      .spyOn(trainingApi, "relaunch")
      .mockResolvedValue({ run_id: "r1", experiment_id: "exp-1" });

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Start a run" }));
    await screen.findByText("exp-1");
    expect(listSplitChoicesSpy).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText("exp-1"));
    expect(
      await screen.findByText(/draws its split again with seed 42 over the labels as they are now/),
    ).toBeInTheDocument();
    expect(listSplitChoicesSpy).toHaveBeenCalledWith("exp-1");

    fireEvent.click(await screen.findByRole("radio", { name: /\/data\/splits/ }));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(relaunchSpy).toHaveBeenCalledWith("exp-1", "/data/splits"));
  });

  it("disables Start while a selected config's data choices are still loading", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: [] });
    vi.spyOn(trainingApi, "listConfigs").mockResolvedValue({
      configs: [
        {
          experiment_id: "exp-1",
          builder: "m:build_detector",
          task: "detection",
          images_dir: "/data/images",
          subject: "leaf",
          created: null,
          state: "created",
          parent_experiment: null,
        },
      ],
    });
    let resolveChoices: (v: SplitChoices) => void = () => {};
    vi.spyOn(trainingApi, "listSplitChoices").mockReturnValue(
      new Promise<SplitChoices>((resolve) => {
        resolveChoices = resolve;
      }),
    );

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Start a run" }));
    fireEvent.click(await screen.findByText("exp-1"));

    expect(await screen.findByRole("button", { name: "Start" })).toBeDisabled();

    resolveChoices({
      as_recorded: { case: "drawn", line: "draws its split again", compatible: true, reason: null },
      manifests: [],
    });
    await waitFor(() => expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled());
  });

  it("shows a per-row split-choices fetch failure as text on the row", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: [] });
    vi.spyOn(trainingApi, "listConfigs").mockResolvedValue({
      configs: [
        {
          experiment_id: "exp-1",
          builder: "m:f",
          task: "detection",
          images_dir: "/data/images",
          subject: "leaf",
          created: null,
          state: "created",
          parent_experiment: null,
        },
      ],
    });
    vi.spyOn(trainingApi, "listSplitChoices").mockRejectedValue(new Error("network error"));

    render(<TrainingTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Start a run" }));
    fireEvent.click(await screen.findByText("exp-1"));

    expect(
      await screen.findByText(/Could not load its data choices: network error/),
    ).toBeInTheDocument();
  });
});

describe("dataPickerFor", () => {
  const choices: SplitChoices = {
    as_recorded: {
      case: "bound",
      line: "on the partition it bound",
      compatible: false,
      reason: "Directory not found: data.images_dir = '/moved'",
    },
    manifests: [
      {
        manifest_dir: "/data/splits",
        enabled: true,
        reason: null,
        seed: 7,
        group_by: "tile_prefix",
        train: 4,
        val: 2,
        calibration: 1,
        other_dates: 3,
        replaced_split_keys: ["seed"],
      },
      {
        manifest_dir: "/data/other",
        enabled: false,
        reason: "split manifest was drawn for subject='bud'",
        seed: null,
        group_by: null,
        train: 0,
        val: 0,
        calibration: 0,
        other_dates: 0,
        replaced_split_keys: [],
      },
    ],
  };

  it("maps the snapshot's own compatibility into the disabled/reason pair the picker renders", () => {
    const picker = dataPickerFor(choices);
    expect(picker?.asRecordedLine).toBe("on the partition it bound");
    expect(picker?.asRecordedDisabled).toBe(true);
    expect(picker?.asRecordedReason).toBe("Directory not found: data.images_dir = '/moved'");
    expect(picker?.absenceMessage).toMatch(/this listing found no other recorded partition/);
  });

  it("maps each manifest's own enabled flag and reason onto the choice the picker renders", () => {
    const picker = dataPickerFor(choices);
    expect(picker?.choices[0]).toMatchObject({
      manifestDir: "/data/splits",
      disabled: false,
      replacedSplitKeys: ["seed"],
    });
    expect(picker?.choices[1]).toMatchObject({
      manifestDir: "/data/other",
      disabled: true,
      reason: "split manifest was drawn for subject='bud'",
    });
  });

  it("renders the seed/group_by/counts line from the manifest's own recorded fields", () => {
    const picker = dataPickerFor(choices);
    render(<div>{picker?.choices[0].label}</div>);
    expect(
      screen.getByText(/seed 7 · tile_prefix · train 4 · val 2 · calibration 1/),
    ).toBeInTheDocument();
    expect(screen.getByText(/3 member\(s\) under other dates/)).toBeInTheDocument();
  });

  it("answers undefined for a row with no choices fetched yet", () => {
    expect(dataPickerFor(undefined)).toBeUndefined();
  });
});

describe("TrainingTab compare", () => {
  function setProjectRoot(root: string | null) {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: root } },
    }));
  }

  // The comparison's own columns repeat a marked run's id in its detail region (table headers),
  // and the single-run header repeats the selected one; the sidebar's own row is always first.
  function rowFor(id: string): HTMLElement {
    return screen.getAllByText(id)[0].closest("li") as HTMLElement;
  }

  it("marking two runs switches the detail region and closes the single-run stream", async () => {
    setProjectRoot("/proj");
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({ run_id: "run-a", status: "running", experiment_id: "exp-a" }),
        run({ run_id: "run-b", status: "running", experiment_id: "exp-b" }),
      ],
    });
    const stopSingle = vi.fn();
    vi.mocked(openTrainingStream).mockReturnValueOnce(stopSingle);
    vi.spyOn(trainingApi, "compare").mockResolvedValue({
      experiments: [{ experiment_id: "exp-a" }, { experiment_id: "exp-b" }] as unknown as Awaited<
        ReturnType<typeof trainingApi.compare>
      >["experiments"],
      count: 2,
      same_dataset_fingerprint: null,
    });

    render(<TrainingTab />);
    await screen.findByText("run-a");
    fireEvent.click(within(rowFor("run-a")).getByText("run-a"));
    await waitFor(() =>
      expect(openTrainingStream).toHaveBeenCalledWith("/proj", "run-a", expect.any(Function)),
    );

    fireEvent.click(within(rowFor("run-a")).getByRole("button", { name: "Compare run-a" }));
    fireEvent.click(within(rowFor("run-b")).getByRole("button", { name: "Compare run-b" }));

    expect(await screen.findByText("Comparing")).toBeInTheDocument();
    expect(stopSingle).toHaveBeenCalled();
  });

  it("caps the marked set and names the reason on a fifth toggle", async () => {
    setProjectRoot("/proj");
    const runs = ["run-1", "run-2", "run-3", "run-4", "run-5"].map((id) =>
      run({ run_id: id, status: "running", experiment_id: `exp-${id}` }),
    );
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs });
    vi.spyOn(trainingApi, "compare").mockResolvedValue({
      experiments: [],
      count: 0,
      same_dataset_fingerprint: null,
    });
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

  it("disables a row's Compare toggle and names the two reasons", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({ run_id: "run-unresolved", status: "running" }),
        run({
          run_id: "run-failed-tracking",
          status: "running",
          experiment_error: "dataset_identity failed: boom",
        }),
      ],
    });

    render(<TrainingTab />);
    await screen.findByText("run-unresolved");

    const unresolvedToggle = within(rowFor("run-unresolved")).getByRole("button", {
      name: "Compare run-unresolved",
    });
    expect(unresolvedToggle).toBeDisabled();
    expect(unresolvedToggle).not.toHaveAttribute("title");
    const unresolvedReason = within(rowFor("run-unresolved")).getByText(
      "experiment not resolved yet",
    );
    expect(unresolvedReason).toBeInTheDocument();
    expect(unresolvedToggle).toHaveAttribute("aria-describedby", unresolvedReason.id);

    const failedToggle = within(rowFor("run-failed-tracking")).getByRole("button", {
      name: "Compare run-failed-tracking",
    });
    expect(failedToggle).toBeDisabled();
    const failedReason = within(rowFor("run-failed-tracking")).getByText(
      "experiment tracking failed: dataset_identity failed: boom",
    );
    expect(failedReason).toBeInTheDocument();
    expect(failedToggle).toHaveAttribute("aria-describedby", failedReason.id);
  });

  it("groups Compare and Cancel as one action group", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "run-grouped", status: "running", experiment_id: "exp-grouped" })],
    });

    render(<TrainingTab />);
    await screen.findByText("run-grouped");

    const group = within(rowFor("run-grouped")).getByRole("group", { name: "Run actions" });
    expect(within(group).getByRole("button", { name: "Compare run-grouped" })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: "Cancel run-grouped" })).toBeInTheDocument();
  });

  it("drops a run from the marked comparison the moment its own reason turns unmarkable, through the one implementation toggleMarked also consults", async () => {
    setProjectRoot("/proj");
    const listRuns = vi.spyOn(trainingApi, "listRuns");
    listRuns.mockResolvedValueOnce({
      runs: [
        run({ run_id: "run-a", status: "running", experiment_id: "exp-a" }),
        run({ run_id: "run-b", status: "running", experiment_id: "exp-b" }),
      ],
    });
    vi.spyOn(trainingApi, "compare").mockResolvedValue({
      experiments: [{ experiment_id: "exp-a" }, { experiment_id: "exp-b" }] as unknown as Awaited<
        ReturnType<typeof trainingApi.compare>
      >["experiments"],
      count: 2,
      same_dataset_fingerprint: null,
    });

    vi.useFakeTimers();
    try {
      render(<TrainingTab />);
      await vi.waitFor(() => expect(screen.getByText("run-a")).toBeInTheDocument());
      fireEvent.click(within(rowFor("run-a")).getByRole("button", { name: "Compare run-a" }));
      fireEvent.click(within(rowFor("run-b")).getByRole("button", { name: "Compare run-b" }));
      await vi.waitFor(() => expect(screen.getByText("Comparing")).toBeInTheDocument());

      // run-b's own record now carries both a resolved id and a tracking failure: a naive
      // `experiment_id` truthiness check would still call it markable, unlike unmarkableReason.
      listRuns.mockResolvedValueOnce({
        runs: [
          run({ run_id: "run-a", status: "running", experiment_id: "exp-a" }),
          run({
            run_id: "run-b",
            status: "running",
            experiment_id: "exp-b",
            experiment_error: "dataset_identity failed: boom",
          }),
        ],
      });
      // Advance one tick of the runs poll.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(RUN_REFRESH_MS);
      });

      expect(
        within(rowFor("run-b")).getByText(
          "experiment tracking failed: dataset_identity failed: boom",
        ),
      ).toBeInTheDocument();
      // Only one run is still markable, so the detail falls back out of the comparison view.
      expect(screen.queryByText("Comparing")).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("names each run's Cancel control with that run's own id", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({ run_id: "train-a", status: "running" }),
        run({ run_id: "train-b", status: "running" }),
      ],
    });

    render(<TrainingTab />);
    await screen.findByText("train-a");

    expect(screen.getByRole("button", { name: "Cancel train-a" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel train-b" })).toBeInTheDocument();
  });

  it("selects the sole remaining run when unmarking drops the marked set to one, instead of stranding it unselected", async () => {
    setProjectRoot("/proj");
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [
        run({ run_id: "run-a", status: "running", experiment_id: "exp-a" }),
        run({ run_id: "run-b", status: "running", experiment_id: "exp-b" }),
      ],
    });
    vi.spyOn(trainingApi, "compare").mockResolvedValue({
      experiments: [{ experiment_id: "exp-a" }, { experiment_id: "exp-b" }] as unknown as Awaited<
        ReturnType<typeof trainingApi.compare>
      >["experiments"],
      count: 2,
      same_dataset_fingerprint: null,
    });

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

  it("counts the cap against the same markable set the header prints, not a stale unmarkable id", async () => {
    setProjectRoot("/proj");
    const allRuns = ["run-1", "run-2", "run-3", "run-4", "run-5"].map((id) =>
      run({ run_id: id, status: "running", experiment_id: `exp-${id}` }),
    );
    const listRuns = vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: allRuns });
    vi.spyOn(trainingApi, "compare").mockResolvedValue({
      experiments: [],
      count: 0,
      same_dataset_fingerprint: null,
    });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    vi.useFakeTimers();
    try {
      render(<TrainingTab />);
      await vi.waitFor(() => expect(screen.getByText("run-1")).toBeInTheDocument());
      for (const id of ["run-1", "run-2", "run-3", "run-4"]) {
        fireEvent.click(within(rowFor(id)).getByRole("button", { name: `Compare ${id}` }));
      }
      await vi.waitFor(() => expect(screen.getByText("4 of 4 runs")).toBeInTheDocument());

      // run-1 turns unmarkable without ever leaving markedRunIds: the header's own count (the
      // markable set, marked.length) drops to 3 while the raw id set still holds four.
      listRuns.mockResolvedValueOnce({
        runs: [
          run({
            run_id: "run-1",
            status: "running",
            experiment_id: "exp-run-1",
            experiment_error: "dataset_identity failed: boom",
          }),
          ...allRuns.slice(1),
        ],
      });
      // Advance one tick of the runs poll.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(RUN_REFRESH_MS);
      });
      expect(screen.getByText("3 of 4 runs")).toBeInTheDocument();

      fireEvent.click(within(rowFor("run-5")).getByRole("button", { name: "Compare run-5" }));

      expect(pushToast).not.toHaveBeenCalledWith(expect.stringContaining("at most 4 runs"));
      await vi.waitFor(() => expect(screen.getByText("4 of 4 runs")).toBeInTheDocument());
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("TrainingTab configs loader", () => {
  it("keeps and shows a failed configs fetch, with a Retry control, in place of the empty line", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({ runs: [] });
    const listConfigsSpy = vi
      .spyOn(trainingApi, "listConfigs")
      .mockRejectedValueOnce(new Error("network error"))
      .mockResolvedValue({ configs: [] });

    render(<TrainingTab />);
    fireEvent.click(screen.getByRole("button", { name: "Start a run" }));

    expect(await screen.findByText("Could not load configs: network error")).toBeInTheDocument();
    expect(screen.queryByText("No config exists in this project yet.")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(listConfigsSpy).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("No config exists in this project yet.")).toBeInTheDocument();
  });
});
