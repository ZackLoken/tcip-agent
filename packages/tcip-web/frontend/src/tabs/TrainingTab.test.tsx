import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { StructuredRefusalError } from "@/api/http";
import {
  openTrainingStream,
  trainingApi,
  type LaunchableConfig,
  type SplitChoices,
  type TrainingRunSummary,
} from "@/api/training";
import { useStore } from "@/store";
import { TrainingTab, dataPickerFor } from "@/tabs/TrainingTab";

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

    fireEvent.click(within(rowFor("run-a")).getByRole("button", { name: "Compare" }));
    fireEvent.click(within(rowFor("run-b")).getByRole("button", { name: "Compare" }));

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
      fireEvent.click(within(rowFor(id)).getByRole("button", { name: "Compare" }));
    }
    fireEvent.click(within(rowFor("run-5")).getByRole("button", { name: "Compare" }));

    expect(pushToast).toHaveBeenCalledWith(expect.stringContaining("at most 4 runs"));
    // The fifth toggle stayed off: the fourth (last accepted) row is still the marked one.
    expect(within(rowFor("run-4")).getByRole("button", { name: "Compare" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(within(rowFor("run-5")).getByRole("button", { name: "Compare" })).toHaveAttribute(
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
      name: "Compare",
    });
    expect(unresolvedToggle).toBeDisabled();
    expect(unresolvedToggle).not.toHaveAttribute("title");
    expect(
      within(rowFor("run-unresolved")).getByText("experiment not resolved yet"),
    ).toBeInTheDocument();

    const failedToggle = within(rowFor("run-failed-tracking")).getByRole("button", {
      name: "Compare",
    });
    expect(failedToggle).toBeDisabled();
    expect(
      within(rowFor("run-failed-tracking")).getByText(
        "experiment tracking failed: dataset_identity failed: boom",
      ),
    ).toBeInTheDocument();
  });

  it("groups Compare and Cancel as one action group", async () => {
    vi.spyOn(trainingApi, "listRuns").mockResolvedValue({
      runs: [run({ run_id: "run-grouped", status: "running", experiment_id: "exp-grouped" })],
    });

    render(<TrainingTab />);
    await screen.findByText("run-grouped");

    const group = within(rowFor("run-grouped")).getByRole("group", { name: "Run actions" });
    expect(within(group).getByRole("button", { name: "Compare" })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: "Cancel run-grouped" })).toBeInTheDocument();
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
