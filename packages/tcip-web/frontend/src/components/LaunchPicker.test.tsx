import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LaunchPicker, NO_OTHER_PARTITION, type LaunchPickerRow } from "@/components/LaunchPicker";
import { thrown } from "@/test/refusals";
import { MOVED_SPLIT_CHOICES, SPLIT_CHOICES, TRAINING_REFUSALS } from "@/test/training";

afterEach(cleanup);

/** The issue the produced refusal of a relaunch names. */
const [ISSUE] = TRAINING_REFUSALS.refused_relaunch.body.detail.issues;

/** The produced partition offered disabled, its record unreadable, and the one offered enabled. */
const DISABLED = SPLIT_CHOICES.selections.find((choice) => choice.reason !== null)!;
const ENABLED = SPLIT_CHOICES.selections.find((choice) => choice.reason === null)!;

/** The disabled partition's reason as the page shows it, its whitespace collapsed. */
const DISABLED_REASON = String(DISABLED.reason).replace(/\s+/g, " ").trim();

/** The radio of the partition under `dir` itself, never of one in a directory beneath it. */
function radioOf(dir: string): HTMLElement {
  return screen.getByRole("radio", {
    name: (name) => name.startsWith(dir) && !/^[/\\]/.test(name.slice(dir.length)),
  });
}

describe("LaunchPicker", () => {
  it("shows the empty message when there are no configs", () => {
    render(
      <LaunchPicker
        list={{
          title: "Configs in this project",
          emptyMessage: "No config exists in this project yet.",
          rows: [],
        }}
      />,
    );
    expect(screen.getByText("No config exists in this project yet.")).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument();
  });

  function row(overrides: Partial<LaunchPickerRow> = {}): LaunchPickerRow {
    return {
      key: "exp-1",
      content: <span>exp-1</span>,
      branchLine: "Its first run, on the data paths it names",
      onStart: vi.fn().mockResolvedValue(undefined),
      ...overrides,
    };
  }

  function renderRow(overrides: Partial<LaunchPickerRow>) {
    render(
      <LaunchPicker
        list={{ title: "Configs in this project", emptyMessage: "none", rows: [row(overrides)] }}
      />,
    );
    fireEvent.click(screen.getByText("exp-1"));
  }

  it("reveals the Start action and its branch line only once a row is selected", () => {
    render(
      <LaunchPicker
        list={{ title: "Configs in this project", emptyMessage: "none", rows: [row()] }}
      />,
    );
    expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("exp-1"));
    expect(screen.getByRole("button", { name: "Start" })).toBeInTheDocument();
    expect(screen.getByText("Its first run, on the data paths it names")).toBeInTheDocument();
  });

  it("shows a refused start's issues under the row, in the tool's own words", async () => {
    const onStart = vi.fn().mockRejectedValue(await thrown(TRAINING_REFUSALS.refused_relaunch));
    renderRow({ onStart });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(screen.getByText(ISSUE)).toBeInTheDocument());
  });

  it("a start that succeeds closes the row's own action", async () => {
    const onStart = vi.fn().mockResolvedValue(undefined);
    renderRow({ onStart });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(onStart).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument(),
    );
  });

  it("renders no Data section for a row that carries none", () => {
    renderRow({});
    expect(screen.queryByText("Data")).not.toBeInTheDocument();
  });

  it('shows "As recorded"\'s own case line and the one-line absence with no offers', () => {
    const { as_recorded: recorded, selections } = MOVED_SPLIT_CHOICES;
    expect(selections).toHaveLength(0);
    renderRow({ data: MOVED_SPLIT_CHOICES });
    expect(screen.getByText(new RegExp(recorded.line))).toBeInTheDocument();
    expect(screen.getByText(NO_OTHER_PARTITION)).toBeInTheDocument();
  });

  it("renders one row per offered selection, labeled by its own recorded fields", () => {
    const offered = ENABLED;
    renderRow({ data: SPLIT_CHOICES });
    expect(
      screen.getByText(
        `seed ${offered.seed} · ${offered.group_by} · train ${offered.train} · val ${offered.val} · calibration ${offered.calibration}`,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(NO_OTHER_PARTITION)).not.toBeInTheDocument();
  });

  it("shows a disabled candidate's own reason", () => {
    renderRow({ data: SPLIT_CHOICES });
    expect(screen.getByText(DISABLED_REASON)).toBeInTheDocument();
    expect(radioOf(DISABLED.selection_dir)).toBeDisabled();
  });

  it("starts with the chosen selection directory once a candidate is selected", async () => {
    const chosen = ENABLED.selection_dir;
    const onStart = vi.fn().mockResolvedValue(undefined);
    renderRow({ onStart, data: SPLIT_CHOICES });
    fireEvent.click(radioOf(chosen));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(onStart).toHaveBeenCalledWith(chosen));
  });

  it("starts with null for the default As recorded choice", async () => {
    const onStart = vi.fn().mockResolvedValue(undefined);
    renderRow({ onStart, data: SPLIT_CHOICES });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(onStart).toHaveBeenCalledWith(null));
  });

  it("disables Start while the checked As recorded choice is disabled, showing its reason once, in the status region", () => {
    const reason = String(MOVED_SPLIT_CHOICES.as_recorded.reason);
    renderRow({ data: MOVED_SPLIT_CHOICES });
    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
    expect(screen.getAllByText(reason)).toHaveLength(1);
    expect(screen.getByRole("status")).toHaveTextContent(reason);
  });

  it("shows a disabled candidate's own inline reason once it is no longer the checked choice, not duplicated into the status region", () => {
    expect(SPLIT_CHOICES.as_recorded.reason).toBeNull();
    renderRow({ data: SPLIT_CHOICES });
    // As recorded (the default checked choice) is enabled, so Start is not blocked and no
    // status region renders; the untouched, disabled candidate still names its own reason once.
    expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled();
    expect(screen.getAllByText(DISABLED_REASON)).toHaveLength(1);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders a refused start's issues once, in the status region under Start, not the row's disabled-choice text", async () => {
    const onStart = vi.fn().mockRejectedValue(await thrown(TRAINING_REFUSALS.refused_relaunch));
    renderRow({ onStart, data: SPLIT_CHOICES });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(screen.getByText(ISSUE)).toBeInTheDocument());
    expect(screen.getAllByText(ISSUE)).toHaveLength(1);
    expect(screen.getByRole("status")).toHaveTextContent(ISSUE);
  });

  it("disables Start and reports checking the data choice while a row's Data choices are still loading", () => {
    renderRow({ data: undefined, dataLoading: true });
    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("checking the data choice");
  });

  it("shows a list's own load failure in place of the empty line, with a Retry control", () => {
    const onRetry = vi.fn();
    render(
      <LaunchPicker
        list={{
          title: "Configs in this project",
          emptyMessage: "No config exists in this project yet.",
          rows: [],
          error: "Could not load configs: network error",
          onRetry,
        }}
      />,
    );
    expect(screen.getByText("Could not load configs: network error")).toBeInTheDocument();
    expect(screen.queryByText("No config exists in this project yet.")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("carries aria-expanded, not aria-pressed, on a row's own disclosure toggle", () => {
    render(
      <LaunchPicker
        list={{ title: "Configs in this project", emptyMessage: "none", rows: [row()] }}
      />,
    );
    const toggle = screen.getByText("exp-1").closest("button") as HTMLElement;
    expect(toggle).toHaveAttribute("aria-expanded");
    expect(toggle).not.toHaveAttribute("aria-pressed");
  });

  it("re-enables Start once a checked choice that is itself enabled is selected", () => {
    // The moved run's refused As recorded beside the offered partitions: no one answer pairs them.
    renderRow({
      data: { as_recorded: MOVED_SPLIT_CHOICES.as_recorded, selections: SPLIT_CHOICES.selections },
    });
    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
    fireEvent.click(radioOf(ENABLED.selection_dir));
    expect(screen.getByRole("button", { name: "Start" })).not.toBeDisabled();
  });

  it("shows the recorded split keys a candidate would replace beside its own row", () => {
    const { replaced_split_keys: replaced } = ENABLED;
    expect(replaced).not.toHaveLength(0);
    renderRow({ data: SPLIT_CHOICES });
    // Each offered candidate, the disabled one included, names the keys it would replace.
    expect(
      screen.getAllByText(new RegExp(`replaces the recorded\\s*${replaced.join(", ")}`)),
    ).toHaveLength(SPLIT_CHOICES.selections.length);
  });

  it("shows a per-row Data-choices fetch failure as text on the row", () => {
    renderRow({ dataError: "Could not load its data choices: network error" });
    expect(screen.getByText("Could not load its data choices: network error")).toBeInTheDocument();
  });

  it("fetches a row's own Data choices only once it is selected", () => {
    const onSelect = vi.fn();
    render(
      <LaunchPicker
        list={{
          title: "Configs in this project",
          emptyMessage: "none",
          rows: [row(), row({ key: "exp-2", content: <span>exp-2</span> })],
        }}
        onSelect={onSelect}
      />,
    );
    expect(onSelect).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("exp-1"));
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith("exp-1");
    fireEvent.click(screen.getByText("exp-1"));
    expect(onSelect).toHaveBeenCalledTimes(1);
  });
});
