import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { StructuredRefusalError } from "@/api/http";
import { LaunchPicker, type LaunchPickerRow } from "@/components/LaunchPicker";

afterEach(cleanup);

function noop() {}

describe("LaunchPicker", () => {
  it("shows the empty message and only the composer row when there are no configs", () => {
    render(
      <LaunchPicker
        list={{
          title: "Configs in this project",
          emptyMessage: "No config exists in this project yet.",
          rows: [],
        }}
        composerLabel="Describe a new one to the agent"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    expect(screen.getByText("No config exists in this project yet.")).toBeInTheDocument();
    expect(screen.getByText("Describe a new one to the agent")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument();
  });

  it("omits the list entirely and offers only the agent row when no list is given", () => {
    render(
      <LaunchPicker
        composerLabel="Describe a new one to the agent"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    expect(screen.queryByText("Configs in this project")).not.toBeInTheDocument();
    expect(screen.getByText("Describe a new one to the agent")).toBeInTheDocument();
  });

  it("sends the composer's request to the agent", () => {
    const onSend = vi.fn();
    render(
      <LaunchPicker
        composerLabel="Describe a new one to the agent"
        request="train something"
        onRequestChange={noop}
        onSend={onSend}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /send to agent/i }));
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("disables the send action until the composer holds text", () => {
    render(
      <LaunchPicker
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    expect(screen.getByRole("button", { name: /send to agent/i })).toBeDisabled();
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

  it("reveals the Start action and its branch line only once a row is selected", () => {
    render(
      <LaunchPicker
        list={{ title: "Configs in this project", emptyMessage: "none", rows: [row()] }}
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("exp-1"));
    expect(screen.getByRole("button", { name: "Start" })).toBeInTheDocument();
    expect(screen.getByText("Its first run, on the data paths it names")).toBeInTheDocument();
  });

  it("shows a refused start's issues under the row, in the tool's own words", async () => {
    const onStart = vi
      .fn()
      .mockRejectedValue(
        new StructuredRefusalError({ issues: ["batch_size must be positive"] }, 422, ""),
      );
    render(
      <LaunchPicker
        list={{ title: "Configs in this project", emptyMessage: "none", rows: [row({ onStart })] }}
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    fireEvent.click(screen.getByText("exp-1"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(screen.getByText("batch_size must be positive")).toBeInTheDocument(),
    );
  });

  it("a start that succeeds closes the row's own action", async () => {
    const onStart = vi.fn().mockResolvedValue(undefined);
    render(
      <LaunchPicker
        list={{ title: "Configs in this project", emptyMessage: "none", rows: [row({ onStart })] }}
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    fireEvent.click(screen.getByText("exp-1"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(onStart).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument(),
    );
  });

  it("renders no Data section for a row that carries none", () => {
    render(
      <LaunchPicker
        list={{ title: "Configs in this project", emptyMessage: "none", rows: [row()] }}
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    fireEvent.click(screen.getByText("exp-1"));
    expect(screen.queryByText("Data")).not.toBeInTheDocument();
  });

  it('shows "As recorded"\'s own case line and the one-line absence with no offers', () => {
    render(
      <LaunchPicker
        list={{
          title: "Configs in this project",
          emptyMessage: "none",
          rows: [
            row({
              data: {
                asRecordedLine:
                  "draws its split again with seed 42 over the labels as they are now",
                choices: [],
                absenceMessage:
                  "this listing found no other recorded partition the config can bind to; the agent can draw one.",
              },
            }),
          ],
        }}
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    fireEvent.click(screen.getByText("exp-1"));
    expect(
      screen.getByText(/draws its split again with seed 42 over the labels as they are now/),
    ).toBeInTheDocument();
    expect(screen.getByText(/this listing found no other recorded partition/)).toBeInTheDocument();
  });

  it("renders one row per offered manifest, with the draw-seed label, only when offers exist", () => {
    render(
      <LaunchPicker
        list={{
          title: "Configs in this project",
          emptyMessage: "none",
          rows: [
            row({
              data: {
                asRecordedLine: "on the partition it bound",
                absenceMessage: "unused",
                choices: [
                  {
                    manifestDir: "/data/splits",
                    label: <span>seed 7 · tile_prefix · train 4 · val 2</span>,
                  },
                ],
              },
            }),
          ],
        }}
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    fireEvent.click(screen.getByText("exp-1"));
    expect(screen.getByText(/seed 7/)).toBeInTheDocument();
    expect(screen.queryByText("unused")).not.toBeInTheDocument();
  });

  it("shows a disabled candidate's own compatibility reason", () => {
    render(
      <LaunchPicker
        list={{
          title: "Configs in this project",
          emptyMessage: "none",
          rows: [
            row({
              data: {
                asRecordedLine: "on the partition it bound",
                absenceMessage: "unused",
                choices: [
                  {
                    manifestDir: "/data/other-subject-splits",
                    label: <span>/data/other-subject-splits</span>,
                    disabled: true,
                    reason:
                      "split manifest was drawn for subject='bud', but this run is subject='leaf'",
                  },
                ],
              },
            }),
          ],
        }}
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    fireEvent.click(screen.getByText("exp-1"));
    expect(screen.getByText(/drawn for subject='bud'/)).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /other-subject-splits/ })).toBeDisabled();
  });

  it("starts with the chosen manifest directory once a candidate is selected", async () => {
    const onStart = vi.fn().mockResolvedValue(undefined);
    render(
      <LaunchPicker
        list={{
          title: "Configs in this project",
          emptyMessage: "none",
          rows: [
            row({
              onStart,
              data: {
                asRecordedLine:
                  "draws its split again with seed 42 over the labels as they are now",
                absenceMessage: "unused",
                choices: [{ manifestDir: "/data/splits", label: <span>/data/splits</span> }],
              },
            }),
          ],
        }}
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    fireEvent.click(screen.getByText("exp-1"));
    fireEvent.click(screen.getByRole("radio", { name: "/data/splits" }));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(onStart).toHaveBeenCalledWith("/data/splits"));
  });

  it("starts with null for the default As recorded choice", async () => {
    const onStart = vi.fn().mockResolvedValue(undefined);
    render(
      <LaunchPicker
        list={{
          title: "Configs in this project",
          emptyMessage: "none",
          rows: [
            row({
              onStart,
              data: {
                asRecordedLine: "on the partition it bound",
                absenceMessage: "unused",
                choices: [{ manifestDir: "/data/splits", label: <span>/data/splits</span> }],
              },
            }),
          ],
        }}
        composerLabel="Describe a new one"
        request=""
        onRequestChange={noop}
        onSend={noop}
      />,
    );
    fireEvent.click(screen.getByText("exp-1"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(onStart).toHaveBeenCalledWith(null));
  });
});
