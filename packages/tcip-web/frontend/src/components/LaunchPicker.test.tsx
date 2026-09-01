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
        list={{ title: "Configs in this project", emptyMessage: "No config exists in this project yet.", rows: [] }}
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
      <LaunchPicker composerLabel="Describe a new one" request="" onRequestChange={noop} onSend={noop} />,
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
      .mockRejectedValue(new StructuredRefusalError({ issues: ["batch_size must be positive"] }, 422, ""));
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
    await waitFor(() => expect(screen.getByText("batch_size must be positive")).toBeInTheDocument());
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
    await waitFor(() => expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument());
  });
});
