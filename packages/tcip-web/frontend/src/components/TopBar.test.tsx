import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

import { TopBar } from "@/components/TopBar";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

afterEach(cleanup);

describe("TopBar canvas-binding notice", () => {
  it("shows nothing when the binding is present and the socket is connected", () => {
    useStore.setState({ wsStatus: "connected", canvasBindingMissing: false });
    render(<TopBar />);
    expect(screen.queryByText(/canvas not synced/i)).not.toBeInTheDocument();
  });

  it("shows a notice when the binding is missing, even while connected", () => {
    useStore.setState({ wsStatus: "connected", canvasBindingMissing: true });
    render(<TopBar />);
    expect(screen.getByText(/canvas not synced, reopen the project/i)).toBeInTheDocument();
  });

  it("clears once canvasBindingMissing is cleared", () => {
    useStore.setState({ wsStatus: "connected", canvasBindingMissing: true });
    render(<TopBar />);
    expect(screen.getByText(/canvas not synced/i)).toBeInTheDocument();

    act(() => useStore.setState({ canvasBindingMissing: false }));
    expect(screen.queryByText(/canvas not synced/i)).not.toBeInTheDocument();
  });

  it("prefers the connection message over the binding one while disconnected", () => {
    useStore.setState({ wsStatus: "disconnected", canvasBindingMissing: true });
    render(<TopBar />);
    expect(screen.getByText(/disconnected, retrying/i)).toBeInTheDocument();
    expect(screen.queryByText(/canvas not synced/i)).not.toBeInTheDocument();
  });
});

describe("TopBar tab strip accessibility", () => {
  it("exposes the strip as a tablist and marks the active tab selected", () => {
    render(<TopBar />);
    const active = useStore.getState().gui.active_tab;
    const tablist = screen.getByRole("tablist");
    const tabs = screen.getAllByRole("tab");
    expect(tabs.length).toBeGreaterThan(1);
    expect(tablist).toContainElement(tabs[0]);
    const selected = tabs.filter((t) => t.getAttribute("aria-selected") === "true");
    expect(selected).toHaveLength(1);
    expect(selected[0]).toHaveTextContent(new RegExp(active, "i"));
  });

  it("moves aria-selected to the clicked tab", () => {
    render(<TopBar />);
    const tuningTab = screen.getByRole("tab", { name: /tuning/i });
    fireEvent.click(tuningTab);
    expect(tuningTab).toHaveAttribute("aria-selected", "true");
  });

  it("points each tab's aria-controls at that tab's own panel id", () => {
    render(<TopBar />);
    const tuningTab = screen.getByRole("tab", { name: /tuning/i });
    expect(tuningTab).toHaveAttribute("id", "tab-tuning");
    expect(tuningTab).toHaveAttribute("aria-controls", "tabpanel-tuning");
  });

  it("keeps only the active tab in the Tab order, a roving tabindex", () => {
    render(<TopBar />);
    const tabs = screen.getAllByRole("tab");
    const active = tabs.filter((t) => t.getAttribute("aria-selected") === "true");
    const inactive = tabs.filter((t) => t.getAttribute("aria-selected") !== "true");
    expect(active).toHaveLength(1);
    expect(active[0]).toHaveAttribute("tabindex", "0");
    inactive.forEach((t) => expect(t).toHaveAttribute("tabindex", "-1"));

    fireEvent.click(screen.getByRole("tab", { name: /tuning/i }));
    expect(screen.getByRole("tab", { name: /tuning/i })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("tab", { name: /training/i })).toHaveAttribute("tabindex", "-1");
  });

  it("moves selection and focus with the arrow keys, wrapping at either end", () => {
    render(<TopBar />);
    const training = screen.getByRole("tab", { name: /training/i });
    fireEvent.click(training);
    training.focus();

    fireEvent.keyDown(training, { key: "ArrowRight" });
    const tuning = screen.getByRole("tab", { name: /tuning/i });
    expect(tuning).toHaveAttribute("aria-selected", "true");
    expect(tuning).toHaveFocus();

    fireEvent.keyDown(tuning, { key: "ArrowLeft" });
    expect(screen.getByRole("tab", { name: /training/i })).toHaveAttribute("aria-selected", "true");

    const annotate = screen.getByRole("tab", { name: /annotate/i });
    fireEvent.click(annotate);
    annotate.focus();
    fireEvent.keyDown(annotate, { key: "ArrowLeft" });
    const last = screen.getAllByRole("tab").at(-1) as HTMLElement;
    expect(last).toHaveAttribute("aria-selected", "true");
  });
});
