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
});
