import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";

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
