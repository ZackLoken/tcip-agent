import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";

import App from "@/App";
import { useStore } from "@/store";

// App's own socket/tab-sync effects reach the network; only the tab/panel wiring is under
// test here, so both are stubbed rather than left to hit a backend that isn't running.
vi.mock("@/api/ws", () => ({
  stateSocket: {
    connect: vi.fn(),
    close: vi.fn(),
    subscribePanel: vi.fn(() => () => {}),
  },
}));
vi.mock("@/hooks/useActiveTabSync", () => ({ useActiveTabSync: vi.fn() }));
// App statically imports the Annotate tab (not code-split), which pulls in Konva; jsdom has no
// canvas backend, so the Konva module itself is stubbed the same way AnnotateTab's own tests do.
vi.mock("konva", () => ({ default: {} }));
// The Training tab fires its own run listing on mount regardless of any project being open;
// stubbed so switching to it here exercises only the tab/panel wiring, not a real fetch.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return {
    ...actual,
    trainingApi: { ...actual.trainingApi, listRuns: vi.fn().mockResolvedValue({ runs: [] }) },
  };
});

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

afterEach(cleanup);

describe("App tab/panel wiring", () => {
  it("labels the active tab's panel by the selected tab button, and points that button at it", async () => {
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);

    const tab = await screen.findByRole("tab", { name: /meta/i });
    const panel = screen.getByRole("tabpanel");
    expect(tab).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
  });

  it("moves the panel's id and labelledby to the newly active tab on selection", async () => {
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);
    await screen.findByRole("tabpanel");

    act(() => useStore.getState().setActiveTab("training"));
    const panel = await screen.findByRole("tabpanel");
    const tab = screen.getByRole("tab", { name: /training/i });
    expect(panel.id).toBe("tabpanel-training");
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
  });
});
