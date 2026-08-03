import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

import { TabBanner } from "@/components/TabBanner";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

afterEach(cleanup);

describe("TabBanner", () => {
  it("renders the active tab's note and nothing for the others", () => {
    useStore.getState().pushBanner("training", "e1", "Two runs are queued behind this one.");
    useStore.getState().setActiveTab("results");
    const { container } = render(<TabBanner />);
    expect(container).toBeEmptyDOMElement();

    cleanup();
    useStore.getState().setActiveTab("training");
    render(<TabBanner />);
    expect(screen.getByText("Two runs are queued behind this one.")).toBeInTheDocument();
  });

  it("stays dismissed when the same event is replayed, and shows a new one", () => {
    useStore.getState().setActiveTab("review");
    useStore.getState().pushBanner("review", "e1", "First note");
    render(<TabBanner />);

    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(screen.queryByText("First note")).not.toBeInTheDocument();

    // The backend replays its event ring on every reconnect: the same id must stay dismissed.
    act(() => useStore.getState().pushBanner("review", "e1", "First note"));
    expect(screen.queryByText("First note")).not.toBeInTheDocument();

    act(() => useStore.getState().pushBanner("review", "e2", "Second note"));
    expect(screen.getByText("Second note")).toBeInTheDocument();
  });

  it("truncates a note long enough to displace the tab below it", () => {
    useStore.getState().setActiveTab("inference");
    useStore.getState().pushBanner("inference", "e1", "x".repeat(400));
    render(<TabBanner />);
    const rendered = screen.getByText(/^x+…$/).textContent ?? "";
    expect(rendered.length).toBeLessThan(400);
  });
});
