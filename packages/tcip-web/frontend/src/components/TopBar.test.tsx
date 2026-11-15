import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { NotificationsButton } from "@/components/Notifications";
import { TopBar } from "@/components/TopBar";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.getState().setUser("grower");
});

afterEach(cleanup);

/** The top bar as App renders it, its notifications button closed. */
function renderTopBar() {
  return render(
    <TopBar notifications={<NotificationsButton open={false} onToggle={() => {}} />} />,
  );
}

describe("TopBar logo", () => {
  it("shows the white side-stacked logo the dark bar calls for", () => {
    renderTopBar();
    const logos = screen.getAllByRole("img", { name: "Savanna Institute" });
    expect(logos).toHaveLength(1);
    expect(logos[0]).toHaveAttribute("src", "/assets/WHITE_SI_logo_SIDESTACKED.png");
  });
});

describe("TopBar before a name is committed", () => {
  it("shows the logo and the notifications button, and neither the tabs nor the agent toggle", () => {
    act(() => useStore.getState().setUser(""));
    renderTopBar();
    expect(screen.getByRole("img", { name: "Savanna Institute" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Notifications" })).toBeInTheDocument();
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Toggle agent terminal" })).not.toBeInTheDocument();
  });
});

describe("TopBar tab strip accessibility", () => {
  it("exposes the strip as a tablist and marks the active tab selected", () => {
    renderTopBar();
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
    renderTopBar();
    const inferenceTab = screen.getByRole("tab", { name: /inference/i });
    fireEvent.click(inferenceTab);
    expect(inferenceTab).toHaveAttribute("aria-selected", "true");
  });

  it("points each tab's aria-controls at that tab's own panel id", () => {
    renderTopBar();
    const inferenceTab = screen.getByRole("tab", { name: /inference/i });
    expect(inferenceTab).toHaveAttribute("id", "tab-inference");
    expect(inferenceTab).toHaveAttribute("aria-controls", "tabpanel-inference");
  });

  it("keeps only the active tab in the Tab order, a roving tabindex", () => {
    renderTopBar();
    const tabs = screen.getAllByRole("tab");
    const active = tabs.filter((t) => t.getAttribute("aria-selected") === "true");
    const inactive = tabs.filter((t) => t.getAttribute("aria-selected") !== "true");
    expect(active).toHaveLength(1);
    expect(active[0]).toHaveAttribute("tabindex", "0");
    inactive.forEach((t) => expect(t).toHaveAttribute("tabindex", "-1"));

    fireEvent.click(screen.getByRole("tab", { name: /inference/i }));
    expect(screen.getByRole("tab", { name: /inference/i })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("tab", { name: /training/i })).toHaveAttribute("tabindex", "-1");
  });

  it("moves selection and focus with the arrow keys, wrapping at either end", () => {
    renderTopBar();
    const training = screen.getByRole("tab", { name: /training/i });
    fireEvent.click(training);
    training.focus();

    fireEvent.keyDown(training, { key: "ArrowRight" });
    const inference = screen.getByRole("tab", { name: /inference/i });
    expect(inference).toHaveAttribute("aria-selected", "true");
    expect(inference).toHaveFocus();

    fireEvent.keyDown(inference, { key: "ArrowLeft" });
    expect(screen.getByRole("tab", { name: /training/i })).toHaveAttribute("aria-selected", "true");

    const first = screen.getAllByRole("tab")[0];
    fireEvent.click(first);
    first.focus();
    fireEvent.keyDown(first, { key: "ArrowLeft" });
    const last = screen.getAllByRole("tab").at(-1) as HTMLElement;
    expect(last).toHaveAttribute("aria-selected", "true");
  });
});
