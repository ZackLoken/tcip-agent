import { cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { dockWithNotificationsOpen, measureWindow, PANE_FOCUS } from "@/test/dock";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Dock with the notifications pane open", () => {
  beforeEach(measureWindow);

  // jsdom computes no stacking: a modal painting above the open pane is read in a browser.
  it("holds the layout and its overlay pane in one isolated stacking layer", async () => {
    const { container } = render(dockWithNotificationsOpen());
    const pane = await PANE_FOCUS.panel();

    const dock = container.firstElementChild as HTMLElement;
    expect(dock).toHaveClass("isolate");
    expect(dock).toContainElement(pane);
  });
});
