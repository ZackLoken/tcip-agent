import { screen } from "@testing-library/react";
import { vi } from "vitest";

import { Dock } from "@/components/Dock";
import { defaultDockModel, toggleNotifications } from "@/lib/dock";

const WINDOW_RECT = {
  x: 0,
  y: 0,
  top: 0,
  left: 0,
  width: 1280,
  height: 800,
  right: 1280,
  bottom: 800,
};

/** Give every element a window's size for the docking layout: jsdom lays nothing out, and
 *  FlexLayout renders a pane only once its container measures one. */
export function measureWindow(): void {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(WINDOW_RECT as DOMRect);
}

/** The elements of the open notifications pane that take focus, by what a person pressed to put
 *  focus there: the pane's heading, which opening the pane focuses, and the pane's own panel,
 *  which a click on its text focuses. Each answers the element once the pane is shown. */
export const PANE_FOCUS = {
  heading: () => screen.findByRole("heading", { name: "Notifications" }),
  panel: () => screen.findByRole("tabpanel", { name: "Notifications" }),
};

/** The docking layout with the notifications pane open over an empty work area, its model made
 *  by the layout's own default and toggle. Needs `measureWindow` first. */
export function dockWithNotificationsOpen() {
  const model = defaultDockModel();
  toggleNotifications(model);
  return <Dock model={model} work={null} />;
}
