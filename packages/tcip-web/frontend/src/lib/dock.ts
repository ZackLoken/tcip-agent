import {
  Actions,
  BorderNode,
  DockLocation,
  type IJsonModel,
  type IJsonTabNode,
  Model,
  type TabNode,
} from "flexlayout-react";

/** The work area's tab, which holds the shell or the picker. */
export const WORK = "work";
const NOTIFICATIONS = "notifications";

const NOTIFICATIONS_TAB: IJsonTabNode = {
  type: "tab",
  id: NOTIFICATIONS,
  name: "Notifications",
  component: NOTIFICATIONS,
  enableDrag: false,
};

/** The default arrangement: the work area alone, with no tab strip and nothing dropped into it,
 *  and the right border, shown only once it holds a pane, which opens a pane over the work area
 *  and keeps it as a strip on that edge while minimized. The notifications area starts closed
 *  and stays in that border. */
const DEFAULT_LAYOUT: IJsonModel = {
  global: { tabEnableRenderOnDemand: false },
  borders: [
    { type: "border", location: "right", borderType: "overlay", enableAutoHide: true, size: 360 },
  ],
  layout: {
    type: "row",
    children: [
      {
        type: "tabset",
        enableTabStrip: false,
        enableDrop: false,
        enableMaximize: false,
        children: [
          {
            type: "tab",
            id: WORK,
            name: "Work",
            component: WORK,
            enableClose: false,
            enableDrag: false,
          },
        ],
      },
    ],
  },
};

export function defaultDockModel(): Model {
  return Model.fromJson(DEFAULT_LAYOUT);
}

/** Whether the notifications area is shown in `model`, open over the work area. */
export function notificationsOpen(model: Model): boolean {
  const node = model.getNodeById(NOTIFICATIONS) as TabNode | undefined;
  return node?.isSelected() ?? false;
}

/** Open the notifications area from its strip, or add it to the right border once it was closed;
 *  close it when it is shown. */
export function toggleNotifications(model: Model): void {
  if (notificationsOpen(model)) {
    model.doAction(Actions.deleteTab(NOTIFICATIONS));
  } else if (model.getNodeById(NOTIFICATIONS)) {
    model.doAction(Actions.selectTab(NOTIFICATIONS));
  } else {
    let border = "";
    model.visitNodes((node) => {
      if (node instanceof BorderNode && node.getLocation() === DockLocation.RIGHT) {
        border = node.getId();
      }
    });
    model.doAction(Actions.addNode(NOTIFICATIONS_TAB, border, DockLocation.CENTER, -1, true));
  }
}
