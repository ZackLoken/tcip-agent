import "flexlayout-react/style/dark.css";

import { Layout, type Model, type TabNode } from "flexlayout-react";
import { type ReactNode, useCallback } from "react";

import { NotificationsPane } from "@/components/Notifications";
import { WORK } from "@/lib/dock";

/** The docking layout under the top bar: `work` fills the work area, and the notifications area
 *  is its pane. */
export function Dock({ model, work }: { model: Model; work: ReactNode }) {
  const factory = useCallback(
    (node: TabNode) => (node.getComponent() === WORK ? work : <NotificationsPane />),
    [work],
  );
  return (
    <div className="relative min-h-0 flex-1">
      <Layout model={model} factory={factory} />
    </div>
  );
}
