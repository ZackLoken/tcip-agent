import { TAB_LABELS } from "@/lib/tabLabels";
import type { TabName } from "@/store/types";

/**
 * The one heading element every tab's main panel carries, so a screen reader lands on a
 * named region when the tab mounts. Visually hidden: the tab strip already shows the tab's
 * name, so a second visible title here would only repeat it.
 */
export function TabHeading({ tab }: { tab: TabName }) {
  return <h1 className="sr-only">{TAB_LABELS[tab]}</h1>;
}
