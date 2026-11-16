import { useStore } from "@/store";
import type { MessageLevel } from "@/store/slices/notifications";

/** Raise `sentence` as a `kind` of the open project about `target`, the thing within the project
 *  it concerns as the person reads it, standing on `key` when one is given. */
export function tell(
  kind: "failure" | "notice" | "job",
  level: MessageLevel,
  sentence: string,
  target: string,
  key?: string,
): void {
  const { raiseNotification, openProject } = useStore.getState();
  raiseNotification({
    kind,
    level,
    text: { sentence },
    context: { project: openProject?.id, target },
    key,
  });
}
