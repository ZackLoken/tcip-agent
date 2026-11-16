import { useStore } from "@/store";
import type { Raised } from "@/store/slices/notifications";

/** Raise `sentence` as a `kind` of the open project about `target`, the thing within the project
 *  it concerns as the person reads it, standing on `key` when one is given; a job states whether
 *  the entry standing on its key is told again (`retell`). */
export function tell(
  kind: Exclude<Raised["kind"], "job">,
  level: Raised["level"],
  sentence: string,
  target: string,
  key?: string,
): void;
export function tell(
  kind: "job",
  level: Raised["level"],
  sentence: string,
  target: string,
  key: string,
  retell: boolean,
): void;
export function tell(
  kind: Raised["kind"],
  level: Raised["level"],
  sentence: string,
  target: string,
  key?: string,
  retell?: boolean,
): void {
  const { raiseNotification, openProject } = useStore.getState();
  const told = { level, text: { sentence }, context: { project: openProject?.id, target }, key };
  raiseNotification(
    kind === "job" ? { ...told, kind, retell: retell === true } : { ...told, kind },
  );
}
