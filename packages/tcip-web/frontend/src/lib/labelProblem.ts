import { useStore } from "@/store";

/** Raise a `/dataset/select` response's `label_problem` as a failure of the project `projectId`,
 *  naming the label document the way the project list's own `label_problem` field does; nothing
 *  when it names none. Every path that installs a new selection through that route,
 *  human-driven or agent-driven, calls this, so the advisory reaches the breeder on whichever
 *  path landed on the corrupt date. */
export function raiseLabelProblem(
  labelProblem: string | null | undefined,
  projectId: string,
): void {
  if (labelProblem) {
    useStore.getState().raiseNotification({
      kind: "failure",
      level: "error",
      text: { sentence: labelProblem },
      context: { project: projectId },
    });
  }
}
