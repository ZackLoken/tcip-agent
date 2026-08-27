import { useStore } from "@/store";

/** Toast a `/dataset/select` response's `label_problem`, naming the label document the way the
 *  project list's own `label_problem` field does. Every path that installs a new selection
 *  through that route, human-driven or agent-driven, calls this, so the advisory reaches the
 *  breeder on whichever path landed on the corrupt date. */
export function toastLabelProblem(labelProblem: string | null | undefined): void {
  if (labelProblem) useStore.getState().pushToast(labelProblem);
}
