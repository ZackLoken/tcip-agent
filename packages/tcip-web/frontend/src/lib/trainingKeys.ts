import type { ListingName } from "@/store/slices/jobListings";

/** The key a training message of `kind` about the run or sweep `id` stands on in the project
 *  `projectId`: a Cancel's failure or a stream's error. */
export function trainingKey(
  kind: "cancel" | "stream",
  projectId: string | undefined,
  id: string,
): string {
  return `training-${kind}:${projectId}:${id}`;
}

/** The key the entry of the job `id`, a run or sweep of the training listing or a job of the
 *  inference listing as `listing` names it, stands on in the project `projectId`. */
export function jobKey(listing: ListingName, projectId: string, id: string): string {
  return `${listing}-job:${projectId}:${id}`;
}
