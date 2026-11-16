/** The key a training message of `kind` about the run or sweep `id` stands on in the project
 *  `projectId`: a Cancel's failure, a stream's error or a run's end. */
export function trainingKey(
  kind: "cancel" | "stream" | "end",
  projectId: string | undefined,
  id: string,
): string {
  return `training-${kind}:${projectId}:${id}`;
}
