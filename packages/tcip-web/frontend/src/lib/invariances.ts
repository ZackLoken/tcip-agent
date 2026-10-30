import type { CaptureInvariances } from "@/api/types.generated";

/** An invariance as the breeder reads it, named by its record field. */
export function invarianceName(name: keyof CaptureInvariances): string {
  return name.replaceAll("_", " ");
}

/** An invariance answer as the breeder reads it: yes or no, or a rotation in degrees. */
export function answerPhrase(value: boolean | number): string {
  return typeof value === "boolean" ? (value ? "yes" : "no") : `${value} degrees`;
}
