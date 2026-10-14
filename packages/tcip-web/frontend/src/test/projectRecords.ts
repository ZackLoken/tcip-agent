import type { CaptureInvariances } from "@/api/types.generated";

/** A sample breeder's answer a readable project record carries: every capture transform leaves
 * the trait's measurement unchanged. */
export const INVARIANCES: CaptureInvariances = {
  horizontal_flip: true,
  vertical_flip: true,
  rotation: 180,
  color: true,
  intensity: true,
  scale: true,
};
