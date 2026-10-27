/** The image the browser reads off the dataset selection. */

import type { CaptureImage } from "@/api/types.generated";
import type { DatasetSelection } from "@/store/types";

/** The image the selection currently points at, or null when no capture or no image is in it. */
export function currentImage(dataset: DatasetSelection | null): CaptureImage | null {
  return dataset?.images[dataset.current_image_index] ?? null;
}
