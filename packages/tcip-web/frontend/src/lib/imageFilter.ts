/** The image stepper's filter: by what each image's label document says about the dataset
 *  subject (every image, the unannotated ones, the ones in progress, or the complete ones), or
 *  the held review queue's images in its rank order. */

import type { CaptureImage, PriorityQueueJob } from "@/api/types.generated";
import { isFinished, type DatasetSelection, type SubjectState } from "@/store/types";

export const IMAGE_FILTERS = ["all", "unannotated", "in_progress", "complete", "queue"] as const;
export type ImageFilter = (typeof IMAGE_FILTERS)[number];

export const IMAGE_FILTER_WORDS: Record<ImageFilter, string> = {
  all: "all",
  unannotated: "unannotated",
  in_progress: "in progress",
  complete: "complete",
  queue: "review queue",
};

/** Whether an image whose document says `state` of the subject passes `filter`; an image whose
 *  state is not known yet passes only `all`. */
export function imageFilterKeeps(
  filter: Exclude<ImageFilter, "queue">,
  state: SubjectState | undefined,
): boolean {
  if (filter === "all") return true;
  if (state === undefined) return false;
  if (filter === "complete") return isFinished(state);
  if (filter === "in_progress") return state === "partial";
  return state === "unannotated";
}

/** The image names of the completed review queue `job` in rank order, when it ranked the capture
 *  `dataset` selects (its dataset root and capture); null for any other capture or no queue. */
export function rankedNames(
  job: PriorityQueueJob | null,
  dataset: DatasetSelection | null,
): string[] | null {
  if (!job?.ranked || job.dataset_root !== dataset?.dataset_root || job.capture !== dataset.date) {
    return null;
  }
  return job.ranked.queue.map((entry) => entry.image);
}

/** The filter the stepper walks: the one `chosen`, except a review queue `ranked` does not hold
 *  for the selected capture (null), which walks every image. */
export function filterInEffect(chosen: ImageFilter, ranked: readonly string[] | null): ImageFilter {
  return chosen === "queue" && ranked === null ? "all" : chosen;
}

/** The indices into `images` of the images `filter` keeps, in the order the stepper walks them:
 *  the list's own for a filter by state, `ranked`'s for the review queue. */
export function filteredIndices(
  images: CaptureImage[],
  states: Record<string, SubjectState>,
  filter: ImageFilter,
  ranked: readonly string[],
): number[] {
  const at = new Map(images.map(({ name }, i) => [name, i]));
  const names =
    filter === "queue"
      ? ranked
      : images.map(({ name }) => name).filter((name) => imageFilterKeeps(filter, states[name]));
  return names.flatMap((name) => at.get(name) ?? []);
}
