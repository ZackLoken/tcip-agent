/** The image stepper's filter by what each image's label document says about the dataset
 *  subject: every image, the unannotated ones, the ones in progress, or the complete ones. */

import { isFinished, type SubjectState } from "@/store/types";

export const IMAGE_FILTERS = ["all", "unannotated", "in_progress", "complete"] as const;
export type ImageFilter = (typeof IMAGE_FILTERS)[number];

export const IMAGE_FILTER_WORDS: Record<ImageFilter, string> = {
  all: "all",
  unannotated: "unannotated",
  in_progress: "in progress",
  complete: "complete",
};

/** Whether an image whose document says `state` of the subject passes `filter`; an image whose
 *  state is not known yet passes only `all`. */
export function imageFilterKeeps(filter: ImageFilter, state: SubjectState | undefined): boolean {
  if (filter === "all") return true;
  if (state === undefined) return false;
  if (filter === "complete") return isFinished(state);
  if (filter === "in_progress") return state === "partial";
  return state === "unannotated";
}

/** The indices into `imageList` of the images `filter` keeps, in list order. */
export function filteredIndices(
  imageList: string[],
  states: Record<string, SubjectState>,
  filter: ImageFilter,
): number[] {
  const kept: number[] = [];
  imageList.forEach((name, i) => {
    if (imageFilterKeeps(filter, states[name])) kept.push(i);
  });
  return kept;
}
