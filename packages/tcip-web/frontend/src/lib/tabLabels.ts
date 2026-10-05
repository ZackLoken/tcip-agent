import type { TabName } from "@/store/types";

/** Every tab's display name, by tab. */
export const TAB_LABELS: Record<TabName, string> = {
  setup: "Setup",
  annotate: "Annotate",
  training: "Training",
  inference: "Inference",
  results: "Results",
  meta: "Meta",
};
