import type { TabName } from "@/store/types";

/** Every tab's display name: the tab strip's own label (TopBar), and what a tab's hidden
 * heading (TabHeading) names a screen reader lands on when the tab mounts. One map, so the
 * two never drift apart under a different name for the same tab. */
export const TAB_LABELS: Record<TabName, string> = {
  annotate: "Annotate",
  review: "Review",
  training: "Training",
  tuning: "Tuning",
  inference: "Inference",
  results: "Results",
  meta: "Meta",
};
