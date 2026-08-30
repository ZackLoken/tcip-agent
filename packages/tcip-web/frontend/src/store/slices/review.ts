import type { StateCreator } from "zustand";

import type { ActionPayload } from "@/api/types.generated";
import type { AppState } from "@/store/appState";
import type { Detection, MatchesResponse } from "@/store/types";

interface ReviewTabState {
  matches: MatchesResponse | null;
  loading: boolean;
  /** One-shot: the detection index a `review_focus` command asked to center on. The reload
   *  effect consumes it once (else it always jumps to first-unreviewed, dropping the agent's
   *  "look at detection N" request). */
  focusDetectionIdx: number | null;
  // Bumped by a review_focus command to force a matches refetch when image + paths are unchanged.
  refetchNonce: number;
}

export interface ReviewSlice {
  /** Review tab derived state. */
  review: ReviewTabState;

  /** Review helpers. */
  setMatches: (matches: MatchesResponse | null) => void;
  setReviewLoading: (loading: boolean) => void;
  setReviewDetectionIdx: (idx: number) => void;
  setReviewFocusIdx: (idx: number | null) => void;
  /** Force a matches refetch even when image/paths are unchanged (re-focus on the open image). */
  bumpReviewRefetch: () => void;
  markDetectionReviewed: (idx: number, action: Exclude<ActionPayload["action"], "swept">) => void;
}

export const createReviewSlice: StateCreator<AppState, [], [], ReviewSlice> = (set) => ({
  review: { matches: null, loading: false, focusDetectionIdx: null, refetchNonce: 0 },

  setMatches: (matches) => set((s) => ({ review: { ...s.review, matches } })),
  setReviewLoading: (loading) => set((s) => ({ review: { ...s.review, loading } })),
  setReviewDetectionIdx: (idx) =>
    set((s) => ({
      gui: { ...s.gui, review: { ...s.gui.review, detection_idx: idx } },
    })),
  setReviewFocusIdx: (idx) => set((s) => ({ review: { ...s.review, focusDetectionIdx: idx } })),
  bumpReviewRefetch: () =>
    set((s) => ({ review: { ...s.review, refetchNonce: s.review.refetchNonce + 1 } })),

  markDetectionReviewed: (idx, action) =>
    set((s) => {
      if (!s.review.matches) return s;
      const next: Detection[] = s.review.matches.detections.slice();
      if (next[idx]) {
        next[idx] = { ...next[idx], reviewed: true, reviewed_action: action };
      }
      return {
        review: {
          ...s.review,
          matches: { ...s.review.matches, detections: next },
        },
      };
    }),
});
