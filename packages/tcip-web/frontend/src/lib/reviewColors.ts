import { useEffect, useState } from "react";

// Review symbology colours (color = outcome; line style = source). User-customisable and
// persisted, so a reviewer can retune TP/FP/FN/under-review to their imagery. Shared here so the
// canvas symbology and the status-bar counts read one palette and stay in lock-step.
export interface ReviewColors {
  tp: string;
  fp: string;
  fn: string;
  active: string;
}

export const DEFAULT_REVIEW_COLORS: ReviewColors = {
  tp: "#4CAF50", // matched
  fp: "#EF5350", // false positive
  fn: "#FFD54A", // missed (gold)
  active: "#00BFFF", // the detection under review, highlighter blue
};

const REVIEW_COLORS_KEY = "tcip.review.colors";
const REVIEW_COLORS_EVENT = "tcip:review-colors";

export function loadReviewColors(): ReviewColors {
  try {
    const raw = localStorage.getItem(REVIEW_COLORS_KEY);
    if (raw) return { ...DEFAULT_REVIEW_COLORS, ...JSON.parse(raw) };
  } catch {
    /* storage disabled, fall back to defaults */
  }
  return DEFAULT_REVIEW_COLORS;
}

function saveReviewColors(colors: ReviewColors): void {
  try {
    localStorage.setItem(REVIEW_COLORS_KEY, JSON.stringify(colors));
  } catch {
    /* storage disabled, colours just won't persist */
  }
  window.dispatchEvent(new CustomEvent<ReviewColors>(REVIEW_COLORS_EVENT, { detail: colors }));
}

/** Shared review palette. Every component using this hook re-renders when the palette changes
 *  (via a same-tab custom event), so recolouring TP in the legend also recolours the TP count. */
export function useReviewColors(): [
  ReviewColors,
  (next: ReviewColors | ((prev: ReviewColors) => ReviewColors)) => void,
] {
  const [colors, setColors] = useState<ReviewColors>(loadReviewColors);
  useEffect(() => {
    const onChange = (e: Event) => {
      const detail = (e as CustomEvent<ReviewColors>).detail;
      if (detail) setColors(detail);
    };
    window.addEventListener(REVIEW_COLORS_EVENT, onChange);
    return () => window.removeEventListener(REVIEW_COLORS_EVENT, onChange);
  }, []);
  const update = (next: ReviewColors | ((prev: ReviewColors) => ReviewColors)) => {
    setColors((prev) => {
      const resolved = typeof next === "function" ? next(prev) : next;
      saveReviewColors(resolved);
      return resolved;
    });
  };
  return [colors, update];
}
