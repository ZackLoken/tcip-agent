/**
 * Recharts takes literal colour strings (not Tailwind classes), so the field-station tokens
 * are mirrored here as hex. Shared by the Training + Results charts so they read as one
 * instrument rather than two ad-hoc palettes. Keep in sync with tailwind.config.ts.
 */
export const CHART = {
  grid: "#33352C", // tcip-border
  axis: "#8C9082", // tcip-muted
  tooltipBg: "#20211B", // tcip-panel
  tooltipBorder: "#33352C", // tcip-border
  legendText: "#E7E5DC", // tcip-fg
} as const;

/** On-brand, visually distinct series palette — SI green + persimmon + the season scale +
 *  the detection blue. Cycled for charts with many series. */
export const CHART_LINE_COLORS = [
  "#507754", // SI green
  "#E6976B", // persimmon
  "#7FA96A", // canopy
  "#C9A24B", // late-summer gold
  "#00BFFF", // detection blue
  "#B48EAD", // muted mauve
  "#7E9CB9", // slate blue
  "#5B6B6A", // dormant gray-green
];
