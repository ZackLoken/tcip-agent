import type { Config } from "tailwindcss";

/**
 * "Field station" identity — a purpose-built instrument for a tree-crop breeding
 * program, not a generic dark dashboard. The surfaces read as bark, soil, and
 * field-notebook paper (warm neutrals) so the Savanna Institute green and persimmon
 * feel botanical rather than "dark IDE + accent". Two tokens are held fixed because
 * the focus-ring accessibility test pins them: `tcip-bg` (#1E1E1E, the ring offset)
 * and `tcip-accent` (#507754 SI_GREEN, the ring colour). The warmth is carried by the
 * text and panel surfaces you actually look at, plus the phenology season-rail
 * signature (see SeasonRail).
 */
const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "tcip-bg": "#1E1E1E", // held fixed (focus-ring offset test)
        "tcip-canvas": "#26271F", // warm-neutral canvas host — annotation colours read true
        "tcip-fg": "#E7E5DC", // field-notebook paper (warm off-white)
        "tcip-muted": "#8C9082", // sage gray
        "tcip-border": "#33352C", // bark border
        "tcip-panel": "#20211B", // bark panel surface
        "tcip-hover": "#282922", // warm raised hover surface (secondary controls)
        "tcip-border-hover": "#454A3B",
        "tcip-accent": "#507754", // SI_GREEN — held fixed (focus-ring colour test)
        "tcip-accent-hover": "#46694A", // darker on hover: white 12px text stays AA (6.2:1)
        "tcip-warn": "#E6976B", // SI_PERSIMMON — the warm / autumn signal
        // Phenology season scale (dormant → bud → canopy → late-summer → fruit),
        // used by the SeasonRail signature. Anchored on the brand green + persimmon.
        "tcip-season-0": "#5B6B6A",
        "tcip-season-1": "#7FA96A",
        "tcip-season-2": "#507754",
        "tcip-season-3": "#C9A24B",
        "tcip-season-4": "#E6976B",
        // Detection tags — kept vivid + distinct (CV correctness).
        "tcip-tp": "#4CAF50",
        "tcip-fp": "#EF5350",
        "tcip-fn": "#FFA726",
        "tcip-pred": "#00BFFF",
        "tcip-focus": "#FFD700",
      },
      fontFamily: {
        sans: [
          "Archivo",
          "Segoe UI",
          "Helvetica Neue",
          "Helvetica",
          "Arial",
          "DejaVu Sans",
          "sans-serif",
        ],
        mono: ["Consolas", "Menlo", "monospace"],
      },
      spacing: {
        topbar: "44px",
        statusbar: "28px",
      },
      keyframes: {
        "tcip-rise": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "tcip-rise": "tcip-rise 0.32s ease-out both",
      },
    },
  },
  plugins: [],
};

export default config;
