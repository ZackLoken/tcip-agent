import type { Config } from "tailwindcss";

/**
 * Palette and typography mirror yolo-annotator (yololabeler) so the TCIP GUI
 * feels identical to the desktop app our breeding colleagues already use.
 */
const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // yololabeler palette
        "tcip-bg": "#1E1E1E",
        "tcip-canvas": "#2D2D2D",
        "tcip-fg": "#E0E0E0",
        "tcip-muted": "#8A8A8A",
        "tcip-border": "#3A3A3A",
        "tcip-panel": "#242424",
        "tcip-accent": "#507754",       // SI_GREEN
        "tcip-accent-hover": "#608864", // ACCENT_HOVER
        "tcip-warn": "#E6976B",         // SI_PERSIMMON
        // Detection tags
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
    },
  },
  plugins: [],
};

export default config;
