import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface Toast {
  id: number;
  message: string;
  level: "error" | "info" | "success";
  /** How many times this exact message (same text, same level) has been pushed since it last
   * appeared; absent (or 1) for one that has never repeated. */
  count?: number;
}

export interface ToastSlice {
  /** Transient user-facing notifications (API failures, etc.). */
  toasts: Toast[];
  pushToast: (message: string, level?: Toast["level"]) => void;
  dismissToast: (id: number) => void;
}

export const createToastSlice: StateCreator<AppState, [], [], ToastSlice> = (set) => ({
  toasts: [],

  pushToast: (message, level = "error") =>
    set((s) => {
      // A repeat of a toast still on screen collapses into it rather than stacking a second,
      // identical one: a flaky poll must not flood the screen with the same sentence.
      const repeat = s.toasts.find((t) => t.message === message && t.level === level);
      if (repeat) {
        return {
          toasts: s.toasts.map((t) => (t === repeat ? { ...t, count: (t.count ?? 1) + 1 } : t)),
        };
      }
      const id = (s.toasts[s.toasts.length - 1]?.id ?? 0) + 1;
      // Cap the stack so a failing poll can't flood the screen.
      return { toasts: [...s.toasts, { id, message, level }].slice(-4) };
    }),
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
});
