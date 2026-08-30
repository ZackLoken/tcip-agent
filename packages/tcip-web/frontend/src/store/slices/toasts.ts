import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface Toast {
  id: number;
  message: string;
  level: "error" | "info" | "success";
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
      const id = (s.toasts[s.toasts.length - 1]?.id ?? 0) + 1;
      // Cap the stack so a failing poll can't flood the screen.
      return { toasts: [...s.toasts, { id, message, level }].slice(-4) };
    }),
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
});
