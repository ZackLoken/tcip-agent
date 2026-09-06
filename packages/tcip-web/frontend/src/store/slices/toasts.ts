import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface Toast {
  id: number;
  message: string;
  level: "error" | "info" | "success";
  /** How many times this exact message (same text, same level) has been pushed since it last
   * appeared; absent (or 1) for one that has never repeated. */
  count?: number;
  /** A channel name; a later push on the same channel replaces this toast under a fresh id
   * instead of stacking beside it. */
  channel?: string;
}

export interface ToastSlice {
  /** Transient user-facing notifications (API failures, etc.). */
  toasts: Toast[];
  /** `channel` replaces the standing toast on that channel under a fresh id (restarting its
   * dismiss timer) instead of stacking; an identical message on it carries the count over. */
  pushToast: (message: string, level?: Toast["level"], channel?: string) => void;
  dismissToast: (id: number) => void;
}

// Monotonic, not array-derived: an id built off the last toast's own could be reused once a
// higher one is dismissed, and a replaced toast reusing an id would never remount its timer.
let nextToastId = 1;

export const createToastSlice: StateCreator<AppState, [], [], ToastSlice> = (set) => ({
  toasts: [],

  pushToast: (message, level = "error", channel) =>
    set((s) => {
      if (channel) {
        // Always replaces the channel's standing toast under a fresh id, identical message
        // included: a repeat must still remount to restart its dismiss timer.
        const standing = s.toasts.find((t) => t.channel === channel);
        const count =
          standing && standing.message === message && standing.level === level
            ? (standing.count ?? 1) + 1
            : undefined;
        const withoutStanding = standing ? s.toasts.filter((t) => t !== standing) : s.toasts;
        const id = nextToastId++;
        return { toasts: [...withoutStanding, { id, message, level, channel, count }].slice(-4) };
      }
      // No channel: today's behavior, an identical toast still on screen collapses in place
      // rather than stacking a second one, so a flaky poll can't flood the screen.
      const repeat = s.toasts.find((t) => t.message === message && t.level === level && !t.channel);
      if (repeat) {
        return {
          toasts: s.toasts.map((t) => (t === repeat ? { ...t, count: (t.count ?? 1) + 1 } : t)),
        };
      }
      const id = nextToastId++;
      // Cap the stack so a failing poll can't flood the screen.
      return { toasts: [...s.toasts, { id, message, level }].slice(-4) };
    }),
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
});
