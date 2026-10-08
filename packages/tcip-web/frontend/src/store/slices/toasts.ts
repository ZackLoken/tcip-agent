import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

/** What a toast says, held one way: its text, or the refusals a notice of refused work names
 *  (each an item and the reason given), never both; `textOf` reads either. */
type ToastContent = {
  level: "error" | "info" | "success";
  /** How many times this exact text (same level) has been pushed since it last appeared;
   * absent (or 1) for one that has never repeated. */
  count?: number;
  /** A channel name; a later push on the same channel replaces this toast under a fresh id
   * instead of stacking beside it. */
  channel?: string;
} & ({ message: string; refusals?: never } | { refusals: Refusal[]; message?: never });

export type Toast = ToastContent & { id: number };

/** The text `toast` shows: its message, or its refusals grouped under each reason. */
export function textOf(toast: ToastContent): string {
  return toast.refusals === undefined ? toast.message : refusalMessage(toast.refusals);
}

/** One refused item and the reason it was refused. */
export interface Refusal {
  item: string;
  reason: string;
}

/** The text of a notice of `refusals`: their items, grouped under each reason. */
function refusalMessage(refusals: Refusal[]): string {
  const reasons = [...new Set(refusals.map((r) => r.reason))];
  return reasons
    .map((reason) => {
      const items = refusals.filter((r) => r.reason === reason).map((r) => r.item);
      return `Not recorded (${reason}): ${items.join(", ")}`;
    })
    .join("; ");
}

export interface ToastSlice {
  /** Transient user-facing notifications (API failures, etc.). */
  toasts: Toast[];
  /** `channel` replaces the standing toast on that channel under a fresh id (restarting its
   * dismiss timer) instead of stacking; an identical message on it carries the count over. */
  pushToast: (message: string, level?: Toast["level"], channel?: string) => void;
  /** Add the refusal of `item` for `reason` to the one notice of refusals on `channel`: the
   *  standing notice's refusals and this one, or this one alone when none stands. */
  pushRefusal: (channel: string, item: string, reason: string) => void;
  dismissToast: (id: number) => void;
}

// Monotonic, not array-derived: an id built off the last toast's own could be reused once a
// higher one is dismissed, and a replaced toast reusing an id would never remount its timer.
let nextToastId = 1;

/** `toasts` with `toast` appended under a fresh id, the stack capped at four so a failing poll
 *  cannot flood the screen. */
function inserted(toasts: Toast[], toast: ToastContent): Toast[] {
  return [...toasts, { ...toast, id: nextToastId++ }].slice(-4);
}

/** `toasts` with `toast` standing on its channel in place of the one there (`inserted`, so
 *  under a fresh id: a repeat must still remount to restart its dismiss timer). */
function onChannel(toasts: Toast[], toast: ToastContent): Toast[] {
  return inserted(
    toasts.filter((t) => t.channel !== toast.channel),
    toast,
  );
}

export const createToastSlice: StateCreator<AppState, [], [], ToastSlice> = (set) => ({
  toasts: [],

  pushRefusal: (channel, item, reason) =>
    set((s) => {
      const standing = s.toasts.find((t) => t.channel === channel);
      const refusals = [...(standing?.refusals ?? []), { item, reason }];
      return { toasts: onChannel(s.toasts, { level: "error", channel, refusals }) };
    }),

  pushToast: (message, level = "error", channel) =>
    set((s) => {
      if (channel) {
        const standing = s.toasts.find((t) => t.channel === channel);
        const count =
          standing && textOf(standing) === message && standing.level === level
            ? (standing.count ?? 1) + 1
            : undefined;
        return { toasts: onChannel(s.toasts, { message, level, channel, count }) };
      }
      // With no channel, an identical toast still on screen collapses in place
      // rather than stacking a second one, so a flaky poll can't flood the screen.
      const repeat = s.toasts.find((t) => textOf(t) === message && t.level === level && !t.channel);
      if (repeat) {
        return {
          toasts: s.toasts.map((t) => (t === repeat ? { ...t, count: (t.count ?? 1) + 1 } : t)),
        };
      }
      return { toasts: inserted(s.toasts, { message, level }) };
    }),
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
});
