import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

/** A note the agent pushed for one panel, carrying the id of the event that delivered it. */
export interface Banner {
  id: string;
  text: string;
}

interface BannerState {
  /** The latest banner per panel; a panel with no entry has none. */
  byPanel: Record<string, Banner | null>;
  /** Banner event ids the human closed. The backend replays its whole per-panel event ring on
   *  every reconnect, so a dismissal has to be keyed on the event, not just cleared: without
   *  this the same note reappears on the next reconnect or reload. */
  dismissed: Set<string>;
}

export interface BannerSlice {
  /** Agent-pushed per-tab notes (see TabBanner). Free text the agent chose, never a computed
   *  verdict about the data. */
  banners: BannerState;
  pushBanner: (panel: string, id: string, text: string) => void;
  dismissBanner: (id: string) => void;
}

export const createBannerSlice: StateCreator<AppState, [], [], BannerSlice> = (set) => ({
  banners: { byPanel: {}, dismissed: new Set<string>() },
  pushBanner: (panel, id, text) =>
    set((s) => ({
      banners: { ...s.banners, byPanel: { ...s.banners.byPanel, [panel]: { id, text } } },
    })),
  dismissBanner: (id) =>
    set((s) => ({
      banners: { ...s.banners, dismissed: new Set(s.banners.dismissed).add(id) },
    })),
});
