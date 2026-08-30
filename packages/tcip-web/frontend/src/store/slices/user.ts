import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface UserSlice {
  /** Current annotator/reviewer identity (persisted). Stamped as created_by/accepted_by on
   * everything this person authors; set on the workspace page, shown in the status bar. */
  user: string;
  setUser: (user: string) => void;
}

export const createUserSlice: StateCreator<AppState, [], [], UserSlice> = (set) => ({
  user: (() => {
    try {
      return localStorage.getItem("tcip.user") ?? "";
    } catch {
      return "";
    }
  })(),
  setUser: (user) => {
    try {
      localStorage.setItem("tcip.user", user);
    } catch {
      /* preference just won't persist */
    }
    set({ user });
  },
});
