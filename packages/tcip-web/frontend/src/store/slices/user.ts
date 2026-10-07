import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface UserSlice {
  /** Current annotator/reviewer identity (persisted), stamped as created_by/accepted_by on
   *  everything this person authors. */
  user: string;
  setUser: (user: string) => void;
}

/** Whether ``name`` names a person; whitespace is no name. */
export const isAnnotatorName = (name: string): boolean => name.trim().length > 0;

export const selectAnnotatorNamed = (s: AppState): boolean => isAnnotatorName(s.user);

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
