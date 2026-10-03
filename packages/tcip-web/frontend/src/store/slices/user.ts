import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface UserSlice {
  /** Current annotator/reviewer identity (persisted), stamped as created_by/accepted_by on
   *  everything this person authors. Changing it closes the open image visit under the person
   *  it was opened by and reopens it under the new one. */
  user: string;
  setUser: (user: string) => void;
}

export const createUserSlice: StateCreator<AppState, [], [], UserSlice> = (set, get) => ({
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
    const visiting = get().sessionTracking.currentImageName;
    get().closeSessionInterval();
    set({ user });
    if (visiting !== null) get().startImageSessionTracking(visiting);
  },
});
