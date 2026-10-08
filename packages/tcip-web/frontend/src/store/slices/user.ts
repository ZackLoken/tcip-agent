import type { StateCreator } from "zustand";

import { PERSON_NAME_RULE } from "@/api/types.generated";
import type { AppState } from "@/store/appState";

export interface UserSlice {
  /** Current annotator/reviewer identity (persisted), stamped as created_by/accepted_by on
   *  everything this person authors. */
  user: string;
  setUser: (user: string) => void;
}

const personName = new RegExp(PERSON_NAME_RULE);

/** Whether ``name`` names a person: the backend's one rule (``PERSON_NAME_RULE``), the rule the
 *  open door refuses on, so a name the field admits is one the door admits and the door's own
 *  refusal of a name is unreachable from here. */
export const isAnnotatorName = (name: string): boolean => personName.test(name);

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
