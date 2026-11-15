import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";
import type { Toast } from "@/store/slices/toasts";

/** A state of the application that holds over time: the state socket's lost connection, a
 *  served bundle its source would not build, and visits whose recording is unconfirmed. */
export type ConditionName = "connection" | "bundle" | "unconfirmed";

/** How loudly each condition is told, on the toasts' levels: the lost connection's pill dot was
 *  the failure red, the stale bundle's the caution gold, and the unconfirmed visits' notice an
 *  error toast. */
const CONDITION_LEVEL: Record<ConditionName, Toast["level"]> = {
  connection: "error",
  bundle: "info",
  unconfirmed: "error",
};

/** One entry of the notifications area: one occurrence of a condition, from when it began. */
export interface NotificationEntry {
  id: number;
  level: Toast["level"];
  /** The condition whose occurrence this entry reports. */
  condition: ConditionName;
  /** When the occurrence began, in milliseconds since the epoch. */
  time: number;
  /** What the condition states now, updated in place while it holds. */
  sentence: string;
}

export interface NotificationsSlice {
  /** The entries the notifications area lists, oldest first; a dismissed entry leaves it. */
  notifications: NotificationEntry[];
  /** Each condition holding now, with the id of the entry its occurrence raised; it stays here
   *  after that entry is dismissed and leaves when the condition ends. */
  holding: Partial<Record<ConditionName, number>>;
  /** The id of the newest entry the person has had the area open over; any later one is new. */
  seenThrough: number;
  /** The entry the live region announces: the one raised last, null before any. */
  announced: NotificationEntry | null;
  /** Hold `condition` stating `sentence`, or end it with null. A condition not holding raises a
   *  new entry, announced once; one holding updates its entry's sentence in place, or nothing
   *  once that entry is dismissed. */
  holdCondition: (condition: ConditionName, sentence: string | null) => void;
  dismissNotification: (id: number) => void;
  /** Count every entry listed now as seen. */
  markNotificationsSeen: () => void;
}

let nextEntryId = 1;

export const createNotificationsSlice: StateCreator<AppState, [], [], NotificationsSlice> = (
  set,
) => ({
  notifications: [],
  holding: {},
  seenThrough: 0,
  announced: null,

  holdCondition: (condition, sentence) =>
    set((s) => {
      const id = s.holding[condition];
      if (sentence === null) {
        if (id === undefined) return s;
        const { [condition]: _ended, ...holding } = s.holding;
        return { holding };
      }
      if (id !== undefined) {
        if (!s.notifications.some((e) => e.id === id && e.sentence !== sentence)) return s;
        return {
          notifications: s.notifications.map((e) => (e.id === id ? { ...e, sentence } : e)),
        };
      }
      const entry: NotificationEntry = {
        id: nextEntryId++,
        level: CONDITION_LEVEL[condition],
        condition,
        time: Date.now(),
        sentence,
      };
      return {
        notifications: [...s.notifications, entry],
        holding: { ...s.holding, [condition]: entry.id },
        announced: entry,
      };
    }),

  dismissNotification: (id) =>
    set((s) => ({ notifications: s.notifications.filter((e) => e.id !== id) })),

  markNotificationsSeen: () =>
    set((s) => {
      const newest = s.notifications.at(-1)?.id ?? 0;
      return newest > s.seenThrough ? { seenThrough: newest } : s;
    }),
});

/** How many listed entries the person has not had the area open over. */
export const selectNewCount = (s: AppState): number =>
  s.notifications.filter((e) => e.id > s.seenThrough).length;

/** The loudest level among the conditions holding now, null while none holds. */
export const selectHoldingLevel = (s: AppState): Toast["level"] | null => {
  const levels = (Object.keys(s.holding) as ConditionName[]).map((c) => CONDITION_LEVEL[c]);
  if (!levels.length) return null;
  return levels.includes("error") ? "error" : levels[0];
};
