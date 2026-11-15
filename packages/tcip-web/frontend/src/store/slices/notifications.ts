import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

/** How loud a message to the person is. */
export type MessageLevel = "error" | "warning" | "info" | "success";

/** How each level is shown and told, read by every message component: its word, its text tone,
 *  the color of its mark, and the politeness its live region announces it with. */
export const LEVELS: Record<
  MessageLevel,
  { word: string; tone: string; mark: string; live: "assertive" | "polite" }
> = {
  error: { word: "Error", tone: "tcip-problem", mark: "bg-tcip-fp", live: "assertive" },
  warning: { word: "Warning", tone: "text-tcip-warn", mark: "bg-tcip-warn", live: "polite" },
  info: { word: "Information", tone: "text-tcip-fg", mark: "bg-tcip-fn", live: "polite" },
  success: { word: "Done", tone: "text-tcip-fg", mark: "bg-tcip-tp", live: "polite" },
};

/** A state of the application that holds over time: the state socket's lost connection, a
 *  served bundle its source would not build, and visits whose recording is unconfirmed. */
export type ConditionName = "connection" | "bundle" | "unconfirmed";

/** How loudly each condition is told. */
const CONDITION_LEVEL: Record<ConditionName, MessageLevel> = {
  connection: "error",
  bundle: "info",
  unconfirmed: "error",
};

/** One refused item and the reason it was refused. */
export interface Refusal {
  item: string;
  reason: string;
}

/** What an entry concerns, each part where its producer has one: the id of the project, and the
 *  name of the target within it (a project's display name, a capture), as the person reads it. */
export interface EntryContext {
  project?: string;
  target?: string;
}

/** An action an entry offers: its label and what it runs. */
export interface EntryAction {
  label: string;
  run: () => void;
}

/** What an entry says: a sentence, or the refusals a notice of refused work names, never both;
 *  `textOf` reads either. */
type EntryText = { sentence: string; refusals?: never } | { refusals: Refusal[]; sentence?: never };

/** The kinds of entry a producer raises, beside a condition's occurrence. */
type RaisedKind = "failure" | "notice" | "job";

/** What an entry is: a condition's occurrence, naming its condition; or a failure, a notice or a
 *  job, standing on the key its producer gave until the key is released or a raise stating
 *  another text takes it, and offering its action only while it stands there. */
type EntryStanding =
  | { kind: "condition"; condition: ConditionName; key?: never; action?: never }
  | { kind: RaisedKind; key?: string; action?: EntryAction; condition?: never };

/** One entry of the notifications area. */
export type NotificationEntry = EntryText &
  EntryStanding & {
    id: number;
    level: MessageLevel;
    /** When it was first raised, in milliseconds since the epoch. */
    time: number;
    /** How many times it was raised. */
    count: number;
    context: EntryContext;
  };

/** What a producer raises: a failure, a notice or a job, stating a sentence or one refusal. */
export interface Raised {
  kind: RaisedKind;
  level: MessageLevel;
  text: { sentence: string } | { refusal: Refusal };
  context: EntryContext;
  key?: string;
  action?: EntryAction;
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
  /**
   * Raise `raised`. On a key a listed entry stands on, a job updates that entry in place, a
   * refusal is added to its refusals, a failure or notice stating the same text raises its count,
   * and one stating another text is a new entry that takes the key, the earlier one staying
   * listed. With no key, or none standing on it, a listed entry standing on no key of the same
   * kind, level, text and context raises its count and takes the key. Anything else is a new
   * entry, announced once. An entry raised again carries the latest raise's action.
   */
  raiseNotification: (raised: Raised) => void;
  /** End what stands on `key`: its entry stays listed until dismissed, standing on no key and
   *  offering no action. */
  releaseNotification: (key: string) => void;
  dismissNotification: (id: number) => void;
  /** Count every entry listed now as seen. */
  markNotificationsSeen: () => void;
}

/** The text of a notice of `refusals`: their items, grouped under each reason. */
function refusalText(refusals: Refusal[]): string {
  const reasons = [...new Set(refusals.map((r) => r.reason))];
  return reasons
    .map((reason) => {
      const items = refusals.filter((r) => r.reason === reason).map((r) => r.item);
      return `Not recorded (${reason}): ${items.join(", ")}`;
    })
    .join("; ");
}

/** The text `entry` shows: its sentence, or its refusals grouped under each reason. */
export function textOf(entry: EntryText): string {
  return entry.refusals === undefined ? entry.sentence : refusalText(entry.refusals);
}

const sameContext = (a: EntryContext, b: EntryContext): boolean =>
  a.project === b.project && a.target === b.target;

/** The listed entry standing on `key`, null when none does. */
export const selectStanding =
  (key: string) =>
  (s: AppState): NotificationEntry | null =>
    s.notifications.find((e) => e.key === key) ?? null;

/** `entry` standing on no key, so offering no action. */
const unkeyed = (entry: NotificationEntry): NotificationEntry =>
  ({ ...entry, key: undefined, action: undefined }) as NotificationEntry;

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
          notifications: s.notifications.map((e) =>
            e.id === id ? ({ ...e, sentence } as NotificationEntry) : e,
          ),
        };
      }
      const entry: NotificationEntry = {
        id: nextEntryId++,
        kind: "condition",
        condition,
        level: CONDITION_LEVEL[condition],
        time: Date.now(),
        count: 1,
        context: {},
        sentence,
      };
      return {
        notifications: [...s.notifications, entry],
        holding: { ...s.holding, [condition]: entry.id },
        announced: entry,
      };
    }),

  raiseNotification: ({ kind, level, text, context, key, action }) =>
    set((s) => {
      const said: EntryText =
        "refusal" in text ? { refusals: [text.refusal] } : { sentence: text.sentence };
      const standing =
        (key === undefined ? null : selectStanding(key)(s)) ??
        s.notifications.find(
          (e) =>
            e.key === undefined &&
            e.kind === kind &&
            e.level === level &&
            textOf(e) === textOf(said) &&
            sameContext(e.context, context),
        );
      const replaced = (next: Partial<NotificationEntry>) => ({
        notifications: s.notifications.map((e) =>
          e === standing ? ({ ...e, key, action, ...next } as NotificationEntry) : e,
        ),
      });
      if (standing && kind === "job") {
        return replaced({ level, context, ...said });
      }
      if (standing?.refusals && "refusal" in text) {
        return replaced({ refusals: [...standing.refusals, text.refusal] });
      }
      if (standing && textOf(standing) === textOf(said)) {
        return replaced({ count: standing.count + 1 });
      }
      const entry = {
        id: nextEntryId++,
        kind,
        key,
        level,
        time: Date.now(),
        count: 1,
        context,
        action,
        ...said,
      } as NotificationEntry;
      return {
        notifications: [...s.notifications.map((e) => (e === standing ? unkeyed(e) : e)), entry],
        announced: entry,
      };
    }),

  releaseNotification: (key) =>
    set((s) =>
      selectStanding(key)(s)
        ? { notifications: s.notifications.map((e) => (e.key === key ? unkeyed(e) : e)) }
        : s,
    ),

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
export const selectHoldingLevel = (s: AppState): MessageLevel | null => {
  const levels = (Object.keys(s.holding) as ConditionName[]).map((c) => CONDITION_LEVEL[c]);
  if (!levels.length) return null;
  return levels.includes("error") ? "error" : levels[0];
};
