import type { StateCreator } from "zustand";

import type { InferenceJob } from "@/api/types.generated";
import type { AppState } from "@/store/appState";

/** How loud a message to the person is. */
export type MessageLevel = "error" | "warning" | "info";

/** How insistently a live region announces what it is given. */
export type Politeness = "assertive" | "polite";

/** How each level is shown and told, read by every message component: its word, its text tone,
 *  the color of its mark, and the politeness its live region announces it with. */
export const LEVELS: Record<
  MessageLevel,
  { word: string; tone: string; mark: string; live: Politeness }
> = {
  error: { word: "Error", tone: "tcip-problem", mark: "bg-tcip-fp", live: "assertive" },
  warning: { word: "Warning", tone: "text-tcip-warn", mark: "bg-tcip-warn", live: "polite" },
  info: { word: "Information", tone: "text-tcip-fg", mark: "bg-tcip-fn", live: "polite" },
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

/** An action an entry offers: its label, the visible text of its button; what it acts on, when
 *  stated, said after the label in the button's accessible name; and what it runs. */
export interface EntryAction {
  label: string;
  actsOn?: string;
  run: () => void;
}

/** What an entry says: a sentence, or the refusals a notice of refused work names, never both;
 *  `textOf` reads either. */
type EntryText = { sentence: string; refusals?: never } | { refusals: Refusal[]; sentence?: never };

/** The kinds of entry a producer raises, beside a condition's occurrence. */
type RaisedKind = "failure" | "notice" | "job";

/** What an entry is: a condition's occurrence, naming its condition; or a failure, a notice or a
 *  job, standing on the key its producer gave, if any, until the key is released, set aside or
 *  resolved or a raise stating another text takes it. It offers its action and runs `onDismiss`
 *  when dismissed until it is taken off a key, keeps the inference job it is about, if any, as
 *  long as it is listed, and is `resolved` from the moment its producer says its cause ended
 *  until it is raised again. A failure set aside for a new attempt names the key it was taken
 *  from (`setAsideFrom`) until that attempt's outcome reaches the key. */
type EntryStanding =
  | {
      kind: "condition";
      condition: ConditionName;
      key?: never;
      action?: never;
      onDismiss?: never;
      inferenceJob?: never;
      resolved?: never;
      setAsideFrom?: never;
    }
  | {
      kind: RaisedKind;
      key?: string;
      action?: EntryAction;
      onDismiss?: () => void;
      inferenceJob?: InferenceJob;
      resolved: boolean;
      setAsideFrom?: string;
      condition?: never;
    };

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
    /** What the live region announces for it: its text and level as first raised, which no later
     *  change of the entry rewrites. */
    announcement: { text: string; level: MessageLevel };
  };

/** What a producer raises: a failure, a notice or a job, stating a sentence or one refusal, what
 *  dismissing its entry does beside taking it off the list (`onDismiss`), and the record of the
 *  inference job it is about, when it is about one (`inferenceJob`). A job states whether an
 *  entry already standing for it is told again (`retell`). */
export type Raised = {
  level: MessageLevel;
  text: { sentence: string } | { refusal: Refusal };
  context: EntryContext;
  key?: string;
  action?: EntryAction;
  onDismiss?: () => void;
  inferenceJob?: InferenceJob;
} & ({ kind: Exclude<RaisedKind, "job">; retell?: never } | { kind: "job"; retell: boolean });

export interface NotificationsSlice {
  /** The entries the notifications area lists, oldest first; a dismissed entry leaves it. */
  notifications: NotificationEntry[];
  /** Each condition holding now, with the id of the entry its occurrence raised; it stays here
   *  after that entry is dismissed and leaves when the condition ends. */
  holding: Partial<Record<ConditionName, number>>;
  /** The id of the newest entry the person has had the area open over; any later one is new. */
  seenThrough: number;
  /** Hold `condition` stating `sentence`, or end it with null. A condition not holding raises a
   *  new entry, announced once; one holding updates its entry's sentence in place, or nothing
   *  once that entry is dismissed. */
  holdCondition: (condition: ConditionName, sentence: string | null) => void;
  /**
   * Raise `raised`. On a key a listed entry stands on, a job updates that entry in place, or
   * with `retell` replaces it by a new entry on the same key, a refusal is added to its
   * refusals, a failure or notice stating the same text raises its count, and one stating
   * another text is a new entry that takes the key, the earlier one staying listed. With no key,
   * or none standing on it, a listed entry standing on no key of the same kind, level, text and
   * context raises its count and takes the key. Anything else is a new entry, announced once. An
   * entry raised again carries the latest raise's action, `onDismiss` and inference job, and is
   * no longer resolved. A raise on a key ends every entry's setting aside from that key, each
   * staying listed as it is.
   */
  raiseNotification: (raised: Raised) => void;
  /** Free `key` for a reason other than the end of its entry's cause: the entry standing on it
   *  stays listed until dismissed, standing on no key, offering no action and running nothing
   *  when dismissed, and keeping its inference job. Nothing for a key no entry stands on. */
  releaseNotification: (key: string) => void;
  /** Free `key` for a new attempt at what its entry's producer failed at: the entry standing on
   *  it is released as `releaseNotification` releases it and, when it is a failure, names `key`
   *  as the key it was set aside from, so the attempt's success marks it resolved and its
   *  failure leaves it as it is. Nothing for a key no entry stands on. */
  setAsideNotification: (key: string) => void;
  /** Say the cause of what `key` stood for ended: the entry standing on it is released as
   *  `releaseNotification` releases it, and it and every entry set aside from `key` are marked
   *  resolved. Nothing when no entry stands on or was set aside from `key`. */
  resolveNotification: (key: string) => void;
  /** Take the entry `id` names off the list, then run its `onDismiss`; nothing for an id no
   *  listed entry carries. */
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

/** `entry` standing on no key, set aside from none, offering no action and running nothing when
 *  dismissed. */
const unkeyed = (entry: NotificationEntry): NotificationEntry =>
  ({
    ...entry,
    key: undefined,
    setAsideFrom: undefined,
    action: undefined,
    onDismiss: undefined,
  }) as NotificationEntry;

/** The change to `s` that passes each listed entry `picked` names through `change`; `s` itself
 *  when it names none. */
const changed = (
  s: AppState,
  picked: (e: NotificationEntry) => boolean,
  change: (e: NotificationEntry) => NotificationEntry,
) =>
  s.notifications.some(picked)
    ? { notifications: s.notifications.map((e) => (picked(e) ? change(e) : e)) }
    : s;

let nextEntryId = 1;

export const createNotificationsSlice: StateCreator<AppState, [], [], NotificationsSlice> = (
  set,
  get,
) => ({
  notifications: [],
  holding: {},
  seenThrough: 0,

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
      const level = CONDITION_LEVEL[condition];
      const entry: NotificationEntry = {
        id: nextEntryId++,
        kind: "condition",
        condition,
        level,
        time: Date.now(),
        count: 1,
        context: {},
        sentence,
        announcement: { text: sentence, level },
      };
      return {
        notifications: [...s.notifications, entry],
        holding: { ...s.holding, [condition]: entry.id },
      };
    }),

  raiseNotification: ({
    kind,
    level,
    text,
    context,
    key,
    action,
    onDismiss,
    inferenceJob,
    retell,
  }) =>
    set((s) => {
      const said: EntryText =
        "refusal" in text ? { refusals: [text.refusal] } : { sentence: text.sentence };
      const listed =
        key === undefined
          ? s.notifications
          : s.notifications.map((e) =>
              e.setAsideFrom === key ? ({ ...e, setAsideFrom: undefined } as NotificationEntry) : e,
            );
      const standing =
        (key === undefined ? undefined : listed.find((e) => e.key === key)) ??
        listed.find(
          (e) =>
            e.key === undefined &&
            e.kind === kind &&
            e.level === level &&
            textOf(e) === textOf(said) &&
            sameContext(e.context, context),
        );
      const replaced = (next: Partial<NotificationEntry>) => ({
        notifications: listed.map((e) =>
          e === standing
            ? ({
                ...e,
                key,
                setAsideFrom: undefined,
                action,
                onDismiss,
                inferenceJob,
                resolved: false,
                ...next,
              } as NotificationEntry)
            : e,
        ),
      });
      const standingJob = kind === "job" && !!standing;
      if (standingJob && !retell) {
        return replaced({ level, context, ...said });
      }
      if (!standingJob && standing?.refusals && "refusal" in text) {
        return replaced({ refusals: [...standing.refusals, text.refusal] });
      }
      if (!standingJob && standing && textOf(standing) === textOf(said)) {
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
        onDismiss,
        inferenceJob,
        resolved: false,
        announcement: { text: textOf(said), level },
        ...said,
      } as NotificationEntry;
      const earlier = listed.flatMap((e) =>
        e !== standing ? [e] : standingJob ? [] : [unkeyed(e)],
      );
      return { notifications: [...earlier, entry] };
    }),

  releaseNotification: (key) => set((s) => changed(s, (e) => e.key === key, unkeyed)),

  setAsideNotification: (key) =>
    set((s) =>
      changed(
        s,
        (e) => e.key === key,
        (e) =>
          ({
            ...unkeyed(e),
            setAsideFrom: e.kind === "failure" ? key : undefined,
          }) as NotificationEntry,
      ),
    ),

  resolveNotification: (key) =>
    set((s) =>
      changed(
        s,
        (e) => e.key === key || e.setAsideFrom === key,
        (e) => ({ ...unkeyed(e), resolved: true }) as NotificationEntry,
      ),
    ),

  dismissNotification: (id) => {
    const entry = get().notifications.find((e) => e.id === id);
    if (!entry) return;
    set((s) => ({ notifications: s.notifications.filter((e) => e !== entry) }));
    entry.onDismiss?.();
  },

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
