import { Bell } from "lucide-react";
import { forwardRef, useId } from "react";

import { dismissLabel } from "@/lib/dismissLabel";
import { useStore } from "@/store";
import {
  type EntryAction,
  LEVELS,
  type NotificationEntry,
  selectHoldingLevel,
  selectNewCount,
  selectStanding,
  textOf,
} from "@/store/slices/notifications";

export const NOTIFICATIONS_HEADING_ID = "notifications-heading";

/** The top bar's notifications button: its mark counts the entries new since the area was last
 *  open and shows a dot while any condition holds, whether or not its entry was dismissed. */
export const NotificationsButton = forwardRef<
  HTMLButtonElement,
  { open: boolean; onToggle: () => void }
>(function NotificationsButton({ open, onToggle }, ref) {
  const fresh = useStore(selectNewCount);
  const holdingLevel = useStore(selectHoldingLevel);
  const name = [
    "Notifications",
    fresh ? `${fresh} new` : null,
    holdingLevel ? "a problem is ongoing" : null,
  ]
    .filter(Boolean)
    .join(", ");

  return (
    <button
      ref={ref}
      onClick={onToggle}
      aria-expanded={open}
      aria-label={name}
      title={name}
      className={`relative flex items-center gap-1.5 px-2.5 h-7 mr-1 rounded text-[13px] font-medium transition-colors ${
        open
          ? "bg-tcip-accent text-white"
          : "bg-transparent text-tcip-muted hover:text-tcip-fg hover:bg-tcip-hover"
      }`}
    >
      {holdingLevel && (
        <span
          aria-hidden="true"
          data-testid="condition-mark"
          className={`h-2 w-2 rounded-full ${LEVELS[holdingLevel].mark}`}
        />
      )}
      <Bell aria-hidden="true" className="tcip-icon" />
      {fresh > 0 && (
        <span
          aria-hidden="true"
          className="min-w-4 h-4 px-1 rounded-full bg-tcip-fg text-tcip-bg text-[11px] leading-4 text-center tabular-nums"
        >
          {fresh}
        </span>
      )}
    </button>
  );
});

/** Where an entry is shown: in the notifications area, or in place beside the control it stands
 *  for, where it is the entry standing on `entryKey`, `afterDismiss` puts focus back, `action`,
 *  when given, is the placing control's own, offered beside the entry's, and `textId`, when
 *  given, is the id its text takes so that control can be described by it. */
type Placement =
  | { placement: "area"; entry: NotificationEntry }
  | {
      placement: "in place";
      entryKey: string;
      afterDismiss: () => void;
      action?: EntryAction;
      textId?: string;
    };

/**
 * The one entry component. In the area it shows the entry's text, level, kind, target, time,
 * count, action and dismiss control, and whether a condition still holds; in place it shows the
 * entry standing on its key, while one is listed, with its text, count, action, the placing
 * control's action and dismiss control, and nothing when none stands.
 */
export function NotificationItem(props: Placement) {
  const standing = useStore((s) =>
    props.placement === "area" ? props.entry : selectStanding(props.entryKey)(s),
  );
  const holding = useStore((s) => s.holding);
  const dismiss = useStore((s) => s.dismissNotification);
  const ownId = useId();
  const textId = (props.placement === "in place" && props.textId) || ownId;
  if (!standing) return null;
  const entry = standing;
  const { word, tone, mark } = LEVELS[entry.level];
  const text = textOf(entry);
  const at = new Date(entry.time);
  const time = at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const count = entry.count > 1 ? `, ${entry.count} times` : "";
  const offer = ({ label, run }: EntryAction) => (
    <button
      type="button"
      className="tcip-btn h-6 shrink-0 px-2 text-[12px]"
      aria-describedby={textId}
      onClick={run}
    >
      {label}
    </button>
  );
  const action = entry.action && offer(entry.action);
  const dismissControl = (
    <button
      type="button"
      className="h-6 w-6 shrink-0 rounded text-tcip-muted hover:bg-tcip-hover hover:text-tcip-fg"
      onClick={() => {
        dismiss(entry.id);
        if (props.placement === "area") document.getElementById(NOTIFICATIONS_HEADING_ID)?.focus();
        else props.afterDismiss();
      }}
      aria-label={`${dismissLabel(text)} (${time})`}
    >
      ×
    </button>
  );

  if (props.placement === "in place") {
    return (
      <div data-testid="entry-in-place" className="flex items-start gap-2 text-[13px]">
        <p id={textId} className={`min-w-0 flex-1 break-words leading-6 ${tone}`}>
          {text}
          {count && <span className="text-tcip-muted">{count}</span>}
        </p>
        {action}
        {props.action && offer(props.action)}
        {dismissControl}
      </div>
    );
  }
  const meta = [word, entry.kind, entry.context.target].filter(Boolean).join(", ");
  return (
    <li className="flex items-start gap-2 py-2.5 pr-2">
      <span aria-hidden="true" className={`w-0.5 self-stretch rounded-full ${mark}`} />
      <div className="min-w-0 flex-1 pl-1">
        <p id={textId} className={`break-words text-[13px] leading-snug ${tone}`}>
          {text}
        </p>
        <p className="mt-1 text-[11px] text-tcip-muted tabular-nums">
          {meta}, <time dateTime={at.toISOString()}>{time}</time>
          {count}
          {entry.condition && (holding[entry.condition] === entry.id ? ", ongoing" : ", ended")}
        </p>
        {action && <div className="mt-2">{action}</div>}
      </div>
      {dismissControl}
    </li>
  );
}

/** The notifications area's content: every listed entry, newest first, each in the one entry
 *  component. */
export function NotificationsPane() {
  const entries = useStore((s) => s.notifications);

  return (
    <section
      aria-labelledby={NOTIFICATIONS_HEADING_ID}
      className="flex min-h-full flex-col bg-tcip-panel text-tcip-fg"
    >
      <h2
        id={NOTIFICATIONS_HEADING_ID}
        tabIndex={-1}
        className="px-3 pt-3 pb-2 text-[13px] font-semibold outline-none focus-visible:ring-2 focus-visible:ring-tcip-focus"
      >
        Notifications
      </h2>
      {entries.length ? (
        <ul className="flex flex-col gap-1 px-2 pb-3">
          {entries.toReversed().map((entry) => (
            <NotificationItem key={entry.id} placement="area" entry={entry} />
          ))}
        </ul>
      ) : (
        <p className="px-3 text-[13px] text-tcip-muted">
          Nothing to report. What the app needs you to know stays listed here until you dismiss it.
        </p>
      )}
    </section>
  );
}

/** The application's one standing live region: it announces each new entry once as it is
 *  raised, at its level's politeness, whether or not the area is open. */
export function NotificationsLiveRegion() {
  const announced = useStore((s) => s.announced);
  const spans = (live: "polite" | "assertive") =>
    announced
      .filter((e) => LEVELS[e.level].live === live)
      .map((e) => <span key={e.id}>{textOf(e)}</span>);
  return (
    <div className="sr-only">
      <div role="status" aria-live="polite">
        {spans("polite")}
      </div>
      <div role="alert" aria-live="assertive">
        {spans("assertive")}
      </div>
    </div>
  );
}
