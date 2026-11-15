import { forwardRef } from "react";

import { dismissLabel } from "@/lib/dismissLabel";
import { useStore } from "@/store";
import {
  type NotificationEntry,
  selectHoldingLevel,
  selectNewCount,
} from "@/store/slices/notifications";
import { LEVELS } from "@/store/slices/toasts";

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
      Notifications
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

function NotificationItem({ entry, ongoing }: { entry: NotificationEntry; ongoing: boolean }) {
  const dismiss = useStore((s) => s.dismissNotification);
  const { word, tone, mark } = LEVELS[entry.level];
  const at = new Date(entry.time);
  const time = at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return (
    <li className="flex items-start gap-2 py-2.5 pr-2">
      <span aria-hidden="true" className={`w-0.5 self-stretch rounded-full ${mark}`} />
      <div className="min-w-0 flex-1 pl-1">
        <p className={`break-words text-[13px] leading-snug ${tone}`}>{entry.sentence}</p>
        <p className="mt-1 text-[11px] text-tcip-muted tabular-nums">
          {word}, <time dateTime={at.toISOString()}>{time}</time>
          {ongoing ? ", ongoing" : ", ended"}
        </p>
      </div>
      <button
        className="h-6 w-6 shrink-0 rounded text-tcip-muted hover:bg-tcip-hover hover:text-tcip-fg"
        onClick={() => {
          dismiss(entry.id);
          document.getElementById(NOTIFICATIONS_HEADING_ID)?.focus();
        }}
        aria-label={`${dismissLabel(entry.sentence)} (${time})`}
      >
        ×
      </button>
    </li>
  );
}

/** The notifications area's content: every listed entry, newest first, each in one entry
 *  component with its level, time, sentence, whether its condition still holds and its dismiss
 *  control. */
export function NotificationsPane() {
  const entries = useStore((s) => s.notifications);
  const holding = useStore((s) => s.holding);

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
            <NotificationItem
              key={entry.id}
              entry={entry}
              ongoing={holding[entry.condition] === entry.id}
            />
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
  const live = announced && LEVELS[announced.level].live;
  return (
    <div className="sr-only">
      <div role="status" aria-live="polite">
        {announced && live === "polite" && <span key={announced.id}>{announced.sentence}</span>}
      </div>
      <div role="alert" aria-live="assertive">
        {announced && live === "assertive" && <span key={announced.id}>{announced.sentence}</span>}
      </div>
    </div>
  );
}
