import { forwardRef, useEffect, useRef } from "react";

import { dismissLabel } from "@/lib/dismissLabel";
import { useStore } from "@/store";
import {
  type NotificationEntry,
  selectHoldingLevel,
  selectNewCount,
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
          className={`h-2 w-2 rounded-full ${holdingLevel === "error" ? "bg-tcip-fp" : "bg-tcip-fn"}`}
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
  const at = new Date(entry.time);
  const time = at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return (
    <li
      className={`flex items-start gap-2 border-l-2 py-2.5 pl-3 pr-2 ${
        entry.level === "error" ? "border-tcip-fp" : "border-tcip-fn"
      }`}
    >
      <div className="min-w-0 flex-1">
        <p
          className={`break-words text-[13px] leading-snug ${
            entry.level === "error" ? "tcip-problem" : "text-tcip-fg"
          }`}
        >
          {entry.sentence}
        </p>
        <p className="mt-1 text-[11px] text-tcip-muted tabular-nums">
          <time dateTime={at.toISOString()}>{time}</time>
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
 *  component with its time, sentence, whether its condition still holds and its dismiss
 *  control. Escape closes the area. */
export function NotificationsPane({ onClose }: { onClose: () => void }) {
  const entries = useStore((s) => s.notifications);
  const holding = useStore((s) => s.holding);
  const area = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = area.current;
    if (!element) return;
    const closeOnEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    element.addEventListener("keydown", closeOnEscape);
    return () => element.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <section
      ref={area}
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
 *  raised, an error assertively and anything else politely, whether or not the area is open. */
export function NotificationsLiveRegion() {
  const announced = useStore((s) => s.announced);
  const assertive = announced?.level === "error";
  return (
    <div className="sr-only">
      <div role="status" aria-live="polite">
        {announced && !assertive && <span key={announced.id}>{announced.sentence}</span>}
      </div>
      <div role="alert" aria-live="assertive">
        {announced && assertive && <span key={announced.id}>{announced.sentence}</span>}
      </div>
    </div>
  );
}
