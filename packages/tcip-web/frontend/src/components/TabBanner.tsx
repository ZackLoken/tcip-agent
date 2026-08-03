import { useStore } from "@/store";

/** Past this, a pushed note would start displacing the tab it sits above. */
const MAX_CHARS = 240;

/**
 * The agent's current note for the tab in view: free text it chose to say, shown as a quiet
 * aside. Deliberately unlike the validity markers elsewhere in the app (which are computed from
 * evidence on disk): this is someone talking, not a verdict about the data. Rendered as plain
 * text, never markup.
 *
 * A pure store read, mounted once by App: the panel subscriptions live there, so this never
 * opens one of its own.
 */
export function TabBanner() {
  const panel = useStore((s) => s.gui.active_tab);
  const banner = useStore((s) => s.banners.byPanel[panel] ?? null);
  const dismissed = useStore((s) => s.banners.dismissed);
  const dismissBanner = useStore((s) => s.dismissBanner);

  if (!banner || !banner.text.trim() || dismissed.has(banner.id)) return null;
  const text =
    banner.text.length > MAX_CHARS ? `${banner.text.slice(0, MAX_CHARS).trimEnd()}…` : banner.text;

  return (
    <div className="flex items-start gap-3 border-l-2 border-tcip-accent/40 pl-3 pr-4 py-2 text-[11px] leading-relaxed text-tcip-muted">
      <span className="flex-1">{text}</span>
      <button
        type="button"
        className="shrink-0 text-tcip-muted hover:text-tcip-fg"
        title="Dismiss this note"
        aria-label="Dismiss this note"
        onClick={() => dismissBanner(banner.id)}
      >
        ✕
      </button>
    </div>
  );
}
